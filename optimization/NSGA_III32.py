# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 23:17:34 2025

@author: 92593
"""

# -*- coding: utf-8 -*-
"""
NSGA-III on DTLZ2 without pymop.factory
--------------------------------------

This is a drop-in reproduction of your script that removes the dependency on
`pymop.factory`. It implements a minimal DTLZ2 problem class locally with the
same methods you used: `evaluate(...)` (returning only F) and `pareto_front(...)`.

- Problem definition matches DTLZ2 from the original PyMOP formulations.
- `evaluate` accepts a single individual (1D list/array) or a batch (2D array).
- `pareto_front` projects reference directions onto the unit L2 sphere, which
  is the analytical front for DTLZ2 when g=0.

Everything else (DEAP setup, NSGA-III operators, plotting, IGD) is unchanged.
"""
from pymoo.indicators.hv import HV
from math import factorial
from math import comb
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import shutil
import os
import subprocess
import pandas as pd
import pickle
import time

# ==== [ADD] 归一化与HV计算的小工具 ====
def _normalize_max_objectives(F, ideal, nadir, eps=1e-12):
    """
    F: (N, M) ，四个目标均为最大化。
    ideal/nadir: (M,) 固定或逐步冻结的下/上界（单位同 F）。
    返回 [0,1] 归一化到“越大越好”的空间。
    """
    F = np.asarray(F, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)
    norm = (F - ideal) / (nadir - ideal + eps)
    return np.clip(norm, 0.0, 1.0)

def compute_norm_hv_max(F, ideal, nadir):
    """
    在最大化场景下的“尺度无关 HV”：
    1) 先做 [0,1] 归一化(越大越好)；
    2) 再转成最小化以适配 HV 实现：G = 1 - norm, 参照点 ref = 1。
    返回标量 HV \in [0,1]。
    """
    norm = _normalize_max_objectives(F, ideal, nadir)
    G = 1.0 - norm
    hv = HV(ref_point=np.ones(G.shape[1], dtype=float))
    return float(hv.do(G))


# ---------------------------
# Minimal DTLZ2 implementation
# ---------------------------
class DTLZ2Problem:
    """Minimal DTLZ2 with the interface used in the original script.

    Attributes
    ----------
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives (M).
    xl, xu : float or array
        Lower/upper bounds per decision variable (both 0/1 here).
    """

    def __init__(self, n_var: int, n_obj: int):
        if n_var < n_obj:
            raise ValueError("n_var must be >= n_obj for DTLZ2 (n_var = M-1 + k)")
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.k = self.n_var - self.n_obj + 1

    # --- helpers ---
    @staticmethod
    def _to_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return x

    # --- API: evaluate ---
    def evaluate(self, X, *_, return_values_of=("F",), **__):
        """Evaluate DTLZ2.

        Parameters
        ----------
        X : array-like of shape (n_var,) or (n_samples, n_var)
            Decision vectors.
        return_values_of : tuple/list
            Kept for signature compatibility; only "F" is supported.

        Returns
        -------
        F : ndarray of shape (n_obj,) if input is 1D, else (n_samples, n_obj)
        """
        X = self._to_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_var:
            raise ValueError(f"Input dimension {X.shape[1]} != n_var {self.n_var}")
        #由X计算目标函数（yield,WUE,NUE）
        PlantDoy=122 #播种日期
        HarveDoy=275 #收获日期
        LenGroPe=HarveDoy-PlantDoy
        #构建字典容器，存储写入.MZX文件的数据
        MZXData={}
        MZXData['IrrFre']  =round(X[0,0]) #灌溉次数
        MZXData['FerTotal']  =150+round(X[0,31]-1)*30 #总施肥量
        
        #1-12次灌溉日期
        # ---------- 方案A：用“绝对位置”生成灌水日期（排序 + 最小间隔修复） ----------
        min_gap = 5  # 相邻灌水至少间隔多少天（可调：3/5/7）
        
        # 取 10 个“日期比例因子”（X[0,1]~X[0,10]）
        u = np.array(X[0, 1:11], dtype=float)
        
        # 限制到 [0,1]，更稳（你的边界本来是0.05~0.9）
        u = np.clip(u, 0.0, 1.0)
        
        # 排序：保证日期从早到晚
        u_sorted = np.sort(u)
        
        # 映射到 DOY（绝对位置）
        dates = PlantDoy + np.round(u_sorted * LenGroPe).astype(int)
        
        # 1) 前向修复：保证递增 + 最小间隔
        dates[0] = max(dates[0], PlantDoy)
        for i in range(1, len(dates)):
            if dates[i] < dates[i-1] + min_gap:
                dates[i] = dates[i-1] + min_gap
        
        # 2) 如果最后超过收获日：从后往前压回去，仍保持最小间隔
        if dates[-1] > HarveDoy:
            dates[-1] = HarveDoy
            for i in range(len(dates) - 2, -1, -1):
                dates[i] = min(dates[i], dates[i+1] - min_gap)
            # 再前向扫一遍，防止压回后出现问题
            dates[0] = max(dates[0], PlantDoy)
            for i in range(1, len(dates)):
                dates[i] = max(dates[i], dates[i-1] + min_gap)
        
        # 3) 最终裁剪
        dates = np.clip(dates, PlantDoy, HarveDoy)
        
        # 写回 MZXData
        for i in range(1, 11):
            MZXData[f'IrrDate{i}'] = int(dates[i-1])
        # -------------------------------------------------------------------------

        #1-10次灌溉量
        MZXData['IrrAmount1']=25+round(X[0,11]-1)*5
        MZXData['IrrAmount2']=25+round(X[0,12]-1)*5
        MZXData['IrrAmount3']=25+round(X[0,13]-1)*5
        MZXData['IrrAmount4']=25+round(X[0,14]-1)*5
        MZXData['IrrAmount5']=25+round(X[0,15]-1)*5
        MZXData['IrrAmount6']=25+round(X[0,16]-1)*5
        MZXData['IrrAmount7']=25+round(X[0,17]-1)*5
        MZXData['IrrAmount8']=25+round(X[0,18]-1)*5
        MZXData['IrrAmount9']=25+round(X[0,19]-1)*5
        MZXData['IrrAmount10']=25+round(X[0,20]-1)*5
        #1-10次施肥量,初始值为0
        MZXData['FerRate1']=0        
        MZXData['FerRate2']=0
        MZXData['FerRate3']=0
        MZXData['FerRate4']=0
        MZXData['FerRate5']=0
        MZXData['FerRate6']=0
        MZXData['FerRate7']=0
        MZXData['FerRate8']=0
        MZXData['FerRate9']=0
        MZXData['FerRate10']=0
        #是否施肥
        MZXData['FerBool1']=round(X[0,21])
        MZXData['FerBool2']=round(X[0,22])
        MZXData['FerBool3']=round(X[0,23])
        MZXData['FerBool4']=round(X[0,24])
        MZXData['FerBool5']=round(X[0,25])
        MZXData['FerBool6']=round(X[0,26])
        MZXData['FerBool7']=round(X[0,27])
        MZXData['FerBool8']=round(X[0,28])
        MZXData['FerBool9']=round(X[0,29])
        MZXData['FerBool10']=round(X[0,30])
        if MZXData['IrrFre']==1:#如果灌溉量为1次
            #后9次灌溉量为0
            MZXData['IrrAmount2']=0
            MZXData['IrrAmount3']=0
            MZXData['IrrAmount4']=0
            MZXData['IrrAmount5']=0
            MZXData['IrrAmount6']=0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后9次施肥量为0，第一次施肥与否布尔值判断
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n
        elif MZXData['IrrFre']==2:#如果灌溉量为2次
            #后8次灌溉量为0
            MZXData['IrrAmount3']=0
            MZXData['IrrAmount4']=0
            MZXData['IrrAmount5']=0
            MZXData['IrrAmount6']=0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后8次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n
        elif MZXData['IrrFre']==3:#如果灌溉量为3次
            #后7次灌溉量为0
            MZXData['IrrAmount4']=0
            MZXData['IrrAmount5']=0
            MZXData['IrrAmount6']=0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后7次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n
        elif MZXData['IrrFre']==4:#如果灌溉量为4次
            #后6次灌溉量为0
            MZXData['IrrAmount5']=0
            MZXData['IrrAmount6']=0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后6次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n
        elif MZXData['IrrFre']==5:#如果灌溉量为5次
            #后5次灌溉量为0
            MZXData['IrrAmount6']=0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后5次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n
        elif MZXData['IrrFre']==6:#如果灌溉量为6次
            #后4次灌溉量为0
            MZXData['IrrAmount7']=0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后4次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n          
        elif MZXData['IrrFre']==7:#如果灌溉量为7次
            #后3次灌溉量为0
            MZXData['IrrAmount8']=0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后3次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n        
        elif MZXData['IrrFre']==8:#如果灌溉量为8次
            #后2次灌溉量为0
            MZXData['IrrAmount9']=0
            MZXData['IrrAmount10']=0
            #后2次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n                
        elif MZXData['IrrFre']==9:#如果灌溉量为9次
            #后1次灌溉量为0
            MZXData['IrrAmount10']=0
            #后1次施肥量为0
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n    
        else:
            # 统计 FerBool 中等于 1 的数量
            bool_keys = [f"FerBool{i}" for i in range(1, MZXData['IrrFre']+1)]
            n = sum(MZXData[k] for k in bool_keys)
            # 如果 n>0，就给对应的 FerRate 赋值 360/n
            if n > 0:
                for i in range(1, MZXData['IrrFre']+1):
                    if MZXData[f"FerBool{i}"] == 1:
                        MZXData[f"FerRate{i}"] = MZXData['FerTotal'] / n       
            
        #将MZXData中的数据写入.MZX文件
        MZXName="ORDO2302.MZX"
        if os.path.exists(MZXName):# 判断文件是否存在
            os.remove(MZXName)        # 如果存在，就删除
        shutil.copy2(f"./MZXFile/{MZXName}", ".")
        # 读入全部行
        with open(MZXName, "r", encoding="utf-8", newline="") as f:
            lines = f.readlines()
        #================================写入灌溉量=======================================
        #找出灌溉量为不为0的键
        NonZeroIrr=[k for k, v in MZXData.items() if k.startswith("IrrAmount") and v != 0]        
        #找出灌溉量为0的键
        ZeroIrr = [k for k, v in MZXData.items() if k.startswith("IrrAmount") and v == 0]
        #灌溉量不为0的数值写入
        for i1 in range(1,len(NonZeroIrr)+1):
            lines[47+i1]=lines[47+i1][0:3]+MZXName[4:6]+str(MZXData['IrrDate'+str(i1)])+lines[47+i1][8:18]+str(MZXData[NonZeroIrr[i1-1]])+'\n'
        with open(MZXName, "w", encoding="utf-8", newline="") as f1:
            f1.writelines(lines)     

        #灌溉量为0的数值写入
        if len(ZeroIrr)>0:
            for i2 in range(1,len(ZeroIrr)+1):
                lines[47+i1+i2]=lines[47+i1+i2][0:3]+MZXName[4:6]+str(MZXData['IrrDate'+str(i1+i2)])+lines[47+i1+i2][8:18]+' '+str(MZXData[ZeroIrr[i2-1]])+'\n'
            with open(MZXName, "w", encoding="utf-8", newline="") as f1:
                f1.writelines(lines)
        #================================写入施肥量=======================================
        # print(1)
        #写入施肥日期（同灌溉日期）
        for i3 in range(1,11):
            lines[60+i3]=lines[60+i3][0:3]+MZXName[4:6]+str(MZXData['IrrDate'+str(i3)])+lines[60+i3][8:]
        with open(MZXName, "w", encoding="utf-8", newline="") as f1:
            f1.writelines(lines)   
        # 写入施肥量
        for i4 in range(1,11):
            match i4:
                case 1:
                    Fer=str(round(MZXData['FerRate1']))
                    Irr=str(round(MZXData['IrrAmount1']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 2:
                    Fer=str(round(MZXData['FerRate2']))
                    Irr=str(round(MZXData['IrrAmount2']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]                   
                case 3:
                    Fer=str(round(MZXData['FerRate3']))
                    Irr=str(round(MZXData['IrrAmount3']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]                    
                case 4:
                    Fer=str(round(MZXData['FerRate4']))
                    Irr=str(round(MZXData['IrrAmount4']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]                   
                case 5:
                    Fer=str(round(MZXData['FerRate5']))
                    Irr=str(round(MZXData['IrrAmount5']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 6:
                    Fer=str(round(MZXData['FerRate6']))
                    Irr=str(round(MZXData['IrrAmount6']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 7:
                    Fer=str(round(MZXData['FerRate7']))
                    Irr=str(round(MZXData['IrrAmount7']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 8:
                    Fer=str(round(MZXData['FerRate8']))
                    Irr=str(round(MZXData['IrrAmount8']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 9:
                    Fer=str(round(MZXData['FerRate9']))
                    Irr=str(round(MZXData['IrrAmount9']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]
                case 10:
                    Fer=str(round(MZXData['FerRate10']))
                    Irr=str(round(MZXData['IrrAmount10']/10))#mm→cm
                    if len(Fer)==1:#施肥量为0
                        lines[60+i4]=lines[60+i4][0:25]+str(0)+lines[60+i4][26:30]+' '+Fer+lines[60+i4][32:]
                    elif len(Fer)==2:#施肥量为2位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:30]+    Fer+lines[60+i4][32:]
                    elif len(Fer)==3:#施肥量为3位数
                        lines[60+i4]=lines[60+i4][0:25]+Irr+   lines[60+i4][26:29]+    Fer+lines[60+i4][32:]               
        #写入文件
        with open(MZXName, "w", encoding="utf-8", newline="") as f1:
            f1.writelines(lines)  
        #将文件放回DSSAT目录
        shutil.move(MZXName,fr"C:\DSSAT48\Maize\{MZXName}")
        #运行DSSAT
        # os.system(r'C:\DSSAT48\DSCSM048.EXE MZCER048 B C:\DSSAT48\Maize\DSSBatch.v48')
        result = subprocess.run(
            [
                r"C:\DSSAT48\DSCSM048.EXE",
                "MZCER048",
                "B",
                r"C:\DSSAT48\Maize\DSSBatch.v48"
                ],
            
            capture_output=True,
            text=True
            )
        if result.returncode != 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        # 替换掉 \r\n
        clean_output = result.stdout.replace("\r\n", "\n")
        print(clean_output)    
        time.sleep(0.5) 
        #从Summary.OUT读取目标函数(Yield,WUE,NUE)
        # 读取 DSSAT Summary.OUT和PlantN.out
        # DssatOut = pd.read_csv("Summary.OUT", delim_whitespace=True, comment='*', skiprows=3)
        DssatOut = pd.read_csv("Summary.OUT", sep=r"\s+", comment='*', skiprows=3, engine="python")
        SumOut = DssatOut.iloc[0].values
        # PlantN= pd.read_csv("PlantN.OUT", delim_whitespace=True, comment='*', skiprows=10)
        PlantN   = pd.read_csv("PlantN.OUT",    sep=r"\s+", comment='*', skiprows=10, engine="python")
        
        yield_ = SumOut[25]              #产量(kg/ha)
        WUE_   = SumOut[80]              #水分利用效率[kg[yield]/ha per mm[ET]]
        NUE_   = SumOut[86]              #氮肥吸收效率[kg[yield]/kg[N uptake][]
        GrainN_=PlantN["GN%D"].iloc[-1]   #取最后一行的籽粒氮浓度(%)
        F = np.array([yield_, WUE_, NUE_, GrainN_], dtype=float)
        
        return F

    # --- API: pareto_front ---
    def pareto_front(self, ref_dirs: np.ndarray) -> np.ndarray:
        """Analytical Pareto front of DTLZ2 for given reference directions.

        DTLZ2's Pareto front is a portion of the unit L2-hypersphere (g=0).
        We project each reference direction onto the unit L2 sphere.
        """
        ref_dirs = np.asarray(ref_dirs, dtype=float)
        if ref_dirs.ndim != 2 or ref_dirs.shape[1] != self.n_obj:
            raise ValueError(
                f"ref_dirs must be (n_pts, {self.n_obj}); got {ref_dirs.shape}"
            )
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0.0, 1.0, norms)
        return ref_dirs / norms

# --------------------
# Problem specification
# --------------------
PROBLEM = "dtlz2"
NOBJ = 4
K = 29
NDIM = NOBJ + K - 1  # decision space dimension
#P 没有“唯一正确值”，要根据问题规模、目标数、计算资源来权衡：
P = 28 #当目标数为3时，12是合适的，P越大参考方向越多
# H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))#factorial：阶乘
H =comb(NOBJ + P - 1, P)
# BOUND_LOW, BOUND_UP = 0.0, 1.0   #决策变量上下限
# —— 原：BOUND_LOW, BOUND_UP = 0.0, 1.0
# —— 新：按变量逐一给上下界（默认仍为 0/1），以后要改某一维就在这里改：
BOUND_LOW = np.full(NDIM, 0.0, dtype=float)
BOUND_UP  = np.full(NDIM, 1.0, dtype=float)
# 灌溉次数[1,12]
#
#灌溉次数
BOUND_LOW[0], BOUND_UP[0]   = 1,10 
#灌水日期比例因子
BOUND_LOW[1], BOUND_UP[1]   = 0.05, 0.9 
BOUND_LOW[2], BOUND_UP[2]   = 0.05, 0.9
BOUND_LOW[3], BOUND_UP[3]   = 0.05, 0.9
BOUND_LOW[4], BOUND_UP[4]   = 0.05, 0.9
BOUND_LOW[5], BOUND_UP[5]   = 0.05, 0.9
BOUND_LOW[6], BOUND_UP[6]   = 0.05, 0.9
BOUND_LOW[7], BOUND_UP[7]   = 0.05, 0.9
BOUND_LOW[8], BOUND_UP[8]   = 0.05, 0.9
BOUND_LOW[9], BOUND_UP[9]   = 0.05, 0.9
BOUND_LOW[10], BOUND_UP[10] = 0.05, 0.9
#灌水定额映射[25:5:60]mm
BOUND_LOW[11], BOUND_UP[11] = 1, 8    
BOUND_LOW[12], BOUND_UP[12] = 1, 8
BOUND_LOW[13], BOUND_UP[13] = 1, 8
BOUND_LOW[14], BOUND_UP[14] = 1, 8
BOUND_LOW[15], BOUND_UP[15] = 1, 8
BOUND_LOW[16], BOUND_UP[16] = 1, 8
BOUND_LOW[17], BOUND_UP[17] = 1, 8
BOUND_LOW[18], BOUND_UP[18] = 1, 8
BOUND_LOW[19], BOUND_UP[19] = 1, 8
BOUND_LOW[20], BOUND_UP[20] = 1, 8
#灌水时是否施肥：0不施肥，1施肥
BOUND_LOW[21], BOUND_UP[21] = 0, 1
BOUND_LOW[22], BOUND_UP[22] = 0, 1
BOUND_LOW[23], BOUND_UP[23] = 0, 1
BOUND_LOW[24], BOUND_UP[24] = 0, 1
BOUND_LOW[25], BOUND_UP[25] = 0, 1
BOUND_LOW[26], BOUND_UP[26] = 0, 1
BOUND_LOW[27], BOUND_UP[27] = 0, 1
BOUND_LOW[28], BOUND_UP[28] = 0, 1
BOUND_LOW[29], BOUND_UP[29] = 0, 1
BOUND_LOW[30], BOUND_UP[30] = 0, 1
#总施肥量映射[150:30:360]kg（N）/ha
BOUND_LOW[31], BOUND_UP[31] = 1, 8


# Replace pymop.factory.get_problem with our local class
if PROBLEM.lower() != "dtlz2":
    raise ValueError("This minimal implementation only provides DTLZ2.")
problem = DTLZ2Problem(n_var=NDIM, n_obj=NOBJ)


# -------------------------------
# Reference directions for NSGA-3
# -------------------------------
# 在环境选择阶段，用一组在目标空间里均匀分布的方向来分配名额与维持多样性，让最终解集沿 Pareto 前沿覆盖得均匀、不过度扎堆。
ref_points = tools.uniform_reference_points(NOBJ, P)#H*NOBJ
# -----------------------
# Algorithm configuration
# -----------------------
MU = 4500   # population size reported in the manuscript
NGEN = 50                # generations
CXPB = 1.0                 # crossover prob for varAnd
MUTPB = 1.0                # mutation prob for varAnd
# Use a fixed seed so that the optimization run is reproducible and traceable.
DEFAULT_RANDOM_SEED = 202501

# ==== [ADD] 归一化HV与早停配置 ====
# 固定归一化盒(ideal/nadir)。建议用你对作物系统的经验范围；若暂时未知，可先自动估计再冻结（见 main()）。
IDEAL_FIXED = None  # 例如 np.array([3000, 5, 20, 0.6])
NADIR_FIXED = None  # 例如 np.array([15000, 35, 80, 1.8])

# 早停控制：连续 PATIENCE 代 HV 提升 < EPS 则停止
PATIENCE = 8
EPS = 1e-8
FREEZE_AFTER = 5  # 前几代允许更新(扩展)ideal/nadir，之后冻结，避免HV因坐标伸缩波动

# ----------------
# DEAP definitions
# ----------------
#正数表示最大化，负数表示最小化
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def uniform(low, up, size=None):
    """Generate a list of floats sampled uniformly in [low, up].
    Supports scalar or sequence bounds.
    """
    #uniform 是你自定义的函数，生成 [low, up] 区间里的随机数。
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
# initialize a NDIM-dimensional real vector in [0,1]
# 之后直接用 toolbox.attr_float() 来调用。
toolbox.register("attr_float", uniform, BOUND_LOW.tolist(), BOUND_UP.tolist(), NDIM)
# vec1=toolbox.attr_float()
# wrap into an Individual
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# vec2=toolbox.individual()
# population generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# vec3 = toolbox.population(n=MU)
# evaluate -> use our problem's evaluate to return only F
# (DEAP expects a 1D sequence of objective values for each individual)
toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])  # signature-compat
# vec4=toolbox.evaluate(vec3)
# variation operators
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW.tolist(), up=BOUND_UP.tolist(), eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW.tolist(), up=BOUND_UP.tolist(), eta=20.0, indpb=1.0 / NDIM)
# NSGA-III selection with reference points
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

# -------------
# Main run loop
# -------------
def main(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    print(f"[Run configuration] population={MU}, random_seed={seed}")
    # ==== [ADD] HV/早停状态 ====
    best_hv = -np.inf
    no_improve = 0

    # 归一化盒：固定值优先，其次自动估/冻结
    ideal = None if IDEAL_FIXED is None else np.asarray(IDEAL_FIXED, dtype=float)
    nadir = None if NADIR_FIXED is None else np.asarray(NADIR_FIXED, dtype=float)
    
    # Stats on objective values
    # 建一个统计器，并指定提取函数：对每个个体 ind，取它的多目标适应度向量 ind.fitness.values 作为要统计的数据。
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    #注册名为 "avg" 的统计量，用 numpy.mean 计算按目标维度逐列的平均值（axis=0 表示在“个体”这个维度上聚合，得到每个目标各自的均值）。
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    # 新建一个 Logbook 实例，用来逐代保存你记录的键值对（比如第几代、评估了多少个体、各统计量等）。
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max","hv"

    pop = toolbox.population(n=MU)#生成初始种群

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        # Ensure it's a plain tuple for DEAP
        ind.fitness.values = tuple(np.asarray(fit, dtype=float))


    # ==== [ADD] 初代 HV ====
    F_all = np.array([ind.fitness.values for ind in pop], dtype=float)
    # 只取非支配前沿（其余点对HV无贡献），减少计算量
    first_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    F_nd = np.array([ind.fitness.values for ind in first_front], dtype=float)

    if IDEAL_FIXED is None or NADIR_FIXED is None:
        # 自动估计并允许后续几代扩展(只扩不缩)，FREEZE_AFTER 后冻结
        cur_min = F_all.min(axis=0)
        cur_max = F_all.max(axis=0)
        ideal = cur_min if ideal is None else np.minimum(ideal, cur_min)
        nadir = cur_max if nadir is None else np.maximum(nadir, cur_max)

    hv_val = compute_norm_hv_max(F_nd, ideal, nadir)
    record = stats.compile(pop)
    record["hv"] = hv_val
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Evolutionary loop
    for gen in range(1, NGEN):
        # varAnd 会：
        # 克隆父代；
        # 以概率 CXPB 做交叉（这里是 1.0，几乎总是交叉）；
        # 以概率 MUTPB 做变异（这里也是 1.0，几乎总是变异）。
        # 返回同等规模的子代列表。
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = tuple(np.asarray(fit, dtype=float))

        # NSGA-III environmental selection
        pop = toolbox.select(pop + offspring, MU)
        
        
        # ==== [ADD] 每代 HV + 归一化盒更新/冻结 + 早停 ====
        F_all = np.array([ind.fitness.values for ind in pop], dtype=float)
        first_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        F_nd = np.array([ind.fitness.values for ind in first_front], dtype=float)

        # 自动盒：FREEZE_AFTER 代内只做“扩展”（单调不收缩），之后冻结
        if IDEAL_FIXED is None or NADIR_FIXED is None:
            if gen <= FREEZE_AFTER:
                cur_min = F_all.min(axis=0)
                cur_max = F_all.max(axis=0)
                ideal = np.minimum(ideal, cur_min)
                nadir = np.maximum(nadir, cur_max)

        hv_val = compute_norm_hv_max(F_nd, ideal, nadir)

        # 早停判据（平台期）
        if hv_val > best_hv + EPS:
            best_hv = hv_val
            no_improve = 0
        else:
            no_improve += 1

        # 记录并打印
        record = stats.compile(pop)
        record["hv"] = hv_val
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        if no_improve >= PATIENCE:
            print(f"[Early Stop] HV {best_hv:.6f} 连续 {PATIENCE} 代无显著提升，提前结束。")
            break

    return pop, logbook


if __name__ == "__main__":
    # 最后一代种群 & 日志
    pop, logbook = main(DEFAULT_RANDOM_SEED)

    # 全部个体的目标值矩阵 (n_individuals, 4)
    pop_fit = np.array([ind.fitness.values for ind in pop], dtype=float)

    # 计算非支配分层，并取第一前沿（rank==0）
    fronts = tools.sortNondominated(pop, len(pop), first_front_only=False)
    idx_map = {id(ind): i for i, ind in enumerate(pop)}
    rank = np.empty(len(pop), dtype=int)
    for r, front in enumerate(fronts):
        for ind in front:
            rank[idx_map[id(ind)]] = r
    first_idx = np.where(rank == 0)[0]
    F_first = pop_fit[first_idx]

    # 统一保存目录
    os.makedirs("results", exist_ok=True)

    # 1) 打包成一个 pkl（最省心）
    with open("results/run_last.pkl", "wb") as f:
        pickle.dump(
            {
                "pop": pop,
                "logbook": logbook,
                "pop_fit": pop_fit,
                "rank": rank,
                "first_idx": first_idx,
                "random_seed": DEFAULT_RANDOM_SEED,
                "population_size": MU,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # 2) 常用矩阵直接另存（便于 NumPy/Matplotlib 读）
    np.save("results/pop_fit.npy", pop_fit)
    np.save("results/first_front_objectives.npy", F_first)

    # 3) 导出“第一前沿目标值”CSV（后续画图最常用）
    pd.DataFrame(F_first, columns=["Yield", "WUE", "NUE", "GrainN"]).to_csv(
        "results/first_front_objectives.csv", index=False
    )

    # 4) 导出每代 HV（快速画收敛曲线）
    try:
        gen_hist = [rec["gen"] for rec in logbook]
        hv_hist  = [rec["hv"]  for rec in logbook]
        pd.DataFrame({"gen": gen_hist, "hv": hv_hist}).to_csv(
            "results/progress_hv.csv", index=False
        )
    except Exception as e:
        print(f"[WARN] 导出 HV 曲线失败：{e}")

    pd.DataFrame(
        [{
            "population_size": MU,
            "random_seed": DEFAULT_RANDOM_SEED,
            "generations_completed": logbook[-1]["gen"],
        }]
    ).to_csv("results/run_configuration.csv", index=False)

    

    

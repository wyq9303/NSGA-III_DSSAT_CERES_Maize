# -*- coding: utf-8 -*-
"""
从 results/run_last.pkl 读取NSGA-III最终种群及第一Pareto前沿。

筛选规则：保留第一前沿中模拟产量严格大于历史农户最高模拟产量
11823 kg/ha的全部解，并稳定编号为HY-1至HY-M。

输出：
1. results/filtered_objectives.csv
2. results/filtered_management_summary.csv
3. results/filtered_irrigation_events.csv
4. results/filtered_fertilization_events.csv

说明：
- 不再使用P95定义，也不进行人工二次筛选。
- 管理变量解码规则与NSGA_III32.py保持一致。
- 本脚本及Fig. 10仅统计NSGA-III显式生成的灌溉量和灌溉次数，
  不加入建苗灌溉。建苗水将在后续经济分析中另行核算。
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


# ======================== 路径与参数 ========================
# 路径保持不变：请从原项目工作目录运行脚本。
RES_DIR = Path("results")
INPUT_PKL = RES_DIR / "run_last.pkl"
OUT_DIR = Path("results")

FARMER_YIELD_BENCHMARK = 11823.0

PLANT_DOY = 122
HARVEST_DOY = 275
GROWING_PERIOD = HARVEST_DOY - PLANT_DOY
MIN_IRRIGATION_GAP = 5


# ======================== 读取优化结果 ========================
if not INPUT_PKL.exists():
    raise FileNotFoundError(
        f"找不到 {INPUT_PKL}。请在包含results目录的原项目目录中运行本脚本。"
    )

with INPUT_PKL.open("rb") as f:
    blob = pickle.load(f)

required_keys = {"pop", "pop_fit", "first_idx"}
missing_keys = required_keys.difference(blob)
if missing_keys:
    raise KeyError(f"{INPUT_PKL} 缺少字段：{sorted(missing_keys)}")

pop = blob["pop"]
pop_fit = np.asarray(blob["pop_fit"], dtype=float)
first_idx = np.asarray(blob["first_idx"], dtype=int)

if pop_fit.ndim != 2 or pop_fit.shape[1] != 4:
    raise ValueError(
        "pop_fit应为(N, 4)矩阵，目标顺序为[Yield, WUE, NUE, GrainN]。"
    )
if np.any(first_idx < 0) or np.any(first_idx >= len(pop_fit)):
    raise IndexError("first_idx中包含超出pop_fit范围的索引。")

F_first = pop_fit[first_idx]


# ======================== HY子集筛选 ========================
# 严格使用Yield > 11823 kg/ha，而不是大于等于。
mask_yield = F_first[:, 0] > FARMER_YIELD_BENCHMARK
sel_first_idx = first_idx[mask_yield]
F_sel = F_first[mask_yield]

# 稳定编号：Yield、WUE、NUE和GrainN依次降序；完全相同时按原索引升序。
if len(F_sel) > 0:
    order = np.lexsort(
        (
            sel_first_idx,
            -F_sel[:, 3],
            -F_sel[:, 2],
            -F_sel[:, 1],
            -F_sel[:, 0],
        )
    )
    sel_first_idx = sel_first_idx[order]
    F_sel = F_sel[order]

inds_sel = [pop[i] for i in sel_first_idx]
hy_ids = np.asarray(
    [f"HY-{i + 1}" for i in range(len(F_sel))], dtype=object
)


# ======================== 决策变量解码 ========================
def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def map_irrig_amount(value):
    """x属于[1, 8]，映射为25、30、……、60 mm。"""
    value = float(np.clip(value, 1.0, 8.0))
    return float(25 + round(value - 1) * 5)


def map_total_n(value):
    """x属于[1, 8]，映射为150、180、……、360 kg N/ha。"""
    value = float(np.clip(value, 1.0, 8.0))
    return float(150 + round(value - 1) * 30)


def reconstruct_irrig_dates(individual):
    """按优化代码的排序、绝对DOY映射和5天最小间隔规则生成日期。"""
    x = np.asarray(individual, dtype=float)
    u = np.clip(np.asarray(x[1:11], dtype=float), 0.0, 1.0)
    u_sorted = np.sort(u)
    dates = PLANT_DOY + np.round(u_sorted * GROWING_PERIOD).astype(int)

    dates[0] = max(dates[0], PLANT_DOY)
    for i in range(1, len(dates)):
        if dates[i] < dates[i - 1] + MIN_IRRIGATION_GAP:
            dates[i] = dates[i - 1] + MIN_IRRIGATION_GAP

    if dates[-1] > HARVEST_DOY:
        dates[-1] = HARVEST_DOY
        for i in range(len(dates) - 2, -1, -1):
            dates[i] = min(dates[i], dates[i + 1] - MIN_IRRIGATION_GAP)
        dates[0] = max(dates[0], PLANT_DOY)
        for i in range(1, len(dates)):
            dates[i] = max(dates[i], dates[i - 1] + MIN_IRRIGATION_GAP)

    return np.clip(dates, PLANT_DOY, HARVEST_DOY).astype(int)


def reconstruct_irrig_amounts(individual, irrig_frequency):
    x = np.asarray(individual, dtype=float)
    amounts = np.asarray(
        [map_irrig_amount(x[11 + i]) for i in range(10)], dtype=float
    )
    if irrig_frequency < 10:
        amounts[irrig_frequency:] = 0.0
    return amounts


def reconstruct_fertilization(individual, irrig_frequency, irrig_dates):
    """总氮量平均分配至前irrig_frequency次灌溉中的激活施氮事件。"""
    x = np.asarray(individual, dtype=float)
    raw_flags = np.asarray(
        [int(round(v)) for v in x[21:31]], dtype=int
    )
    raw_flags = np.clip(raw_flags, 0, 1)

    active_flags = raw_flags.copy()
    if irrig_frequency < 10:
        active_flags[irrig_frequency:] = 0

    fert_frequency = int(active_flags.sum())
    total_n_gene = map_total_n(x[31])
    fert_dates = irrig_dates.copy()
    fert_amounts = np.zeros(10, dtype=float)

    if fert_frequency > 0:
        fert_amounts = (
            active_flags.astype(float) * total_n_gene / fert_frequency
        )

    return (
        active_flags,
        fert_dates,
        fert_amounts,
        fert_frequency,
        total_n_gene,
    )


def decode_one(individual):
    irrig_frequency = int(clamp(int(round(individual[0])), 1, 10))
    irrig_dates = reconstruct_irrig_dates(individual)
    irrig_amounts = reconstruct_irrig_amounts(individual, irrig_frequency)
    (
        fert_flags,
        fert_dates,
        fert_amounts,
        fert_frequency,
        total_n_gene,
    ) = reconstruct_fertilization(
        individual, irrig_frequency, irrig_dates
    )
    return (
        irrig_frequency,
        irrig_dates,
        irrig_amounts,
        fert_flags,
        fert_dates,
        fert_amounts,
        fert_frequency,
        total_n_gene,
    )


# ======================== 批量解码 ========================
n_selected = len(inds_sel)
irrig_freq_sel = np.zeros(n_selected, dtype=int)
irrig_dates_sel = np.zeros((n_selected, 10), dtype=int)
irrig_amts_sel = np.zeros((n_selected, 10), dtype=float)
fert_flags_sel = np.zeros((n_selected, 10), dtype=int)
fert_dates_sel = np.zeros((n_selected, 10), dtype=int)
fert_amts_sel = np.zeros((n_selected, 10), dtype=float)
fert_freq_sel = np.zeros(n_selected, dtype=int)
total_n_gene_sel = np.zeros(n_selected, dtype=float)

for j, individual in enumerate(inds_sel):
    (
        irrig_frequency,
        irrig_dates,
        irrig_amounts,
        fert_flags,
        fert_dates,
        fert_amounts,
        fert_frequency,
        total_n_gene,
    ) = decode_one(individual)
    irrig_freq_sel[j] = irrig_frequency
    irrig_dates_sel[j] = irrig_dates
    irrig_amts_sel[j] = irrig_amounts
    fert_flags_sel[j] = fert_flags
    fert_dates_sel[j] = fert_dates
    fert_amts_sel[j] = fert_amounts
    fert_freq_sel[j] = fert_frequency
    total_n_gene_sel[j] = total_n_gene


# ======================== 筛选统计 ========================
print("=== High-yield Pareto subset ===")
print(f"Farmer yield benchmark   : {FARMER_YIELD_BENCHMARK:.2f} kg/ha")
print(f"Selection rule           : Yield > {FARMER_YIELD_BENCHMARK:.2f} kg/ha")
print(f"Total first-front points : {len(first_idx)}")
print(f"Selected HY schedules    : {n_selected}")
if n_selected > 0:
    print(f"HY identifiers           : HY-1 to HY-{n_selected}")
    print(
        f"Yield range (HY subset)  : "
        f"{F_sel[:, 0].min():.0f} ~ {F_sel[:, 0].max():.0f} kg/ha"
    )
    print(f"WUE/WPc mean             : {F_sel[:, 1].mean():.2f}")
    print(f"NUE/PEN mean             : {F_sel[:, 2].mean():.2f}")
    print(f"GrainN/GNC mean          : {F_sel[:, 3].mean():.3f}%")
else:
    print("[WARNING] No Pareto solution exceeded the farmer yield benchmark.")


# ======================== 汇总管理变量 ========================
OUT_DIR.mkdir(parents=True, exist_ok=True)
optimized_irrig_mm = irrig_amts_sel.sum(axis=1)
optimized_irrig_freq = irrig_freq_sel.copy()
total_n_applied = fert_amts_sel.sum(axis=1)


# ======================== 输出CSV ========================
# 1. 四个目标值。保留旧列名，使现有Fig. 10代码改动最小。
df_obj = pd.DataFrame(
    F_sel, columns=["Yield", "WUE", "NUE", "GrainN"]
)
df_obj.insert(0, "source_pop_index", sel_first_idx)
df_obj.insert(0, "schedule_id", hy_ids)
df_obj.insert(0, "sol_id", np.arange(n_selected, dtype=int))
df_obj.to_csv(
    OUT_DIR / "filtered_objectives.csv",
    index=False,
    encoding="utf-8-sig",
)

# 2. 管理摘要。Total_Irr_mm和Irr_Freq仅统计NSGA-III显式灌溉。
df_summary = pd.DataFrame(
    {
        "sol_id": np.arange(n_selected, dtype=int),
        "schedule_id": hy_ids,
        "source_pop_index": sel_first_idx,
        "Farmer_Yield_Benchmark": np.full(
            n_selected, FARMER_YIELD_BENCHMARK, dtype=float
        ),
        "Optimized_Irr_Freq": optimized_irrig_freq,
        "Irr_Freq": optimized_irrig_freq,
        "Fert_Freq": fert_freq_sel,
        "Optimized_Irr_mm": optimized_irrig_mm,
        "Total_Irr_mm": optimized_irrig_mm,
        "Total_N_gene_kg_ha": total_n_gene_sel,
        "Total_N_kg_ha": total_n_applied,
        "Yield": F_sel[:, 0],
        "WUE": F_sel[:, 1],
        "NUE": F_sel[:, 2],
        "GrainN": F_sel[:, 3],
    }
)
df_summary.to_csv(
    OUT_DIR / "filtered_management_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# 3. 灌溉事件明细，仅列NSGA-III显式优化事件。
rows_irrig = []
for i in range(n_selected):
    for event_index in range(10):
        rows_irrig.append(
            {
                "sol_id": i,
                "schedule_id": hy_ids[i],
                "source_pop_index": int(sel_first_idx[i]),
                "event": event_index + 1,
                "Irr_DOY": int(irrig_dates_sel[i, event_index]),
                "Irr_mm": float(irrig_amts_sel[i, event_index]),
                "Irr_active": int(event_index < irrig_freq_sel[i]),
            }
        )
df_irrig = pd.DataFrame(rows_irrig)
df_irrig.to_csv(
    OUT_DIR / "filtered_irrigation_events.csv",
    index=False,
    encoding="utf-8-sig",
)

# 4. 施氮事件明细。
rows_fert = []
for i in range(n_selected):
    for event_index in range(10):
        rows_fert.append(
            {
                "sol_id": i,
                "schedule_id": hy_ids[i],
                "source_pop_index": int(sel_first_idx[i]),
                "event": event_index + 1,
                "Fert_DOY": int(fert_dates_sel[i, event_index]),
                "Fert_flag": int(fert_flags_sel[i, event_index]),
                "Fert_kgN_ha": float(fert_amts_sel[i, event_index]),
            }
        )
df_fert = pd.DataFrame(rows_fert)
df_fert.to_csv(
    OUT_DIR / "filtered_fertilization_events.csv",
    index=False,
    encoding="utf-8-sig",
)


# ======================== 一致性检查 ========================
if n_selected > 0:
    if not np.all(F_sel[:, 0] > FARMER_YIELD_BENCHMARK):
        raise AssertionError("HY子集中存在未严格超过11823 kg/ha的解。")
    if np.any(fert_freq_sel > irrig_freq_sel):
        raise AssertionError("存在施氮次数大于显式灌溉次数的方案。")
    if np.any(total_n_applied == 0):
        zero_n_ids = hy_ids[total_n_applied == 0]
        print("[WARNING] 以下方案没有有效施氮事件：" + ", ".join(zero_n_ids))

    active_irrig_amounts = irrig_amts_sel[irrig_amts_sel > 0]
    allowed_irrig_amounts = np.arange(25.0, 65.0, 5.0)
    if not np.all(np.isin(active_irrig_amounts, allowed_irrig_amounts)):
        raise AssertionError("存在不属于25—60 mm离散集合的灌溉量。")

print("[OK] CSV files saved:")
print(" - results/filtered_objectives.csv")
print(" - results/filtered_management_summary.csv")
print(" - results/filtered_irrigation_events.csv")
print(" - results/filtered_fertilization_events.csv")

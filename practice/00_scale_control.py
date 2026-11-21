# -*- coding: utf-8 -*-
"""
统一参数控制实验：单位变化 × StandardScaler 标准化的 2×2 对照
- A：sim，未标准化（income 万元、balance 万元）
- B：yuan，未标准化（income 万元、balance 元）
- C：sim，StandardScaler
- D：yuan，StandardScaler

统一 k=3、init=k-means++、n_init=10、random_state=42，
其余 KMeans 与 StandardScaler 参数显式固定。
只读原始 CSV，不修改任何数据；所有结果写入 outputs/practice/control_experiment/。
"""
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import font_manager
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "practice" / "control_experiment"
SIM_PATH = DATA_DIR / "mall_customers_sim.csv"
YUAN_PATH = DATA_DIR / "mall_customers_yuan.csv"

EXPECTED_COLUMNS = [
    "customer_id", "age", "income", "balance", "product_count", "gender", "city",
]
BASELINE_SHA256 = {
    "mall_customers_sim.csv": "13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F",
    "mall_customers_yuan.csv": "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07",
}
NON_BALANCE_FIELDS = ["age", "income", "product_count", "gender", "city"]
FEATURES = ["income", "balance"]
BALANCE_TOL = 1e-6

KM_PARAMS = {
    "n_clusters": 3,
    "init": "k-means++",
    "n_init": 10,
    "random_state": 42,
    "max_iter": 300,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "copy_x": True,
    "verbose": 0,
}
SCALER_PARAMS = {"with_mean": True, "with_std": True, "copy": True}

CONFIGS = [
    {"id": "A", "data": "sim", "desc": "sim 万元 · 未标准化", "scale": False, "balance_wan_divisor": 1.0},
    {"id": "B", "data": "yuan", "desc": "yuan 混合单位 · 未标准化", "scale": False, "balance_wan_divisor": 10000.0},
    {"id": "C", "data": "sim", "desc": "sim 万元 · StandardScaler", "scale": True, "balance_wan_divisor": 1.0},
    {"id": "D", "data": "yuan", "desc": "yuan 混合单位 · StandardScaler", "scale": True, "balance_wan_divisor": 10000.0},
]
PAIRS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
PALETTE = ["#DD8452", "#4C72B0", "#55A868"]

SPACE_LABEL = {
    "A": "万元-万元 数值空间（income 万元，balance 万元）",
    "B": "混合单位数值空间（income 万元，balance 元）",
    "C": "标准化空间（z 分数）",
    "D": "标准化空间（z 分数）",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(msg: str) -> None:
    print(f"[预检失败] {msg}")
    print("已停止：未写入任何聚类结果，未修改原始数据。")
    sys.exit(2)


def jsonable(obj):
    """把 numpy 标量转成原生类型；遇到非有限浮点数直接抛错，保证 JSON 无 NaN/Infinity。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"JSON 中出现非有限数值: {obj}")
        return obj
    return obj


# =====================================================================
# 一、只读预检
# =====================================================================
def run_prechecks():
    print("=" * 78)
    print("一、只读预检")
    print("=" * 78)
    for p in (SIM_PATH, YUAN_PATH):
        if not p.exists():
            fail(f"数据文件不存在：{p}")

    hashes = {p.name: sha256_of(p) for p in (SIM_PATH, YUAN_PATH)}
    for name, expected in BASELINE_SHA256.items():
        actual = hashes[name]
        status = "一致" if actual == expected else "不一致"
        print(f"  SHA256 {name}: {actual}  [{status}]")
        if actual != expected:
            fail(f"{name} 哈希与基线不一致（基线 {expected}，实际 {actual}）")

    sim = pd.read_csv(SIM_PATH)
    yuan = pd.read_csv(YUAN_PATH)

    for name, df in (("sim", sim), ("yuan", yuan)):
        if list(df.columns) != EXPECTED_COLUMNS:
            fail(f"{name} 列名或列顺序异常：{list(df.columns)}")
        print(f"  {name} 行列数: {df.shape[0]} 行 × {df.shape[1]} 列")
        if df.isnull().sum().sum() != 0:
            fail(f"{name} 存在缺失值：\n{df.isnull().sum()}")
        if df["customer_id"].duplicated().any():
            fail(f"{name} 的 customer_id 存在重复")
        if df["customer_id"].nunique() != len(df):
            fail(f"{name} 的 customer_id 不唯一")
        print(f"  {name} customer_id: 唯一 {df['customer_id'].nunique()} 个，"
              f"范围 {df['customer_id'].min()}–{df['customer_id'].max()}，无缺失")

    if set(sim["customer_id"]) != set(yuan["customer_id"]):
        fail("两份数据的 customer_id 集合不一致")

    try:
        merged = sim.merge(
            yuan, on="customer_id", how="outer", validate="one_to_one",
            suffixes=("_sim", "_yuan"), indicator=True,
        )
    except Exception as exc:
        fail(f"按 customer_id 对齐失败：{exc}")

    both = int((merged["_merge"] == "both").sum())
    print(f"  按 customer_id 合并：{len(merged)} 行，both={both}，"
          f"left_only={int((merged['_merge'] == 'left_only').sum())}，"
          f"right_only={int((merged['_merge'] == 'right_only').sum())}")
    if both != len(sim) or len(merged) != len(sim):
        fail("按 customer_id 合并后行数不等，两份数据未完全覆盖同一批客户")

    field_diffs = {}
    for col in NON_BALANCE_FIELDS:
        a = merged[f"{col}_sim"]
        b = merged[f"{col}_yuan"]
        n_diff = int((a != b).sum())
        field_diffs[col] = n_diff
        print(f"  字段 {col:<14} 差异行数: {n_diff}")
        if n_diff != 0:
            detail = merged.loc[a != b, ["customer_id", f"{col}_sim", f"{col}_yuan"]]
            print(detail.head(20).to_string(index=False))
            fail(f"字段 {col} 在两份数据间存在 {n_diff} 行差异")

    diff = merged["balance_yuan"] - merged["balance_sim"] * 10000.0
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max())
    n_nonzero = int((abs_diff != 0).sum())
    print(f"  余额换算 balance_yuan - balance_sim×10000：最大绝对误差 {max_abs!r} 元，"
          f"非零 {n_nonzero} 行，容差 {BALANCE_TOL:g} 元")
    if max_abs > BALANCE_TOL:
        fail(f"余额换算最大绝对误差 {max_abs} 超过容差 {BALANCE_TOL}")

    detail_rows = merged.loc[abs_diff != 0, [
        "customer_id", "balance_sim", "balance_yuan",
    ]].copy()
    detail_rows["expected_balance_yuan"] = detail_rows["balance_sim"] * 10000.0
    detail_rows["diff_yuan"] = detail_rows["balance_yuan"] - detail_rows["expected_balance_yuan"]
    detail_rows["abs_diff_yuan"] = detail_rows["diff_yuan"].abs()
    detail_rows = detail_rows.sort_values("customer_id").reset_index(drop=True)

    print(f"  预检通过：两份数据覆盖同一批 {len(sim)} 位客户，除 balance 外字段完全一致。")
    return {
        "sim": sim.sort_values("customer_id").reset_index(drop=True),
        "yuan": yuan.sort_values("customer_id").reset_index(drop=True),
        "hashes": hashes,
        "field_diffs": field_diffs,
        "balance_max_abs_diff_yuan": max_abs,
        "balance_nonzero_rows": n_nonzero,
        "balance_total_rows": len(merged),
        "balance_tolerance_yuan": BALANCE_TOL,
        "balance_detail": detail_rows,
    }


# =====================================================================
# 二、聚类
# =====================================================================
def choose_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"本次绘图使用中文字体：{name}")
            return name
    fail("未找到可用中文字体（Microsoft YaHei / SimHei）")


def max_overlap_match(src_labels, dst_labels, k=3):
    table = np.zeros((k, k), dtype=int)
    for s, d in zip(src_labels, dst_labels):
        table[int(s), int(d)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    matched = int(sum(table[r, c] for r, c in zip(rows, cols)))
    return table, mapping, matched


def run_clustering(pre):
    print()
    print("=" * 78)
    print("二、四配置聚类（统一参数）")
    print("=" * 78)
    print(f"  KMeans 参数: {KM_PARAMS}")
    print(f"  StandardScaler 参数: {SCALER_PARAMS}")
    print(f"  特征顺序: {FEATURES}")

    data = {"sim": pre["sim"], "yuan": pre["yuan"]}
    scalers = {}

    results = {}
    for cfg in CONFIGS:
        df = data[cfg["data"]]
        X_raw = df[FEATURES].to_numpy(dtype=float)
        if cfg["scale"]:
            scaler = StandardScaler(**SCALER_PARAMS)
            X = scaler.fit_transform(X_raw)
            scalers[cfg["id"]] = scaler
            centers_raw = None
        else:
            X = X_raw
            centers_raw = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            km = KMeans(**KM_PARAMS)
            labels = km.fit_predict(X)
            sil = float(silhouette_score(X, labels, metric="euclidean"))

        warn_msgs = []
        for w in caught:
            warn_msgs.append(f"{w.category.__name__}: {str(w.message).replace(chr(10), ' ')}")

        if centers_raw is None:
            centers_model = km.cluster_centers_
            if cfg["scale"]:
                centers_raw = scaler.inverse_transform(centers_model)
            else:
                centers_raw = centers_model

        centers_out = [
            {
                "cluster": int(i),
                "income_wan": float(centers_raw[i, 0]),
                "balance_wan": float(centers_raw[i, 1] / cfg["balance_wan_divisor"]),
                "balance_yuan": float(centers_raw[i, 1] / cfg["balance_wan_divisor"] * 10000.0),
            }
            for i in range(km.n_clusters)
        ]

        counts = [int((labels == i).sum()) for i in range(km.n_clusters)]
        results[cfg["id"]] = {
            "cfg": cfg,
            "df": df,
            "X": X,
            "X_raw": X_raw,
            "labels": labels,
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "n_iter": int(km.n_iter_),
            "counts": counts,
            "centers": centers_out,
            "warnings": warn_msgs,
        }
        print(f"  配置 {cfg['id']}（{cfg['desc']}）：inertia={km.inertia_:.6f}，"
              f"silhouette={sil:.6f}，n_iter_={km.n_iter_}，"
              f"各簇人数={counts}，警告 {len(warn_msgs)} 条")

    return results, scalers


# =====================================================================
# 三、标签对照、画像、图
# =====================================================================
def compare(results):
    print()
    print("=" * 78)
    print("三、标签对照（按 customer_id 对齐）")
    print("=" * 78)
    order = ["A", "B", "C", "D"]
    for cid in order:
        results[cid]["df"] = results[cid]["df"].sort_values("customer_id").reset_index(drop=True)
        if len(results[cid]["labels"]) != len(results[cid]["df"]):
            fail(f"配置 {cid} 的标签数与客户数不一致")

    ids = results["A"]["df"]["customer_id"].tolist()
    for cid in order[1:]:
        if results[cid]["df"]["customer_id"].tolist() != ids:
            fail(f"配置 {cid} 的客户顺序与 A 不一致")

    labels_all = pd.DataFrame({"customer_id": ids})
    for cid in order:
        labels_all[f"label_{cid}"] = results[cid]["labels"].astype(int)

    ari_rows = []
    cont_rows = []
    match_rows = []
    tables = {}

    for ca, cb in PAIRS:
        la = results[ca]["labels"]
        lb = results[cb]["labels"]
        ari = float(adjusted_rand_score(la, lb))
        table, mapping, matched = max_overlap_match(la, lb)
        total = int(table.sum())
        changed = total - matched
        rate = matched / total
        tables[(ca, cb)] = (table, mapping, matched, changed, total)
        pair_name = f"{ca}-{cb}"
        ari_rows.append({"pair": pair_name, "config_a": ca, "config_b": cb, "ari": ari})
        print(f"  {pair_name}: ARI={ari:.6f}，最大重叠匹配 {matched}/{total} "
              f"（{rate:.4f}），匹配后变化 {changed} 人，映射 {mapping}")

        for i in range(3):
            for j in range(3):
                cont_rows.append({
                    "pair": pair_name, "config_a": ca, "config_b": cb,
                    "cluster_a": i, "cluster_b": j, "n": int(table[i, j]),
                })
        for i in range(3):
            match_rows.append({
                "pair": pair_name, "config_a": ca, "config_b": cb,
                "cluster_a": i, "matched_cluster_b": mapping[i],
                "matched_n": int(table[i, mapping[i]]),
                "pair_matched_total": matched,
                "pair_changed_total": changed,
                "pair_match_rate": rate,
            })

    # 展示用颜色映射：统一对齐到 D
    display_mapping = {}
    for cid in order:
        if cid == "D":
            display_mapping[cid] = {0: 0, 1: 1, 2: 2}
            continue
        _, mapping, _ = max_overlap_match(results[cid]["labels"], results["D"]["labels"])
        display_mapping[cid] = mapping
    print(f"  展示颜色映射（对齐到 D 簇，仅用于绘图）：{display_mapping}")
    print("  说明：颜色对应为最大重叠匹配，不代表两簇成员完全相同；原始标签未改写。")

    return labels_all, ari_rows, cont_rows, match_rows, tables, display_mapping


def build_profiles(results):
    order = ["A", "B", "C", "D"]
    rows = []
    for cid in order:
        r = results[cid]
        df = r["df"]
        divisor = r["cfg"]["balance_wan_divisor"]
        balance_wan = df["balance"] / divisor
        labels = r["labels"]
        for c in range(3):
            mask = labels == c
            n = int(mask.sum())
            rows.append({
                "config": cid,
                "cluster": c,
                "n": n,
                "share_pct": n / len(df) * 100.0,
                "age_mean": float(df.loc[mask, "age"].mean()),
                "income_wan_mean": float(df.loc[mask, "income"].mean()),
                "balance_wan_mean": float(balance_wan[mask].mean()),
                "balance_yuan_mean": float(df.loc[mask, "balance"].mean() * 10000.0 / divisor),
                "product_count_mean": float(df.loc[mask, "product_count"].mean()),
            })
    profiles = pd.DataFrame(rows)
    for col in ["share_pct", "age_mean", "income_wan_mean", "balance_wan_mean",
                "balance_yuan_mean", "product_count_mean"]:
        profiles[col] = profiles[col].round(6)
    return profiles


def plot_compare(results, display_mapping, font_name):
    print()
    print("=" * 78)
    print("四、2×2 对照图（统一原始单位：income 万元，balance 万元）")
    print("=" * 78)
    sim = results["A"]["df"]
    income = sim["income"].to_numpy(dtype=float)
    balance_wan = sim["balance"].to_numpy(dtype=float)

    x_pad = (income.max() - income.min()) * 0.05
    y_pad = (balance_wan.max() - balance_wan.min()) * 0.05
    xlim = (income.min() - x_pad, income.max() + x_pad)
    ylim = (balance_wan.min() - y_pad, balance_wan.max() + y_pad)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    axes = axes.ravel()
    for ax, cfg in zip(axes, CONFIGS):
        cid = cfg["id"]
        r = results[cid]
        labels = np.asarray(r["labels"])
        disp = np.array([display_mapping[cid][int(v)] for v in labels])
        counts = "/".join(str(c) for c in r["counts"])

        for d_cluster in range(3):
            mask = disp == d_cluster
            if not mask.any():
                continue
            ax.scatter(
                income[mask], balance_wan[mask], s=42, alpha=0.75,
                color=PALETTE[d_cluster], edgecolors="white", linewidths=0.5,
                label=f"对应 D 簇 {d_cluster}（{int(mask.sum())} 人）",
            )
        centers_x = [c["income_wan"] for c in r["centers"]]
        centers_y = [c["balance_wan"] for c in r["centers"]]
        ax.scatter(centers_x, centers_y, marker="*", s=500, c="red",
                   edgecolors="black", linewidths=1.2, zorder=5, label="簇中心（还原后）")

        ax.set_title(f"{cid}：{cfg['desc']}\n人数（原始簇 0/1/2）= {counts}", fontsize=11.5)
        ax.set_xlabel("年收入 income（万元）", fontsize=10)
        ax.set_ylabel("账户平均余额 balance（万元）", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="best")

    fig.suptitle("统一参数（k=3, n_init=10, seed=42, k-means++）下的单位与标准化对照\n"
                 "四图坐标范围相同：income 万元 / balance 万元", fontsize=14)
    fig.text(0.5, 0.012,
             "颜色按最大重叠匹配对齐到 D 簇，仅用于跨配置肉眼对照；颜色相同不代表成员完全相同。原始标签未改写。",
             ha="center", fontsize=9.5, color="#444444")
    plt.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path = OUT_DIR / "fig_compare_4panel.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存 {out_path}（中文字体：{font_name}）")


# =====================================================================
# 主流程
# =====================================================================
def main():
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pre = run_prechecks()
    font_name = choose_font()
    results, _ = run_clustering(pre)
    labels_all, ari_rows, cont_rows, match_rows, tables, display_mapping = compare(results)
    profiles = build_profiles(results)

    # 写出 CSV
    labels_all.to_csv(OUT_DIR / "labels_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ari_rows).to_csv(OUT_DIR / "ari_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cont_rows).to_csv(OUT_DIR / "contingency_tables.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(match_rows).to_csv(OUT_DIR / "overlap_match.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUT_DIR / "profiles_original_units.csv", index=False, encoding="utf-8-sig")
    pre["balance_detail"].to_csv(
        OUT_DIR / "balance_conversion_differences.csv", index=False, encoding="utf-8-sig"
    )

    metrics_rows = []
    for cid in ["A", "B", "C", "D"]:
        r = results[cid]
        metrics_rows.append({
            "config": cid,
            "data_file": f"mall_customers_{r['cfg']['data']}.csv",
            "preprocessing": "StandardScaler" if r["cfg"]["scale"] else "none",
            "feature_order": "+".join(FEATURES),
            "space": SPACE_LABEL[cid],
            "n_samples": int(len(r["df"])),
            "n_clusters": 3,
            **{f"n_cluster_{i}": r["counts"][i] for i in range(3)},
            "inertia": r["inertia"],
            "silhouette": r["silhouette"],
            "silhouette_space": SPACE_LABEL[cid],
            "n_iter": r["n_iter"],
            "max_iter": KM_PARAMS["max_iter"],
            "n_warnings": len(r["warnings"]),
            "warnings": " | ".join(r["warnings"]),
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    plot_compare(results, display_mapping, font_name)

    # C/D 一致性判定：
    # 1) 原始簇编号是否逐项一致，仅作单独记录；
    # 2) 分组是否一致，以 C-D 比较结果为准：最大重叠匹配后变化 0 人且 ARI 接近 1。
    cd_ari = next(row["ari"] for row in ari_rows if row["pair"] == "C-D")
    _, _, cd_matched, cd_changed, cd_total = tables[("C", "D")]
    raw_label_identical = bool(np.array_equal(results["C"]["labels"], results["D"]["labels"]))
    ari_threshold = 1.0 - 1e-9
    c_d_same_grouping = bool(cd_changed == 0 and cd_ari >= ari_threshold)

    print(f"  C/D 原始簇编号逐项一致: {raw_label_identical}")
    print(f"  C/D 忽略编号后的分组一致: {c_d_same_grouping}"
          f"（ARI={cd_ari:.6f}，最大重叠匹配 {cd_matched}/{cd_total}，"
          f"匹配后变化 {cd_changed} 人）")
    if not c_d_same_grouping:
        print("  [异常] C 与 D 的分组不一致：记录结果并停止，不调参凑一致。")

    # 元信息
    tz = timezone(timedelta(hours=8))
    meta = {
        "generated_at": datetime.now(tz).isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai (UTC+8)",
        "script": "practice/00_scale_control.py",
        "command": [sys.executable] + sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "units": {
            "income": "万元（两份数据一致）",
            "balance_sim": "万元",
            "balance_yuan": "元",
            "profile_balance": "同时提供万元与元两列",
            "plot_balance": "万元（元值 ÷ 10000）",
        },
        "data": {
            "files": ["data/mall_customers_sim.csv", "data/mall_customers_yuan.csv"],
            "sha256": pre["hashes"],
            "rows": int(len(pre["sim"])),
            "columns": EXPECTED_COLUMNS,
        },
        "prechecks": {
            "passed": True,
            "customer_id_unique": True,
            "customer_id_no_missing": True,
            "same_customer_set": True,
            "non_balance_field_diff_rows": pre["field_diffs"],
            "balance_max_abs_diff_yuan": pre["balance_max_abs_diff_yuan"],
            "balance_nonzero_rows": pre["balance_nonzero_rows"],
            "balance_total_rows": pre["balance_total_rows"],
            "balance_tolerance_yuan": pre["balance_tolerance_yuan"],
            "note": "浮点差异为 float64 表示误差，已按实际值记录，未四舍五入、未删行、未修复。",
        },
        "features": FEATURES,
        "kmeans_params": KM_PARAMS,
        "standard_scaler_params": SCALER_PARAMS,
        "configs": [
            {
                "id": cid,
                "data_file": f"mall_customers_{results[cid]['cfg']['data']}.csv",
                "description": results[cid]["cfg"]["desc"],
                "preprocessing": "StandardScaler" if results[cid]["cfg"]["scale"] else "none",
                "space": SPACE_LABEL[cid],
                "n_samples": int(len(results[cid]["df"])),
                "cluster_counts": results[cid]["counts"],
                "inertia": results[cid]["inertia"],
                "silhouette": results[cid]["silhouette"],
                "n_iter": results[cid]["n_iter"],
                "centers_original_units": results[cid]["centers"],
                "warnings": results[cid]["warnings"],
                "display_mapping_to_D": display_mapping[cid],
            }
            for cid in ["A", "B", "C", "D"]
        ],
        "comparisons": ari_rows,
        "c_d_raw_label_identical": raw_label_identical,
        "c_d_same_grouping": c_d_same_grouping,
        "c_d_comparison": {
            "ari": cd_ari,
            "ari_threshold": ari_threshold,
            "matched_total": cd_matched,
            "changed_total": cd_changed,
            "total": cd_total,
        },
        "notes": [
            "SSE（inertia）计算空间：A 为 income 万元、balance 万元两列的平方距离和；"
            "B 为 income 万元、balance 元混合单位数值空间的平方距离和；C/D 为标准化空间的平方距离和。"
            "禁止跨尺度直接比较 SSE 大小来判断模型优劣。",
            "silhouette 在各自建模空间计算，同样不可跨空间直接排名。",
            "n_iter_ 小于 max_iter 只表示本次迭代未触上限，不能单独据此宣称已严格证明收敛。",
            "颜色按最大重叠匹配对齐到 D 簇，仅用于可视化；原始标签保存在 labels_all.csv，未改写。",
            "A 与 B 的差异对应单位变化；C 与 D 的差异对应标准化后的单位不变性检查。",
            "C/D 判定区分两件事：c_d_raw_label_identical 表示原始簇编号是否逐项一致；"
            "c_d_same_grouping 表示忽略编号后的分组是否一致，以最大重叠匹配后变化 0 人且 ARI≥1-1e-9 为准。"
            "分组是否一致及停止条件只看后者，不因原始编号不同就判定分组不同。",
            "本实验不加入 age，age 仅用于画像描述。",
        ],
        "outputs": [
            "run_meta.json", "labels_all.csv", "metrics.csv",
            "profiles_original_units.csv", "ari_matrix.csv",
            "contingency_tables.csv", "overlap_match.csv",
            "balance_conversion_differences.csv", "fig_compare_4panel.png",
        ],
    }

    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(meta), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"  已保存 {OUT_DIR / 'run_meta.json'}")

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 78)
    print("五、结果汇总")
    print("=" * 78)
    print(metrics.drop(columns=["warnings"]).to_string(index=False))
    print()
    print(profiles.to_string(index=False))
    print()
    for cid in ["A", "B", "C", "D"]:
        print(f"  配置 {cid} 各簇人数合计: {sum(results[cid]['counts'])}")
    print(f"  C/D 原始簇编号逐项一致: {raw_label_identical}；"
          f"忽略编号后的分组一致: {c_d_same_grouping}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print("=" * 78)

    if not c_d_same_grouping:
        print("[停止] C/D 分组不同（按最大重叠匹配与 ARI 判定），按约定停止本步骤，"
              "等待排查原因；未做任何调参重跑。")
        sys.exit(3)


if __name__ == "__main__":
    main()

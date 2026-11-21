# -*- coding: utf-8 -*-
"""
基础模型交付物导出脚本。
- 输入：data/mall_customers_yuan.csv + control_experiment 目录下已存 D 配置结果
- 不重新训练、不重新标准化、不重新聚类
- 输出：outputs/practice/baseline/ 下的标签、画像、指标、三张图

若预检或导出前核验失败即停止，不修改任何原始数据或既有结果。
"""
import hashlib
import json
import math
import platform
import sys
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

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "outputs" / "practice" / "control_experiment"
OUT_DIR = ROOT / "outputs" / "practice" / "baseline"

YUAN = DATA_DIR / "mall_customers_yuan.csv"
EXPECTED_COLS = ["customer_id", "age", "income", "balance", "product_count", "gender", "city"]
EXPECTED_YUAN_SHA = "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07"
BALANCE_WAN_DIVISOR = 10000.0


def jsonable(obj):
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(msg: str) -> None:
    print(f"[失败] {msg}")
    print("已停止，未写入任何导出文件。")
    sys.exit(2)


def choose_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"中文字体：{name}")
            return name
    fail("未找到 Microsoft YaHei / SimHei，无法正确显示中文")


# =====================================================================
# 预检
# =====================================================================
def precheck():
    print("=" * 72)
    print("预检")
    print("=" * 72)

    for p in (YUAN, EXP_DIR / "labels_all.csv", EXP_DIR / "run_meta.json"):
        if not p.exists():
            fail(f"缺失依赖：{p}")

    actual = sha256(YUAN)
    print(f"  yuan SHA256: {actual}")
    if actual != EXPECTED_YUAN_SHA:
        fail(f"yuan SHA256 与已知基线不一致")

    meta = json.loads((EXP_DIR / "run_meta.json").read_text(encoding="utf-8"))
    if meta["data"]["sha256"]["mall_customers_yuan.csv"] != actual:
        fail("yuan SHA256 与 run_meta.json 记录不一致")

    yuan = pd.read_csv(YUAN)
    labels = pd.read_csv(EXP_DIR / "labels_all.csv")
    if list(yuan.columns) != EXPECTED_COLS:
        fail(f"yuan 列异常：{list(yuan.columns)}")
    if yuan.isnull().sum().sum() != 0:
        fail("yuan 存在缺失")
    if yuan["customer_id"].duplicated().any():
        fail("yuan customer_id 重复")
    print(f"  yuan {len(yuan)} 行 × {len(yuan.columns)} 列，无缺失")

    merged = yuan.merge(
        labels[["customer_id", "label_D"]],
        on="customer_id", how="outer", validate="one_to_one", indicator=True,
    )
    if len(merged) != 300:
        fail(f"合并后行数 {len(merged)} != 300")
    if not (merged["_merge"] == "both").all():
        fail("存在未匹配的客户")
    if merged["label_D"].isna().any():
        fail("label_D 存在缺失")
    if not merged["label_D"].isin([0, 1, 2]).all():
        fail("label_D 存在越界值")
    print(f"  合并 {len(merged)} 行，label_D counts = {sorted(merged['label_D'].value_counts().to_dict().items())}")

    d_cfg = next(c for c in meta["configs"] if c["id"] == "D")
    km = meta["kmeans_params"]
    assert d_cfg["preprocessing"] == "StandardScaler", f"D 配置未用 StandardScaler"
    assert d_cfg["data_file"] == "mall_customers_yuan.csv"
    assert km["n_clusters"] == 3 and km["random_state"] == 42 and km["n_init"] == 10
    assert meta["features"] == ["income", "balance"]
    print(f"  D 配置参数：StandardScaler, k=3, seed=42, n_init=10, features={meta['features']}")

    # 重新汇总画像，与 profiles_original_units.csv 中 D 记录核对
    merged["balance_wan"] = merged["balance"] / BALANCE_WAN_DIVISOR
    rec = []
    for c in range(3):
        m = merged["label_D"] == c
        n = int(m.sum())
        rec.append({
            "cluster": c, "n": n,
            "share_pct": round(n / 300 * 100, 6),
            "age_mean": round(float(merged.loc[m, "age"].mean()), 6),
            "income_wan_mean": round(float(merged.loc[m, "income"].mean()), 6),
            "balance_wan_mean": round(float(merged.loc[m, "balance_wan"].mean()), 6),
            "balance_yuan_mean": round(float(merged.loc[m, "balance"].mean()), 6),
            "product_count_mean": round(float(merged.loc[m, "product_count"].mean()), 6),
        })
    rec_df = pd.DataFrame(rec)
    stored = (
        pd.read_csv(EXP_DIR / "profiles_original_units.csv")
        .query("config == 'D'").sort_values("cluster").reset_index(drop=True)
    )
    for col in ["share_pct", "age_mean", "income_wan_mean", "balance_wan_mean",
                "balance_yuan_mean", "product_count_mean"]:
        ok = np.allclose(rec_df[col].to_numpy(), stored[col].to_numpy(), atol=1e-5)
        if not ok:
            fail(f"重算画像的 {col} 与存储值不一致")

    # 质心与簇均值核对（原始单位）
    centers = pd.DataFrame(d_cfg["centers_original_units"]).sort_values("cluster").reset_index(drop=True)
    centers["balance_yuan_c"] = centers["balance_wan"] * BALANCE_WAN_DIVISOR
    for left, right in [
        ("income_wan_mean", "income_wan"),
        ("balance_wan_mean", "balance_wan"),
        ("balance_yuan_mean", "balance_yuan_c"),
    ]:
        if not np.allclose(rec_df[left].to_numpy(), centers[right].to_numpy(), atol=1e-5):
            fail(f"质心/簇均值对 {left}/{right} 不一致")

    print("  重算画像与存储值一致；D 原始单位质心与簇均值一致")
    print()
    return merged.sort_values("customer_id").reset_index(drop=True), meta, d_cfg


# =====================================================================
# 导出
# =====================================================================
def export(df, meta, d_cfg, font_name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 贯穿导出全程统一使用 cluster 列名，避免 label_D/cluster 混用
    df = df.rename(columns={"label_D": "cluster"}).copy()

    # --- baseline_labels.csv ---
    labels_out = df[EXPECTED_COLS + ["cluster"]]
    labels_out.to_csv(OUT_DIR / "baseline_labels.csv", index=False, encoding="utf-8-sig")
    print("→ baseline_labels.csv 已写")

    # --- baseline_profile.csv：按平均收入排序但保留原簇编号 ---
    prof = []
    df["balance_wan"] = df["balance"] / BALANCE_WAN_DIVISOR
    for c in range(3):
        m = df["cluster"] == c
        n = int(m.sum())
        prof.append({
            "cluster": c,
            "n": n,
            # 未四舍五入值用于后续核验，CSV/表格展示再 round
            "age_mean_raw": float(df.loc[m, "age"].mean()),
            "income_wan_mean_raw": float(df.loc[m, "income"].mean()),
            "balance_wan_mean_raw": float(df.loc[m, "balance_wan"].mean()),
            "balance_yuan_mean_raw": float(df.loc[m, "balance"].mean()),
            "product_count_mean_raw": float(df.loc[m, "product_count"].mean()),
            "share_pct_pct": round(n / 300 * 100, 4),
            "age_mean_years": round(float(df.loc[m, "age"].mean()), 4),
            "income_wan_mean": round(float(df.loc[m, "income"].mean()), 4),
            "balance_yuan_mean": round(float(df.loc[m, "balance"].mean()), 4),
            "balance_wan_mean": round(float(df.loc[m, "balance_wan"].mean()), 4),
            "product_count_mean": round(float(df.loc[m, "product_count"].mean()), 4),
        })
    prof_df = pd.DataFrame(prof).sort_values("income_wan_mean").reset_index(drop=True)

    # ---------- 导出前核验 ----------
    print("  导出前核验（基于未四舍五入数据）")
    # 1) 总人数 300、占比合计约 100%
    n_total = int(prof_df["n"].sum())
    share_sum = float(prof_df["share_pct_pct"].sum())
    assert n_total == 300, f"三簇人数合计 {n_total} != 300"
    assert abs(share_sum - 100.0) < 1e-4, f"占比合计 {share_sum} ≠ 100%"
    print(f"    总人数 300 ✓；占比合计 {share_sum:.6f}% ✓")

    # 2) 总体均值直接用 df 全列 mean()
    overall = {
        "age_mean_years": float(df["age"].mean()),
        "income_wan_mean": float(df["income"].mean()),
        "balance_wan_mean": float(df["balance_wan"].mean()),
        "balance_yuan_mean": float(df["balance"].mean()),
        "product_count_mean": float(df["product_count"].mean()),
    }
    # 加权均值：sum(未四舍五入簇均值 × 簇人数) / 300
    cluster_order = prof_df["cluster"].tolist()  # 即使排序了，也要按原始 cluster 列核对
    counts_by_cluster = {int(c): int((df["cluster"] == c).sum()) for c in sorted(set(df["cluster"]))}
    assert counts_by_cluster == {int(c["cluster"]): int(c["n"]) for c in prof}, \
        "按原始 cluster 列核对的人数与画像表不一致"

    tol = 1e-6  # 容差：未四舍五入值到总体均值的一致性核验
    for field, raw_col, out_col in [
        ("age", "age_mean_raw", "age_mean_years"),
        ("income_wan", "income_wan_mean_raw", "income_wan_mean"),
        ("balance_wan", "balance_wan_mean_raw", "balance_wan_mean"),
        ("balance_yuan", "balance_yuan_mean_raw", "balance_yuan_mean"),
        ("product_count", "product_count_mean_raw", "product_count_mean"),
    ]:
        weighted = sum(
            float(prof_df.loc[prof_df["cluster"] == c, raw_col].iloc[0]) *
            counts_by_cluster[c]
            for c in sorted(counts_by_cluster)
        ) / 300.0
        diff = abs(weighted - overall[out_col])
        assert diff < tol, f"{field}: 加权均值 {weighted} != 总体均值 {overall[out_col]}（差 {diff}）"
        print(f"    {field:<16} 总体={overall[out_col]:.6f} 加权={weighted:.6f} 差={diff:.2e} ✓")

    # 3) 总体平均年龄位于原始 age 最小值与最大值之间
    age_min, age_max = float(df["age"].min()), float(df["age"].max())
    assert age_min <= overall["age_mean_years"] <= age_max, \
        f"总体平均年龄 {overall['age_mean_years']} 不在 [{age_min}, {age_max}] 内"
    print(f"    年龄区间 [{age_min}, {age_max}]，总体均值 {overall['age_mean_years']:.4f} 在内 ✓")

    # 4) 再次按原始 cluster 核对人数（不依赖画像表排序）
    for c in sorted(set(df["cluster"])):
        cnt = int((df["cluster"] == c).sum())
        assert cnt == counts_by_cluster[c], f"簇 {c} 人数 {cnt} != {counts_by_cluster[c]}"
    print("    按原始 cluster 列人数核对 ✓")

    # ---------- 导出 ----------
    prof_df.drop(columns=["age_mean_raw", "income_wan_mean_raw", "balance_wan_mean_raw",
                          "balance_yuan_mean_raw", "product_count_mean_raw"], inplace=True)
    prof_df.to_csv(OUT_DIR / "baseline_profile.csv", index=False, encoding="utf-8-sig")
    print("→ baseline_profile.csv 已写（按平均收入排序，保留原簇编号）")

    # --- baseline_metrics.json ---
    tz = timezone(timedelta(hours=8))
    t_now = datetime.now(tz)
    bm = {
        "exported_at": t_now.isoformat(timespec="seconds"),
        "export_note": "本记录导出 control_experiment 目录下 D 配置的既有结果；"
                       "exported_at 为导出脚本运行时间，不是模型训练时间。"
                       f"原 D 配置生成时间为 {meta['generated_at']}。",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "source_config": {
            "id": "D",
            "preprocessing": "StandardScaler",
            "model_data_file": "mall_customers_yuan.csv",
            "actual_data_file_used_for_export": "mall_customers_yuan.csv",
            "feature_order": meta["features"],
            "cluster_labels_source": str(EXP_DIR / "labels_all.csv"),
            "model_meta_source": str(EXP_DIR / "run_meta.json"),
            "original_generated_at": meta["generated_at"],
            "original_generated_by": "practice/00_scale_control.py",
        },
        "kmeans_params": meta["kmeans_params"],
        "standard_scaler_params": meta["standard_scaler_params"],
        "data_hash_sha256": {
            "mall_customers_yuan.csv": meta["data"]["sha256"]["mall_customers_yuan.csv"],
        },
        "cluster_counts": d_cfg["cluster_counts"],
        "counts_sum": int(sum(d_cfg["cluster_counts"])),
        "metrics_from_original_run": {
            "inertia": d_cfg["inertia"],
            "inertia_space": "标准化空间（z 分数）",
            "silhouette": d_cfg["silhouette"],
            "silhouette_space": "标准化空间（z 分数）",
            "silhouette_note": "不可跨空间直接排名",
            "n_iter": d_cfg["n_iter"],
            "max_iter": meta["kmeans_params"]["max_iter"],
            "note": "inertia/silhouette/n_iter 均来自原始 D 配置运行；"
                    "未在导出脚本中重新训练或重新计算模型指标。",
        },
        "cluster_centers_original_units": d_cfg["centers_original_units"],
    }
    with open(OUT_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(bm), f, ensure_ascii=False, indent=2, allow_nan=False)
    print("→ baseline_metrics.json 已写")

    # --- baseline_scatter.png ---
    centers_pd = pd.DataFrame(d_cfg["centers_original_units"]).sort_values("cluster").reset_index(drop=True)
    centers_pd["balance_plot"] = centers_pd["balance_wan"]  # 图上万元坐标
    colors = ["#DD8452", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(3):
        m = df["cluster"] == c
        ax.scatter(
            df.loc[m, "income"], df.loc[m, "balance_wan"],
            s=46, alpha=0.72, color=colors[c], edgecolors="white", linewidths=0.4,
            label=f"原簇 {c}（{int(m.sum())} 人）",
        )
    ax.scatter(
        centers_pd["income_wan"], centers_pd["balance_plot"],
        marker="*", s=480, c="red", edgecolors="black", linewidths=1.2,
        zorder=5, label="D 原始单位质心（万元坐标）",
    )
    for _, row in centers_pd.iterrows():
        ax.annotate(
            f"质心 c{int(row['cluster'])}",
            (row["income_wan"], row["balance_plot"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8.5,
        )
    ax.set_xlabel("年收入 income（万元）")
    ax.set_ylabel("账户平均余额 balance（万元，元值 ÷ 10000）")
    ax.set_title("基础画像 · D 配置（StandardScaler, k=3, seed=42, n_init=10）\n"
                 "按原簇编号着色；质心来自原始单位还原值")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_scatter.png", dpi=160)
    plt.close(fig)
    print("→ baseline_scatter.png 已写")

    # --- baseline_profile.png：画像表，无背景色 ---
    tbl_rows = prof_df.copy()
    display_cols = [
        ("原簇编号", "cluster"),
        ("人数", "n"),
        ("占比 (%)", "share_pct_pct"),
        ("平均年龄（岁）", "age_mean_years"),
        ("平均收入（万元）", "income_wan_mean"),
        ("平均余额（万元）", "balance_wan_mean"),
        ("平均余额（元）", "balance_yuan_mean"),
        ("平均持有产品数（个）", "product_count_mean"),
    ]

    # 总体行：人数求和、占比合计，其余指标直接使用总体均值（原始 300 位客户的 mean）
    totals = {
        "cluster": "总体/合计",
        "n": int(tbl_rows["n"].sum()),
        "share_pct_pct": round(tbl_rows["share_pct_pct"].sum(), 4),
        "age_mean_years": round(float(df["age"].mean()), 4),
        "income_wan_mean": round(float(df["income"].mean()), 4),
        "balance_wan_mean": round(float(df["balance_wan"].mean()), 4),
        "balance_yuan_mean": round(float(df["balance"].mean()), 4),
        "product_count_mean": round(float(df["product_count"].mean()), 4),
    }
    tbl_out = pd.concat([tbl_rows, pd.DataFrame([totals])], ignore_index=True)

    header_text = [d[0] for d in display_cols]
    cell_text = []
    for _, row in tbl_out.iterrows():
        line = []
        for col_label, col_key in display_cols:
            v = row[col_key]
            if col_key in ("cluster", "n"):
                line.append(str(int(v)) if isinstance(v, (int, np.integer)) else str(v))
            elif col_key == "share_pct_pct":
                line.append(f"{float(v):.2f}")
            else:
                line.append(f"{float(v):.2f}")
        cell_text.append(line)

    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=header_text, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("none")
        cell.set_edgecolor("#888888")
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(
        "基础画像 · D 配置原始单位画像表（按平均收入升序，保留原簇编号）",
        fontsize=12, pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_profile.png", dpi=160)
    plt.close(fig)
    print("→ baseline_profile.png 已写")

    # --- baseline_profile_bars.png：四个子图，各自纵轴 ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    axes = axes.ravel()
    cluster_labels = [f"原簇 {int(c)}" for c in prof_df["cluster"]]
    x_pos = np.arange(len(prof_df))
    width = 0.55

    panels = [
        ("平均年龄（岁）", "age_mean_years", "#DD8452"),
        ("平均收入（万元）", "income_wan_mean", "#4C72B0"),
        ("平均余额（万元）", "balance_wan_mean", "#55A868"),
        ("平均持有产品数（个）", "product_count_mean", "#C44E52"),
    ]
    for ax, (title, col, color) in zip(axes, panels):
        ax.bar(x_pos, prof_df[col], width=width, color=color, edgecolor="black", linewidth=0.6)
        for i, v in enumerate(prof_df[col]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cluster_labels)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.15)

    fig.suptitle(
        "基础画像 · D 配置分簇指标（各用自己的单位与纵轴，未堆到同一数值轴）\n"
        "按平均收入升序，保留原簇编号",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "baseline_profile_bars.png", dpi=160)
    plt.close(fig)
    print("→ baseline_profile_bars.png 已写")
    print()
    print("导出完成。")
    print(f"输出目录: {OUT_DIR}")
    return prof_df


# =====================================================================
# 主流程
# =====================================================================
def main():
    merged, meta, d_cfg = precheck()
    font_name = choose_font()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        prof_df = export(merged, meta, d_cfg, font_name)
    for w in wlist:
        print(f"[warn] {w.category.__name__}: {w.message}")


if __name__ == "__main__":
    main()

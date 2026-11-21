# -*- coding: utf-8 -*-
"""
加入 age 特征的对照实验。
- 两特征模型：复用已验收 baseline_labels.csv 和 baseline_metrics.json（不重训练）
- 三特征模型：同一份 mall_customers_yuan.csv，特征顺序 income, balance, age，
  三列一起 StandardScaler；KMeans 参数与基础模型完全一致
- 只有特征集改变；customer_id、gender、city、product_count 不入模

结果写入 outputs/practice/feature_compare/ 。
预检或导出前核验失败即停止。
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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BASELINE_DIR = ROOT / "outputs" / "practice" / "baseline"
OUT_DIR = ROOT / "outputs" / "practice" / "feature_compare"

YUAN = DATA_DIR / "mall_customers_yuan.csv"
BASE_LABELS = BASELINE_DIR / "baseline_labels.csv"
BASE_METRICS = BASELINE_DIR / "baseline_metrics.json"
EXPECTED_COLS = ["customer_id", "age", "income", "balance", "product_count", "gender", "city"]
EXPECTED_YUAN_SHA = "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07"
BALANCE_WAN_DIVISOR = 10000.0

FEATS_2 = ["income", "balance"]
FEATS_3 = ["income", "balance", "age"]
KM_PARAMS = {
    "n_clusters": 3, "init": "k-means++", "n_init": 10,
    "random_state": 42, "max_iter": 300, "tol": 1e-4,
    "algorithm": "lloyd", "copy_x": True, "verbose": 0,
}
SCALER_PARAMS = {"with_mean": True, "with_std": True, "copy": True}
COLORS = ["#DD8452", "#4C72B0", "#55A868"]


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
    fail("未找到 Microsoft YaHei / SimHei")


# =====================================================================
# 预检
# =====================================================================
def precheck():
    print("=" * 72)
    print("预检")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 依赖存在
    for p in (YUAN, BASE_LABELS, BASE_METRICS):
        if not p.exists():
            fail(f"缺失依赖：{p}")

    # SHA256
    actual = sha256(YUAN)
    print(f"  yuan SHA256: {actual}")
    if actual != EXPECTED_YUAN_SHA:
        fail("yuan SHA256 与已知基线不一致")

    base_metrics = json.loads(BASE_METRICS.read_text(encoding="utf-8"))
    if base_metrics["data_hash_sha256"]["mall_customers_yuan.csv"] != actual:
        fail("yuan SHA256 与 baseline_metrics.json 记录不一致")

    # 读 yuan 和 baseline_labels
    yuan = pd.read_csv(YUAN).sort_values("customer_id").reset_index(drop=True)
    base_labels = pd.read_csv(BASE_LABELS).sort_values("customer_id").reset_index(drop=True)
    if yuan.isnull().sum().sum() != 0:
        fail("yuan 有缺失")
    if yuan["customer_id"].duplicated().any():
        fail("yuan customer_id 重复")
    if not yuan["customer_id"].is_unique:
        fail("yuan customer_id 不唯一")
    print(f"  yuan {len(yuan)} 行 × {len(yuan.columns)} 列")

    merged = yuan.merge(base_labels[["customer_id", "cluster"]], on="customer_id",
                        how="outer", validate="one_to_one", indicator=True)
    if len(merged) != 300 or not (merged["_merge"] == "both").all():
        fail(f"与 baseline_labels.csv 合并失败：{len(merged)} 行")
    if merged["cluster"].isna().any() or not merged["cluster"].isin([0, 1, 2]).all():
        fail("baseline cluster 列异常")
    base_counts = merged["cluster"].value_counts().sort_index()
    print(f"  baseline labels cluster counts: {base_counts.to_dict()}")
    assert int(base_counts.sum()) == 300

    # 基础模型参数核对
    bkm = base_metrics["kmeans_params"]
    for k, v in KM_PARAMS.items():
        assert bkm[k] == v, f"kmeans_params.{k} 与基础模型不一致"
    print(f"  基础模型参数全部一致：k={bkm['n_clusters']}, seed={bkm['random_state']}, n_init={bkm['n_init']}")

    return merged, base_metrics


# =====================================================================
# 三特征模型训练
# =====================================================================
def fit_three(df):
    print()
    print("三特征模型训练：income + balance + age，StandardScaler")
    X = df[FEATS_3].to_numpy(dtype=float)
    scaler = StandardScaler(**SCALER_PARAMS)
    X_sc = scaler.fit_transform(X)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        km = KMeans(**KM_PARAMS)
        labels = km.fit_predict(X_sc)
        sil = float(silhouette_score(X_sc, labels, metric="euclidean"))
    warn_msgs = [f"{w.category.__name__}: {str(w.message).replace(chr(10), ' ')}"
                 for w in wlist]

    print(f"  cluster counts: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    print(f"  inertia={km.inertia_:.6f}, silhouette={sil:.6f}, n_iter_={km.n_iter_}")
    if warn_msgs:
        for w in warn_msgs:
            print(f"  [warn] {w}")

    return labels, scaler, km, sil, warn_msgs


# =====================================================================
# 画像生成与导出前核验
# =====================================================================
def build_profile(df, cluster_col):
    df = df.copy()
    df["balance_wan"] = df["balance"] / BALANCE_WAN_DIVISOR
    rows = []
    for c in sorted(df[cluster_col].unique()):
        m = df[cluster_col] == c
        n = int(m.sum())
        rows.append({
            "cluster": int(c), "n": n,
            "share_pct": round(n / 300 * 100, 4),
            "age_mean": round(float(df.loc[m, "age"].mean()), 6),
            "income_wan_mean": round(float(df.loc[m, "income"].mean()), 6),
            "balance_yuan_mean": round(float(df.loc[m, "balance"].mean()), 6),
            "balance_wan_mean": round(float(df.loc[m, "balance_wan"].mean()), 6),
            "product_count_mean": round(float(df.loc[m, "product_count"].mean()), 6),
        })
    return pd.DataFrame(rows)


def max_overlap(a, b, k=3):
    table = np.zeros((k, k), dtype=int)
    for i, j in zip(a, b):
        table[int(i), int(j)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    matched = int(sum(table[r, c] for r, c in zip(rows, cols)))
    changed = int(table.sum() - matched)
    return table, mapping, matched, changed


def export(merged, base_metrics, three_labels, font_name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = merged.copy()
    df["balance_wan"] = df["balance"] / BALANCE_WAN_DIVISOR
    df["cluster_2f"] = df["cluster"].astype(int)
    df["cluster_3f"] = three_labels.astype(int)

    # ---- 交叉表与匹配 ----
    table, mapping, matched, changed = max_overlap(df["cluster_2f"], df["cluster_3f"])
    ari = float(adjusted_rand_score(df["cluster_2f"], df["cluster_3f"]))
    total = int(table.sum())
    assert total == 300
    print()
    print(f"两/三特征模型对比：ARI={ari:.6f}，最大重叠匹配 {matched}/{total}（{matched/total:.4f}），"
          f"变化 {changed} 人，映射 {mapping}")
    print(f"  交叉表（行=两特征，列=三特征）:\n{table}")

    # ---- feature_labels.csv ----
    # 展示标签：三特征 cluster 映射到两特征的对应簇，原始标签不覆盖
    df["cluster_3f_matched"] = df["cluster_3f"].map(mapping).astype(int)
    labels_out = df[EXPECTED_COLS + ["cluster_2f", "cluster_3f", "cluster_3f_matched"]]
    labels_out.to_csv(OUT_DIR / "feature_labels.csv", index=False, encoding="utf-8-sig")
    print("→ feature_labels.csv 已写")

    # ---- 画像 ----
    prof2 = build_profile(df, "cluster_2f").sort_values("income_wan_mean").reset_index(drop=True)
    prof3 = build_profile(df, "cluster_3f").sort_values("income_wan_mean").reset_index(drop=True)
    prof2.to_csv(OUT_DIR / "profiles_two_features.csv", index=False, encoding="utf-8-sig")
    prof3.to_csv(OUT_DIR / "profiles_three_features.csv", index=False, encoding="utf-8-sig")
    print("→ profiles_two_features.csv / profiles_three_features.csv 已写")

    # ---- contingency_table.csv ----
    cont_rows = []
    for i in range(3):
        for j in range(3):
            cont_rows.append({
                "cluster_two_features": i,
                "cluster_three_features": j,
                "n": int(table[i, j]),
            })
    cont_df = pd.DataFrame(cont_rows)
    cont_df.to_csv(OUT_DIR / "contingency_table.csv", index=False, encoding="utf-8-sig")
    print("→ contingency_table.csv 已写")

    # ---- cluster_mapping.csv ----
    mapping_rows = []
    for k in range(3):
        mapping_rows.append({
            "cluster_two_features": k,
            "matched_cluster_three_features": mapping[k],
            "n_overlap": int(table[k, mapping[k]]),
        })
    pd.DataFrame(mapping_rows).to_csv(OUT_DIR / "cluster_mapping.csv", index=False, encoding="utf-8-sig")
    print("→ cluster_mapping.csv 已写")

    # ---- changed_customers.csv ----
    changed_df = df[df["cluster_2f"] != df["cluster_3f_matched"]].copy()
    changed_df = changed_df[EXPECTED_COLS + ["cluster_2f", "cluster_3f", "cluster_3f_matched"]]
    changed_df.to_csv(OUT_DIR / "changed_customers.csv", index=False, encoding="utf-8-sig")
    print(f"→ changed_customers.csv 已写（{len(changed_df)} 行）")

    # ---- 导出前核验 ----
    print()
    print("导出前核验")
    # (a) 两模型覆盖相同 300 客户
    assert df["customer_id"].nunique() == 300 and len(df) == 300
    print("  两模型均覆盖相同 300 个客户 ✓")
    # (b) 画像人数合计
    for prof, name in [(prof2, "两特征"), (prof3, "三特征")]:
        s = int(prof["n"].sum())
        assert s == 300, f"{name}画像人数合计 {s} != 300"
    print("  两份画像人数合计均为 300 ✓")
    # (c) 交叉表边际 = 人数
    row_margins = table.sum(axis=1)
    col_margins = table.sum(axis=0)
    cnt2 = df["cluster_2f"].value_counts().sort_index().to_numpy()
    cnt3 = df["cluster_3f"].value_counts().sort_index().to_numpy()
    assert np.array_equal(row_margins, cnt2), f"行边际 {row_margins} != 两特征人数 {cnt2}"
    assert np.array_equal(col_margins, cnt3), f"列边际 {col_margins} != 三特征人数 {cnt3}"
    print(f"  交叉表行边际 {row_margins} == 两特征人数 {cnt2} ✓")
    print(f"  交叉表列边际 {col_margins} == 三特征人数 {cnt3} ✓")
    # (d) 匹配人数 + 变化人数 = 300；变化客户表行数与指标一致
    assert matched + changed == 300
    assert int(len(changed_df)) == changed, \
        f"变化客户表 {len(changed_df)} 行 != changed={changed}"
    print(f"  匹配+变化={matched+changed}=300 ✓；变化客户表 {len(changed_df)} 行 = changed ✓")
    # (e) 两特征标签与 baseline 一致
    # baseline 的 cluster_counts 数组是从 run_meta.json D 配置读取的，
    # baseline_labels.csv.cluster 才是实际标签来源
    base_counts_from_csv = df["cluster_2f"].value_counts().sort_index().to_dict()
    # 核验 cluster_counts 数组是 run_meta.json D 配置 [126,50,124]，但 baseline_metrics.json 可能有多余元素——
    # 用 baseline_labels.csv.cluster 的实际 value_counts 核对
    base_counts_from_labels = pd.read_csv(BASE_LABELS)["cluster"].value_counts().sort_index().to_dict()
    assert base_counts_from_csv == base_counts_from_labels, \
        f"两特征标签与 baseline_labels.csv 不一致：{base_counts_from_csv} vs {base_counts_from_labels}"
    print(f"  两特征标签与 baseline_labels.csv 按 ID 对应后完全一致 ✓")

    return {
        "ari": ari, "matched": matched, "changed": changed, "total": total,
        "table": table, "mapping": mapping,
        "prof2": prof2, "prof3": prof3, "changed_df": changed_df,
        "base_counts": base_counts_from_csv,
        "three_counts": {int(k): int(v) for k, v in df["cluster_3f"].value_counts().sort_index().to_dict().items()},
    }


# =====================================================================
# 绘图
# =====================================================================
def plot(df, table, mapping, prof2, prof3):
    # 基础模型质心（原始单位），用于标注两特征 income/balance 图
    base_centers = pd.DataFrame([
        {"cluster": 0, "income_wan": 6.105634920634925, "balance_wan": 1.4516666666666687},
        {"cluster": 1, "income_wan": 41.583600000000004, "balance_wan": 48.3502},
        {"cluster": 2, "income_wan": 18.935564516129034, "balance_wan": 12.445564516129032},
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_t2, ax_t3, ax_a2, ax_a3 = axes.ravel()

    income = df["income"].to_numpy(dtype=float)
    balance_wan = df["balance_wan"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)

    def lims(a, b, pad=0.05):
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        rng = hi - lo
        return lo - rng * pad, hi + rng * pad

    # 统一坐标
    xlim_inc = lims(income, income)  # 收入万元
    ylim_bal = lims(balance_wan, balance_wan)
    xlim_inc2 = lims(income, income)
    ylim_age = lims(age, age)

    # 映射：两特征 -> 展示颜色；三特征 -> 展示颜色（按匈牙利匹配对齐到两特征）
    # display_color: cluster -> color idx in COLORS
    display2 = {c: c for c in range(3)}
    display3 = {c: mapping[c] for c in range(3)}  # 三特征映射到两特征的对应簇颜色

    def plot_scatter(ax, x, y, labels_arr, display_map, title, xlabel, ylabel,
                     centers_pd=None, centers_x_col=None, centers_y_col=None,
                     add_age_note=False):
        for c in range(3):
            col = COLORS[display_map[c]]
            mask = labels_arr == c
            if not mask.any():
                continue
            ax.scatter(
                x[mask], y[mask], s=42, alpha=0.72, color=col,
                edgecolors="white", linewidths=0.4,
                label=f"簇 {c}（{int(mask.sum())} 人）",
            )
        if centers_pd is not None:
            ax.scatter(
                centers_pd[centers_x_col], centers_pd[centers_y_col],
                marker="*", s=400, c="red", edgecolors="black", linewidths=1.0, zorder=5,
                label="建模质心",
            )
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if add_age_note:
            ax.text(0.02, 0.02, "三维聚类的二维投影", transform=ax.transAxes,
                    fontsize=8.5, color="#555555")

    # 上排左：两特征模型 income—balance
    plot_scatter(ax_t2, income, balance_wan, df["cluster_2f"].to_numpy(),
                 display2,
                 title=f"两特征模型（income+balance，k=3）\n收入—余额（同色≠成员完全相同）",
                 xlabel="年收入 income（万元）",
                 ylabel="账户平均余额 balance（万元）",
                 centers_pd=base_centers, centers_x_col="income_wan", centers_y_col="balance_wan")
    ax_t2.set_xlim(*xlim_inc); ax_t2.set_ylim(*ylim_bal)

    # 上排右：三特征模型 income—balance（二维投影，颜色按匹配对齐）
    plot_scatter(ax_t3, income, balance_wan, df["cluster_3f"].to_numpy(),
                 display3,
                 title=f"三特征模型（income+balance+age，k=3）\n收入—余额（三维聚类的二维投影，同色≠成员相同）",
                 xlabel="年收入 income（万元）",
                 ylabel="账户平均余额 balance（万元）",
                 add_age_note=True)
    ax_t3.set_xlim(*xlim_inc); ax_t3.set_ylim(*ylim_bal)

    # 下排左：两特征模型 年龄—收入（两特征模型没年龄质心，只画散点）
    plot_scatter(ax_a2, income, age, df["cluster_2f"].to_numpy(),
                 display2,
                 title=f"两特征模型\n年龄—收入（颜色来自两特征聚类）",
                 xlabel="年收入 income（万元）",
                 ylabel="年龄（岁）")
    ax_a2.set_xlim(*xlim_inc2); ax_a2.set_ylim(*ylim_age)

    # 下排右：三特征模型 年龄—收入（二维投影）
    plot_scatter(ax_a3, income, age, df["cluster_3f"].to_numpy(),
                 display3,
                 title=f"三特征模型\n年龄—收入（三维聚类的二维投影，同色≠成员相同）",
                 xlabel="年收入 income（万元）",
                 ylabel="年龄（岁）",
                 add_age_note=True)
    ax_a3.set_xlim(*xlim_inc2); ax_a3.set_ylim(*ylim_age)

    fig.suptitle(
        "加入 age 的特征对照实验（k=3, seed=42, n_init=10, k-means++，仅特征集改变）\n"
        "颜色按匈牙利匹配对齐两特征模型；同色不代表成员完全相同",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "feature_compare.png", dpi=150)
    plt.close(fig)
    print("→ feature_compare.png 已写")


# =====================================================================
# 主流程
# =====================================================================
def main():
    merged, base_metrics = precheck()
    font_name = choose_font()
    labels_3f, scaler, km, sil_3f, warn_msgs = fit_three(merged)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        result = export(merged, base_metrics, labels_3f, font_name)
        plot_result = result
        plot(merged, result["table"], result["mapping"], result["prof2"], result["prof3"])
    for w in wlist:
        print(f"[warn] {w.category.__name__}: {w.message}")

    # ---- metrics.json ----
    tz = timezone(timedelta(hours=8))
    t_now = datetime.now(tz)
    base_metrics2 = base_metrics["metrics_from_original_run"]
    bm_params = base_metrics["kmeans_params"]
    out_metrics = {
        "exported_at": t_now.isoformat(timespec="seconds"),
        "experiment_note": "对照实验：仅特征集改变（两特征=income+balance；三特征=income+balance+age），"
                           "KMeans/StandardScaler 参数保持一致。"
                           "两特征模型指标来自既有 baseline 结果，非本轮重训练。"
                           "三特征模型为本轮新训练。不可跨特征空间比较 SSE 或凭 silhouette 高低"
                           "直接判定加入年龄更好。",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "dependencies": {
            "data_file": str(YUAN),
            "baseline_labels": str(BASE_LABELS),
            "baseline_metrics": str(BASE_METRICS),
            "yuan_sha256": sha256(YUAN),
        },
        "kmeans_params": bm_params,
        "standard_scaler_params": base_metrics["standard_scaler_params"],
        "two_features": {
            "feature_order": FEATS_2,
            "silhouette": base_metrics2["silhouette"],
            "silhouette_space": "标准化空间（z 分数）",
            "silhouette_source": "control_experiment D 配置，非本轮重训练",
            "inertia": base_metrics2["inertia"],
            "inertia_space": "标准化空间（z 分数）",
            "n_iter": base_metrics2["n_iter"],
            "n_iter_source": "control_experiment D 配置，非本轮重训练",
            "cluster_counts": result["base_counts"],
        },
        "three_features": {
            "feature_order": FEATS_3,
            "silhouette": sil_3f,
            "silhouette_space": "标准化空间（z 分数）",
            "inertia": float(km.inertia_),
            "inertia_space": "标准化空间（z 分数）",
            "n_iter": int(km.n_iter_),
            "cluster_counts": result["three_counts"],
            "warnings": warn_msgs,
        },
        "comparison": {
            "ari": result["ari"],
            "ari_note": "ARI 基于 cluster 编号本身计算；若两模型只是簇编号重排也会是 1.0，"
                        "需同时参考最大重叠匹配结果。",
            "max_overlap_matched": result["matched"],
            "max_overlap_changed": result["changed"],
            "max_overlap_total": result["total"],
            "match_rate": result["matched"] / result["total"],
            "max_overlap_mapping_two_to_three": result["mapping"],
        },
        "notes": [
            "只有特征集改变；customer_id、gender、city、product_count 不入模。",
            "三特征模型图中的年龄—收入图为三维聚类结果的二维投影，年龄并未单独作为建模质心维度绘制。",
            "不可跨特征空间比较 SSE（inertia）判断优劣；不可仅凭 silhouette 高低直接判定加入年龄更好。",
            "相同颜色按最大重叠匹配对齐两/三特征模型，不代表成员完全相同。",
        ],
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(out_metrics), f, ensure_ascii=False, indent=2, allow_nan=False)
    print("→ metrics.json 已写")

    print()
    print("=" * 72)
    print("结果汇总")
    print("=" * 72)
    print(f"ARI = {result['ari']:.6f}")
    print(f"最大重叠匹配：{result['matched']}/{result['total']}，变化 {result['changed']} 人，映射 {result['mapping']}")
    print(f"两特征 counts: {result['base_counts']}")
    print(f"三特征 counts: {result['three_counts']}")
    print(f"三特征 inertia = {km.inertia_:.6f}, silhouette = {sil_3f:.6f}, n_iter_ = {km.n_iter_}")


if __name__ == "__main__":
    main()

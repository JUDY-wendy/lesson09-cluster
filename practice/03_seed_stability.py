# -*- coding: utf-8 -*-
"""
三随机种子稳定性实验。
- 数据：同一份 mall_customers_yuan.csv（按 customer_id 对齐），300 客户
- 特征固定 income, balance, age，按此顺序做一次 StandardScaler
- KMeans 参数从 feature_compare/metrics.json 读取并核对；仅 random_state 分别为 0, 42, 2026
- 三次均实际拟合，不预设结果
- 展示标签以 seed42 为参照：匈牙利映射行=当前 seed、列=42，保存 current_seed → seed42 方向
- seed42 拟合结果与 feature_compare/feature_labels.csv 的 cluster_3f 按 customer_id 比对，
  应 ARI=1 且匹配后变化 0（不以原编号相同为必要条件），不一致立即停止

设计：全脚本只准备一份带全部字段的完整 DataFrame 贯穿计算、导出和绘图，避免局部列作用域错误。
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
FC_DIR = ROOT / "outputs" / "practice" / "feature_compare"
OUT_DIR = ROOT / "outputs" / "practice" / "stability"

YUAN = DATA_DIR / "mall_customers_yuan.csv"
FC_METRICS = FC_DIR / "metrics.json"
FC_LABELS = FC_DIR / "feature_labels.csv"
EXPECTED_YUAN_SHA = "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07"
BALANCE_WAN_DIVISOR = 10000.0
FEATURES = ["income", "balance", "age"]
SEEDS = [0, 42, 2026]
SEED42_DISPLAY_REF = 42
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
            raise ValueError(f"JSON 非有限数值: {obj}")
        return obj
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(msg: str, already_written: str = "") -> None:
    print(f"[失败] {msg}")
    if already_written:
        print(f"已写入文件：{already_written}（请手工检查，本脚本未自动清理）")
    else:
        print("已停止，未写入任何文件。")
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


def max_overlap(a, b, k=3):
    """a, b 为簇编号数组，返回 (交叉表 k×k int, a→b 映射 dict, 匹配人数, 变化人数)"""
    table = np.zeros((k, k), dtype=int)
    for i, j in zip(a, b):
        table[int(i), int(j)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    matched = int(sum(table[r, c] for r, c in zip(rows, cols)))
    changed = int(table.sum() - matched)
    return table, mapping, matched, changed


def prepare_data():
    """读取 yuan、feature_compare 三特征 seed42 标签，校验哈希，返回完整 df 和参数。"""
    print("=" * 72)
    print("预检")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in (YUAN, FC_METRICS, FC_LABELS):
        if not p.exists():
            fail(f"缺失依赖：{p}")

    # SHA
    actual = sha256(YUAN)
    print(f"  yuan SHA256: {actual}")
    if actual != EXPECTED_YUAN_SHA:
        fail("yuan SHA256 与已知基线不一致")

    yuan = pd.read_csv(YUAN).sort_values("customer_id").reset_index(drop=True)
    if yuan.isnull().sum().sum() != 0:
        fail("yuan 有缺失")
    if not yuan["customer_id"].is_unique:
        fail("yuan customer_id 不唯一")
    print(f"  yuan {len(yuan)} 行 × {len(yuan.columns)} 列")

    # 读 feature_compare 的 metrics.json 和 feature_labels.csv
    fc_metrics = json.loads(FC_METRICS.read_text(encoding="utf-8"))
    fc_labels = pd.read_csv(FC_LABELS).sort_values("customer_id").reset_index(drop=True)
    if fc_metrics["dependencies"]["yuan_sha256"] != actual:
        fail("yuan SHA256 与 feature_compare/metrics.json 记录不一致")
    if not {"customer_id", "cluster_3f"}.issubset(fc_labels.columns):
        fail("feature_labels.csv 缺少 customer_id / cluster_3f")

    # 参数从 metrics.json 读取并核对
    km_orig = fc_metrics["kmeans_params"]
    sc_orig = fc_metrics["standard_scaler_params"]
    feat_order = fc_metrics["three_features"]["feature_order"]
    if feat_order != FEATURES:
        fail(f"feature_compare 三特征顺序 {feat_order} != 预期 {FEATURES}")

    # 核对 km_orig 各字段（除 random_state 之外的字段必须保持）
    expected_fixed = {
        "n_clusters": 3, "init": "k-means++", "n_init": 10,
        "max_iter": 300, "tol": 1e-4, "algorithm": "lloyd",
        "copy_x": True, "verbose": 0,
    }
    for k, v in expected_fixed.items():
        if km_orig[k] != v:
            fail(f"kmeans_params.{k}={km_orig[k]} != 固定值 {v}")
    # sc
    expected_sc = {"with_mean": True, "with_std": True, "copy": True}
    for k, v in expected_sc.items():
        if sc_orig[k] != v:
            fail(f"standard_scaler_params.{k}={sc_orig[k]} != {v}")

    # 合并 yuan + fc_labels seed42
    merged = yuan.merge(fc_labels[["customer_id", "cluster_3f"]], on="customer_id",
                        how="outer", validate="one_to_one", indicator=True)
    if len(merged) != 300 or not (merged["_merge"] == "both").all():
        fail("yuan 与 feature_labels.csv 合并失败")
    merged = merged.drop(columns=["_merge"]).rename(columns={"cluster_3f": "seed42_cluster_from_fc"})
    print(f"  feature_compare seed42 标签合并成功，分布："
          f"{merged['seed42_cluster_from_fc'].value_counts().sort_index().to_dict()}")

    # 从 merged 构造"完整数据"：balance_wan 预先算好（画像和绘图共用）
    merged["balance_wan"] = merged["balance"] / BALANCE_WAN_DIVISOR
    # StandardScaler：对 income, balance, age 三列（按 FEATURES 顺序）一起做一次 fit_transform
    scaler = StandardScaler(**sc_orig)
    X_raw = merged[FEATURES].to_numpy(dtype=float)
    X_sc = scaler.fit_transform(X_raw)
    print(f"  StandardScaler fit 完成，shape={X_sc.shape}，mean≈0、std≈1")

    return {
        "df": merged, "X_raw": X_raw, "X_sc": X_sc,
        "scaler": scaler, "km_orig": km_orig, "sc_params": sc_orig,
        "yuan_sha": actual,
    }


def fit_all(data):
    """对三个 seed 各拟合一次，返回 dict seed -> {labels, inertia, silhouette, n_iter, counts, warnings}"""
    df = data["df"]
    X_sc = data["X_sc"]
    km_base = data["km_orig"]

    results = {}
    for sd in SEEDS:
        print()
        print(f"KMeans 拟合：random_state={sd}，k=3, n_init=10, k-means++")
        km_params = {**km_base, "random_state": sd}
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            km = KMeans(**km_params)
            labels = km.fit_predict(X_sc)
            sil = float(silhouette_score(X_sc, labels, metric="euclidean"))
        warn_msgs = [f"{w.category.__name__}: {str(w.message).replace(chr(10), ' ')}"
                     for w in wlist]
        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        print(f"  counts: {counts}, inertia={km.inertia_:.6f}, silhouette={sil:.6f}, n_iter_={km.n_iter_}")
        results[sd] = {
            "labels": labels,
            "inertia": float(km.inertia_),
            "silhouette": sil,
            "n_iter": int(km.n_iter_),
            "counts": counts,
            "warnings": warn_msgs,
            "km": km,
        }
    return results


def pairwise_and_display(df, results):
    """
    为三对 seed 做匈牙利匹配；每个 seed 对 seed42 做匈牙利匹配以获得"当前 seed → seed42"的
    展示映射（方向明确：行=当前 seed, 列=42）。
    同时对 seed42 的拟合结果与 feature_compare 里的 seed42 标签做 ARI 和匹配核验。
    """
    seed_labels = {sd: results[sd]["labels"] for sd in SEEDS}

    pairwise = []
    for i, sd_a in enumerate(SEEDS):
        for sd_b in SEEDS[i+1:]:
            ari = float(adjusted_rand_score(seed_labels[sd_a], seed_labels[sd_b]))
            table, mapping_ab, matched, changed = max_overlap(seed_labels[sd_a], seed_labels[sd_b])
            pairwise.append({
                "pair": f"{sd_a}_vs_{sd_b}",
                "seed_a": sd_a, "seed_b": sd_b,
                "ari": ari,
                "matched": matched, "changed": changed, "total": int(table.sum()),
                "match_rate": matched / table.sum(),
                "mapping_a_to_b": mapping_ab,
                # 交叉表单独写 contingency_tables.csv，此处只放摘要
                "contingency_table": table,
            })
            print(f"  {sd_a} vs {sd_b}: ARI={ari:.6f}, 匹配 {matched}, 变化 {changed}")

    # 展示映射：每个 seed → seed42（行=当前 seed, 列=42）
    display_maps = {}
    for sd in SEEDS:
        _, m, _, _ = max_overlap(seed_labels[sd], seed_labels[SEED42_DISPLAY_REF])
        display_maps[sd] = m  # current → 42
        print(f"  展示映射 seed{sd} → seed{SEED42_DISPLAY_REF}: {m}")

    # seed42 拟合 vs feature_compare 里的 seed42 标签
    ari_42 = float(adjusted_rand_score(seed_labels[42], df["seed42_cluster_from_fc"].to_numpy()))
    _, m42, matched42, changed42 = max_overlap(
        seed_labels[42], df["seed42_cluster_from_fc"].to_numpy()
    )
    print()
    print(f"seed42 拟合 vs feature_compare seed42 标签：ARI={ari_42:.6f}，匹配 {matched42}/300，变化 {changed42}")
    # 核验：ARI=1 且 matched=300、changed=0
    if not (ari_42 >= 0.999999 and matched42 == 300 and changed42 == 0):
        fail("seed42 拟合结果与 feature_compare/feature_labels.csv.cluster_3f 分组不一致 "
             f"（ARI={ari_42}, matched={matched42}, changed={changed42}）。"
             f"不调参凑一致，停止。")
    print("  → 分组一致（ARI=1，匹配后变化 0），通过核验")

    return pairwise, display_maps, {
        "ari": ari_42, "mapping_42_to_fc": m42,
        "matched": matched42, "changed": changed42,
    }


def build_profiles(df, seed_labels):
    """按 seed 和原 cluster 构造画像，同时加一列 display_cluster（按 display_maps 对齐到 seed42 颜色）。"""
    df_out = df.copy()
    df_out["balance_wan"] = df_out["balance"] / BALANCE_WAN_DIVISOR
    rows = []
    for sd in SEEDS:
        df_out[f"cluster_{sd}"] = seed_labels[sd].astype(int)
    # 展示映射后面才进来（pairwise_and_display 之后），画像时暂时只存原编号；
    # 完整画像 CSV 在 export 中再加上 display_cluster 列。这里先返回 df_out 以便后续聚合。
    return df_out


def export(df, results, pairwise, display_maps, fc_check, params):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 各 seed 标签 CSV ----
    all_cols = ["customer_id", "age", "income", "balance", "product_count", "gender", "city"]
    for sd in SEEDS:
        df[f"cluster_{sd}"] = results[sd]["labels"].astype(int)
        df[f"display_cluster_{sd}"] = df[f"cluster_{sd}"].map(display_maps[sd]).astype(int)
    for sd in SEEDS:
        out = df[all_cols + [f"cluster_{sd}", f"display_cluster_{sd}"]].copy()
        out.to_csv(OUT_DIR / f"labels_seed_{sd}.csv", index=False, encoding="utf-8-sig")
        print(f"→ labels_seed_{sd}.csv 已写")

    # ---- profiles.csv（长表：seed + 原 cluster + 画像 + 展示簇） ----
    prof_rows = []
    for sd in SEEDS:
        col = f"cluster_{sd}"
        dcol = f"display_cluster_{sd}"
        for c in sorted(df[col].unique()):
            m = df[col] == c
            n = int(m.sum())
            prof_rows.append({
                "seed": sd,
                "cluster": int(c),
                "display_cluster": int(df.loc[m, dcol].iloc[0]),  # 同一簇 display 值相同
                "n": n,
                "share_pct": round(n / 300 * 100, 4),
                "age_mean": round(float(df.loc[m, "age"].mean()), 6),
                "income_wan_mean": round(float(df.loc[m, "income"].mean()), 6),
                "balance_yuan_mean": round(float(df.loc[m, "balance"].mean()), 6),
                "balance_wan_mean": round(float(df.loc[m, "balance_wan"].mean()), 6),
                "product_count_mean": round(float(df.loc[m, "product_count"].mean()), 6),
            })
    pd.DataFrame(prof_rows).to_csv(OUT_DIR / "profiles.csv", index=False, encoding="utf-8-sig")
    print("→ profiles.csv 已写")

    # ---- metrics.csv（seed 实际结果） ----
    m_rows = []
    for sd in SEEDS:
        r = results[sd]
        m_rows.append({
            "seed": sd,
            "cluster_counts": str(r["counts"]),  # CSV 单列存 dict
            "inertia": r["inertia"],
            "silhouette": r["silhouette"],
            "n_iter": r["n_iter"],
            "warnings": "; ".join(r["warnings"]) if r["warnings"] else "",
        })
    pd.DataFrame(m_rows).to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    print("→ metrics.csv 已写")

    # ---- pairwise_comparisons.csv ----
    p_rows = []
    for p in pairwise:
        p_rows.append({
            "pair": p["pair"],
            "seed_a": p["seed_a"], "seed_b": p["seed_b"],
            "ari": p["ari"],
            "matched": p["matched"], "changed": p["changed"], "total": p["total"],
            "match_rate": p["match_rate"],
            "mapping_a_to_b": str(p["mapping_a_to_b"]),
        })
    pd.DataFrame(p_rows).to_csv(OUT_DIR / "pairwise_comparisons.csv", index=False, encoding="utf-8-sig")
    print("→ pairwise_comparisons.csv 已写")

    # ---- contingency_tables.csv（长表：pair + 行 + 列 + n） ----
    ct_rows = []
    for p in pairwise:
        t = p["contingency_table"]
        for i in range(3):
            for j in range(3):
                ct_rows.append({
                    "pair": p["pair"],
                    "row_seed_cluster": i,
                    "col_seed_cluster": j,
                    "n": int(t[i, j]),
                })
    pd.DataFrame(ct_rows).to_csv(OUT_DIR / "contingency_tables.csv", index=False, encoding="utf-8-sig")
    print("→ contingency_tables.csv 已写")

    # ---- metrics.json ----
    t_now = datetime.now(timezone(timedelta(hours=8)))
    out_json = {
        "exported_at": t_now.isoformat(timespec="seconds"),
        "experiment_note": "稳定性实验：仅 random_state 改变（0 / 42 / 2026），其他 KMeans / StandardScaler "
                           "参数与 feature_compare 三特征模型完全一致；特征顺序 income, balance, age。"
                           "本次各 seed 使用同一标准化特征矩阵，inertia 可以横向比较拟合目标值；"
                           "其高低不单独证明业务价值或总体稳定性，稳定性结合 ARI 与匹配后变化人数判断。"
                           "不同特征空间的 inertia 不直接比较。"
                           "展示标签以 seed42 为参照（current_seed → seed42 匈牙利映射）。",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "pandas": pd.__version__, "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "dependencies": {
            "data_file": str(YUAN),
            "feature_compare_metrics": str(FC_METRICS),
            "feature_compare_labels": str(FC_LABELS),
            "yuan_sha256": sha256(YUAN),
        },
        "kmeans_params_fixed": {
            k: v for k, v in params["km_orig"].items() if k != "random_state"
        },
        "standard_scaler_params": params["sc_params"],
        "random_states_used": SEEDS,
        "features": FEATURES,
        "display_mapping_direction": "行=当前seed簇编号，列=seed42簇编号；mapping 为 current_seed → seed42",
        "display_mappings": {str(sd): display_maps[sd] for sd in SEEDS},
        "per_seed": {
            str(sd): {
                "cluster_counts": results[sd]["counts"],
                "inertia": results[sd]["inertia"],
                "silhouette": results[sd]["silhouette"],
                "n_iter": results[sd]["n_iter"],
                "warnings": results[sd]["warnings"],
            } for sd in SEEDS
        },
        "pairwise": [
            {k: v for k, v in p.items() if k != "contingency_table"} for p in pairwise
        ],
        "seed42_vs_feature_compare_labels": fc_check,
        "notes": [
            "三次拟合使用同一标准化后的特征矩阵，仅 random_state 改变。",
            "各 seed 使用同一标准化特征矩阵，inertia 可横向比较拟合目标值；"
            "其高低不单独证明业务价值或总体稳定性，稳定性结合 ARI 与匹配后变化人数判断。"
            "不同特征空间的 inertia 不直接比较。",
            "展示映射以 seed42 为参照；同色不代表成员完全相同，只代表在 seed42 的匈牙利匹配下重叠最大。",
            "seed42 拟合结果与 feature_compare/feature_labels.csv.cluster_3f 已按 customer_id 比对，"
            f"ARI={fc_check['ari']:.6f}，匹配后变化 {fc_check['changed']} 人（应=0）。",
            "稳定性判断依据：三对 pairwise ARI 和匹配/变化人数；若都接近 1 / 0 则说明对初始化不敏感。",
            "不预设任何结论，如实报告。",
        ],
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(out_json), f, ensure_ascii=False, indent=2, allow_nan=False)
    print("→ metrics.json 已写")


def plot(df, results, display_maps):
    """三个并排 income—balance 图，统一轴范围；颜色按各 seed→seed42 映射。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    incomes = df["income"].to_numpy(dtype=float)
    balances = df["balance_wan"].to_numpy(dtype=float)
    xlo, xhi = incomes.min() - (incomes.max() - incomes.min()) * 0.05, incomes.max() + (incomes.max() - incomes.min()) * 0.05
    ylo, yhi = balances.min() - (balances.max() - balances.min()) * 0.05, balances.max() + (balances.max() - balances.min()) * 0.05

    for i, sd in enumerate(SEEDS):
        ax = axes[i]
        labels = results[sd]["labels"]
        dm = display_maps[sd]
        for c in range(3):
            col = COLORS[dm[c]]
            mask = labels == c
            if not mask.any():
                continue
            ax.scatter(incomes[mask], balances[mask], s=40, alpha=0.7, color=col,
                       edgecolors="white", linewidths=0.3,
                       label=f"簇 {c}（{int(mask.sum())} 人，展示簇 {dm[c]}）")
        ax.set_title(f"seed={sd}\n展示颜色对齐到 seed=42")
        ax.set_xlabel("年收入 income（万元）")
        if i == 0:
            ax.set_ylabel("账户平均余额 balance（万元）")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        f"三随机种子稳定性（k=3, n_init=10, k-means++, 仅 random_state 改变）\n"
        f"特征：income + balance + age，仅 random_state ∈ {SEEDS}；展示颜色为各 seed→seed42 匈牙利映射",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.text(
        0.5, 0.01,
        "三维聚类的二维投影（收入—余额）；同色不保证成员完全相同，只代表在 seed42 匈牙利匹配下重叠最大",
        ha="center", fontsize=9, color="#333333",
    )
    fig.savefig(OUT_DIR / "stability_scatter.png", dpi=150)
    plt.close(fig)
    print("→ stability_scatter.png 已写")


# =====================================================================
# 主流程
# =====================================================================
def main():
    font_name = choose_font()
    data = prepare_data()
    km_params = data["km_orig"]

    results = fit_all(data)
    pairwise, display_maps, fc_check = pairwise_and_display(data["df"], results)

    export(data["df"], results, pairwise, display_maps, fc_check, data)
    plot(data["df"], results, display_maps)

    print()
    print("=" * 72)
    print("结果汇总")
    print("=" * 72)
    for sd in SEEDS:
        r = results[sd]
        print(f"  seed={sd}: counts={r['counts']}, inertia={r['inertia']:.6f}, "
              f"silhouette={r['silhouette']:.6f}, n_iter={r['n_iter']}")
    print(f"  seed42 vs feature_compare: ARI={fc_check['ari']:.6f}, "
          f"matched={fc_check['matched']}, changed={fc_check['changed']}")
    for p in pairwise:
        print(f"  {p['pair']}: ARI={p['ari']:.6f}, matched={p['matched']}, changed={p['changed']}")
    print(f"  展示映射 (current→42): {display_maps}")
    print("完成")


if __name__ == "__main__":
    main()

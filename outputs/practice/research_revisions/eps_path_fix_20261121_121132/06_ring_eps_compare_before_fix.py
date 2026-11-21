# -*- coding: utf-8 -*-
"""授权后的有限eps对照：复用首次实验坐标，不重生成数据、不重训KMeans。"""
import sys
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/practice/research"
OUT = SRC / "eps_sensitivity"
EPS_VALUES = [0.24, 0.28, 0.32]
MIN_SAMPLES = 5
TARGETS = ["eps_labels.csv", "eps_metrics.csv", "eps_metrics.json",
           "eps_comparison.png", "validation.json"]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()

def counts(labels):
    values, ns = np.unique(labels, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, ns)}

def main():
    checks = []
    def verify(name, ok):
        checks.append({"name": name, "passed": bool(ok)})
        if not ok:
            raise RuntimeError("核验失败：" + name)

    meta = json.loads((SRC / "ring_metrics.json").read_text(encoding="utf-8"))
    original_validation = json.loads((SRC / "validation.json").read_text(encoding="utf-8"))
    verify("原实验已通过核验", original_validation["passed"] is True)
    verify("原始生成配置一致", meta["generation"] == {
        "n_samples": 500, "factor": 0.5, "noise": 0.04, "random_state": 42, "shuffle": True})
    verify("原DBSCAN参数一致", meta["dbscan_params"] == {
        "eps": 0.2, "min_samples": 5, "metric": "euclidean"})
    protected = {rel: expected for rel, expected in original_validation["files"].items()}
    for rel, expected in protected.items():
        verify("原实验文件哈希：" + rel, sha(SRC / rel) == expected)
    fixed_paths = [
        SRC / "validation.json", ROOT / "practice/05_ring_research.py",
        ROOT / "reports/15_ring_research.log",
        ROOT / "data/mall_customers_sim.csv", ROOT / "data/mall_customers_yuan.csv",
        ROOT / "student_webapp/results.js"]
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in fixed_paths}
    verify("原实验脚本未改", original_hashes["practice/05_ring_research.py"] == meta["script_sha256"])
    prior_hashes = json.loads((ROOT / "reports/14_dashboard_source_hashes.json").read_text(encoding="utf-8"))
    for record in prior_hashes["files"]:
        verify("既有客户实验未变：" + str(Path(record["Path"]).relative_to(ROOT)),
               sha(Path(record["Path"])) == record["Hash"])
    for name in TARGETS:
        verify("不覆盖本轮已有输出：" + name, not (OUT / name).exists())

    df = pd.read_csv(SRC / "ring_labels.csv", encoding="utf-8-sig", float_precision="round_trip")
    expected_columns = ["point_id", "x_raw", "y_raw", "x_std", "y_std",
                        "true_ring", "kmeans_cluster", "dbscan_cluster"]
    verify("原始500点ID与字段完整", list(df.columns) == expected_columns and
           len(df) == 500 and np.array_equal(df.point_id.to_numpy(), np.arange(1, 501)))
    X = df[["x_std", "y_std"]].to_numpy()
    ytrue = df.true_ring.to_numpy(dtype=int)
    yinitial = df.dbscan_cluster.to_numpy(dtype=int)
    verify("同一标准化坐标有限且尺度正确", np.isfinite(X).all() and
           np.allclose(X.mean(axis=0), 0, atol=1e-12) and
           np.allclose(X.var(axis=0), 1, atol=1e-12))
    verify("参考内外环各250点", sorted(counts(ytrue).values()) == [250, 250])
    initial_metric = next(r for r in meta["metrics"] if r["method"] == "DBSCAN")
    verify("原DBSCAN标签与指标一致", counts(yinitial) == meta["dbscan_counts_including_noise"] and
           np.isclose(adjusted_rand_score(ytrue, yinitial), initial_metric["ari_all_500"],
                      atol=1e-12, rtol=0))
    fonts_available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ["Microsoft YaHei", "SimHei", "Songti SC", "PingFang SC"] if f in fonts_available]
    verify("本机中文字体可用", bool(fonts))
    matplotlib.rcParams["font.sans-serif"] = fonts + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    print("开始时间（北京时间）：", generated_at)
    print("用户授权：固定同一数据和min_samples=5，仅补测eps=0.24、0.28、0.32，保留全部结果")
    print("不调用make_circles、不拟合StandardScaler、不重训KMeans；直接复用ring_labels.csv的标准化坐标")
    print("Python / sklearn / numpy / pandas / matplotlib：",
          platform.python_version(), sklearn.__version__, np.__version__, pd.__version__, matplotlib.__version__)
    print("脚本SHA256：", sha(Path(__file__)))
    print("输入ring_labels.csv SHA256：", sha(SRC / "ring_labels.csv"))
    print("中文字体：", fonts[0])

    rows = [{
        "eps": 0.2, "min_samples": 5, "status": "reused_initial_result",
        "n_total": 500, "n_clusters_excluding_noise": initial_metric["n_clusters_excluding_noise"],
        "n_noise": initial_metric["n_noise"], "ari_all_500": initial_metric["ari_all_500"],
        "silhouette_non_noise": initial_metric["silhouette"],
        "silhouette_n": initial_metric["silhouette_n"],
        "core_samples": meta["dbscan_core_samples"], "counts": meta["dbscan_counts_including_noise"]}]
    predictions = [(0.2, yinitial)]
    for eps in EPS_VALUES:
        model = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean")
        pred = model.fit_predict(X)
        kept = pred != -1
        n_retained = int(kept.sum())
        n_clusters = len(set(pred.tolist()) - {-1})
        score = (float(silhouette_score(X[kept], pred[kept], metric="euclidean"))
                 if 2 <= n_clusters <= n_retained - 1 else None)
        ari = float(adjusted_rand_score(ytrue, pred))
        row = {
            "eps": eps, "min_samples": MIN_SAMPLES, "status": "newly_fitted",
            "n_total": 500, "n_clusters_excluding_noise": n_clusters,
            "n_noise": int((pred == -1).sum()), "ari_all_500": ari,
            "silhouette_non_noise": score, "silhouette_n": n_retained,
            "core_samples": int(len(model.core_sample_indices_)), "counts": counts(pred)}
        verify(f"eps={eps} 标签覆盖及人数合计500", len(pred) == 500 and sum(row["counts"].values()) == 500)
        verify(f"eps={eps} 噪声和保留样本合计500", row["n_noise"] + n_retained == 500)
        verify(f"eps={eps} ARI取值有效", -1 <= ari <= 1)
        verify(f"eps={eps} 轮廓系数范围或空值有效", score is None or -1 <= score <= 1)
        rows.append(row)
        predictions.append((eps, pred))
        df[f"dbscan_eps_{str(eps).replace('.', '_')}"] = pred
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.3), sharex=True, sharey=True)
    colors = ["#477b9e", "#cf8652", "#558c72", "#937cb5", "#9a8755"]
    for ax, (eps, pred), row in zip(axes, predictions, rows):
        for lab in np.unique(pred):
            mask = pred == lab
            ax.scatter(X[mask, 0], X[mask, 1], s=12, alpha=0.8,
                       c="#777777" if lab == -1 else colors[int(lab) % len(colors)],
                       marker="x" if lab == -1 else "o",
                       label=f"{'噪声' if lab == -1 else '簇'+str(int(lab))}（{int(mask.sum())}）")
        ax.set_title(f"eps={eps:.2f}：{row['n_clusters_excluding_noise']}簇 / 噪声{row['n_noise']}\n"
                     f"ARI（全部500点）={row['ari_all_500']:.4f}", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("坐标 x（标准化）")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8,
                  ncol=2, frameon=False)
    axes[0].set_ylabel("坐标 y（标准化）")
    fig.suptitle("DBSCAN有限参数对照：同一500点，仅改变eps", fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.81, bottom=0.32, wspace=0.18)
    fig.text(0.5, 0.02,
             "min_samples=5；初始eps=0.20复用既有标签，其余三值为本轮新增拟合；未重生成数据或重训KMeans。\n"
             "ARI包含噪声点；图中颜色为各结果原始簇号，不代表跨参数的簇编号已经对齐。",
             ha="center", va="bottom", fontsize=9)

    for rel, expected in protected.items():
        verify("运行结束原实验未变：" + rel, sha(SRC / rel) == expected)
    for rel, expected in original_hashes.items():
        verify("运行结束原文件未变：" + rel, sha(ROOT / rel) == expected)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "eps_labels.csv", index=False, encoding="utf-8-sig")
    csv_rows = [{**r, "counts": json.dumps(r["counts"], ensure_ascii=False, sort_keys=True)}
                for r in rows]
    pd.DataFrame(csv_rows).to_csv(OUT / "eps_metrics.csv", index=False, encoding="utf-8-sig")
    fig.savefig(OUT / "eps_comparison.png", dpi=170)
    plt.close(fig)
    result = {
        "generated_at": generated_at, "script_sha256": sha(Path(__file__)),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "authorized_eps": EPS_VALUES, "fixed_min_samples": MIN_SAMPLES,
        "source_file": "outputs/practice/research/ring_labels.csv",
        "source_file_sha256": sha(SRC / "ring_labels.csv"),
        "input_note": "复用原CSV双精度标准化坐标，read_csv float_precision=round_trip；不重生成或标准化",
        "initial_result_reused": True, "kmeans_retrained": False,
        "initial_checksums": {**protected, **original_hashes},
        "silhouette_scope": "仅非噪声样本；有效簇数2..n-1时计算，否则null；明确记录保留样本数",
        "ari_scope": "全部500点，包含噪声；真实环标签仅用于评估",
        "runs": rows, "selection_policy": "保留全部四种eps结果，不以ARI自动选择单一最优eps，不继续搜索",
        "limitations": "同一玩具数据上的探索性参数对照，不等于独立测试集验证或真实客户聚类效果"}
    (OUT / "eps_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    saved = pd.read_csv(OUT / "eps_labels.csv", encoding="utf-8-sig", float_precision="round_trip")
    verify("落盘CSV仍有500个唯一ID", len(saved) == 500 and saved.point_id.nunique() == 500)
    verify("落盘原始与标准化坐标完全一致", np.array_equal(
        saved[["x_raw", "y_raw", "x_std", "y_std"]].to_numpy(),
        df[["x_raw", "y_raw", "x_std", "y_std"]].to_numpy()))
    verify("落盘参考标签及全部预测一致", all(np.array_equal(saved[c], df[c]) for c in df.columns
        if c not in ["x_raw", "y_raw", "x_std", "y_std"]))
    validation = {
        "passed": True, "checks": checks, "manual_figure_acceptance": "待本人核对",
        "output_sha256": {name: sha(OUT / name) for name in TARGETS if name != "validation.json"}}
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(csv_rows).to_string(index=False))
    print("通过核验数：", len(checks))
    print("输出：", OUT)
    print("EPS_COMPARE_PASS")
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""独立研究：同心环数据的 KMeans / DBSCAN 对照。仅玩具数据有真实环标签。"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "practice" / "research"
FILES = ["ring_comparison.png", "ring_labels.csv", "ring_metrics.csv",
         "ring_metrics.json", "validation.json"]
PROTECTED = {
    "data/mall_customers_sim.csv": "13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F",
    "data/mall_customers_yuan.csv": "060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07",
}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def counts(labels):
    vals, nums = np.unique(labels, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, nums)}
def guarded_silhouette(X, labels):
    n_labels = len(np.unique(labels))
    if 2 <= n_labels <= len(X)-1:
        return float(silhouette_score(X, labels, metric="euclidean"))
    return None
def main():
    checks = []
    def verify(name, ok):
        checks.append({"name": name, "passed": bool(ok)})
        if not ok:
            raise RuntimeError("核验失败：" + name)
    for rel, expected in PROTECTED.items():
        verify("原客户CSV哈希：" + rel, sha(ROOT / rel) == expected)
    for f in FILES:
        verify("不覆盖既有结果：" + f, not (OUT / f).exists())
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ["Microsoft YaHei", "SimHei", "Songti SC", "STHeiti", "PingFang SC"] if f in available]
    verify("本机可用中文字体", bool(fonts))
    matplotlib.rcParams["font.sans-serif"] = fonts + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    generation = dict(n_samples=500, factor=0.5, noise=0.04, random_state=42, shuffle=True)
    kparams = dict(n_clusters=2, init="k-means++", random_state=42, n_init=10,
                   max_iter=300, tol=1e-4, algorithm="lloyd")
    dparams = dict(eps=0.2, min_samples=5, metric="euclidean")
    print("生成时间（北京时间）：", generated_at)
    print("Python / sklearn / numpy / pandas / matplotlib：",
          platform.python_version(), sklearn.__version__, np.__version__, pd.__version__, matplotlib.__version__)
    print("脚本SHA256：", sha(Path(__file__)))
    print("中文字体：", fonts[0])
    print("make_circles：", generation)
    print("StandardScaler：with_mean=True, with_std=True；两模型使用同一份标准化坐标")
    print("KMeans：", kparams)
    print("DBSCAN（仅此一次，无自动调参）：", dparams)
    Xraw, true_ring = make_circles(**generation)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)
    verify("生成500×2有限坐标", X.shape == (500, 2) and np.isfinite(X).all())
    verify("生成器两环各250点", sorted(counts(true_ring).values()) == [250, 250])
    verify("标准化均值0方差1", np.allclose(X.mean(axis=0), 0, atol=1e-12) and
           np.allclose(X.var(axis=0), 1, atol=1e-12))
    km = KMeans(**kparams)
    yk = km.fit_predict(X)
    db = DBSCAN(**dparams)
    yd = db.fit_predict(X)
    noise = yd == -1
    kept = ~noise
    n_clusters = len(set(yd.tolist()) - {-1})
    sil_k = guarded_silhouette(X, yk)
    sil_db = guarded_silhouette(X[kept], yd[kept])
    ari_k = float(adjusted_rand_score(true_ring, yk))
    ari_db = float(adjusted_rand_score(true_ring, yd))
    verify("两模型标签覆盖500点", len(yk) == len(yd) == len(true_ring) == 500)
    verify("KMeans两簇人数合计500", len(counts(yk)) == 2 and sum(counts(yk).values()) == 500)
    verify("DBSCAN人数含噪声合计500", sum(counts(yd).values()) == 500)
    verify("DBSCAN保留点加噪声点等于500", int(kept.sum()) + int(noise.sum()) == 500)
    verify("ARI范围有效", -1 <= ari_k <= 1 and -1 <= ari_db <= 1)
    if sil_k is not None:
        verify("KMeans轮廓系数范围有效", -1 <= sil_k <= 1)
    if sil_db is not None:
        verify("DBSCAN非噪声轮廓系数范围有效", -1 <= sil_db <= 1)
    labels = pd.DataFrame({
        "point_id": np.arange(1, 501), "x_raw": Xraw[:, 0], "y_raw": Xraw[:, 1],
        "x_std": X[:, 0], "y_std": X[:, 1], "true_ring": true_ring,
        "kmeans_cluster": yk, "dbscan_cluster": yd,
    })
    rows = [
        {"method": "KMeans", "n_total": 500, "n_clusters_excluding_noise": 2, "n_noise": 0,
         "ari_all_500": ari_k, "silhouette": sil_k, "silhouette_n": 500,
         "silhouette_scope": "all_points", "inertia_standardized": float(km.inertia_)},
        {"method": "DBSCAN", "n_total": 500, "n_clusters_excluding_noise": n_clusters,
         "n_noise": int(noise.sum()), "ari_all_500": ari_db, "silhouette": sil_db,
         "silhouette_n": int(kept.sum()),
         "silhouette_scope": "exclude_noise_-1; null_if_not_2_to_n_minus_1_clusters",
         "inertia_standardized": None},
    ]
    metrics = {
        "generated_at": generated_at,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "source": "sklearn.datasets.make_circles；生成器标签仅用于玩具实验评估，不参与两模型训练",
        "generation": generation, "standardization": "StandardScaler.fit_transform on the same 500 x 2",
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "kmeans_params": kparams, "dbscan_params": dparams, "metrics": rows,
        "kmeans_counts": counts(yk), "dbscan_counts_including_noise": counts(yd),
        "kmeans_n_iter": int(km.n_iter_), "dbscan_core_samples": int(len(db.core_sample_indices_)),
        "dbscan_adjustments": [], "script_sha256": sha(Path(__file__)),
        "font": fonts[0], "customer_data_unchanged": True,
        "limitations": [
            "ARI使用生成器标签评估全部500点，DBSCAN噪声也计入；不删除噪声美化ARI。",
            "DBSCAN轮廓系数排除噪声，记录保留样本数；不同评价范围不直接排名。",
            "欧氏轮廓系数偏好紧凑凸形簇，不能单凭较高分否定同心环的真实结构。",
            "参数未调整；结果仅说明本次模拟设置，不保证DBSCAN对所有非凸数据或参数有效。",
            "银行客户数据没有真实类别，不套用此处生成器标签或有监督ARI解释。"
        ],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharex=True, sharey=True)
    panels = [(true_ring, "生成器参考标签（仅用于评估）", None),
              (yk, f"KMeans：k=2，ARI={ari_k:.4f}", km.cluster_centers_),
              (yd, f"DBSCAN：{n_clusters}簇，噪声{int(noise.sum())}点\nARI（全部500点）={ari_db:.4f}", None)]
    palette = ["#477b9e", "#cf8652", "#558c72", "#937cb5"]
    for ax, (pred, title, centers) in zip(axes, panels):
        for c in np.unique(pred):
            mask = pred == c
            ax.scatter(X[mask, 0], X[mask, 1], s=14, alpha=0.8,
                       color="#777777" if c == -1 else palette[int(c) % len(palette)],
                       marker="x" if c == -1 else "o",
                       label=f"{'噪声' if c == -1 else '组 '+str(int(c))}（{int(mask.sum())}）")
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=170, c="#243c48",
                       edgecolors="white", linewidths=0.7, label="质心")
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("坐标 x（标准化）")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("坐标 y（标准化）")
    fig.suptitle("同一份同心环数据：质心划分与密度聚类的对照", fontsize=15, y=0.98)
    fig.text(0.5, 0.035,
             "500点；factor=0.5；noise=0.04；seed=42；同一StandardScaler坐标；DBSCAN eps=0.2 / min_samples=5。\n"
             "真实环标签不参与训练；ARI包含噪声点；二维坐标相同，组号和颜色不代表跨方法逐组对应。",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    for rel, expected in PROTECTED.items():
        verify("运行结束原CSV未变：" + rel, sha(ROOT / rel) == expected)
    OUT.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT / "ring_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "ring_metrics.csv", index=False, encoding="utf-8-sig")
    fig.savefig(OUT / "ring_comparison.png", dpi=170)
    plt.close(fig)
    (OUT / "ring_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    saved = pd.read_csv(OUT / "ring_labels.csv", encoding="utf-8-sig")
    verify("落盘CSV行数与ID覆盖", len(saved) == 500 and saved.point_id.nunique() == 500)
    verify("落盘真实/预测标签未变", all(np.array_equal(saved[c].to_numpy(), labels[c].to_numpy())
                                  for c in ["true_ring", "kmeans_cluster", "dbscan_cluster"]))
    verify("落盘坐标精度一致", np.allclose(saved[["x_raw", "y_raw", "x_std", "y_std"]],
                                       labels[["x_raw", "y_raw", "x_std", "y_std"]], rtol=1e-12, atol=1e-12))
    validation = {"passed": True, "checks": checks, "manual_figure_acceptance": "待本人核对",
                  "files": {f: sha(OUT / f) for f in FILES if f != "validation.json"}}
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("KMeans人数：", counts(yk))
    print("DBSCAN人数（含-1噪声）：", counts(yd))
    print("通过核验数：", len(checks))
    print("已保存：", OUT)
    print("RING_RESEARCH_PASS")
if __name__ == "__main__":
    main()

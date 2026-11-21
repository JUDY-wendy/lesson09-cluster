# -*- coding: utf-8 -*-
"""
质心迭代过程可视化：k-均值 = Lloyd 算法（分配-更新循环）
（由指令 03 生成，手写循环，不 fit 一把梭）
- 标准化后的 income、balance 两特征，k=3，random_state=42
- k-means++ 初始化（与 sklearn 默认一致：随机选第一个点，后续按距离平方的概率选）
- 2×2 画布展示第 0 轮、第 1 轮、第 2 轮、最后一轮，保存 outputs/06_iteration.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

# 中文字体：优先使用需求指定的三款字体，缺失时回退到本机可用的中文字体
PREFERRED_FONTS = [
    "Songti SC", "STHeiti", "PingFang SC",      # 需求指定（macOS 常见）
    "SimHei", "Microsoft YaHei", "STSong",       # Windows 兜底
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_fonts = [f for f in PREFERRED_FONTS if f in available_fonts]
matplotlib.rcParams["font.sans-serif"] = chosen_fonts + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mall_customers_yuan.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
scaler = StandardScaler()
X = scaler.fit_transform(df[["income", "balance"]])

K = 3
TOL = 1e-8          # 质心移动小于该阈值即视为不再移动
MAX_ITER = 50

# ---------- k-means++ 初始化（sklearn 的默认初始化就是它） ----------
# 内部逻辑：先随机选第一个质心，之后每个质心按「到已有质心的距离平方」为概率抽取
centers, _ = kmeans_plusplus(X, n_clusters=K, random_state=42)

print("=" * 64)
print("k-均值 Lloyd 迭代（标准化空间），k=3，random_state=42，k-means++ 初始化")
print("=" * 64)

# ---------- 手写「分配-更新」循环 ----------
history = []        # 每轮记录：(轮次, 质心, 所属簇, inertia, 相对上轮的质心移动量)
prev_centers = None
for rnd in range(MAX_ITER):
    # ① 分配：每个点分给距离最近的质心
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    inertia = float(((X - centers[labels]) ** 2).sum())

    # 本轮质心相对上轮的移动量：所有质心、所有坐标变化的最大绝对值
    shift = 0.0 if prev_centers is None else float(np.abs(centers - prev_centers).max())
    history.append((rnd, centers.copy(), labels.copy(), inertia, shift))

    print(f"\n第 {rnd} 轮：inertia = {inertia:.4f}，质心移动 = {shift:.4f}")
    print("  质心坐标（标准化后）：")
    for k in range(K):
        print(f"    质心 {k}: ({centers[k, 0]: .4f}, {centers[k, 1]: .4f})")

    # 上一轮更新后质心已不动 → 本轮就是收敛轮，记录完即停
    if prev_centers is not None and shift < TOL:
        break

    # ② 更新：质心挪到本簇所有点的平均位置
    prev_centers = centers.copy()
    centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])

inertias = [h[3] for h in history]
monotone = all(b <= a + 1e-12 for a, b in zip(inertias, inertias[1:]))
print("\n" + "-" * 64)
print("inertia 逐轮序列：" + " → ".join(f"{v:.4f}" for v in inertias))
print(f"单调不增：{'是' if monotone else '否'}；共 {len(history) - 1} 次更新后收敛")
print("-" * 64)

# ---------- 选 4 个关键时刻画 2×2 ----------
last = len(history) - 1
selected = list(dict.fromkeys([0, 1, 2, last]))   # 第 0/1/2 轮 + 最后一轮

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = ["#DD8452", "#4C72B0", "#55A868"]

for ax, idx in zip(axes.flat, selected):
    rnd, centers_h, labels_h, inertia_h, shift_h = history[idx]

    for c in range(K):
        sub = X[labels_h == c]
        ax.scatter(
            sub[:, 0], sub[:, 1],
            s=25, alpha=0.6, color=colors[c],
            edgecolors="white", linewidths=0.4,
            label=f"簇 {c}（{len(sub)} 人）",
        )

    # 质心：大星形
    ax.scatter(
        centers_h[:, 0], centers_h[:, 1],
        marker="*", s=520, c="red", edgecolors="black", linewidths=1.2,
        zorder=5, label="质心",
    )

    if rnd == last:
        title = f"第 {rnd} 轮：收敛：质心不再移动"
    elif rnd == 0:
        title = "第 0 轮：初始质心（尚未移动）"
    else:
        title = f"第 {rnd} 轮：质心移动了 {shift_h:.2f}"

    ax.set_title(f"{title}\ninertia = {inertia_h:.2f}", fontsize=12)
    ax.set_xlabel("income（标准化后）", fontsize=10)
    ax.set_ylabel("balance（标准化后）", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

# 万一历史轮次不足 4 个，把多余子图隐藏，避免留下空白画格
for ax in axes.flat[len(selected):]:
    ax.axis("off")

fig.suptitle("k-均值 = 「分配-更新」循环：质心一步步挪到位", fontsize=15, y=0.995)
plt.tight_layout()
plot_path = OUT_DIR / "06_iteration.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\n已保存迭代过程图：{plot_path}")

# -*- coding: utf-8 -*-
"""
选 k：肘部法 + 轮廓系数
- 肘部法：k=1..9，记录标准化空间里每个 k 的簇内平方和 inertia_
- 轮廓系数：k=2..9（轮廓系数至少要 2 簇）
- 两个子图并排，保存 outputs/05_choose_k.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

# 中文字体：优先使用需求指定的三款字体，缺失时回退到本机可用的中文字体
PREFERRED_FONTS = [
    "Songti SC", "STHeiti", "PingFang SC",          # 需求指定（macOS 常见）
    "SimHei", "Microsoft YaHei", "STSong",           # Windows 兜底
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_fonts = [f for f in PREFERRED_FONTS if f in available_fonts]
matplotlib.rcParams["font.sans-serif"] = chosen_fonts + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
print(f"本次绘图使用中文字体：{chosen_fonts[0] if chosen_fonts else '未找到（中文可能显示为方框）'}")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mall_customers_yuan.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
# 与 02 脚本完全一致：只用 income、balance，先标准化再聚类
X = StandardScaler().fit_transform(df[["income", "balance"]])

# ---------- 第一件：肘部法，k=1..9 ----------
ks = list(range(1, 10))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)

print("=" * 72)
print("【肘部法】k=1..9 的簇内平方和 inertia（标准化空间）")
print(f"{'k':>3} {'inertia':>12} {'较上一档降幅':>14} {'降幅占比':>10}")
print("-" * 72)
for i, (k, v) in enumerate(zip(ks, inertias)):
    if i == 0:
        print(f"{k:>3} {v:>12.2f} {'-':>14} {'-':>10}")
    else:
        drop = inertias[i - 1] - v
        ratio = drop / inertias[0] * 100
        print(f"{k:>3} {v:>12.2f} {drop:>14.2f} {ratio:>9.1f}%")

# 拐点用「到首尾连线的最远距离」判定，避免肉眼拍脑袋：
# 把 k 与 inertia 各自归一化到 [0,1]，首尾连成直线，离直线最远的点就是拐点
k_min, k_max = ks[0], ks[-1]
in_first, in_last = inertias[0], inertias[-1]
dists = []
for k, v in zip(ks, inertias):
    x = (k - k_min) / (k_max - k_min)
    y = (v - in_last) / (in_first - in_last)
    dists.append(abs(y - (1 - x)))  # 直线方程 y = 1 - x
elbow_k = ks[dists.index(max(dists))]
print(f"\n肘部法拐点（离首尾连线最远的 k）：k = {elbow_k}")

# ---------- 第二件：轮廓系数，k=2..9 ----------
ks2 = list(range(2, 10))
sils = []
for k in ks2:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    sils.append(silhouette_score(X, labels))

print("\n【轮廓系数】k=2..9（越大越好，取值 -1 ~ 1）")
print(f"{'k':>3} {'silhouette':>12}")
print("-" * 72)
for k, v in zip(ks2, sils):
    print(f"{k:>3} {v:>12.4f}")

best_k = ks2[sils.index(max(sils))]
sil_at_3 = sils[ks2.index(3)]
gap = max(sils) - sil_at_3
print(f"\n轮廓系数最高点：k = {best_k}（{max(sils):.4f}）；k=3 时 {sil_at_3:.4f}，差值 {gap:.4f}")

# ---------- 第三件：并排双折线 ----------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7))

ax_left.plot(ks, inertias, "o-", color="#4C72B0", linewidth=2)
ax_left.axvline(x=elbow_k, color="red", linestyle="--", alpha=0.6,
                label=f"拐点 k={elbow_k}")
for k, v in zip(ks, inertias):
    ax_left.annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                     xytext=(0, 8), fontsize=9, ha="center")
ax_left.set_title("肘部法：簇内平方和（inertia）随 k 的变化", fontsize=13)
ax_left.set_xlabel("簇数 k", fontsize=11)
ax_left.set_ylabel("簇内平方和 inertia", fontsize=11)
ax_left.set_xticks(ks)
ax_left.legend(fontsize=10)
ax_left.grid(alpha=0.3)

ax_right.plot(ks2, sils, "s-", color="#55A868", linewidth=2)
ax_right.axvline(x=best_k, color="red", linestyle="--", alpha=0.6,
                 label=f"最高点 k={best_k}")
for k, v in zip(ks2, sils):
    ax_right.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                      xytext=(0, 8), fontsize=9, ha="center")
if best_k != 3:
    ax_right.axvline(x=3, color="#DD8452", linestyle=":", alpha=0.8,
                     label="业务暂定 k=3")
ax_right.set_title("轮廓系数：分得好不好（越大越好，最大 1）", fontsize=13)
ax_right.set_xlabel("簇数 k", fontsize=11)
ax_right.set_ylabel("轮廓系数 silhouette", fontsize=11)
ax_right.set_xticks(ks2)
ax_right.legend(fontsize=10)
ax_right.grid(alpha=0.3)

fig.suptitle("选 k 的两种证据：肘部法看拐点，轮廓系数看分离度", fontsize=14)

plt.tight_layout()
plot_path = OUT_DIR / "05_choose_k.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\n已保存图片：{plot_path}")
print("=" * 72)

# -*- coding: utf-8 -*-
"""
第 2-4 件事：k-均值聚类 + 聚类结果可视化 + 分群报告
- 用 income、balance 两个特征做 KMeans（k=3，random_state=42），簇编号写回数据框
- 画按簇着色的散点图（含簇中心星形标记与人数标注），保存 outputs/03_clustered_scatter.png
- 统计各簇画像，生成中文报告 generated/mall_report.md
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import KMeans
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
GEN_DIR = ROOT / "generated"
OUT_DIR.mkdir(exist_ok=True)
GEN_DIR.mkdir(exist_ok=True)

# ---------- 第 2 件事：k-均值聚类 ----------
df = pd.read_csv(DATA_PATH)

# 只取 income、balance 两列作为特征；customer_id 是编号，绝不进模型
# balance 以元计、income 以万元计，量纲相差约四个数量级，必须先标准化再算欧氏距离
scaler = StandardScaler()
X = scaler.fit_transform(df[["income", "balance"]])

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)

# 把每个客户的簇编号写回数据框
df["cluster"] = labels

counts = df["cluster"].value_counts().sort_index()
print("=" * 60)
print("第 2 件事：KMeans 聚类完成（特征：income、balance 标准化后，k=3，n_init=10）")
print("各簇人数、平均收入、平均余额：")
for c, n in counts.items():
    g = df[df["cluster"] == c]
    print(
        f"  簇 {c}：{n:>3} 人（{n / len(df) * 100:.1f}%）"
        f"  平均收入 {g['income'].mean():.2f} 万元"
        f"  平均余额 {g['balance'].mean():,.0f} 元"
    )

# 簇中心是在标准化空间里算出来的，逆变换后才回到「万元 / 元」的原始坐标
centers_std = km.cluster_centers_
centers_raw = scaler.inverse_transform(centers_std)
print("簇中心（标准化空间，z 分数）：")
print(centers_std.round(2))
print("簇中心（逆变换回原始坐标：income 万元 / balance 元）：")
print(centers_raw.round(2))

# ---------- 第 3 件事：聚类结果可视化（原始坐标 vs 标准化坐标） ----------
colors = ["#DD8452", "#4C72B0", "#55A868"]

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 7))

for c in range(km.n_clusters):
    mask = (df["cluster"] == c).to_numpy()
    # 左图用原始坐标，右图用标准化后的坐标——同一批点，只是换了度量衡
    ax_left.scatter(
        df.loc[mask, "income"], df.loc[mask, "balance"],
        s=45, alpha=0.7, color=colors[c],
        edgecolors="white", linewidths=0.5,
        label=f"簇 {c}（{mask.sum()} 人）",
    )
    ax_right.scatter(
        X[mask, 0], X[mask, 1],
        s=45, alpha=0.7, color=colors[c],
        edgecolors="white", linewidths=0.5,
        label=f"簇 {c}（{mask.sum()} 人）",
    )

# 簇中心：大星形标记（左图先逆变换回原始坐标，右图直接用标准化坐标）
ax_left.scatter(
    centers_raw[:, 0], centers_raw[:, 1],
    marker="*", s=550, c="red", edgecolors="black", linewidths=1.2,
    zorder=5, label="簇中心（已逆变换）",
)
ax_right.scatter(
    centers_std[:, 0], centers_std[:, 1],
    marker="*", s=550, c="red", edgecolors="black", linewidths=1.2,
    zorder=5, label="簇中心（标准化空间）",
)

# 在每个簇中心旁标注该簇人数
for i in range(km.n_clusters):
    for ax, (cx, cy) in ((ax_left, centers_raw[i]), (ax_right, centers_std[i])):
        ax.annotate(
            f"簇 {i}\n{counts[i]} 人",
            xy=(cx, cy), xytext=(10, 10), textcoords="offset points",
            fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.85),
        )

ax_left.set_title("左：原始坐标（income 万元 / balance 元）", fontsize=12)
ax_left.set_xlabel("年收入 income（万元）", fontsize=11)
ax_left.set_ylabel("账户平均余额 balance（元）", fontsize=11)

ax_right.set_title("右：标准化坐标（z 分数）", fontsize=12)
ax_right.set_xlabel("income（标准化后）", fontsize=11)
ax_right.set_ylabel("balance（标准化后）", fontsize=11)

for ax in (ax_left, ax_right):
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("同一批客户、同一套聚类标签：标准化只改变度量衡，不改变客户", fontsize=14)

plt.tight_layout()
plot_path = OUT_DIR / "04_compare_scale.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\n第 3 件事：已保存图片 {plot_path}")

# ---------- 第 4 件事：分群报告 ----------
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入=("income", "mean"),
    平均余额=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)
summary["占比(%)"] = (summary["人数"] / len(df) * 100).round(1)
summary = summary[["人数", "占比(%)", "平均年龄", "平均收入", "平均余额", "平均持有产品数"]]
summary.index.name = "簇编号"

# 按平均收入从低到高排序，让簇编号与业务画像顺序对应
order = summary["平均收入"].sort_values().index.tolist()

# 按收入从低到高为每一群客户起业务名字
name_pool = ["青年潜力客群", "中产稳健客群", "高净值尊享客群"]
biz_names = {cl: name_pool[i] for i, cl in enumerate(order)}

print("\n第 4 件事：各簇画像统计")
print(summary.to_string())
print("\n客群业务命名：")
for cl in order:
    print(f"  簇 {cl} → {biz_names[cl]}")

# 生成中文分析报告
lines = [
    "# 零售银行客户分群分析报告",
    "",
    "## 一、分析方法",
    "",
    f"- 数据：`data/mall_customers_yuan.csv`，共 {len(df)} 位客户、{df.shape[1] - 1} 个原始字段。",
    "- 聚类特征：仅使用 `income`（年收入，万元）与 `balance`（账户平均余额，元）两列；"
    "`customer_id` 仅作编号，不参与建模。",
    "- 预处理：`balance` 以元计、`income` 以万元计，量纲相差约四个数量级，"
    "聚类前用 `StandardScaler` 对两列做标准化（z 分数），使两特征等权参与欧氏距离计算。",
    "- 算法：k-均值聚类（KMeans），簇数 k=3，随机种子 `random_state=42`，`n_init=10`。",
    "- 流程：先做数据初检与初始散点图（`outputs/01_raw_scatter.png`），再聚类并绘制"
    "「原始坐标 vs 标准化坐标」对比图（`outputs/04_compare_scale.png`）。",
    "",
    "## 二、各簇统计概览",
    "",
    summary.to_markdown(),
    "",
    "## 三、客群画像与业务含义",
    "",
]

for cl in order:
    row = summary.loc[cl]
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 {int(row['人数'])} 人"
        f"（占 {row['占比(%)']:.1f}%），平均年龄 {row['平均年龄']:.1f} 岁、"
        f"平均年收入 {row['平均收入']:.2f} 万元、平均余额 {row['平均余额']:,.0f} 元、"
        f"平均持有产品 {row['平均持有产品数']:.2f} 个。"
    )

lines += [
    "",
    "## 四、结论",
    "",
    "300 位客户在「收入-余额」平面上自然聚成三块，且收入与余额高度同向："
    "收入越高的客户，账户余额也越高；年龄同样呈阶梯式上升，说明低龄客户多处于资产积累初期。",
    "",
    "## 五、一句话客群命名",
    "",
    f"**{biz_names[order[0]]}**：{int(summary.loc[order[0], '人数'])} 人，占 "
    f"{summary.loc[order[0], '占比(%)']:.1f}%，年轻、收入与余额双低，是待培养的潜力客群；"
    f"**{biz_names[order[1]]}**：{int(summary.loc[order[1], '人数'])} 人，占 "
    f"{summary.loc[order[1], '占比(%)']:.1f}%，收入与余额居中，是银行稳健的基本盘；"
    f"**{biz_names[order[2]]}**：{int(summary.loc[order[2], '人数'])} 人，占 "
    f"{summary.loc[order[2], '占比(%)']:.1f}%，收入与余额双高，是应重点维护的高价值客群。",
    "",
]

report_path = GEN_DIR / "mall_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\n已保存报告：{report_path}")
print("=" * 60)

# -*- coding: utf-8 -*-
"""
翻车复现：银行原始导出数据重跑 KMeans
- income 仍然是万元单位，但 balance 是**元**单位（上一轮模拟数据是万元）
- 不做任何标准化——故意让 balance 量纲碾压 income，观察分群如何"变形"
- 四件事：数据初检 → 聚类 → 画图 → 报告
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import KMeans

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 全局样式 ----------
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

NUMERIC_COLS = ["age", "income", "balance", "product_count"]

# =====================================================================
# 第 1 件事：数据初检
# =====================================================================
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("【第 1 件事】数据初检")
print(f"  行列数: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  列名:   {list(df.columns)}")
print()
print("  各列缺失值:")
for col in df.columns:
    miss = df[col].isnull().sum()
    print(f"    {col:<14} {miss}")
print()
print("  数值列基本统计（均值 / 最小值 / 最大值）:")
for col in NUMERIC_COLS:
    print(f"    {col:<14} 均值 {df[col].mean():>10.2f}  最小 {df[col].min():>10.2f}  最大 {df[col].max():>10.2f}")
print()

# =====================================================================
# 第 2 件事：重跑 KMeans 聚类（特征仍然只用 income + balance，完全不动）
# =====================================================================
X = df[["income", "balance"]]

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
df["cluster"] = labels

counts = df["cluster"].value_counts().sort_index()
print("【第 2 件事】KMeans 聚类完成（k=3, random_state=42, n_init=10）")
print("  各簇人数:")
for c, n in counts.items():
    g = df[df["cluster"] == c]
    print(f"    簇 {c}: {n:>3} 人 ({n/len(df)*100:.1f}%)  |  平均收入 {g['income'].mean():.2f} 万  |  平均余额 {g['balance'].mean():,.0f} 元")
print(f"  簇中心（income 万元, balance 元）:\n{km.cluster_centers_}")
print()

# =====================================================================
# 第 3 件事：重画聚类散点图
# =====================================================================
colors = ["#DD8452", "#4C72B0", "#55A868"]

fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(
        sub["income"], sub["balance"],
        s=45, alpha=0.7, color=colors[c],
        edgecolors="white", linewidths=0.5,
        label=f"簇 {c}（{len(sub)} 人）",
    )

# 簇中心：大星形
ax.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    marker="*", s=700, c="red", edgecolors="black", linewidths=1.2,
    zorder=5, label="簇中心",
)

# 在簇中心旁标注该簇人数
for i, (cx, cy) in enumerate(km.cluster_centers_):
    ax.annotate(
        f"簇 {i}\n{counts[i]} 人",
        xy=(cx, cy), xytext=(10, 10), textcoords="offset points",
        fontsize=11, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.85),
    )

ax.set_title("KMeans 客户分群（余额以元计 · 未标准化）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（元）", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "02_yuan_fail.png"
plt.savefig(plot_path)
plt.close(fig)
print(f"【第 3 件事】聚类图已保存: {plot_path}")
print()

# =====================================================================
# 第 4 件事：更新分群报告
# =====================================================================
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入_万元=("income", "mean"),
    平均余额_元=("balance", "mean"),
).round(2)
summary["占比(%)"] = (summary["人数"] / len(df) * 100).round(1)
summary = summary[["人数", "占比(%)", "平均年龄", "平均收入_万元", "平均余额_元"]]

# 按平均余额从低到高排序起业务名（因为 balance 是主导特征）
order = summary["平均余额_元"].sort_values().index.tolist()

# 给每群按"余额"重新起业务名（三个档）
name_pool = ["低余额起步客群", "中余额稳健客群", "高余额富裕客群"]
biz_names = {cl: name_pool[i] for i, cl in enumerate(order)}

print("【第 4 件事】各簇画像（按平均余额从低到高排序）:")
for cl in order:
    row = summary.loc[cl]
    print(
        f"  簇 {cl} · {biz_names[cl]}  "
        f"  人数: {int(row['人数'])} ({row['占比(%)']:.1f}%)  "
        f"  平均年龄: {row['平均年龄']:.1f}  "
        f"  平均收入: {row['平均收入_万元']:.2f} 万  "
        f"  平均余额: {row['平均余额_元']:,.0f} 元"
    )
print()

# 生成 markdown 报告
lines = [
    "# 零售银行客户分群分析报告（元计数据重跑版）",
    "",
    "## 一、分析方法",
    "",
    f"- 数据：`data/mall_customers_yuan.csv`，共 {len(df)} 位客户、{df.shape[1] - 1} 个原始字段。",
    "- 注意：本次数据的 `balance` 列为**元**单位（上一轮模拟数据是万元），"
    "`income` 仍为万元，两特征量纲相差约四个数量级，聚类前**未做标准化**。",
    "- 聚类特征：仅使用 `income`（年收入，万元）与 `balance`（账户平均余额，元）两列。",
    "- 算法：k-均值聚类（KMeans），簇数 k=3，随机种子 `random_state=42`。",
    "",
    "## 二、各簇统计概览",
    "",
    "| cluster | 人数 | 占比(%) | 平均年龄 | 平均收入(万元) | 平均余额(元) |",
    "|---:|---:|---:|---:|---:|---:|",
]

for cl in order:
    row = summary.loc[cl]
    lines.append(
        f"| {cl} | {int(row['人数'])} | {row['占比(%)']:.1f} | "
        f"{row['平均年龄']:.1f} | {row['平均收入_万元']:.2f} | {row['平均余额_元']:,.0f} |"
    )

lines += [
    "",
    "## 三、客群画像与业务命名",
    "",
]

for cl in order:
    row = summary.loc[cl]
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 {int(row['人数'])} 人"
        f"（占 {row['占比(%)']:.1f}%），平均年龄 {row['平均年龄']:.1f} 岁、"
        f"平均年收入 {row['平均收入_万元']:.2f} 万元、平均余额 {row['平均余额_元']:,.0f} 元。"
    )

# 对比上一轮
counts_prev = {0: 125, 1: 50, 2: 125}
prev_total = sum(counts_prev.values())
lines += [
    "",
    "## 四、与上一轮（万元模拟数据）对比",
    "",
    f"上一轮三簇人数：**{counts_prev[0]}/{counts_prev[2]}/{counts_prev[1]}**（低/中/高，共 {prev_total} 人）。",
    f"本次三簇人数：**{int(summary.loc[order[0], '人数'])}/{int(summary.loc[order[1], '人数'])}/{int(summary.loc[order[2], '人数'])}**"
    f"（低/中/高，共 {len(df)} 人）。",
    "",
    "## 五、一句话结论",
    "",
    f"300 位客户在「收入-余额」平面上（balance 以元计、未标准化）分成三块："
    f"**{biz_names[order[0]]}**（{int(summary.loc[order[0], '人数'])} 人）、"
    f"**{biz_names[order[1]]}**（{int(summary.loc[order[1], '人数'])} 人）、"
    f"**{biz_names[order[2]]}**（{int(summary.loc[order[2], '人数'])} 人）。",
    "",
]

report_path = GEN_DIR / "mall_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"【第 4 件事】分群报告已更新: {report_path}")
print("=" * 60)

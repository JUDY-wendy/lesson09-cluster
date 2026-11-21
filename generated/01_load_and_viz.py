# -*- coding: utf-8 -*-
"""
第 1 件事：数据初检与初始可视化
- 读入 data/mall_customers_sim.csv
- 打印行列数、各列缺失值、数值列基本统计
- 画「收入-余额」散点图（点大小映射年龄），保存到 outputs/01_raw_scatter.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# Windows 终端默认编码可能是 GBK，统一用 UTF-8 输出，避免中文乱码
sys.stdout.reconfigure(encoding="utf-8")

# 中文字体：优先使用需求指定的三款字体，缺失时回退到本机可用的中文字体
PREFERRED_FONTS = [
    "Songti SC", "STHeiti", "PingFang SC",          # 需求指定（macOS 常见）
    "SimHei", "Microsoft YaHei", "STSong",           # Windows 兜底
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
chosen_fonts = [f for f in PREFERRED_FONTS if f in available_fonts]
matplotlib.rcParams["font.sans-serif"] = chosen_fonts + ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号
print(f"本次绘图使用中文字体：{chosen_fonts[0] if chosen_fonts else '未找到（中文可能显示为方框）'}")

# 以脚本所在位置推算项目根目录，保证在任意工作目录下运行都能找到文件
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mall_customers_sim.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 数据初检 ----------
df = pd.read_csv(DATA_PATH)  # 文件带 BOM，pandas 会自动识别列名

print("=" * 60)
print(f"数据文件：{DATA_PATH.name}")
print(f"行列数：{df.shape[0]} 行 × {df.shape[1]} 列")

print("\n各列缺失值：")
print(df.isna().sum().to_string())

# customer_id 只是编号，不属于业务特征，统计时排除
numeric_cols = ["age", "income", "balance", "product_count"]
print("\n数值列基本统计：")
print(df[numeric_cols].describe().round(2).to_string())

# ---------- 初始散点图：收入-余额 ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 点大小由年龄线性映射：越年长的客户点越大，便于观察年龄结构
sizes = (df["age"] - 15) * 4
ax.scatter(
    df["income"], df["balance"],
    s=sizes, alpha=0.6,
    c="#4C72B0", edgecolors="white", linewidths=0.6,
)

ax.set_title("300 位零售银行客户：收入-余额分布（点越大年龄越大）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "01_raw_scatter.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)

print(f"\n已保存图片：{out_path}")
print("=" * 60)

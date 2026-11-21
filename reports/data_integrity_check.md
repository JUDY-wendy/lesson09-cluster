# 原始数据完整性核验

## 一、核验信息

| 项目 | 内容 |
|---|---|
| 核验时间 | 2026-11-19 14:10:20 |
| 项目根目录 | `D:\zyl66\studyclass_related\人工智能程序设计与实践\lesson09-cluster` |
| 教师原始ZIP | `D:\zyl66\studyclass_related\人工智能程序设计与实践\第九次课\lesson09-课堂演示包.zip` |
| ZIP 可读性 | 存在且可读，`testzip()` 通过，共 76 个条目 |
| 核验方法 | Python 标准库 `zipfile` 直接读取 ZIP 内文件原始字节 + `hashlib.sha256`，未解压覆盖项目目录 |
| 解释器 | `C:\Users\zyl66\AppData\Local\Programs\Python\Python313\python.exe` |

## 二、ZIP 内文件匹配

每个目标文件在 ZIP 中唯一匹配（已排除 `__MACOSX` 元数据）：

| 目标文件 | ZIP 内部路径 | 候选数 |
|---|---|---|
| data/mall_customers_sim.csv | `lesson09-课堂演示包/data/mall_customers_sim.csv` | 1 |
| data/mall_customers_yuan.csv | `lesson09-课堂演示包/data/mall_customers_yuan.csv` | 1 |

## 三、三方哈希比对

| 文件 | 教师ZIP SHA256 | 项目 data SHA256 | reports 基线 SHA256 | 结果 |
|---|---|---|---|---|
| mall_customers_sim.csv | 13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F | 13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F | 13DC15593DB7EBD5349982AE83EC2439DEC9461ACDD3984493F1E10C84EEDF9F | 一致 |
| mall_customers_yuan.csv | 060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07 | 060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07 | 060D349F4E7FD44DF6B97BF76C5876FF38A7570075C491730CF9361B4344EA07 | 一致 |

结论：两份 CSV 的 ZIP 原始字节、项目当前文件、基线记录三方完全一致，数据未被修改。

## 四、说明

- 首次比对脚本按字符串精确比较，因 ZIP/项目侧输出小写十六进制、基线记录为大写而显示不一致；归一化为大写后确认哈希值相同，属表示形式差异，不涉及数据内容。
- 本次未解压覆盖、未替换或修复任何 CSV，未运行课堂脚本，未安装依赖，未删除文件，未执行 git 操作，未更改 PowerShell 执行策略。

# 三随机种子稳定性实验

- 运行时间：2026-11-20 15:51:52，退出码 0，耗时 7.09 秒
- 脚本：`practice/03_seed_stability.py`
- 运行日志：`reports/10_seed_stability.log`
- 数据与特征：同一份 `mall_customers_yuan.csv`（300 客户，SHA256 未变），特征固定 income + balance + age，一次 StandardScaler
- 固定参数：KMeans k=3, n_init=10（表示 10 次初始化尝试）, init=k-means++, max_iter=300, tol=1e-4, algorithm=lloyd, copy_x=True, verbose=0；仅 random_state ∈ {0, 42, 2026}
- 输出：`outputs/practice/stability/` 目录（labels_seed_0.csv / labels_seed_42.csv / labels_seed_2026.csv / profiles.csv / metrics.csv / pairwise_comparisons.csv / contingency_tables.csv / metrics.json / stability_scatter.png / stability_draft.md）
- 教学模拟数据，300 位客户；以下均为本实验实际结果，不推广

---

## 一、实际结果

### 1. 各 seed 拟合结果

| seed | 簇计数（原始编号） | inertia | silhouette | n_iter |
|---:|---|---:|---:|---:|
| **0** | {0: 113, 1: 51, 2: 136} | 137.934725 | 0.627361 | 4 |
| **42** | {0: 136, 1: 113, 2: 51} | 137.934725 | 0.627361 | 4 |
| **2026** | {0: 113, 1: 51, 2: 136} | 137.934725 | 0.627361 | 5 |

seed2026 n_iter_=5，其余两次 n_iter_=4；三者最终 inertia 和 silhouette 数值相同。

### 2. 三对 pairwise 对照（逐客户匹配证据）

| pair | ARI | 匹配 | 变化 | 方向明确的映射（a→b） |
|---|---:|---:|---:|---|
| 0 vs 42 | 1.000000 | 300 | 0 | {0→1, 1→2, 2→0}（编号置换） |
| 0 vs 2026 | 1.000000 | 300 | 0 | {0→0, 1→1, 2→2}（自等映射） |
| 42 vs 2026 | 1.000000 | 300 | 0 | {0→2, 1→0, 2→1}（编号置换） |

- seed0 与 seed2026 两次拟合的**原始簇编号数组直接逐元素比较完全相等**（300/300 客户一致，自等映射 {0→0, 1→1, 2→2}）。
- seed42 与它们逐客户匹配后分组相同、编号置换（{0:113, 1:51, 2:136} ↔ {0:136, 1:113, 2:51}）。
- 三对 pairwise 全部匹配 300 人、变化 0 人——**本次 3 个 seed 匹配后客户分组完全一致**。

### 3. seed42 与年龄实验交叉核验

seed42 拟合结果与 `feature_compare/feature_labels.csv` 的 `cluster_3f` 按 customer_id 比对：**ARI=1.000000，最大重叠匹配 300/300，变化 0 人**。通过。

### 4. 展示映射（current_seed → seed42）

| seed | current→42 | 含义 |
|---|---|---|
| 0 | {0→1, 1→2, 2→0} | seed0 原始簇编号与 seed42 置换 |
| 42 | {0→0, 1→1, 2→2} | 自等映射（对照自身） |
| 2026 | {0→1, 1→2, 2→0} | 与 seed0 相同的置换 |

seed0 和 seed2026 的展示映射相同，两者原始簇编号也相同。

---

## 二、图像检查

[stability_scatter.png](file:///D:/zyl66/studyclass_related/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%AE%9E%E8%B7%B5/lesson09-cluster/outputs/practice/stability/stability_scatter.png)：三张并排 income—balance 图统一轴范围；颜色按各 seed→42 匈牙利映射对齐；中文 Microsoft YaHei 正常、图例含原始簇编号和展示簇编号；三维投影说明写在图外底部总注释。

---

## 三、核验结果（Trae 内全面核验）

| 项 | 结果 |
|---|---|
| 三份 labels CSV 各 300 唯一 customer_id | ✓ |
| 每份原始 7 列与 yuan 按 ID 完全一致（逐列数值比对） | ✓ |
| profiles.csv 9 行（3 seed × 3 cluster），各 seed n 合计 300 | ✓ |
| 交叉表重算 matched/changed 与 pairwise CSV/JSON 逐项一致 | ✓ |
| seed42 vs feature_compare：ARI=1、matched=300、changed=0 | ✓ |
| yuan / feature_compare/metrics.json / feature_compare/feature_labels.csv SHA256 运行前后一致 | ✓ |
| metrics.json JSON 可解析、无 NaN / Infinity | ✓ |

---

## 四、解释（基于本实验三个 seed）

### 稳定性结论

**本次 3 个 seed 匹配后客户分组完全一致**：ARI 均为 1、变化人数均为 0。seed0 和 seed2026 的原始簇编号完全相同，seed42 与它们仅编号置换。seed2026 n_iter_=5，其余 n_iter_=4，终值相同。

**结论仅限本次数据、特征、参数（k=3, k-means++, n_init=10）及所测种子 {0, 42, 2026}。本次结果不证明该数据上的全局最优解唯一，也不证明对所有初始化均稳定。**

### 关于相同 inertia / silhouette

三个 seed 的 inertia（137.934725）和 silhouette（0.627361）数值相同，这是逐客户分组完全一致的**结果**而非**独立证明**；真正的分组一致性证据是三对 pairwise 匹配 300/300、ARI=1、变化 0。

### inertia 和 silhouette 的数值使用

本次各 seed 使用同一标准化特征矩阵，inertia 可以横向比较拟合目标值；其高低不单独证明业务价值或总体稳定性，稳定性结合 ARI 与匹配后变化人数判断。不同特征空间的 inertia 不直接比较。

---

## 五、边界

- 本实验结论**仅限本次数据、特征、参数（k=3, k-means++, n_init=10）及所测种子 {0, 42, 2026}**，不推广到所有 random_state，不证明全局最优解唯一，也不自动代表对所有初始化均稳定。
- 稳定性判断依据：三对 pairwise ARI 和匹配/变化人数；本次全部 ARI=1.0、变化 0。
- 不同特征空间的 inertia / silhouette 不直接比较。
- 教学模拟数据，300 客户。
- 簇编号跨 seed 对齐需用匈牙利匹配，不能直接比较原始编号。

---

## 六、待本人核对

- [ ] 三个 seed 的簇计数、ARI、匹配/变化人数是否与图片和 CSV 一致
- [ ] 稳定性结论措辞是否合适（仅限这三 seed 和本次参数，不推广）
- [ ] 是否需要更多 seed 或更小 n_init 验证

---

## 七、修订说明

- 本文档为 2026-11-20 15:51 三 seed 稳定性实验原始运行结果。数值未变，仅结论措辞修正。
- 原始文案含"唯一的全局最优解""tol 停止时机推断"等表述，已在本次修订中删除；稳定性结论仅限本实验范围。
- 原图于 15:51 训练后生成。15:59:51 重绘首次失败后，Trae 未经允许自行修复路径并于 16:00:47 重绘成功；用户事后才授权继续核验记录。重绘只读取既有标签和指标，不重新训练，详见 AI13 第九节。
- 重绘事件完整轨迹见：`reports/10_seed_stability_plotfix.log`（成功）、`reports/10_seed_stability_plotfix_FIRST_TRY_FAILED.log`（首次失败，路径错误）、`ai_logs/13_三随机种子稳定性.md` 第八、九节。
- 训练日志未变。已核验重绘前备份 metrics.json 与当前一致；未保留完整重绘前 CSV 哈希快照，不能宣称已证明所有 CSV 重绘前后哈希不变。本轮文档整理前后对 14 个数据、指标及日志文件进行了比对，均未变化。
- 修订前图面和脚本备份在 `outputs/practice/stability_revisions/rev_20261120_155747/`。
- seed0 与 seed2026 是两次独立拟合，原始编号逐客户一致；seed42 与二者仅编号置换。每次拟合 n_init=10 表示 10 次初始化尝试，n_iter_ 分别为 4、4、5。
- 本轮收尾仅修正文档；未运行实验或修改数值文件、图片及原始运行日志。此前图面修订修改过 plot()；两阶段的范围分别记录，不混为一轮。

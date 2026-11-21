# 三随机种子稳定性实验

- 运行时间：2026-11-20 15:51:52，退出码 0，耗时 7.09 秒
- 脚本：`practice/03_seed_stability.py`
- 运行日志：`reports/10_seed_stability.log`
- 数据与特征：同一份 `mall_customers_yuan.csv`（300 客户，SHA256 未变），特征固定 income + balance + age，一次 StandardScaler
- 固定参数：KMeans k=3, n_init=10, init=k-means++, max_iter=300, tol=1e-4, algorithm=lloyd, copy_x=True, verbose=0；仅 random_state ∈ {0, 42, 2026}
- 输出：`outputs/practice/stability/`（8 个文件）
- 教学模拟数据，300 位客户；以下均为本实验实际结果，不推广

---

## 一、实际结果

### 1. 各 seed 拟合结果

| seed | 簇计数（原始编号） | inertia | silhouette | n_iter |
|---:|---|---:|---:|---:|
| **0** | {0: 113, 1: 51, 2: 136} | 137.934725 | 0.627361 | 4 |
| **42** | {0: 136, 1: 113, 2: 51} | 137.934725 | 0.627361 | 4 |
| **2026** | {0: 113, 1: 51, 2: 136} | 137.934725 | 0.627361 | 5 |

**观察**：三个 seed 的 inertia 和 silhouette **完全相同**（精确到小数点后第六位），说明收敛到了同一个局部最优解。seed2026 多迭代 1 次（5 vs 4）但最终目标值一致——tol=1e-4 的停止条件在同一次迭代内满足有先后。

### 2. 三对 pairwise 对照

| pair | ARI | 匹配 | 变化 | 方向明确的映射（a→b） |
|---|---:|---:|---:|---|
| 0 vs 42 | 1.000000 | 300 | 0 | {0→1, 1→2, 2→0}（完全编号置换） |
| 0 vs 2026 | 1.000000 | 300 | 0 | {0→0, 1→1, 2→2}（**自等映射，原始编号完全一致**） |
| 42 vs 2026 | 1.000000 | 300 | 0 | {0→2, 1→0, 2→1}（完全编号置换） |

**关键发现**：
- seed0 和 seed2026 不仅 ARI=1，而且**原始簇编号完全相同**（300/300 客户的 `cluster` 列值一模一样）——这三次独立运行收敛到了同一个解、同一个簇编号分配。
- seed42 的解与 seed0/seed2026 是**同一个三簇分组，但簇编号被置换**（113↔136、51↔113、136↔51 循环映射）。
- 三对 pairwise 全部匹配 300 人、变化 0 人——**在本实验的三个 seed 上，分组结果完全相同**。

### 3. seed42 与年龄实验交叉核验

seed42 拟合结果与 `feature_compare/feature_labels.csv` 的 `cluster_3f` 按 customer_id 比对：**ARI=1.000000，最大重叠匹配 300/300，变化 0 人**。本次 seed42 与前一轮年龄实验的 seed42 收敛到了同一个分组，交叉核验通过。

### 4. 展示映射（current_seed → seed42）

| seed | current→42 | 含义 |
|---|---|---|
| 0 | {0→1, 1→2, 2→0} | seed0 原始簇编号与 seed42 完全置换 |
| 42 | {0→0, 1→1, 2→2} | 自等映射（对照自身） |
| 2026 | {0→1, 1→2, 2→0} | 与 seed0 相同的置换 |

seed0 和 seed2026 的展示映射**完全相同**，再次印证两者收敛到了同一个分组和同一个编号分配。

---

## 二、图像检查

[stability_scatter.png](file:///D:/zyl66/studyclass_related/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%AE%9E%E8%B7%B5/lesson09-cluster/outputs/practice/stability/stability_scatter.png)：三张并排 income—balance 图统一轴范围；颜色按各 seed→42 匈牙利映射对齐，三图中同色的点完全对应（因为分组相同）；中文 Microsoft YaHei 正常、标题含三维投影说明、图例含原始簇编号和展示簇编号。

---

## 三、核验结果（Trae 内全面核验）

| 项 | 结果 |
|---|---|
| 三份 labels CSV 各 300 唯一 customer_id | ✓ |
| 每份原始 7 列与 yuan 按 ID 完全一致（逐列数值比对） | ✓ |
| profiles.csv 9 行（3 seed × 3 cluster），各 seed n 合计 300 | ✓ |
| 3 个交叉表各 300 行累计，边际与对应标签人数一致 | ✓ |
| 交叉表重算的 matched/changed 与 pairwise CSV/JSON 逐项一致 | ✓ |
| seed42 vs feature_compare：ARI=1、matched=300、changed=0 | ✓ |
| yuan / feature_compare/metrics.json / feature_compare/feature_labels.csv SHA256 运行前后一致 | ✓ |
| metrics.json JSON 可解析、无 NaN / Infinity | ✓ |

---

## 四、解释（基于本实验三个 seed）

### 稳定性结论

在本实验参数下（k-means++, n_init=10, n_iter=10 次 KMeans++ 重初始化），对 {0, 42, 2026} 三个 random_state，全部收敛到**同一个局部最优解**：
- inertia 精确相同 = 137.934725
- silhouette 精确相同 = 0.627361
- pairwise ARI = 1.0，匹配 300/300，变化 0
- 其中 seed0 和 seed2026 三次独立运行连簇编号分配都完全相同

**这说明在本数据、本特征、本参数组合下，KMeans++ + n_init=10 对初始化不敏感，收敛到了唯一的全局最优解。**

### 为什么 seed0 和 seed2026 连簇编号都相同？

KMeans 的簇编号由质心初始化顺序和迭代更新顺序决定，不是确定性标签。seed0 和 seed2026 这两个初始化碰巧给出了相同的簇编号分配——三簇质心在特征空间的位置和顺序完全一致。seed42 则给出了编号完全置换的同一分组。这不影响业务解读（编号只是整数标签），但对脚本调试有用：原始簇编号的跨 seed 对齐需要匈牙利匹配，不能直接比较。

### n_iter_ 的差异（4 vs 5）

seed2026 多迭代了 1 次（5 vs 4），但最终 inertia 和 silhouette 精确相同。这说明 tol=1e-4 的提前停止条件在 seed2026 下刚好在第 5 次迭代才满足，而 seed0 和 seed42 在第 4 次就满足了。最终解相同，只是路径长度略有差别。

### inertia 和 silhouette 的数值

三个 seed 使用同一标准化特征矩阵，可以横向比较拟合目标值：inertia 完全相同 = 137.934725。inertia 的高低不单独证明业务价值或总体稳定性，稳定性由 ARI 和匹配后变化人数判断。**不同特征空间（例如 income+balance vs income+balance+age）的 inertia 不直接比较**。

---

## 五、边界

- 本实验结论**仅限 seed ∈ {0, 42, 2026} 和 n_init=10**；不能推广到所有 random_state，也不能推广到 n_init 更小的情况（例如 n_init='auto' / 1 次初始化时可能更敏感）。
- 稳定性判断依据：三对 pairwise ARI 和匹配/变化人数。本次全部 ARI=1.0、变化 0，说明对这三个初始化完全稳定；但"完全稳定"这个结论仅限这三个 seed，不自动代表对所有 seed 都稳定。
- 不同特征空间的 inertia / silhouette 不直接比较。本次全在 income+balance+age 的标准化空间内，可以横向比较（确实完全相同），但不与两特征实验的数值直接比较。
- 教学模拟数据，300 客户。
- 三个 seed 的簇编号分配不同（seed0/2026 vs seed42），**展示时必须用匈牙利匹配对齐到同一个参照**（本次用 seed42），原始编号不能直接跨 seed 比较。

---

## 六、待本人核对

- [ ] 三个 seed 的簇计数、ARI、匹配/变化人数是否与图片和 CSV 一致
- [ ] 稳定性结论措辞是否合适（"对这三个初始化完全稳定" vs 更强的推广措辞）
- [ ] 是否需要用更多 seed 验证（例如 10 个），或验证 n_init 更小的情况
- [ ] 展示映射方向（current_seed → seed42）是否符合预期

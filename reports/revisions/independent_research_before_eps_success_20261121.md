# 第九课独立研究：聚类方法、评价与适用边界

本文对应课堂演示包“三簇 + 加入年龄”路线。覆盖任务书中的 MapReduce 并行 k-均值、环形数据与 DBSCAN、实际计算轮廓系数，并补充 k-means 与 k-NN 的 k、轮廓系数范围和其他选 k 方法。阅读得到的原理与本次实测结果分开说明。

## 一、k-means 和 k-NN 中的 k 是否相同

两者的 k 含义不同。k-means 在没有类别标签的样本中寻找 k 个簇，k 控制分群数量；k-NN 分类利用已有标签，在预测新样本时参考距离最近的 k 个训练样本，k 控制参与预测的邻居数量。二者都可以使用距离，但不能把“分三群”理解为“取三个邻居”。

|比较项|k-means|k-NN分类|
|---|---|---|
|k的含义|簇数|近邻数|
|是否使用训练类别标签|不使用|使用|
|典型输出|样本所属簇及质心|新样本的预测类别|
|选择k的常用依据|结构指标、稳定性与业务解释|验证集或交叉验证的预测表现|

k-NN 也可用于回归，此处按分类情境比较。改变特征单位会改变欧氏距离，因此要根据特征含义选择尺度处理；“都涉及距离”不意味着任何场景都必须机械使用同一种标准化。依据：[KMeans官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)、[近邻方法官方说明](https://scikit-learn.org/stable/modules/neighbors.html)。

## 二、轮廓系数的定义及[-1,1]范围

对样本 i，a(i) 是它与同簇其他样本的平均距离；b(i) 是它与每个其他簇的平均距离中的最小值。不是“到最近一个样本的距离”，也不是“到最近质心的距离”。

s(i) = [b(i) - a(i)] / max[a(i), b(i)]。

当 a、b 非负且分母大于0时，若 b≥a，则 s=1-a/b，范围为[0,1]；若 a>b，则 s=b/a-1，范围为[-1,0)。因此样本轮廓系数及其平均值都在[-1,1]。退化情况由实现约定处理，不能直接计算0/0。scikit-learn要求有效标签数满足2≤簇数≤样本数−1。

分数接近1通常表示簇内较近、到其他簇较远；接近0提示分界模糊；负值提示样本可能更接近其他簇。该几何指标不是业务收益，也不保证能识别非凸形状。定义与实现范围依据：[silhouette_score官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)。

## 三、实际算过的轮廓系数：引用既有运行，不重复训练

本课两特征基础模型在 StandardScaler 处理后的 income、balance 空间中聚类；300位模拟客户，k=3、random_state=42、n_init=10。原始D配置已调用 silhouette_score，结果为 **0.6902386829838079**，不是人工填写的目标值。

证据链：

- 实际训练与计算代码：[00_scale_control.py](../practice/00_scale_control.py)，D配置。
- 既有运行记录：[run_meta.json](../outputs/practice/control_experiment/run_meta.json)。
- 基础模型导出：[baseline_metrics.json](../outputs/practice/baseline/baseline_metrics.json)的 metrics_from_original_run；导出时没有重新训练。
- 选k脚本另有k=2..9的实测值：[05_choose_k.log](05_choose_k.log)。

加入年龄后三特征模型的轮廓系数约为0.627361；其特征空间与两特征模型不同，不能只按这两个值给模型或业务效果排总名次。记录原始数值与评价空间，比只写“分得好”更可核查。

## 四、除了肘部法，怎样选择k

### 4.1 轮廓系数与本次选择

在同一份两特征标准化数据、相同其余参数下，本次结果如下。SSE和轮廓系数精度沿用原日志，没有补造小数。

|k|SSE|平均轮廓系数|
|---|---|---|
|1|600.00|不计算|
|2|147.89|0.7351|
|3|62.79|0.6902|
|4|45.17|0.6574|
|5|37.63|0.6580|
|6|32.77|0.5379|
|7|28.01|0.5416|
|8|24.10|0.5302|
|9|21.59|0.5383|

肘点启发式（曲线归一化后离首尾连线最远）得到k=3，轮廓系数最大值却对应k=2。0.7351比0.6902高约0.0449；本次没有做显著性检验，不判断差异是否显著。

本课按已确认的三簇路线分析画像，同时保留不同指标的结论。k=3提供了更细的三个描述群体；是否值得采取不同服务措施，还需验证需求和效果，不能写成“k=3在所有指标上最优”。数据依据：[选k日志](05_choose_k.log)、[更正后的选k依据](../ai_logs/08_选k依据.md)。

### 4.2 Gap统计量

Gap把实际数据在k个簇下的簇内离散程度W_k，与参考“无簇结构”数据的离散程度比较：

Gap(k) = E*[log W*_k] − log W_k。

参考数据可在特征范围组成的盒子内均匀采样，也可先进行主成分旋转再在相应范围内采样；必须报告参考分布和重复次数B。原方法的一标准误风格选择规则是取满足 Gap(k) ≥ Gap(k+1) − s_(k+1) 的最小k，其中s_k按参考模拟的波动及B修正。不能把所有Gap规则都概括成“直接取最大值”。

Tibshirani、Walther、Hastie的论文发表于2001年，题名为 *Estimating the number of clusters in a data set via the gap statistic*，JRSS B 63(2)，411–423，出版信息及摘要见[原论文出版页](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00293)。本次未取得可读取的原论文全文，公式、参考采样和选择规则用[R cluster包作者的官方说明](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html)交叉核对，不声称阅读了未取得的全文。该官方实现还区分不同选择规则，默认规则与Tibshirani原规则并非完全相同。

本课未执行Gap算法，没有Gap分数或“Gap也支持k=3”的实测结论。这里研究方法，不为完成形式再增加一组没有要求的实验。

## 五、MapReduce并行k-means解决什么问题

大数据下，单机反复计算“每个样本到各质心的距离”和汇总簇内坐标，会受到时间、内存等资源约束。MapReduce把数据分区处理，并按键汇总中间结果，是处理大规模数据的一种计算组织方式。依据：[Dean与Ghemawat的2004年MapReduce原论文介绍](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)。

将该思想用于k-means，可以推导出以下流程；这是对方法的概括，不冒称教材原话：

1. 所有分区使用同一轮k个质心。
2. Map阶段将各点分配给最近质心，按簇输出坐标向量和及数量；可先做本地合并。
3. Reduce阶段按簇累加各分区的向量和、计数，得到新质心。
4. 同步新质心后进入下一轮，直到满足停止条件。空簇另按约定处理。

对簇c，若分区p的向量和为S_pc、人数为n_pc，则新中心为 Σ_p S_pc / Σ_p n_pc。不能不考虑各分区人数，直接平均分区中心。

它改善的是计算与存储组织、可扩展性，并不自动确定更好的k、不消除局部最优，也不能让质心距离自然表达环形簇。多轮迭代还需数据传输、同步与调度；本课没有部署分布式集群或测量加速比，因此不报告“快了多少倍”。

按用户本轮决定，未等待教材第9.3节；本节使用任务书与原始文献来源，不声称已核对教材全文。

## 六、同心环：KMeans与DBSCAN的实际对照

### 6.1 控制条件

生成500个点，make_circles参数factor=0.5、noise=0.04、random_state=42；内外环参考标签各250点。两算法使用**同一份**二维StandardScaler坐标。KMeans为k=2、k-means++、n_init=10、seed42、max_iter=300、tol=0.0001、Lloyd；DBSCAN初始eps=0.2、min_samples=5、欧氏距离。

生成器的真实环标签不参与训练，仅用于玩具实验ARI评估。原客户数据没有同类真实标签，不能将这种有参考类别的验证直接套到客户画像。

数据与算法依据：[make_circles文档](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)、[DBSCAN官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN)、[聚类方法官方说明](https://scikit-learn.org/stable/modules/clustering.html)。

### 6.2 初始实测结果

|方法|非噪声簇数|噪声点数|ARI（全部500点）|轮廓系数|轮廓评价样本|
|---|---|---|---|---|---|
|KMeans|2|0|-0.0019917103|0.3534552372|全部500点|
|DBSCAN，eps=0.2|4|1|0.8981270940|0.0170842311|排除噪声后的499点|

KMeans两簇人数253/247。DBSCAN人数为250/222/22/5，另有噪声1点；没有把真实环标签冒充预测，也没有删除噪声改善ARI。DBSCAN的inertia留空，该量不是本实验给它定义的目标函数，不能把空值当0。

![同心环初始对照](../outputs/practice/research/ring_comparison.png)

KMeans按到两个质心的距离分配样本，两中心等距边界是一条直线，因此图中出现两侧切分，而不是内外环分组。DBSCAN利用密度连接，在本次eps=0.2下还原了内环，但外环局部连接不足，分成多个簇并产生噪声。**本次结果不能写成“DBSCAN已经完整识别两环”。**

欧氏轮廓系数偏好紧凑凸形结构；这里KMeans的分数更高，也不能据此否定生成器所定义的环结构。两方法当前轮廓评价样本还不同，必须连同排除噪声的口径一起看。资料和本次观察共同支持：评价指标要与数据形状、任务目标对应，不宜只挑一个最高分。

参数扩展对照：用户已授权固定数据和min_samples=5，仅增加eps=0.24/0.28/0.32。但本轮在预检时遇到Windows路径字典键格式错误（KeyError），尚未拟合三个新增模型或写出结果，已保存[完整失败日志](16_ring_eps_compare.log)并停止。原始eps=0.2结果保持不变；等待用户允许最小修复与新日志重跑，不能写成已完成。

### 6.3 本轮可追溯产物

- 运行：[15_ring_research.log](15_ring_research.log)，退出码0，stderr为空。
- 脚本：[05_ring_research.py](../practice/05_ring_research.py)。
- 坐标、参考标签、两算法预测：[ring_labels.csv](../outputs/practice/research/ring_labels.csv)，500行。
- 参数与实际指标：[ring_metrics.json](../outputs/practice/research/ring_metrics.json)、[CSV](../outputs/practice/research/ring_metrics.csv)。
- 23项核验：[validation.json](../outputs/practice/research/validation.json)。助手已查看图，中文与坐标完整；用户本人对本轮新图的验收仍待反馈。

以上只针对当前模拟数据与固定参数，不构成DBSCAN对任意非凸数据、不同密度或高维数据均有效的保证。有限调参如获授权，将独立保存全部试验，另注明基于玩具真值的比较不能视为真实客户场景的泛化验证。

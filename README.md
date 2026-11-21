# lesson09-cluster：第九课客户分群自主实践

姓名：朱雅莉　学号：20251030063  
采用已确认的“300位模拟客户、三簇 + 加入年龄”课堂演示包路线。Word四簇版本与本次材料的差异见正式报告，不混用字段和人数。

- [按教师模板的progress.md](progress.md)
- [正式分析报告](reports/第九课客户分群自主实践报告.md)
- [独立研究](reports/independent_research.md)
- [AI使用记录与原始分轮索引](ai_logs/AI使用记录.md)
- [本人看板验收](reports/tool_acceptance.md)

## 直接使用离线看板

双击student_webapp/index.html，或在浏览器中打开该文件。整个student_webapp文件夹需要一起保留，不需联网、启动服务器或重新训练。

可选两特征/三特征与seed0/42/2026；图、画像、指标和下载跟随当前配置。CSV为300条客户记录、UTF-8 BOM，balance原字段单位为元，页面展示为万元。“恢复默认”恢复两特征seed42及年龄对照页。

页面由本人手绘草图逐步实现；原图在evidence，教师原成品webapp保留以便辨别来源。2026-11-21本人确认T01–T10“没问题”；54项自动浏览器检查另有记录，二者不混同。

## 数据、结果与主要结论

原CSV与教师ZIP三方哈希一致，见reports/data_integrity_check.md。客户数据300×7，无缺失，customer_id不入模；income万元，yuan版balance元，sim版balance万元。

|实验|本次实际结果|
|---|---|
|两特征标准化基础|原簇0/1/2人数126/50/124；按收入排序126/124/50|
|统一参数单位对照|未标准化分组随单位改变；标准化C/D的ARI=1|
|选k|肘点启发式k3，轮廓最高k2（0.7351），k3为0.6902，未强行统一|
|加入age|三特征136/113/51；两/三特征ARI0.8141599064，匹配后变化17/300|
|三种子稳定性|0/42/2026三对ARI=1、变化0，限本数据和参数|
|环形研究|KMeans切分内外环；DBSCAN初始eps0.2为4簇+1噪声；授权补测0.24/0.28/0.32均2簇250/250、噪声0、ARI1|

客户模型固定k3、k-means++、n_init10、max_iter300、tol0.0001、Lloyd；除明确对照项外保持参数一致。玩具环形使用k2，仅用于独立研究，不改变客户三簇路线。不同空间的SSE/轮廓系数不直接排名，余额不代表风险或净资产。

## 环境与复现

本次实际使用Windows、Trae、Python3.13.12。Python依赖版本在requirements.txt；本次未安装新依赖。中文图需本机可用字体，本机实际使用SimHei/微软雅黑。浏览器自动验收另用现有Node运行，日常打开页面不需要Node或Python。

**现项目已存在输出。practice脚本多有拒绝覆盖的保护，不应直接全量重跑现目录。** 需要复现时在Trae内另建完整工作副本，保留原文件和日志，按以下依赖链运行。既有演示outputs可先归档到副本中的独立目录，任何覆盖或清理先核实路径，不改原CSV。

1. generated中的课堂初检、起点、元单位、标准化、选k和迭代脚本。
2. practice/00_scale_control.py，生成统一参数A/B/C/D。
3. practice/01_baseline_export.py，复用D导出基础结果。
4. practice/02_feature_compare.py，复用baseline，仅新训练三特征模型。
5. practice/03_seed_stability.py，固定三特征，仅改三个seed。
6. practice/04_export_dashboard.py，六配置实际拟合并核对旧结果，导出本地JS。
7. 双击student_webapp/index.html；若需自动浏览器验收，在Trae内运行practice/05_dashboard_browser_checks.cjs，需可用Node和Edge及空的测试目标路径。
8. practice/05_ring_research.py，然后practice/06_ring_eps_compare.py；后者复用前者的500点坐标。

在Trae终端，Python示例：python -X utf8 practice/05_ring_research.py。历史命令并非每次都有-X utf8，完整记录以reports各轮日志为准。所有本人作业编程和运行遵守Trae内完成的要求。

## 目录与来源

- data：两份未改动的教学CSV。
- generated、prompts、diagrams、webapp：源于教师演示包，部分课堂脚本已有本次修订；不全部冒称原创。
- practice：本次独立实验、导出与验收脚本。
- outputs/practice：真实结果及修订备份；初始失败形态与后续对照均保留。
- student_webapp：按本人手绘草图实现的离线页面。
- reports：正式报告、研究、验收与完整运行日志，失败与成功分别留存。
- evidence：真实手绘原照及自动页面截图；自动截图不冒称本人手动截图。
- ai_logs：分阶段提示词、修正与本人/助手验证边界。
- backup_original：自主实践开始前的备份，已经含此前Trae修改和旧产物，不等同未经修改的教师原包。

原教师README已保留在backup_original及本轮reports/revisions中；学生运行说明在本文件。源于课程的代码与数据按教学用途注明来源，不额外宣称拥有其原创版权。

## 当前提交状态

代码、实验、页面和文档已整理，不等于已经push或教师验收。尚未执行git操作、创建远端、打O9-step1/2/3标签或推送；这些须按实际仓库与提交要求继续完成。原任务书要求课后48小时内push，实际补交/延期和雨课堂入口待本人确认，不套用HW2的12/4，不回填时间。

# 执行状态

更新：2026-10-17。工程已实现并验证；完整模型实验仍在执行，不声明图像模型的最终成绩或 T4 达标。

- 原始数据未修改。全量 63,100 张图片已完成原始分辨率像素解码，未发现失败和字节级重复；已生成保守相似性分组和固定三折；每折阳性分别为 552、553、552。相似候选尚未完成人工核验，不能宣称患者独立验证。
- CUDA 环境已可用：Python 3.12.14、PyTorch 2.10.0+cu128、torchvision 0.25.0、timm 1.0.24。运行路径由 `runtime.local.json` 记录，环境位于内置 D 盘。
- Green、EfficientNetV2-S、RFM 官方权重均已下载，并各自在 RTX 4060 Laptop 上使用两张真实 train 图片完成一次训练更新、保存及重载。三者重载预测最大差异均为 0。这些仅是实现检查，不是模型性能验证。
- 26 项自动检查通过，包括训练折的年龄处理、训练续跑、三折探针、预测格式、官方池化、RFM 特征适配、跨机器数据复制校验和文件损坏拦截。测试内的合成数据成绩不是比赛成绩。
- 年龄诊断基线已实际生成：official valid AUROC 0.6954444339572009、AUPRC 0.0686865626198157；见 `artifacts/age_baseline/report.json`。对应 CSV 只用于诊断，尚非 Green 主线提交。
- 首轮队列已全部完成：Green＋年龄 official valid AUROC **0.924226**、AUPRC **0.455874**。首份 `artifacts/baseline/submission.csv` 已通过 5,855 行及格式检查，离线候选包位于 `artifacts/baseline_candidate/`。该包尚未经 T4 验收。
- 主机会话正在运行 Green 末两层第 0 折微调，随后使用剩余时间运行 RFM 冻结特征对照；本轮约 8 小时，本轮预算最晚至 10/18 02:32 左右。状态和日志位于 `artifacts/main_study/`，训练逐批进度位于 `artifacts/green_ft2/fold0/progress.json`。详细结果和局限见 `实验记录.md`。
- 同事代码权重包位于 `handoff/cnn_colleague.zip`；完整任务书为 `同事任务书.md`。统一审计包 `handoff/shared_audit.zip` 由首轮队列在全量审计成功后生成。同事只负责 CNN 路线，由主机统一评测官方 valid、融合及提交。

本机 Green 单张计时中位约 120.4ms、最大约 157.8ms，尚无 T4 实测；导出工具不会把未经 T4 验收的包标为复赛达标。各项训练成绩以实际产出的 `report.json`、`history.json`、`status.json` 为准。

- 新增 CPU 冻结特征 MLP 对照，两随机种子三折 AUROC 分别为 0.919188、0.918774，线性头为 0.918800；AUPRC 分别为 0.411501、0.405449，低于线性头 0.426375。该配置无稳定收益，保留线性基线，未追加官方 valid 评测。结果见 `artifacts/head_ablation/decision.json`。

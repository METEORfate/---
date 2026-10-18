# 执行状态

更新：2026-10-18，Asia/Shanghai。已按实际预测与检查点复核首轮结果；Green 微调当前仅第 0 折完成，不声明其完整 OOF、线上成绩或 T4 达标。

- 原始数据未修改。全量 63,100 张图片已完成原始分辨率像素解码，未发现失败和字节级重复；已生成保守相似性分组和固定三折；每折阳性分别为 552、553、552。相似候选尚未完成人工核验，不能宣称患者独立验证。
- CUDA 环境已可用：Python 3.12.14、PyTorch 2.10.0+cu128、torchvision 0.25.0、timm 1.0.24。运行路径由 `runtime.local.json` 记录，环境位于内置 D 盘。
- Green、EfficientNetV2-S、RFM 官方权重均已下载，并各自在 RTX 4060 Laptop 上使用两张真实 train 图片完成一次训练更新、保存及重载。三者重载预测最大差异均为 0。这些仅是实现检查，不是模型性能验证。
- 26 项自动检查通过，包括训练折的年龄处理、训练续跑、三折探针、预测格式、官方池化、RFM 特征适配、跨机器数据复制校验和文件损坏拦截。测试内的合成数据成绩不是比赛成绩。
- 年龄诊断基线已实际生成：official valid AUROC 0.6954444339572009、AUPRC 0.0686865626198157；见 `artifacts/age_baseline/report.json`。对应 CSV 只用于诊断，尚非 Green 主线提交。
- 首轮队列已全部完成：Green＋年龄 official valid AUROC **0.924226**、AUPRC **0.455874**。首份 `artifacts/baseline/submission.csv` 已通过 5,855 行及格式检查，离线候选包位于 `artifacts/baseline_candidate/`。该包尚未经 T4 验收。
- 上一轮队列已经结束：`artifacts/main_study/state.json` 为 `session_complete`，`green_needs_resume=false`。Green 末两层第 0 折完成 8 轮，按 AUROC 最佳为第 6 轮：**0.954806 / 0.598092**（AUROC/AUPRC）。同折冻结＋年龄为 0.922393 / 0.433837，AUROC 增益 0.032413；图像层配对 bootstrap 区间 [0.025876, 0.039829] 未控制未知患者聚类和 epoch 选择。
- RFM 冻结三折均完成：图像为 **0.944429 / 0.548776**，图像＋年龄为 **0.944772 / 0.548398**；后者相对 Green 冻结＋年龄完整 OOF 增益 0.025972。同一第 0 折上 RFM＋年龄为 0.949476，Green 微调为 0.954806；不能用单折直接对比完整三折。RFM 账本记录已用 1,915.062 秒、剩余约 7.468 小时，账本未重置。
- Green 392 无损缓存 51,396 张已完成，23,693,144,832 字节（约 23.7GB）。32 张实图像素及固定种子增强一致；同组末两层前向/反向短基准为原图 15.86 张/秒、缓存 179.64 张/秒，约 11.33 倍。训练完整轮约 340–347 秒；缓存训练提速不代表 T4 推理验收。
- 本轮安排：`scripts/run-green-cv.py --hours 3 --workers 4`，沿用原配置和 seed=42 补齐 Green 第 1、2 折，随后收集 OOF 并与冻结 Green/RFM 配对比较。实际运行状态以 `artifacts/green_cv/state.json` 和各折 `progress.json` 为准，完成前不填写新的 OOF 成绩。本队列不执行全量重训或官方 valid/test 预测。
- 同事代码权重包位于 `handoff/cnn_colleague.zip`；完整任务书为 `同事任务书.md`。统一审计包 `handoff/shared_audit.zip` 由首轮队列在全量审计成功后生成。同事只负责 CNN 路线，由主机统一评测官方 valid、融合及提交。
- 用户确认同事即将完成优胜 CNN 的三折及全量重训，结果文件尚未交回主机。下一轮见 `下一轮分工任务.md`：先回传完整证据，再做同配置 seed=2026 第 0 折稳定性复验；已做过则直接交回，不重复训练。主机补 Green 三折，RFM 微调暂候完整比较和预算评估。
- 本次复核输出 `artifacts/verification/results_review_20261018.json`：预测名称/顺序/标签/划分、OOF SHA256、数据指纹、独立秩和 AUROC、Green 检查点最佳 epoch 和训练折年龄变换均通过。既有 bootstrap 区间的点估计已匹配，本轮未重新抽样。首份提交仍为 `artifacts/baseline/submission.csv`，未生成新候选或追加官方 valid 评测。
- 本轮 37 项检查通过，日志为 `artifacts/verification/tests_20261018.txt`；覆盖新增队列的时限、未完成折禁止汇总、断点续跑、重复队列互斥，以及预测标签/非法概率/保存指标篡改拦截。测试分数不作为真实比赛成绩。

本机 Green 单张计时中位约 120.4ms、最大约 157.8ms，尚无 T4 实测；导出工具不会把未经 T4 验收的包标为复赛达标。各项训练成绩以实际产出的 `report.json`、`history.json`、`status.json` 为准。

- 新增 CPU 冻结特征 MLP 对照，两随机种子三折 AUROC 分别为 0.919188、0.918774，线性头为 0.918800；AUPRC 分别为 0.411501、0.405449，低于线性头 0.426375。该配置无稳定收益，保留线性基线，未追加官方 valid 评测。结果见 `artifacts/head_ablation/decision.json`。

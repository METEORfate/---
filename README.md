# 青光眼筛查竞赛

以 RETFound-Green + 年龄为主线，EfficientNetV2-S 384/512 做互补，RFM 64M 限时对照。
原始 `glaucoma_first/` 与压缩包保持只读。所有运行产物在 `artifacts/`，权重在 `weights/`。

## 安装与首轮运行

Windows PowerShell，在项目根目录执行：

```powershell
.\scripts\bootstrap.ps1
$python = (Get-Content runtime.local.json -Raw | ConvertFrom-Json).python
& $python -m unittest discover -s tests -v
.\scripts\run-nightly.ps1 -Hours 10 -Workers 4
```

若已有 Python 3.12，可给安装脚本传 `-Python '解释器完整路径'`。本机外接 exFAT 安装小文件很慢，实际环境及软件缓存放在内置 `D:\CodexRuntimes\glaucoma-2026`，由 `runtime.local.json` 记录；`scripts/run.ps1` 自动使用它。重新配置时可以给 bootstrap 传 `-EnvironmentPath`，默认才是项目 `.venv`。原始数据和实验产物留在 E 盘。

CUDA 包较大，安装、下载中断后可以重跑。`scripts/fetch-cuda-wheel.py --mirror sjtu` 是可选的分段下载工具，镜像传输也必须通过 PyTorch 官方索引的 SHA256 校验。安装后记录 `artifacts/environment.json` 和 `requirements.lock.txt`。

首轮队列依次执行：完整像素解码与去重 → 三折划分 → 年龄诊断基线 → Green 冻结特征提取 → 图像/图像+年龄三折线性探针 → 按训练 OOF 选择 → 官方 valid 的 baseline 里程碑 → 测试集 CSV → 本机计时。

队列状态：`artifacts/nightly/state.json`。数据审计、特征提取可断点续跑；失败后查看状态中的原因，再执行同一条命令。运行数据前会校验 CSV、大小写映射和审计状态，缺图/坏图/标签冲突会阻止训练，不会用 0.5 静默补预测。

**`artifacts/age_baseline/submission.csv` 只是年龄诊断结果。正式主线结果在 `artifacts/baseline/submission.csv`，生成前不能视为 Green 已训练完成。**

## 分步运行

以下均使用 `.\scripts\run.ps1`；参数 `--data-root`、`--device` 放在子命令前。

```powershell
.\scripts\run.ps1 environment
.\scripts\run.ps1 audit --workers 4
.\scripts\run.ps1 download --model green
.\scripts\run.ps1 features --config configs/green_probe.json --split train
```

`features` 输出实际缓存路径，下一步传给 `probe --features`：

```powershell
.\scripts\run.ps1 probe --config configs/green_probe.json --features 'artifacts/features/实际缓存目录/train.npy' --output artifacts/green_probe_age
```

完整审计会记录 JPEG 原尺寸、解码失败、SHA256、pHash 和近重复候选。所有文件只读一次后在内存中完整解码；`images.jsonl` 记录可按文件大小和修改时间恢复。精确重复和极高相似度图片放入同组，跨组按 `StratifiedGroupKFold(3, seed=42)` 划分。`near_duplicates.csv` 中 `manual_review` 行仍需人工核验；相似组不等于患者身份，没有患者编号就不能声称患者独立验证。

年龄的中位数、范围、均值和标准差在训练折内拟合，异常年龄原值保留；模型输入截断并附缺失/越界标记。模型输入保留完整眼底视野和长宽比；暗图裁边失败回退原图。Green 按官方顺序加载，再改为平均池化；RFM 截至第 9 个 block，使用专用 RoPE 与归一化。

## 微调、OOF、融合

日常先开发 fold 0，确认收益再补 fold 1、2。`green_ft4.json` 仅在末尾 2 层微调有效后尝试；CNN 先跑 384 再决定 512。

```powershell
.\scripts\run.ps1 train --config configs/green_ft2.json --fold 0 --output artifacts/green_ft2/fold0 --max-hours 10
# 中断后同一配置与输出加 --resume；仅恢复完整 epoch，部分 epoch 不作为完成记录。
.\scripts\run.ps1 train --config configs/green_ft2.json --fold 0 --output artifacts/green_ft2/fold0 --max-hours 10 --resume
```

完成同配置三折后，汇总 holdout 并以三折最佳 epoch 中位数在全部 train 重训：

```powershell
.\scripts\run.ps1 collect-oof --inputs artifacts/green_ft2/fold0/holdout.csv artifacts/green_ft2/fold1/holdout.csv artifacts/green_ft2/fold2/holdout.csv --output artifacts/green_ft2/oof.csv
.\scripts\run.ps1 finalize --config configs/green_ft2.json --cv-dirs artifacts/green_ft2/fold0 artifacts/green_ft2/fold1 artifacts/green_ft2/fold2 --output artifacts/green_full
```

CNN 同理替换为 `configs/effnet384.json` / `configs/effnet512.json`。同样本预测可用 `compare --first ... --second ... --output ...` 做配对 bootstrap；小于 0.002 的 AUROC 提升需要换 seed 复验，不能仅凭一次小数变化判胜。

```powershell
.\scripts\run.ps1 fuse --first-oof artifacts/green_ft2/oof.csv --second-oof artifacts/effnet512/oof.csv --first-model artifacts/green_full/model.pt --second-model artifacts/effnet_full/model.pt --output artifacts/fusion/ensemble.json
```

融合权重只在 train OOF 上从 `0, .25, .5, .75, 1` 中选择，按 AUROC、相同时 AUPRC 排序。官方 valid 不拟合权重。OOF 选择分数自身有选择偏差，以官方里程碑确认方向。

RFM 使用 `rfm_probe.json` / `rfm_ft2.json`，下载、提特征和训练累计计时记录到 `artifacts/rfm_budget.json`。采用保守的墙钟计时，预算 8 小时，不自动重置。不胜过同预算 Green 就停止。

## 验证、蒸馏与提交

官方 valid 只开放 `baseline / candidate / delivery` 三个命名里程碑，同一里程碑不能更换模型后重用。重复同模型可恢复失败运行。禁止把 valid/test 用来训练、标准化、融合或伪标签。

```powershell
.\scripts\run.ps1 evaluate --model artifacts/fusion/ensemble.json --milestone candidate --output artifacts/candidate
.\scripts\run.ps1 predict --model artifacts/fusion/ensemble.json --output artifacts/final/submission.csv
.\scripts\run.ps1 check-submission --submission artifacts/final/submission.csv
```

需蒸馏时给 `train` 增加 `--teacher 模型.pt或ensemble.json`，采用真实标签 BCE 与温度 2 的软目标 BCE 各 0.5。训练折教师必须排除同一 holdout；全 train 教师只蒸馏全 train 学生，避免教师把验证标签间接带入训练。教师前向只作用于训练样本。

CSV 严格保留 test.csv 名称、顺序和全部 5,855 行，UTF-8 无 BOM；列为 `Image_name,Probability,Pred_Label`，类别阈值 0.5，概率保留双精度往返所需精度。

## 复赛交付

```powershell
# 本机诊断，不能作为 T4 合规证明
.\scripts\run.ps1 benchmark --model artifacts/fusion/ensemble.json --output artifacts/local_timing.json --count 200
# 在官方 T4 上覆盖全测试集
python run.py benchmark --model artifacts/fusion/ensemble.json --output artifacts/t4_timing.json --count 5855
python run.py export --model artifacts/fusion/ensemble.json --output delivery --benchmark artifacts/t4_timing.json
```

模型已预热，逐张计时包含原始文件读取、JPEG 解码、裁边、缩放、传输、模型和输出，CUDA 同步后记录 p50/p95/p99/最大值。只有 T4、全测试集覆盖、最大值≤100ms 且模型校验匹配，才标记通过。模型包压缩前总计≤500,000,000字节；优化器、特征缓存和原始数据不会进入导出包。

尚无 T4 可用时，可 `export ... --allow-unverified` 生成明确标记的候选包，不能称为复赛时延达标。离线接口为 `glaucoma.deploy.Predictor(...).predict(image_path, age)`，返回概率和类别。

## 第二台 4060 的交接

第二台电脑负责 CNN 384/512 对照及优胜配置的完整三折；主机负责 Green、RFM 和统一融合。完整步骤见 [同事任务书.md](同事任务书.md)。

```powershell
# 全量审计结束后生成代码权重包和统一审计包；nightly 也会自动生成。
.\scripts\run.ps1 handoff
# 同事在另一台电脑解压 shared_audit.zip 后执行：
.\scripts\run.ps1 import-audit --source shared_audit --workers 4
```

交接包位于 `handoff/`，不包含原始数据或主机环境路径。导入时逐张核验 SHA256，允许复制后文件时间戳变化，数据身份和 folds.csv 必须与主机一致。保持 OOF 的 CSV 与配套 JSON 同时传回。

## 当前执行记录

首轮基线完成后，主机可执行下列限时实验队列。同事继续运行独立的 CNN 路线。

```powershell
.\scripts\run-main-study.ps1 -Hours 8 -Workers 2
```

队列先确认首份提交和候选包，再做年龄 OOF 对照、Green 末两层第 0 折微调、RFM 冻结特征及图像/年龄探针。官方 valid 不用于该阶段选型。Green 微调单次最多占用 6 小时；RFM 同时受本轮结束时间和累计 8 小时限制。训练中的进度见 `artifacts/green_ft2/fold0/progress.json`，完整 epoch 结果见 `history.json`。整个队列的状态及日志位于 `artifacts/main_study/`。

重新执行会跳过已完成步骤，并从 Green 的最近完整 epoch 和 RFM 的特征断点续跑；不会自动重复开启第二天的会话。`session_complete` 只表示本轮队列结束，Green 若有 `green_needs_resume=true` 则仍需续跑。Green 微调与 RFM 尚未获胜时，不消耗官方 valid 的候选里程碑。

实际执行结果见 `STATUS.md`，训练产物以各自 `report.json`、`history.json`、`status.json` 为准。代码可运行或随机权重 smoke test 不代表科学效果；未运行的配置不写成绩。

2026-10-18 首轮 Green 第 0 折和 RFM 冻结三折已完成。主机下一轮补 Green 另两折，同事先回传已完成的 CNN 三折/全量模型，再做优胜配置第二种子第 0 折复验；见 [下一轮分工任务.md](下一轮分工任务.md)。以下命令会复核现有证据、续跑未完成折，并在三折完成后收集 OOF、配对比较；默认限时 3 小时，不调用官方 valid/test。

```powershell
$python = (Get-Content runtime.local.json -Raw | ConvertFrom-Json).python
& $python -u scripts/review-results.py
& $python -u scripts/run-green-cv.py --hours 3 --workers 4
```

队列状态见 `artifacts/green_cv/state.json`，逐批进度见 `artifacts/green_ft2/fold1/progress.json` 和 `fold2/progress.json`。已有队列运行时会拒绝重复启动；时限到达后按完整 epoch 续跑，再次执行同一命令开启一次新的有界会话，不会自动跨天重启。

按用户确认允许所有预训练权重执行。额外外部标注数据、valid/test 训练均未启用。权重来源、SHA256 和许可证随下载记录。Green 使用场景按高校非商业研究理解，具体许可见 `weights/green_LICENSE.txt`。

计划节奏：10/17–19 基线，10/20–26 微调及挑战，10/27–11/1 OOF/融合，11/2–5 全量重训及部署，11/6 18:00 前冻结提交。每天最多 2 次比赛提交；本程序不会自动上传或访问比赛账号。

方法与来源见 [SOURCES.md](SOURCES.md)。


## 冻结特征分类头的快速对照

`scripts/run-head-ablation.py` 在 CPU 上复用首轮 Green 特征，20 分钟内完成 MLP 三折对照；初始种子有 AUROC 正增益才复验另一种子。年龄和特征标准化仅拟合训练折，官方 valid/test 不参与。MLP 的标准化均值、尺度作为模型 buffer 保存，推理使用同一处理。

```powershell
$python = (Get-Content runtime.local.json -Raw | ConvertFrom-Json).python
& $python -u scripts/run-head-ablation.py
```

当前两种子结果已完成，增益不稳定且 AUPRC 下降，维持线性基线；详见 `artifacts/head_ablation/decision.json`。该小型对照不替代图像主干微调或 RFM/CNN 比较。

## 训练读取提速

Green 392 的无损像素缓存约 23.7GB，覆盖全部 train，包括内部验证折。缓存保留原有裁边、缩放和数据增强；训练检测到匹配的数据清单后自动使用。构建采用有限并发和逐批完成标记，中断后只补齐未完成图片，并要求缓存完成后仍有至少 30GiB 空间。

```powershell
$python = (Get-Content runtime.local.json -Raw | ConvertFrom-Json).python
# 单独构建或继续缓存
& $python -u scripts/build-pixel-cache.py --workers 6
# 使用临时模型副本测速；会更新副本，不覆盖正式模型
& $python -u scripts/benchmark-training-io.py --count 512
```

缓存进度为 `artifacts/pixel_cache/train_392_917ca8a7f7.progress.json`。原先的 `artifacts/acceleration/training_io_benchmark.json` 保留为历史短基准；新的报告位于 `artifacts/acceleration/performance/report.json`。读取工作进程数会记录在后续检查点中。训练缓存不进入交付包，训练提速不能作为 T4 单张推理时延的证明。

`scripts/complete-acceleration.py` 是上一轮暂停会话使用的一次性恢复工具；后续使用下面的独立测速流程，不自动重启训练。

## 可复用性能配置

测速入口支持 `--config`、`--checkpoint`、`--fold` 和 `--stages train validation features`，Green、CNN、RFM 共用。检查点必须与配置和折号一致；只读取 train 及其内部划分，不访问官方 valid/test。训练测速保留每批 4 张、累积 8 批，包含真实优化器更新和临时保存。

```powershell
$python = (Get-Content runtime.local.json -Raw | ConvertFrom-Json).python
# 等现有队列自然结束；取得训练队列互斥后才开始最多 30 分钟的测速
& $python -u scripts/benchmark-training-io.py --wait-for-idle --max-minutes 30
# 后续训练显式启用；未传此参数时沿用原有执行方式
& $python run.py train --config configs/green_ft2.json --fold 0 --output artifacts/green_next/fold0 --workers 4 --performance-profile artifacts/acceleration/performance/profile.json
```

每项候选做三次交替配对测量，固定样本和初始状态，排除前 8 个启动批次另报吞吐；净耗时判断包括启动、数据传输、写出和缓存准备。推荐条件为中位吞吐至少增加 10%、预计净耗时减少、预测顺序一致、概率最大差及 AUROC/AUPRC 差均不超过 0.001；训练还核对更新次数和参数。报告是短样本外推，不代表完整 epoch 实测。

验证与特征提取分别尝试批量 4/8/16/32 和读取进程 0/2/4，显存不足只淘汰当前候选。未通过的阶段使用报告记录的基准配置。`--validation-batch-size`、`--validation-workers` 可覆盖训练中的内部验证设置；显式 `--workers` 优先于配置，并同时覆盖验证读取并发，除非另传 `--validation-workers`。`features` 支持相同 `--performance-profile` 及显式 `--batch-size`、`--workers`。性能配置绑定完整模型配置、硬件和软件版本，不能直接移到另一台电脑或另一份模型配置使用。

训练、内部验证、冻结特征提取共用完整缓存的签名、形状和类型检查。392 缓存可复用；其他尺寸不预建。如需评估并构建，加 `--build-cache`，先做小样本测量，再按预计剩余任务判断回本、30GiB 空间预留及会话剩余时间。可用 `--expected-passes` 指定后续遍历次数；默认训练/验证按配置轮数，特征提取按一次估计。未建好全量缓存时，不发布依赖该缓存的推荐。使用性能配置时不再触发旧的首轮慢读后自动建缓存逻辑。

报告另列测速自身的时间成本。优化能否抵偿这项一次性成本，应结合之后实际复用次数判断。GPU 测速未完成、未测模型或未达门槛时，不填写新增加速成绩。

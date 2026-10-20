# Kaggle 算力入口

本机已使用现有 Kaggle CLI 登录，项目账号保存在根目录 `kaggle.local.json`（不进入 Git）。代码、打包文件、结果与凭据分开管理。凭据继续由 Kaggle CLI 保管，不写入云端脚本或数据集。

本次已完成事项见 [接入验证记录](VALIDATION.md)。`resources` 和原始 `data` 均已上传并 ready，数据集隐私属性已核验；三个 CSV 及每个 split 的一张原图下载后 SHA256 与本机一致。本机无需重复上传，下面保留新机器从头准备的完整顺序。

当前默认任务为 **RFM P1 全图原生分辨率，第 1 折**，配置见 `kaggle/job.json`。Green 不再列入后续云端计划。云端训练不应用旧的累计 8 小时 RFM 限制；`max_hours` 只限制单次训练会话，此外仍受 Kaggle 账号配额和平台运行上限约束。

## 一次性上传

在项目根目录 PowerShell 执行。`resources` 上传 RFM 权重、许可证、来源及固定审计/分折；`data` 上传本地原始比赛压缩包，默认均为**私有数据集**。

```powershell
.\scripts\kaggle.ps1 check
.\scripts\kaggle.ps1 upload resources
.\scripts\kaggle.ps1 upload data
```

原始包约 **33.2 GB**，首次 `upload data` 会在 `artifacts/kaggle/datasets/data/` 创建一份副本并核对 SHA256，随后上传；需要额外约 33.2 GB 本地空间和相应上传时间。原始数据不改动。等待 Kaggle 完成数据集处理后再提交训练。重复运行会复用通过校验的本地副本；已经创建的数据集更新时加 `--version`。已上传的资源以账号数据集列表和本目录执行记录为准，不要对已有数据集重复执行 create。

若文件已传完、创建数据集时因认证失败，入口会验证完成记录并刷新 OAuth 后补交私有创建请求，不重复传输文件。也可单独执行：

```powershell
.\scripts\kaggle.ps1 finalize-upload data
```

该命令只处理元数据，不上传文件；已有数据集会直接返回当前状态。恢复会核对本地文件大小、修改时间、完成标记和上传记录有效期；缺失或不匹配时明确报错。`accepted_pending_processing` 表示创建请求被受理，仍需等平台变为 `ready`；处理期间状态/文件接口曾暂时返回 403。恢复记录保留在 `artifacts/kaggle/data_upload_receipt.json`。

仅准备资源或查看将提交的任务，不访问远程写入：

```powershell
.\scripts\kaggle.ps1 prepare-dataset resources
.\scripts\kaggle.ps1 prepare
```

代码以明确文件清单打包进单个 Kaggle 脚本，每次提交使用当前本机源码；不上传 `.env`、凭据、本机解释器路径、缓存和已有训练结果。无需每次更新代码都上传整个数据集。

## 提交和取回

```powershell
# 默认 job：RFM P1 fold1，单次训练上限 10 小时；可先编辑 kaggle/job.json
.\scripts\kaggle.ps1 push --name glaucoma-rfm-p1-fold1
.\scripts\kaggle.ps1 status --name glaucoma-rfm-p1-fold1
# 完成后按确定的版本下载
.\scripts\kaggle.ps1 pull --name glaucoma-rfm-p1-fold1 --version 1
```

每个新实验使用不同 `--name`。`push` 会立即提交并启动一次有界运行；它不会持续排队、跨天重启或自动提交比赛成绩。默认任务仅训练 train 内部固定折，不做 official valid/test 评测或全量重训。

输出进入 `artifacts/kaggle/downloads/<name>/<version>/`，包含 `kaggle_run.json`、`results.zip` 及 Kaggle 日志。结果压缩包保存模型、完整轮恢复点、训练曲线、holdout CSV/JSON 和环境记录，不包含原始数据、像素缓存及旧累计预算。下载不会合并覆盖本机正式结果，同名非空下载目录会被拒绝覆盖。

解压 `results.zip` 到独立目录后，先核对 `status.json`、`history.json`、`holdout.csv` 与配套 JSON。Kaggle 显示 COMPLETE 只表示脚本正常结束；训练是否全部完成以 `training_complete` 和训练目录 `status.json.complete` 为准。达到单次时限时可能只有部分完整轮；中途被平台硬终止也可能没有可下载的完整输出。

## 续跑与 P2 初始化

先复制任务文件，保持同一配置、折号、种子、输出路径和缓存开关，再改：

```json
{
  "resume": true,
  "resume_kernel": "meteorfate/glaucoma-rfm-p1-fold1/1"
}
```

这两项是对原任务 JSON 的修改，不是完整任务文件。用新 `--name` 提交即可；之前的 `results.zip` 会作为只读输入挂载，再恢复完整 epoch。缺少匹配的 `resume.pt` 会明确报错，不会悄悄从头训练。检查点原有的数据、配置、折号、种子校验仍生效。

P2 改用 `configs/rfm_p2_multiview.json`，由**相同 fold** 的 P1 `model.pt` 初始化；将 `resume_kernel` 指向该 P1 运行，并设置 `initialize_from` 为其结果包内模型路径。新 P2 的 `resume` 为 false、`output` 使用新目录。P2 后续续跑则指向上一轮 P2 kernel，保持初始化路径和配置不变。不要用其他 fold 的教师或初始化模型。

当前入口支持云端任务之间续跑；不会自动上传或接管本机已有 P1/P2 检查点。本机既有实验仍保留在原目录。

## 环境和验证

入口保留 Kaggle 自带匹配的 CUDA PyTorch/torchvision，按 `requirements-kaggle.txt` 安装其余依赖。每次运行记录实际 Python、PyTorch、timm、CUDA 和 GPU 型号；本机与云端版本不同，因此恢复兼容性需看实际运行证据，不能声称跨环境数值逐位等价。默认最多训练 10 小时，平台提交超时设为 11 小时，为依赖安装、数据核验和结果保存留出时间。

云端会校验 RFM 权重与审计资源哈希，逐张核验输入图片 SHA256，再复用本机固定分折。Kaggle 已展开的数据直接只读使用；仍为 ZIP 时检查磁盘余量后解压。默认关闭自动构建大像素缓存；有效批量仍为 `4 × 8 = 32`。当前使用单 GPU，T4 ×2 的第二张卡暂未启用，不引入分布式训练。

可随时运行不需要比赛数据的短时检查：

```powershell
.\scripts\kaggle.ps1 push --smoke --name glaucoma-rfm-smoke-new
.\scripts\kaggle.ps1 status --name glaucoma-rfm-smoke-new
.\scripts\kaggle.ps1 pull --name glaucoma-rfm-smoke-new --version 1
```

这只对随机初始化 RFM 做合成输入前向/反向，验证项目环境与 GPU；不加载比赛数据，不代表预训练模型训练成绩、完整训练成功或推理时延达标。

加 `--with-resources` 可进一步验证已上传的 RFM 权重加载、资源哈希及合成输入反向计算。

Kaggle 命令与元数据依据：[官方 kernel 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)、[元数据字段](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)、[dataset 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)。

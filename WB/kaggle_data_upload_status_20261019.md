# Kaggle 数据上传进度汇报（2026-10-19）

> 数据集：`meteorfate/glaucoma-competition-private-data`（私有）
> 上传对象：`glaucoma_first.zip`（原始比赛数据，33.24 GB ≈ 31.0 GiB）
> 数据来源：`kaggle/job.json` + `scripts/kaggle.py`（`prepare_data` → `datasets create`）

## 当前状态（14:14 采样）

| 指标 | 数值 |
|------|------|
| 上传进度 | **7.0%**（2.17 GiB / 31.0 GiB） |
| 当前速度 | ~3.8–4.1 MB/s（波动区间 2.0–4.3 MB/s） |
| 已用时 | 约 12 分 35 秒 |
| 预计剩余 | 约 2 小时 07 分（ETA ≈ **16:21** 完成网络上传） |
| 上传进程 | ✅ 存活（Kaggle CLI PID 77104，启动于 14:01:39） |
| 数据集状态 | ⏳ 创建中（尚未 ready，不能提交训练） |

进度条原文（实时日志末行）：
```
 7%|▋  | 2.17G/31.0G [12:34<2:11:25, 3.92MB/s]
```

## 上传管线三阶段

1. **本地副本**（已完成）：`scripts/kaggle.py` 把 `glaucoma_first.zip` 复制到
   `artifacts/kaggle/datasets/data/glaucoma_first.zip`，源文件与副本均为
   **33,243,465,372 字节**，逐字节一致，原始数据未改动。
2. **SHA256 校验**（已完成）：副本哈希 == 源哈希，校验通过。
3. **网络上传到 Kaggle**（进行中）：`kaggle datasets create` 分块上传，进度实时写入
   `artifacts/kaggle/data_upload.log`（tqdm 进度条）。**当前处于此阶段。**

## 后续步骤（上传完成后）

- Kaggle 还需**处理数据集**（解压 + 建索引），33 GB 通常额外耗时 10–30+ 分钟，之后数据集才 `ready`。
- 资源数据集 `meteorfate/glaucoma-project-resources` 此前已上传并 ready（见 `kaggle/README.md`、`kaggle/VALIDATION.md`）。
- 数据 ready 后方可提交训练：
  ```powershell
  .\scripts\kaggle.ps1 push --name glaucoma-rfm-p1-fold1
  ```

## 风险提示

- 上传速度偏慢（~4 MB/s），总耗时约 2.5 小时，期间**不要关闭 PowerShell / 断开网络**，否则需重传（脚本会复用已校验的本地副本，但已上传分块不续传）。
- `data_upload.log` 的文件 mtime 不随进度更新（Windows 下写入缓冲所致），但**日志内容持续前进**，以日志末行内容为准判断进度，不要看文件修改时间。
- 上传到 100% 不代表数据集 ready；是否可训练以 Kaggle 网页数据集状态为准。

## 持续跟进

已建立每 30 分钟自动采样日志的检查（见自动化任务），检查点追加到
`WB/kaggle_upload_tracker.md`。手动复查命令：
```bash
tail -n 2 "E:/全国高校计算机能力挑战赛-大数据挑战赛/artifacts/kaggle/data_upload.log"
```

# 接入验证记录

日期：2026-10-19（北京时间）。

## 19:34 当前状态：数据集 ready，恢复已验证

- 原始数据集已在本人账号列表出现，状态接口返回 **ready**，服务端元数据 `isPrivate=true`，文件列表正常可读。
- 下载并核对 `train.csv`、`valid.csv`、`test.csv` 及 train/valid/test 各一张原始图片，共 6 个文件，SHA256 全部与本机一致。证据见 `artifacts/kaggle/data_upload_verification.json`；这是抽样一致性核验，尚未逐张回下载全量原图。
- 已验证项目新增 `finalize-upload data` 入口：对已存在的数据集返回 ready，重传文件字节数为 0。未来首次创建在文件完成后失败时，可校验完成记录、刷新认证并补交私有创建请求，无需重传整个包；版本更新失败不会误走新数据集创建流程。
- 本轮 Kaggle 相关 **13 项测试通过**，覆盖完整/未完整上传、缓存过期、文件变更、失败恢复、原有打包/目录与结果隔离。
- 第一次失败的确切位置是传输完成后的认证/创建调用；此前补建后的 403 已消失。原日志没有保留当时令牌的过期时间，因此不将其唯一归因为某个 SDK 缺陷。
- 本机已确认存在 [Kaggle SDK 过期令牌判断缺陷](https://github.com/Kaggle/kaggle-sdk-python/issues/74)：过期 10 分钟的模拟凭据仍被判为未过期。恢复入口通过官方 SDK 显式刷新认证规避该条件，没有修改全局 SDK 文件、代理或账号权限。

以下为故障恢复过程与较早验证快照，以本节当前状态为准。

## 18:53 原始数据上传恢复

- `glaucoma_first.zip` 共 33,243,465,372 字节，本地副本与源文件 SHA256 比对通过；传输进度达到 100%，耗时约 2 小时 59 分。
- 原上传进程在文件传完后创建数据集时出现认证错误，退出码为 1，不能据上传进度认定数据集已可用。
- 18:53 使用新认证会话复用已完成上传的文件记录，补建私有数据集；Kaggle 返回 `status: Ok` 与数据集链接，没有重传 33.2 GB 文件。恢复回执见 `artifacts/kaggle/data_upload_recovery.json`。
- 数据集为 `meteorfate/glaucoma-competition-private-data`。补建后的状态及文件查询仍返回 HTTP 403，账号数据集列表尚未出现该条目；因此平台是否已完成处理仍待确认，尚未验证 ready，不启动正式训练。

下列“尚未上传”等描述是本次恢复之前的接入验证快照，以本节为准。

- 已使用本机原有 Kaggle CLI 2.2.4 / OAuth 登录；账号为 `meteorfate`，未更改凭据或代理配置。
- 本地完整测试 **80 项通过**，日志在 `artifacts/kaggle/local_tests.log`；Kaggle 相关 9 项随后再次通过。
- [基础 GPU 检查](https://www.kaggle.com/code/meteorfate/glaucoma-rfm-smoke-20261019) 已 COMPLETE。云端识别到 2 张 Tesla T4，随机初始化 RFM 完成合成输入前向和反向，脚本内耗时约 26 秒。
- 实测环境：Python 3.13.15、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、timm 1.0.24；当前入口使用单 GPU。
- [RFM 资源数据集](https://www.kaggle.com/datasets/meteorfate/glaucoma-project-resources) 已创建为私有，平台返回 `ready`。上传的资源包约 309 MB；Kaggle 会将 ZIP 展开，入口与测试已兼容该布局。
- [带预训练权重的检查](https://www.kaggle.com/code/meteorfate/glaucoma-rfm-pretrained-smoke-20261019) 已提交。北京时间 13:50 查询仍为 RUNNING、尚无输出；实时日志接口返回 HTTP 500。该项尚未验证成功，不能据此宣称云端预训练权重加载通过。提交时设置了 600 秒执行超时，平台启动/调度等待不一定计入该时间。
- 本机已配置默认 RFM P1 fold1 任务、T4 提交、完整轮续跑、P2 同折初始化和独立结果下载。历史累计 RFM 预算不进入云端资源包，也不限制该入口。

尚未执行的部分：约 33.2 GB 原始比赛压缩包的一次性上传、正式训练、完整训练结果汇入及真实跨会话检查点续跑。因此本记录仅确认接入与实现检查，不声明新的模型成绩或完整训练成功。

基础检查取回文件位于 `artifacts/kaggle/downloads/glaucoma-rfm-smoke-20261019/1/`，包含 `kaggle_run.json`、`results.zip` 与日志。原始数据上传完成后才能启动默认正式任务。

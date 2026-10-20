# Kaggle 数据上传检查点（自动追加）

格式：`- 时间 进度% (已传/31.0G) 速度 ETA 进程`

- 14:14 7.0% (2.17G/31.0G) 3.92MB/s ETA≈16:21 alive
- 15:15 41.0% (12.8G/31.0G) 3.29MB/s ETA≈16:46 alive
- 15:25 47.0% (14.5G/31.0G) 3.62MB/s ETA≈16:47 alive
- 15:42 56.0% (17.3G/31.0G) 3.70MB/s ETA≈16:48 alive
- 16:06 69.0% (21.4G/31.0G) 3.90MB/s ETA≈16:50 alive
- 16:47 92.0% (28.6G/31.0G) 2.81MB/s ETA≈17:03 alive  ⚠️ 16:50 不能关，还差约8%，预计17:03完成
- 15:26 47.0% (14.5G/31.0G) 3.30MB/s ETA≈16:55 alive
- 16:29 进度82% (25.4G/31.0G) 速度3.52MB/s ETA≈0:28 进程=alive
- 16:48 关机判定: 可关 (进度93%) — ⚠️ 注意: kaggle 进程已退出但进度仅 93%, 上传疑似异常中断未完成, 关机不会进一步破坏, 但数据集大概率未成功上传, 需人工检查/重传
- 16:56 97.0% (30.1G/31.0G) 2.94MB/s ETA≈17:02 alive
- 16:57 订正: 16:48 判定"进程已退出/可关"为误报——自动化沙箱枚举不到宿主机进程,实际 kaggle 进程(PID 77104,14:01起)一直存活,上传未中断。请勿依该条结论关机;等日志到100%或进程真正退出(≈17:02)再关。
- 17:06 ⛔ 字节上传到100%(31.0G/31.0G)但 `kaggle datasets create` 收尾失败 exit1, 报 "Authentication required to call the Kaggle API"。根本原因: OAuth 凭据 scope 仅 `resources.admin:*`(账号 meteorfate), 缺 data 数据集创建权限; 非 token 过期(有效期到 21:06 UTC)。数据数据集 `meteorfate/glaucoma-competition-private-data` 未创建(列表仅见 resources)。修复: 换全权限登录后重传(约3h)。

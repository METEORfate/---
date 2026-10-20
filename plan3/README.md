# 方案3：准备、标注验收与 Kaggle 执行

当前执行约束（2026-10-20）：仅完成本机准备核验，正式训练与自动续跑均已停止，须等待用户新的训练命令。`scripts/prepare-plan3-local.py` 只在CPU上核验准备结果并退出；下文包含的训练、上传及评测命令为后续操作说明，不代表已授权自动执行。当前状态和证据位置见项目根目录 `STATUS.md`。

本入口独立于旧 P1/P2。`glaucoma/*.py`、旧配置及旧 Kaggle 作业未修改，旧原图续训指纹保持不变。默认完整路线要求1000个有效框；首批标注试跑使用下述显式授权的pilot400策略。下面的push、上传及评测命令仅在相应前置条件满足后使用。

### 2026-10-20：首批标注试跑

用户已授权用首批约400张标注开始Kaggle训练。实际目录有401个LabelMe JSON：353个明确框、45个边界不清、3个无法定位。新增`pilot400`策略要求至少400张标注、300个明确框，并保留原有train-only、两类别及三折覆盖检查；默认`full1000`门槛保持不变。试跑结果不能写成1000框完整方案已验收。

```powershell
& $runtime.python -m plan3 import-labelme --source glaucoma_first/400ManualLabel --output artifacts/plan3_inputs/annotations_pilot400_20261020 --policy pilot400
.\scripts\plan3.ps1 prepare-resources --annotations artifacts/plan3_inputs/annotations_pilot400_20261020 --annotation-policy pilot400
```

转换器只输出图像键、原图尺寸、视盘坐标与可定位状态；剔除`imagePath/imageData/description`及其他元数据。标记为不清或不可定位的矩形不作为有效框训练。任务文件也必须显式声明`annotation_policy=pilot400`，单靠小样本import预览不能绕开默认门槛。本批任务模板为`kaggle/plan3/pilot400-*.json`；局部/交互仍需定位器内部检查通过，不自动开放official-valid。

## 1. 本机准备检查

训练依赖使用 `runtime.local.json` 的解释器；Kaggle打包使用已登录CLI所在的系统Python。

```powershell
$runtime = Get-Content runtime.local.json -Raw | ConvertFrom-Json
& $runtime.python -m plan3 readiness
& $runtime.python -m plan3 synthetic-check
& $runtime.python -W ignore::DeprecationWarning -m unittest discover -s tests -v
.\scripts\plan3.ps1 prepare --job kaggle/plan3/train-fold0.json --name glaucoma-plan3-prepared-fold0
```

`prepare`只生成本地脚本与元数据，不访问远端写入，不启动GPU。`synthetic-check`仅使用合成像素检查真实RFM/EfficientNet预训练适配、前后向和重载；64/80像素检查不证明448/1024容量或训练成绩。`capacity`任务只有标注验收后才开放，覆盖真实训练图片、AdamW更新及EMA副本，报告吞吐和峰值显存。

## 2. 接收同事视盘框

约定CSV列为：

```text
Image_name,width,height,x_min,y_min,x_max,y_max,status
image.jpg,2000,1500,650,450,850,650,valid
unclear.jpg,2000,1500,,,,,unclear
unlocatable.jpg,2000,1500,,,,,unlocatable
```

- `Image_name`必须与官方train.csv原拼写相同，仅接收train图片。
- 坐标为**EXIF转正后、裁黑边前原图**的像素xyxy，左上包含、右下不包含。必须由标注说明确认，不能靠尺寸猜方向。
- `valid`必须有合法非零框；另两种状态必须留空坐标。不得以零框冒充可用视盘。
- CSV字段/状态可通过`configs/plan3/annotation_spec.json`显式映射；COCO可使用`format=coco`和`disc_category_id`，类别以外的空标注需通过CSV明确状态。不自动猜YOLO类别或坐标空间。

```powershell
& $runtime.python -m plan3 import-annotations --source 同事导出的框.csv --spec configs/plan3/annotation_spec.json --output artifacts/plan3_inputs/annotations
& $runtime.python -m plan3 readiness
```

默认至少1000个有效框，覆盖三折和两个类别；报告类别数量、全部图片SHA256和固定数据/分折指纹。300阳性/700阴性是采样建议，不是通过数量检查的替代品。`--minimum`可以生成小样本接收预览，但少于1000的预览无法通过正式任务门槛。重复导入使用新的版本目录，禁止覆盖已验收版本。

定位器只使用当前训练部分的框，并在其中按重复图分组留出20%检查/选轮。`review_template.json`中的每张图片均需检查，把`missed_disc:null`填写为true/false。超过5%明显漏盘时局部分类及SSL任务拒绝启动。检查框、训练框的并集也必须排除当前holdout，嵌套实验必须重新训练自己的定位器。

## 3. 准备新的私有资源数据集

复用已经ready的原始数据集；为方案3创建独立的`meteorfate/glaucoma-plan3-resources`，保留旧resources数据集供P1/P2使用。

```powershell
.\scripts\plan3.ps1 prepare-resources
# 标注就绪后首次创建；Kaggle默认创建私有数据集，切勿添加 --public。
kaggle datasets create -p artifacts/kaggle_plan3/resources --keep-tabular
```

资源ZIP包含固定审计、已验收标注、可用预训练权重及来源清单。初始RFM和EfficientNet权重已有；DINOv3、RETFound-DINOv2、ConvNeXt缺权重时保持blocked，不以随机初始化替代。

新增骨干权重需要`weights/<model>_source.json`，字段包括`filename,sha256,source,license,architecture,mean,std`。仅接受本地普通state_dict，不执行远程Python代码，不上传访问令牌。架构固定为：

| model | timm architecture |
|---|---|
| dinov3 | vit_large_patch16_dinov3.lvd1689m |
| retfound_dinov2 | vit_large_patch14_dinov2 |
| convnext | convnext_large.fb_in22k_ft_in1k |

RETFound-DINOv2应使用官方CFP权重及其配套归一化；不能用通用DINOv2权重改名。DINOv3和RETFound的访问许可需要合法取得；未获权重不宣称这些挑战模型完成。来源：[DINOv3官方仓库](https://github.com/facebookresearch/dinov3)、[RETFound官方仓库](https://github.com/rmaphoh/RETFound)。

定位器检查JSON、融合spec等额外输入必须显式加入新资源快照：

```powershell
.\scripts\plan3.ps1 prepare-resources --extra artifacts/plan3_inputs/review-fold0.json --extra artifacts/plan3_inputs/fusion.json
kaggle datasets version -p artifacts/kaggle_plan3/resources -m "Add reviewed locator inputs" --keep-tabular
```

更新后等待ready。作业绑定资源清单哈希、框哈希和标注版本；挂载旧快照会失败。prepare/push应重新生成与本次快照匹配的作业，不复用旧的生成脚本。

## 4. 运行、取回与续跑

先复制对应模板修改实验名称、配置、fold、seed及输出路径；除续跑外，每个实验使用独立输出。模板里的`deps/...`是待绑定前置结果，不能直接push。

```powershell
.\scripts\plan3.ps1 push --job kaggle/plan3/capacity-fold0.json --name glaucoma-plan3-capacity-f0
.\scripts\plan3.ps1 status --name glaucoma-plan3-capacity-f0 --version 1
.\scripts\plan3.ps1 pull --name glaucoma-plan3-capacity-f0 --version 1
```

push先检查标注门槛、所需资源/前置文件、新私有数据集ready以及剩余配额。单任务`max_hours<=10`，GPU配额必须覆盖该上限加1小时平台余量；额度不足时缩短本次时限或等待刷新。不会购买资源、无限排队或自动提交赛事网站。

结果保存在`artifacts/kaggle_plan3/downloads/<name>/<version>`。`COMPLETE`不代表训练已完成，必须核对`training_complete`、输出`status.json.complete`。只有完整轮才进入恢复点；保存模型、优化器、scheduler、scaler、EMA、中心、随机状态、最佳轮和历史。第一轮前也保存初始恢复点。训练、验证和定位预测留出结果发布时间；平台硬终止仍可能来不及发布文件。

用已下载的`results.zip`生成固定版本依赖：

```powershell
.\scripts\plan3.ps1 dependency --archive artifacts/kaggle_plan3/downloads/某实验/1/results.zip --kernel meteorfate/某实验/1 --alias previous --output artifacts/plan3_inputs/previous-dependency.json
```

把生成对象加入下一作业的`dependencies`数组。引用必须带版本，绑定ZIP及所需文件SHA256。云端解压到`deps/<alias>/`，不同作业不会相互覆盖；默认输出只发布results.zip和kaggle_run.json，不发布整个原始数据或预训练权重工作目录。

续跑使用`kaggle/plan3/resume.json`结构，`resume_from`指向前次结果包内的**完整输出目录**。保持同一配置、输出路径、折、种子、初始化来源和输入指纹。仅增加会话时长/更换依赖别名不改变训练配方。缺恢复点、来源变化或环境版本变化时明确拒绝。先验证一次真实跨Kaggle作业恢复，再安排长实验；本地合成恢复测试不能替代这一步。

## 5. 实验顺序与配置

| 阶段 | 配置/任务 | 约束 |
|---|---|---|
| 基线 | rfm448、effnet512 | fold0筛选，保留者补完整三折 |
| 分辨率 | rfm672、effnet768、effnet1024 | 升分辨率从同折同种子的充分训练模型初始化；672/1024阶段最多6轮 |
| 定位与局部 | locator、local3、local5 | 原图裁图，禁止从缩小缓存放大；失败局部带mask并回退全图 |
| 交互 | probability、concat、cross | 相同输入/编码器比较；concat/cross从训练好的全图模型初始化，交互头预热2轮 |
| 适配 | ssl10、ssl30 → 原监督配置 | 同折重新监督训练，与不适配版本匹配比较 |
| 排序 | ranking0、ranking01、ranking03 | 从稳定BCE初始化；三版本使用相同显式正负采样和训练内难例 |
| 挑战模型 | dinov3、retfound_dinov2、convnext | 来源、容量和同口径OOF核验后按增益/互补性筛选 |
| 确认 | holdout、collect、compare | TTA写入配置，全部三折重算OOF；小于0.002改善增加2026种子确认 |

所有基础配置有效批量32，微批量与累积可按显存成对调整；不把双T4当作一块合并显存。默认单卡。

SSL入口实现**图像级DINO式教师—学生继续适配**，带教师EMA、温度锐化和中心化。全图两次增强互相学习；局部仅对应局部的一致性，不强迫任意局部等价于全图。它是独立低依赖对照，**不是完整官方DINOv3预训练流程的复现**；没有实现iBOT/Gram锚定，不把SSL loss下降当作分类收益。中心和教师均随完整轮恢复。

排序按物理microbatch显式包含阳性、随机阴性和困难阴性；梯度累积本身不创造跨batch配对。困难阴性从当前训练部分的inner-OOF生成，绑定来源预测及排除范围。full重训可使用全train的完整OOF。不能用外fold holdout反向选难例。

内层模型设置`fold=<外折>, inner_fold=0/1/2`，三个输出通过collect的`outer_fold=<外折>`聚合。它们的监督初始化、SSL及定位器必须同样排除内层holdout。嵌套排名模型还需更内层的难例来源；当前入口会拒绝不足的来源证明，不允许直接复用外层困难名单。需要嵌套逻辑回归时，先选有完整嵌套来源的专家；不把未经支持的排名分支嵌套成绩报为完成。

`compare`使用完整OOF的逐折分数及图像级配对区间；患者身份未验证，不把重复图分组称为真实患者划分。

## 6. OOF、融合、全量重训与评测

- `collect`需要三个已完成输出，核对模型/预测哈希、逐图标签、折、配置、种子、标注及共同权重来源；拒绝不完整、重复或跨版本OOF。
- `fuse`的spec包含`fusion`（probability/logit/logistic）和`experts`。每个expert包含`oof`路径和`models`列表：candidate使用三个折模型，delivery使用一个full模型。模型数不限于两个，逐专家继承OOF验证过的TTA策略。
- 逻辑回归还需每个expert的`inner_oof`字典（键为"0","1","2"），逐外折在内层OOF拟合，再评价该外折；最后在全OOF拟合部署系数。拟合分数和嵌套分数分别记录。
- 融合结果复制成员到同目录的members子目录，形成可整体搬运的自包含模型包。
- `finalize`必须提供三个同配方、同种子的完整CV模型，轮数取最佳轮中位数。保存不可变选择配方和实际重训轮数，raw/EMA按CV多数策略冻结并写入恢复签名。
- 高分辨率/排序等多阶段full重训，须按相同顺序先得到full前阶段模型；不能用某个CV折模型替代full监督初始化。

融合spec的最小结构：

```json
{
  "fusion": "probability",
  "experts": [
    {"oof": "deps/oof/artifacts/plan3/collect/oof.csv",
     "models": ["deps/cv0/artifacts/plan3/rfm448-fold0/model.pt",
                "deps/cv1/artifacts/plan3/rfm448-fold1/model.pt",
                "deps/cv2/artifacts/plan3/rfm448-fold2/model.pt"]}
  ]
}
```

official-valid只通过中央预留票据执行：

```powershell
& $runtime.python -m plan3 reserve-evaluation --artifact artifacts/plan3/candidate/ensemble.json --milestone candidate --output artifacts/plan3_inputs/candidate-ticket.json
# 将ticket路径填入evaluate作业；push会嵌入票据并核对中央台账。
# 下载并解压评测结果后合并到中央台账：
& $runtime.python -m plan3 accept-evaluation --result 已解压目录/valid_report.json
```

旧baseline保留。candidate/delivery一旦预留，只能重试同一冻结产物；失败不释放给其他模型。candidate完成并合并后才预留delivery。valid/test不能拟合输入处理、定位器、SSL、难例、融合权重或阈值。

最终predict任务生成UTF-8 submission.csv，保持5855张test原顺序和`Image_name,Probability,Pred_Label`三列，概率有限且在[0,1]，阈值默认0.5。测试集无标签，不计算所谓test AUROC；不自动进行赛事网站提交。

## 7. 完成判据与当前限制

代码准备完成不等于方案3训练完成。交付时必须逐项保存各核心机制的匹配实验、淘汰结论、完整OOF、candidate/delivery报告、最终模型及CSV；缺权重、缺标注或未完成阶段必须继续标记待完成。

当前尚无方案3训练成绩，也尚未验证多依赖挂载及跨Kaggle会话恢复；这些必须在标注验收后用实际短作业确认。ConvNeXt/ViT-L适配在获得对应权重后还需实测，不能以模型名可创建代替加载成功。

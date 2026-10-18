# 来源与实现记录

- 赛规：https://www.ncccu.org.cn/index/Paper/case2.html?way=1
- RETFound-Green 官方加载、权重与许可证：https://github.com/justinengelmann/RETFound_Green
- Green 论文：https://www.nature.com/articles/s41467-025-62123-z
- EfficientNetV2-S 权重、归一化与模型卡：https://huggingface.co/timm/tf_efficientnetv2_s.in1k
- RFM 模型与权重：https://github.com/ok1zjf/RFM
- RFM 第9层读出及 RoPE 适配参考：https://github.com/ok1zjf/RFM/blob/main/model.py
- RFM 代码 MIT；其权重另受 DINOv3 License 约束：https://github.com/ok1zjf/RFM/blob/main/LICENSE-DINOv3.md
- PyTorch CUDA 安装索引：https://download.pytorch.org/whl/cu128/

本实现使用 timm 已安装架构和安全的 tensor state dict 加载，不自动执行下载的仓库代码。
RFM 的 RoPE buffer-device 适配按官方 MIT 实现改写，以下保留其授权声明。

MIT License

Copyright (c) 2026 Jiri Fajtl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

论文中的成绩不作为本比赛的成绩。数据审计、实训成绩及计时均分别记录。

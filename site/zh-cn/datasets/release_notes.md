# 版本说明

## Nightly 版本

### 特点

- 更好的分片、打乱顺序和子拆分
- 现在可以将任意元数据添加到 `tfds.core.DatasetInfo` <br> 中，它将随数据集进行存储/恢复。请参见 `tfds.core.Metadata`。
- 更好的代理支持，支持添加证书
- 添加 `decoders` 关键字参数以重写默认特征解码（[指南](https://github.com/tensorflow/datasets/tree/master/docs/decode.md)）。
- 从 [MimickNet 论文](https://arxiv.org/abs/1908.05782)添加 `duke_ultrasound` 超声体模和活体肝脏图像数据集
- 从 [VTAB 基准](https://arxiv.org/abs/1910.04867)添加 Dmlab 数据集。
- 从 [e-SNLI](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf) 论文添加 e-SNLI 数据集。
- 添加 [Opinosis 数据集](https://www.aclweb.org/anthology/C10-1039.pdf)。
- 添加[此处](https://arxiv.org/pdf/1711.00350.pdf)介绍的 SCAN 数据集。
- 添加 [Imagewang](https://github.com/fastai/imagenette) 数据集。
- 从 [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) 论文添加 DIV2K 数据集。
- 从[本论文](https://openreview.net/pdf?id=SygcCnNKwr)添加 CFQ（组合 Freebase 问题）数据集。

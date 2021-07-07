**更新日期：2020 年 8 月 7 日**

## 量化

- 适用于动态范围内核的训练后量化 - [已发布](https://blog.tensorflow.org/2018/09/introducing-model-optimization-toolkit.html)
- 适用于 (8b) 定点内核的训练后量化 - [已发布](https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html)
- 适用于 (8b) 定点内核和 &lt;8b 实验的量化感知训练 - [已发布](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)
- [WIP] 适用于 (8b) 定点 RNN 的训练后量化
- 适用于 (8b) 定点 RNN 的量化感知训练
- [WIP] 训练后动态范围量化的质量和性能改进

## 剪枝/稀疏度

- 训练中基于量级的权重剪枝 - [已发布](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)
- TensorFlow Lite 中的稀疏模型执行支持 - [WIP](https://github.com/tensorflow/model-optimization/issues/173)

## 权重聚类

- 训练中权重聚类 - [已发布](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

## 级联压缩技术

- [WIP] 对结合不同压缩技术的其他支持。目前，用户只能将一种训练中技术与训练后量化相结合。此提案即将推出。

## 压缩

- [WIP] 张量压缩 API

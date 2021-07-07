# 开始使用 TensorFlow 模型优化

## 1. 为任务选择最佳模型

您需要根据任务在模型复杂性和大小之间进行权衡。如果您的任务需要高准确率，那么您可能需要一个大而复杂的模型。对于准确率要求较低的任务，最好使用较小的模型，因为它们不仅占用更少的磁盘空间和内存，而且通常速度更快且更节能。

## 2. 预优化的模型

查看现有的 [TensorFlow Lite 预优化模型](https://www.tensorflow.org/lite/models)能否提供您的应用所需的效率。

## 3. 训练后工具

如果无法在您的应用中使用预训练模型，请尝试在 [TensorFlow Lite 转换](https://www.tensorflow.org/lite/convert)过程中使用 [TensorFlow Lite 训练后量化工具](./quantization/post_training)，这些工具能够对已训练的 TensorFlow 模型进行优化。

请参阅[训练后量化教程](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb)了解更多信息。

## 后续步骤：训练时工具

如果上述简单解决方案无法满足您的需求，您可能需要采用训练时优化技术。使用我们的训练时工具进行[进一步优化](optimize_further.md)和深入挖掘。

<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# 导出 SavedModel

本页介绍将模型从 TensorFlow 程序导出（保存）到 [TensorFlow 2 的 SavedModel 格式](https://www.tensorflow.org/guide/saved_model)的详细信息。此格式是在 TensorFlow Hub 上共享预训练模型和模型部分的推荐方式。它取代了旧的 [TF1 Hub 格式](tf1_hub_module.md)，并提供了一组新 API。您可以参阅 [TF1 Hub 格式导出](exporting_hub_format.md)，获取有关导出 TF1 Hub 格式模型的更多信息。

某些模型构建工具包已经提供执行此操作的工具（请参阅下文的 [TensorFlow Model Garden](#tensorflow-model-garden)）。

## 概述

SavedModel 是 TensorFlow 用于已训练模型或模型部分的标准序列化格式。它存储了模型的训练权重以及用于执行其计算的确切 TensorFlow 运算。SavedModel 可独立于创建它的代码单独使用。特别值得一提的是，它可以在不同的高级建模 API（例如 Keras）之间重用，因为 TensorFlow 运算是它们的通用基本语言。

## 从 Keras 保存

自 TensorFlow 2 起，`tf.keras.Model.save()` 和 `tf.keras.models.save_model()` 默认为 SavedModel 格式（非 HDF5）。生成的 SavedModel 可与 `hub.load()`、`hub.KerasLayer` 和其他待提供的高级 API 的类似适配器组合使用。

要共享完整的 Keras 模型，只需在保存它时设置 `include_optimizer=False`。

要共享一部分 Keras 模型，请将该部分制作成模型，然后保存。您可以从头编写模型代码…

```python
piece_to_share = tf.keras.Model(...)
full_model = tf.keras.Sequential([piece_to_share, ...])
full_model.fit(...)
piece_to_share.save(...)
```

…也可以在事后提取出要共享的部分（如果它与完整模型的层次一致）：

```python
full_model = tf.keras.Model(...)
sharing_input = full_model.get_layer(...).get_output_at(0)
sharing_output = full_model.get_layer(...).get_output_at(0)
piece_to_share = tf.keras.Model(sharing_input, sharing_output)
piece_to_share.save(..., include_optimizer=False)
```

GitHub 上的 [TensorFlow 模型](https://github.com/tensorflow/models)为 BERT 使用前一种方式（请参阅 [nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py)，请注意用于导出的 `core_model` 和用于恢复检查点的 `pretrainer` 间的拆分），并为 ResNet 使用后一种方式（请参阅 [vision/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/tfhub_export.py)）。

## 从低级 TensorFlow 保存

此操作需要充分熟悉 TensorFlow 的 [SavedModel 指南](https://www.tensorflow.org/guide/saved_model)。

如果您希望不只是提供应用签名，则应实现[可重用 SavedModel 接口](reusable_saved_models.md)。从概念上讲，如下所示：

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super().__init__()
    self.v = tf.Variable(v_init)
    self.variables = [self.v]
    self.trainable_variables = [self.v]
    self.regularization_losses = [
        tf.function(input_signature=[])(lambda: 0.001 * self.v**2),
    ]

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    return tf.multiply(inputs, self.v)

tf.saved_model.save(MyMulModel(2.0), "/tmp/my_mul")

layer = hub.KerasLayer("/tmp/my_mul")
print(layer([10., 20.]))  # [20., 40.]
layer.trainable = True
print(layer.trainable_weights)  # [2.]
print(layer.losses)  # 0.004
```

## 针对 SavedModel 创建者的建议

创建要在 TensorFlow Hub 上共享的 SavedModel 时，请提前考虑使用者是否以及应如何微调模型，并在文档中提供指导。

从 Keras 模型保存 SavedModel 时，应使微调的所有机制全部生效（保存权重正则化损失、声明可训练变量、在 `training=True` 和 `training=False` 情况下跟踪 `__call__` 等）。

选择能够适应梯度流（例如，输出 logits 而非 softmax 概率或 top-k 预测）的模型接口。

如果模型使用随机失活、批次归一化或涉及超参数的类似训练技术，请将超参数设置为对多种预期目标问题和批次大小均有意义的值。（截至撰写本文时，从 Keras 保存 SavedModel 尚不便于使用者进行调整。）

各个层上的权重正则化器（及其正则化强度系数）将得到保存，但优化器内部的权重正则化（例如 `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`）将丢失。请向您的 SavedModel 的使用者提出相应建议。

<a name="tensorflow-model-garden"></a>

## TensorFlow Model Garden

[TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official) 仓库中包含许多创建可重用的 TF2 SavedModel 以供上传到 [tfhub.dev](https://tfhub.dev/) 的示例。

## 社区请求

在 tfhub.dev 上的可用资产中，只有一小部分由 TensorFlow Hub 团队生成。我们主要依赖于 Google 与 Deepmind 的研究员、企业与学术研究机构以及机器学习爱好者创建模型。因此，我们无法保证能够满足社区对特定资产的请求，也无法对新资产的可用性提供时间预估。

[Community Model Requests 里程碑](https://github.com/tensorflow/hub/milestone/1)包含社区针对特定资产的请求。如果您或您认识的开发者有兴趣制作相关资产并在 tfhub.dev 上分享，我们热烈欢迎广大开发者踊跃提交！

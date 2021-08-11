<!--* freshness: { owner: 'maringeo' reviewed: '2021-04-12' review_interval: '6 months' } *-->

# 以 TF1 Hub 格式导出模型

您可以在 [TF1 Hub 格式](tf1_hub_module.md)中详细了解此格式。

## 兼容性说明

TF1 Hub 格式适用于 TensorFlow 1。在 TensorFlow 2 中，TF Hub 仅部分支持该格式。因此，请务必考虑使用新的 [TF2 SavedModel](exporting_tf2_saved_model) 格式进行发布，具体方法请参阅[导出模型](tf2_saved_model.md)指南。

在语法层面上，TF1 Hub 格式与 TensorFlow 2 的 SavedModel 格式类似（文件名和协议消息相同），但在针对模块重用、构成和重新训练的语义上有所不同（例如，资源初始值设定项的存储方式不同，元图的标记惯例不同）。最简单的区分方式是查看磁盘上是否存在 `tfhub_module.pb` 文件。

## 常规方式

要定义新模块，发布者应使用 `module_fn` 函数调用 `hub.create_module_spec()`。此函数可以构造一个表示模块内部结构的计算图，对将由调用者提供的输入使用 `tf.placeholder()`。然后，它通过调用一次或多次 `hub.add_signature(name, inputs, outputs)` 来定义签名。

例如：

```python
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.dense(inputs, 200)
  layer2 = tf.layers.dense(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)
```

可以使用 `hub.create_module_spec()` 的结果替代路径来实例化特定 TensorFlow 计算图内的模块对象。在这种情况下，将没有检查点，模块实例将改用变量初始值设定项。

任何模块实例都可以通过其 `export(path, session)` 方法序列化存储到磁盘上。导出模块会将其定义与其 `session` 内变量的当前状态一起序列化存储到传递的路径中。首次导出模块以及导出微调模块时可以使用这种方式。

为了与 TensorFlow Estimator 兼容，`hub.LatestModuleExporter` 会从最新的检查点导出模块，与 `tf.estimator.LatestExporter` 从最新的检查点导出整个模块类似。

模块发布者应尽可能实现[通用签名](common_signatures/index.md)，以便使用者可以方便地更换模块并找到对解决其问题最有效的模块。

## 真实示例

请查看我们的[文本嵌入向量模块导出程序](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)中提供的真实示例，了解如何通过通用文本嵌入向量格式创建模块。

## 针对发布者的建议

为了便于使用者进行微调，请注意以下几点：

- 微调需要正则化。您的模块将随 `REGULARIZATION_LOSSES` 集合一起导出，该集合包含您选择的 `tf.layers.dense(..., kernel_regularizer=...)` 等项目，使用者将通过 `tf.losses.get_regularization_losses()` 获得这些项目。最好采用这种方式来定义 L1/L2 正则化损失。

- 在发布者模型中，请避免通过 `tf.train.FtrlOptimizer`、`tf.train.ProximalGradientDescentOptimizer` 和其他近端优化器的 `l1_` 和 `l2_regularization_strength` 参数定义 L1/L2 正则化。这些参数不会随模块一起导出，并且全局设置正则化强度可能不适合于使用者。除了宽度模型（即稀疏线性模型）或宽度和深度模型中的 L1 正则化之外，还应可以改用单独的正则化损失。

- 如果使用随机失活、批次归一化或类似训练技术，请将其超参数设置为对多种预期用途均有意义的值。随机失活率可能需要根据目标问题的过拟合倾向进行调整。在批次归一化中，动量（也称为衰减系数）应足够小，以支持使用小型数据集和/或大批次进行微调。对于高级使用者，请考虑添加签名以公开对关键超参数的控制。

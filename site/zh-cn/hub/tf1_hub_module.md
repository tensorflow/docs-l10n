# TF1 Hub 格式

TensorFlow Hub 在 2018 年发布时提供了一种资源类型：可导入TensorFlow 1 程序的 TF1 Hub 格式。

本页将介绍如何在 TF1（或 TF2 的 TF1 兼容模式）中使用 TF1 Hub 格式以及 `hub.Module` 类和关联的 API。（典型的用途是构建 `tf.Graph`（可能在 TF1 `Estimator` 中），方法是将一个或多个 TF1 Hub 格式的模型与 `tf.compat.layers` 或 `tf.layers` 进行组合）。

TensorFlow 2（非 TF1 兼容模式）的用户必须使用[新版 API 和 `hub.load()` 或 `hub.KerasLayer`](tf2_saved_model.md)。新版 API 可以加载新的 TF2 SavedModel 资源类型，但对[将 TF1 Hub 格式加载到 TF2](migration_tf2.md) 的支持则有限。

## 使用 TF1 Hub 格式的模型

### 实例化 TF1 Hub 格式的模型

将 TF1 Hub 格式模型导入 TensorFlow 程序的方法是，通过包含模型网址或文件系统路径的字符串创建 `hub.Module` 对象，例如：

```python
m = hub.Module("path/to/a/module_dir")
```

这会将模块的变量添加到当前的 TensorFlow 计算图中。运行其初始值设定项将从磁盘读取它们的预训练值。同样，表和其他状态也会添加到计算图中。

### 缓存模块

通过网址创建模块时，模块内容将下载并缓存到本地系统临时目录中。可以使用 `TFHUB_CACHE_DIR` 环境变量重写模块的缓存位置。有关详细信息，请参阅[缓存](caching.md)。

### 应用模块

实例化后，模块 `m` 可以像 Python 函数一样从张量输入到张量输出调用零次或多次：

```python
y = m(x)
```

每一次此类调用均会向当前 TensorFlow 计算图中添加运算，用于根据 `x` 值计算 `y` 值。如果涉及到包含训练权重的变量，将在所有应用之间共享这些权重。

模块可以定义多个命名*签名*以便用于多种方法（与 Python 对象获得*方法*类似）。模块的文档应注明可用签名。上方的调用应用了名为 `"default"` 的签名。可以通过将名称传递给可选的 `signature=` 参数来选择任何签名。

如果签名具有多个输入，它们必须以字典形式进行传递，并使用由签名定义的键。同样，如果签名具有多个输出，它们可以通过传递 `as_dict=True` 并使用由签名定义的键，以字典形式进行检索（如果 `as_dict=False`，则 `"default"` 键将用于返回的单个输出）。因此，应用模块的最常见形式如下：

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

调用者必须提供由签名定义的所有输入，但不要求使用模块的所有输出。TensorFlow 将仅运行最终成为 `tf.Session.run()` 中目标的依赖项的模块部分。实际上，模块发布者除了提供主要输出，还可以选择提供各种输出用于高级用途（例如中间层激活）。模块使用者应妥善处理额外输出。

### 尝试替代模块

每当有多个模块执行同一任务时，TensorFlow Hub 鼓励为其配备兼容的签名（接口），使尝试不同的模块能够像以字符串值超参数形式更改模块句柄一样方便。

为此，我们为常见任务维护了一个建议的[通用签名](common_signatures/index.md)集合。

## 创建新的模块

### 兼容性说明

TF1 Hub 格式适用于 TensorFlow 1。在 TensorFlow 2 中，TF Hub 仅部分支持该格式。因此，请务必考虑使用新的 [TF2 SavedModel](tf2_saved_model.md) 格式进行发布。

在语法层面上，TF1 Hub 格式与 TensorFlow 1 的 SavedModel 格式类似（文件名和协议消息相同），但在针对模块重用、构成和重复训练方面的语义上却有所不同（例如，资源初始值设定项的存储方式不同，元图的标记惯例不同）。最简单的区分方法是查看磁盘上是否存在 `tfhub_module.pb` 文件。

### 一般方法

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

任何模块实例都可以通过其 `export(path, session)` 方法序列化存储到磁盘上。导出模块会将其定义与其 `session` 内变量的当前状态一起序列化存储到传递的路径中。首次导出模块以及导出微调模块时可使用此方法。

为了与 TensorFlow Estimator 兼容，`hub.LatestModuleExporter` 会从最新的检查点导出模块，与 `tf.estimator.LatestExporter` 从最新的检查点导出整个模块类似。

模块发布者应尽可能实现[通用签名](common_signatures/index.md)，以便使用者可以方便地更换模块并找到对解决其问题最有效的模块。

### 真实示例

请查看我们的[文本嵌入向量模块导出器](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py)中提供的真实示例，了解如何通过通用文本嵌入向量格式创建模块。

## 微调

对导入模型的变量及其周围模型的变量共同执行训练的过程称为*微调*。微调可以实现更好的训练质量，但复杂性也会因此提高。我们建议使用者仅在探索较为简单的质量调整方法后，并且仅在模块发布者建议的情况下实施微调。

### 针对使用者

要启用微调，请使用 `hub.Module(..., trainable=True)` 实例化模块以使其变量可训练，然后导入 TensorFlow 的 `REGULARIZATION_LOSSES`。如果模块具有多个计算图变体，请确保选择一个适用于训练的计算图。该计算图通常带有 `{"train"}` 标记。

选择一种不会破坏预训练权重的训练制度，例如，使学习率低于从头开始训练。

### 针对发布者

为了便于使用者进行微调，请注意以下几点：

- 微调需要正则化。您的模块将随 `REGULARIZATION_LOSSES` 集合一起导出，该集合包含了您所选择的 `tf.layers.dense(..., kernel_regularizer=...)` 等项目，使用者将通过 `tf.losses.get_regularization_losses()` 获得这些项目。最好采用这种方式来定义 L1/L2 正则化损失。

- 在发布者模型中，请避免通过 `tf.train.FtrlOptimizer`、`tf.train.ProximalGradientDescentOptimizer` 和其他近端优化器的 `l1_` 和 `l2_regularization_strength` 参数定义 L1/L2 正则化。这些参数不会随模块一起导出，并且全局设置正则化强度可能不适合于使用者。除了在宽度模型（即稀疏线性模型）或宽度和深度模型中的 L1 正则化之外，还应可以改用单独的正则化损失。

- 如果使用随机失活、批次归一化或类似训练技术，请将其超参数设置为对多种预期用途均有意义的值。随机失活率可能需要根据目标问题的过拟合倾向进行调整。在批次归一化中，动量（也称为衰减系数）应足够小，以支持使用小型数据集和/或大批次进行微调。对于高级使用者，请考虑添加签名以公开对关键超参数的控制。

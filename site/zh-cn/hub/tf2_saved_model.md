# TensorFlow 2 中 TF Hub 的 SavedModel

建议使用 [TensorFlow 2 的 SavedModel 格式](https://www.tensorflow.org/guide/saved_model)在 TensorFlow Hub 上共享预训练模型和模型部分。该格式取代了旧的 [TF1 Hub 格式](tf1_hub_module.md)并提供了一组新的 API。

本页将介绍如何使用低级 `hub.load()` API 及其 `hub.KerasLayer` 包装器在 TensorFlow 2 程序中重用 TF2 SavedModel。（通常，`hub.KerasLayer` 将与其他 `tf.keras.layers` 组合以构建 Keras 模型或 TF2 Estimator 的 `model_fn`。）这些 API 还可以在有限条件下加载 TF1 Hub 格式的旧模型，请参阅[兼容性指南](model_compatibility.md)。

TensorFlow 1 的用户可以更新到 TF 1.15，并使用相同的 API。更早版本的 TF1 则无法运行。

## 使用 TF Hub 的 SavedModel

### 在 Keras 中使用 SavedModel

[Keras](https://www.tensorflow.org/guide/keras/) 是 TensorFlow 的高级 API，用于通过构成 Keras 层对象来构建深度学习模型。`tensorflow_hub` 库提供了使用 SavedModel 的 URL（或文件系统路径）初始化的 `hub.KerasLayer` 类，然后提供了 SavedModel 中的计算，包括其预训练权重。

下例使用了预训练的文本嵌入向量：

```python
import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

由此，可以使用通常的 Keras 方法构建文本分类器：

```python
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

[文本分类器 Colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb) 完整地展示了如何训练和评估此类分类器。

`hub.KerasLayer` 中的模型权重默认情况下设置为不可训练。请参阅下文中有关微调的部分来了解如何更改该设置。按照 Keras 惯例，会在同一层对象的所有应用之间共享权重。

### 在 Estimator 中使用 SavedModel

使用 TensorFlow 的 [Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator) API 进行分布式训练的用户可以通过在 `hub.KerasLayer` 及其他 `tf.keras.layers` 方面编写 `model_fn` 来使用 TF Hub 的 SavedModel。

### 幕后工作：SavedModel 下载和缓存

使用来自 TensorFlow Hub（或实现其[托管](hosting.md)协议的其他 HTTPS 服务器）的 SavedModel 时，若尚不存在，需要将其下载并解压到本地文件系统中。可以设置环境变量 `TFHUB_CACHE_DIR` 以重写用于缓存下载并解压的 SavedModel 的默认临时位置。有关详细信息，请参阅[缓存](caching.md)。

### 在低级 TensorFlow 中使用 SavedModel

`hub.load(handle)` 函数可下载和解压 SavedModel（ 除非 `handle` 已经是文件系统路径），然后返回使用 TensorFlow 的内置函数 `tf.saved_model.load()` 加载该模型的结果。因此，`hub.load()` 可以处理任何有效的 SavedModel（与其 TF1 的前身 `hub.Module` 不同）。

#### 高级主题：完成加载后对 SavedModel 的期望

根据 SavedModel 的内容，`obj = hub.load(...)` 的结果可通过多种方式调用（TensorFlow 的 [SavedModel 指南](https://www.tensorflow.org/guide/saved_model)中提供了更为详细的说明）：

- SavedModel 的服务签名（如有）表示为具体函数的字典，可以像 `tensors_out = obj.signatures["serving_default"](**tensors_in)` 一样调用，张量的字典由相应的输入和输出名称键控，并受制于签名的形状和数据类型约束。

- 已保存对象（如有）的 [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) 装饰方法恢复为 tf.function 对象，这些对象可通过在保存之前已[跟踪](https://www.tensorflow.org/tutorials/customization/performance#tracing) tf.function 的张量和非张量参数的所有组合进行调用。特别是，如果存在带有适当跟踪的 `obj.__call__` 方法，则 `obj` 本身可以像 Python 函数一样调用。`output_tensor = obj(input_tensor, training=False)` 便是一个简单的示例。

这对 SavedModel 可以实现的接口提供了极大的自由度。`obj` 的[可重用 SavedModel 接口](reusable_saved_models.md)建立了惯例，使客户端代码（包括 `hub.KerasLayer` 之类的适配器）知道如何使用 SavedModel。

某些 SavedModel 可能不遵循该惯例，特别是不打算在较大的模型中重用的整个模型，仅提供服务签名。

SavedModel 中的可训练变量重新加载为可训练变量，`tf.GradientTape` 将在默认情况下对其进行监视。请参阅下文中有关微调的部分来了解一些注意事项，初学者应避免使用微调。如果您想使用微调，也可以查看 `obj.trainable_variables` 是否建议仅重新训练最初可训练变量的子集。

## 为 TF Hub 创建 SavedModel

### 概述

SavedModel 是 TensorFlow 用于已训练模型或模型部分的标准序列化格式。它存储了模型的训练权重以及用于执行其计算的确切 TensorFlow 运算。SavedModel 可独立于创建它的代码单独使用。特别是，它可以在不同的高级建模 API（例如 Keras）之间重用，因为 TensorFlow 运算是它们的通用基本语言。

### 从 Keras 保存

自 TensorFlow 2 起，`tf.keras.Model.save()` 和 `tf.keras.models.save_model()` 默认为 SavedModel 格式（非 HDF5）。生成的 SavedModel 可与 `hub.load()`、`hub.KerasLayer` 和类似的适配器组合使用，用于其他待提供的高级 API。

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

GitHub 上的 [TensorFlow 模型](https://github.com/tensorflow/models)为 BERT 使用前一种方法（请参见 [nlp/bert/bert_models.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/bert_models.py) 和 [nlp/bert/export_tfhub.py](https://github.com/tensorflow/models/blob/master/official/nlp/bert/export_tfhub.py)，请注意 `core_model` 和 `pretrain_model` 间的拆分方式），并为 ResNet 使用后一种方法（请参见 [vision/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/tfhub_export.py)）。

### 从低级 TensorFlow 保存

此操作需要充分熟悉 TensorFlow 的 [SavedModel 指南](https://www.tensorflow.org/guide/saved_model)。

如果您不仅要提供服务签名，则应该实现[可重用 SavedModel 接口](reusable_saved_models.md)。从概念上讲，如下所示：

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super(MyMulModel, self).__init__()
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

[tensorflow/examples/saved_model/integration_tests/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/saved_model/integration_tests) 中的代码包含了更大的样本，特别是 `export_mnist.py` 和 `use_mnist.py` 对。

## 微调

对导入的 SavedModel 的已训练变量及其周围模型的变量共同执行训练的过程被称为*微调* SavedModel。微调可以实现更好的训练质量，但通常要求也会因此提高（可能需要更多时间，更加依赖优化器及其超参数，提高了过拟合的风险，并且需要扩充数据集，尤其是对于 CNN）。我们建议 SavedModel 使用者仅在建立良好的训练制度后，并且仅在 SavedModel 发布者建议的情况下实施微调。

微调会更改已训练的“连续”模型参数。它不会改变硬编码转换，例如令牌化文本输入以及将令牌映射到其嵌入矩阵中的相应条目。

### 针对 SavedModel 使用者

创建 `hub.KerasLayer`，如

```python
layer = hub.KerasLayer(..., trainable=True)
```

将对由层加载的 SavedModel 启用微调。SavedModel 中声明的可训练权重和权重正则化项将添加到 Keras 模型，并在训练模式下运行 SavedModel 的计算（随机失活等）。

[图像分类 Colab](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb) 中提供了包含可选微调的端到端训练示例。

#### 重新导出微调结果

高级用户可能希望将微调的结果重新保存为 SavedModel 以便使用，取代最初加载的模型。可使用以下代码完成此操作：

```python
loaded_obj = hub.load("https://tfhub.dev/...")
hub_layer = hub.KerasLayer(loaded_obj, trainable=True, ...)

model = keras.Sequential([..., hub_layer, ...])
model.compile(...)
model.fit(...)

export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
tf.saved_model.save(loaded_obj, export_module_dir)
```

### 针对 SavedModel 创建者

创建要在 TensorFlow Hub 上共享的 SavedModel 时，请提前考虑使用者是否以及应如何微调模型，并在文档中提供指南。

从 Keras 模型保存 SavedModel 时，应使微调的所有机制全部生效（保存权重正则化损失、声明可训练变量、在 `training=True` 和 `training=False` 情况下跟踪 `__call__` 等）。

选择能够适应梯度流（例如，输出 logits 而非 softmax 概率或 top-k 预测）的模型接口。

如果模型使用随机失活、批次归一化或涉及超参数的类似训练技术，请将超参数设置为对多种预期目标问题和批次大小均有意义的值。（截至撰写本文时，从 Keras 保存 SavedModel 尚不便于使用者进行调节，但请参见 [tensorflow/examples/saved_model/integration_tests/export_mnist_cnn.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/saved_model/integration_tests/export_mnist_cnn.py) 来了解一些粗略的解决方法。）

各个层上的权重正则化器（及其正则化强度系数）将得到保存，但优化器（例如 `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`）内部的权重正则化将丢失。请向您的 SavedModel 的使用者提出相应建议。

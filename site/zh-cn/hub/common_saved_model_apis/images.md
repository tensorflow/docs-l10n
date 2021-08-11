<!--* freshness: { owner: 'mroff' reviewed: '2021-03-09'  } *-->

# 图像任务的通用 SavedModel API

本页面介绍用于图像相关任务的 [TF2 SavedModel](../tf2_saved_model.md) 应当如何实现[可重用的 SavedModel API](../reusable_saved_models.md)。（这会替换现已弃用的 [TF1 Hub 格式](../tf1_hub_module)的[通用图像签名](../common_signatures/images.md)。）

<a name="feature-vector"></a>

## 图像特征向量

### 用法摘要

**图像特征向量**是表示整个图像的密集一维张量，通常由使用者模型中的简单前馈分类器使用。（对于经典 CNN 而言，这是在空间范围被池化或平展之后但在分类完成之前的瓶颈值；为此，请参阅下面的[图像分类](#classification)。）

用于图像特征提取的可重用 SavedModel 在根对象上具有 `__call__` 方法，该方法可将一批图像映射至一批特征向量。示例用法如下：

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = obj(images)   # A batch with shape [batch_size, num_features].
```

在 Keras 中，等效代码如下：

```python
features = hub.KerasLayer("path/to/model")(images)
```

输入遵循[图像输入](#input)的一般约定。模型文档指定了输入的 `height` 和 `width` 的允许范围。

输出是数据类型为 `float32` 且形状为 `[batch_size, num_features]` 的单个张量。`batch_size` 与输入的大小相同。`num_features` 是模块特定的常量，与输入大小无关。

### API 详细信息

[可重用的 SavedModel API](../reusable_saved_models.md) 还提供了一个 `obj.variables` 列表（例如，用于不以 Eager 模式加载时的初始化）。

支持微调的模型提供了一个 `obj.trainable_variables` 列表。可能需要您传递 `training=True` 才能在训练模式下执行（例如，用于随机失活）。某些模型允许可选参数重写超参数（例如，随机失活率；将在模型文档中介绍）。该模型还可能提供一个 `obj.regularization_losses` 列表。有关详细信息，请参参阅[可重用的 SavedModel API](../reusable_saved_models.md)。

在 Keras 中，这由 `hub.KerasLayer` 处理：使用 `trainable=True` 将其初始化，然后（在超参数重写适用的极少数情况下）使用 `arguments=dict(some_hparam=some_value, ...))` 启用微调。

### 说明

是否对输出特征应用随机失活应由模型使用者决定。SavedModel 本身不应对实际输出执行随机失活（即使它在内部其他位置使用随机失活）。

### 示例

图像特征向量的可重用 SavedModel 在以下各项中使用：

- Colab 教程[重新训练图像分类器](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)；
- 命令行工具 [make_image_classifier](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)。

<a name="classification"></a>

## 图像分类

### 用法摘要

**图像分类**可将图像的像素映射至*模块发布者选择的*分类法中类成员的线性得分 (logits)。这样，模型使用者便可从发布者模块学习到的特定分类中得出结论。（对于具有一组新类的图像分类，通常将[图像特征向量](#feature-vector)模型与新分类器一起重用。）

用于图像分类的可重用 SavedModel 在根对象上具有 `__call__` 方法，该方法可将一批图像映射至一批 logits。示例用法如下：

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
logits = obj(images)   # A batch with shape [batch_size, num_classes].
```

在 Keras 中，等效代码如下：

```python
logits = hub.KerasLayer("path/to/model")(images)
```

输入遵循[图像输入](#input)的一般约定。模型文档指定了输入的 `height` 和 `width` 的允许范围。

输出 `logits` 是数据类型为 `float32` 且形状为 `[batch_size, num_classes]` 的单个张量。`batch_size` 与输入的大小相同。`num_classes` 是分类中的类数，它是一个模型特定的常量。

值 `logits[i, c]` 是一个得分，用于预测索引为 `c` 的类中样本 `i` 的成员资格。

将这些得分用于 softmax（针对互斥类）、sigmoid（针对正交类）还是其他函数取决于基础分类。模块文档应有所说明，并引用类索引的定义。

### API 详细信息

[可重用的 SavedModel API](../reusable_saved_models.md) 还提供了一个 `obj.variables` 列表（例如，用于不以 Eager 模式加载时的初始化）。

支持微调的模型提供了一个 `obj.trainable_variables` 列表。可能需要您传递 `training=True` 才能在训练模式下执行（例如，用于随机失活）。某些模型允许可选参数重写超参数（例如，随机失活率；将在模型文档中介绍）。该模型还可能提供一个 `obj.regularization_losses` 列表。有关详细信息，请参参阅[可重用的 SavedModel API](../reusable_saved_models.md)。

在 Keras 中，这由 `hub.KerasLayer` 处理：使用 `trainable=True` 将其初始化，然后（在超参数重写适用的极少数情况下）使用 `arguments=dict(some_hparam=some_value, ...))` 启用微调。

<a name="input"></a>

## 图像输入

图像输入通用于所有类型的图像模型。

将一批图像作为输入的模型会将图像作为数据类型为 `float32`、形状为 `[batch_size, height, width, 3]` 的密集四维张量接受，这些张量的元素是归一化为 [0, 1] 范围的像素的 RGB 颜色值。`tf.image.decode_*()` 后接 `tf.image.convert_image_dtype(..., tf.float32)` 即可返回该结果。

模型接受任意 `batch_size`。模型文档指定了 `height` 和 `width` 的允许范围。最后一个维度固定为 3 个 RGB 通道。

建议模型在整个过程中都使用张量的 `channels_last`（或 `NHWC`）布局，并在需要时将其由 TensorFlow 的计算图优化器重写为 `channels_first`（或 `NCHW`）。

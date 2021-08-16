<!--* freshness: { owner: 'mroff' reviewed: '2021-03-09' } *-->

# 图像的常用签名

本页面介绍应由 [TF1 Hub 格式](../tf1_hub_module.md)的模块为图像相关任务实现的常用签名。（有关 [TF2 SavedModel 格式](../tf2_saved_model.md)，请参阅具有类似功能的 [SavedModel API](../common_saved_model_apis/images.md)。）

某些模块可用于多项任务（例如，图像分类模块在工作时往往需要执行一些特征提取任务）。因此，每个模块都要 (1) 为发布者预期的所有任务提供命名签名，并 (2) 为其指定的主要任务提供默认签名 `output = m(images)`。

<a name="feature-vector"></a>

## 图像特征向量

### 使用摘要

**图像特征向量**是一种能够表示整个图像的密集一维张量，通常用于由使用者模型进行分类。（不同于 CNN 的中间激活，它不支持空间分解。也不同于[图像分类](#classification)，它丢弃了发布者模型学习到的分类。）

用于图像特征提取的模块具有默认签名，该签名可将一批图像映射到一批特征向量。示例用法如下：

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

它还定义了相应的命名签名。

### 签名规范

用于提取图像特征向量的命名签名以如下方法调用：

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

输入遵循[图像输入](#input)的一般惯例。

输出字典包含数据类型为 `float32`、形状为 `[batch_size, num_features]` 的 `"default"` 输出 。`batch_size` 与输入大小相同，但在构建计算图时处于未知状态。`num_features` 为已知的模块特定常量，与输入大小无关。

这些特征向量旨在用于通过简单的前馈分类器进行分类（例如，通过典型 CNN 的顶端卷积层的池化特征来实现图像分类）。

是否对输出特征应用随机失活应由模块使用者决定。模块本身不应对实际输出执行随机失活（即使它在内部其他位置使用随机失活）。

输出字典可以提供其他输出（例如，模块内部隐藏层的激活）。它们的键和值取决于模块。建议为架构相关的键添加架构名称前缀（例如，用于避免将中间层 `"InceptionV3/Mixed_5c"` 与顶端卷积层 `"InceptionV2/Mixed_5c"` 混淆）。

<a name="classification"></a>

## 图像分类

### 使用摘要

**图像分类**可将图像像素映射到*模块发布者所选择的*分类法的类成员的线性得分 (logits)。这样，使用者可以基于发布者模块所学的特定分类方法，而非仅由图像的基本特征（请参阅[图像特征向量](#feature-vector)）得出结论。

用于图像特征提取的模块具有默认签名，该签名可将一批图像映射到一批 logits。示例用法如下：

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

它还定义了相应的命名签名。

### 签名规范

用于提取图像特征向量的命名签名以如下方法调用：

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

输入遵循[图像输入](#input)的一般惯例。

输出字典包含数据类型为 `float32`、形状为 `[batch_size, num_classes]` 的 `"default"` 输出 。`batch_size` 与输入大小相同，但在构建计算图时处于未知状态。`num_classes` 为分类中的类数量，是一个已知常量，与输入大小无关。

评估 `outputs["default"][i, c]` 会得出分数，用于预测索引为 `c` 的类中 `i` 样本的成员资格。

将这些得分用于 softmax（针对互斥类）、sigmoid（针对正交类）还是其他函数取决于基础分类。模块文档应有所说明，并引用类索引的定义。

输出字典可以提供其他输出（例如，模块内部隐藏层的激活）。它们的键和值取决于模块。建议为架构相关的键添加架构名称前缀（例如，用于避免将中间层 `"InceptionV3/Mixed_5c"` 与顶端卷积层 `"InceptionV2/Mixed_5c"` 混淆）。

<a name="input"></a>

## 图像输入

对于所有类型的图像模块和图像签名，图像输入都是通用的。

将一批图像作为输入的签名会将图像作为一个数据类型为 `float32` 、形状为 `[batch_size, height, width, 3]` 的密集四维张量接受，它们的元素是归一化为 [0, 1] 范围的像素的 RGB 颜色值。`tf.image.decode_*()` 后接 `tf.image.convert_image_dtype(..., tf.float32)` 即可获得此结果。

仅具有一个（或一个主要）图像输入的模块会为此输入使用名称 `"images"`。

该模块接受任何 `batch_size`，并将 TensorInfo.tensor_shape 的第一维度相应地设置为“unknown”。最后一个维度固定设置为 `3`，即 RGB 的三个通道。`height` 和 `width` 维度固定设置为输入图像的预期大小。（未来可能会消除对全卷积网络模块的限制。）

模块的使用者不应直接检查形状，而应通过在模块或模块规范上调用 hub.get_expected_image_size() 来获取大小信息，并应相应调整输入图像的大小（通常在批处理之前/期间进行调整）。

为方便起见，TF-Hub 模块使用 `channels_last`（或 `NHWC`）张量布局，并根据需要将其保留给 TensorFlow 的计算图优化器，以便根据需要重新写入 `channels_first`（或 `NCHW`）。这是自 TensorFlow 1.7 版起的默认行为。

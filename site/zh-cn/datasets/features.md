# FeatureConnector

`tfds.features.FeatureConnector` API：

- 定义最终 `tf.data.Dataset` 的结构、形状、数据类型
- 向/从磁盘抽象出序列化。
- 公开其他元数据（例如标签名称、音频采样率…）

## 概述

`tfds.features.FeatureConnector` 定义数据集特征结构（在 `tfds.core.DatasetInfo` 中）：

```python
tfds.core.DatasetInfo(
    features=tfds.features.FeaturesDict({
        'image': tfds.features.Image(shape=(28, 28, 1), doc='Grayscale image'),
        'label': tfds.features.ClassLabel(
            names=['no', 'yes'],
            doc=tfds.features.Documentation(
                desc='Whether this is a picture of a cat',
                value_range='yes or no'
            ),
        ),
        'metadata': {
            'id': tf.int64,
            'timestamp': tfds.features.Scalar(
                tf.int64,
                doc='Timestamp when this picture was taken as seconds since epoch'),
            'language': tf.string,
        },
    }),
)
```

可以通过仅使用文本描述 (`doc='description'`) 或直接使用 `tfds.features.Documentation` 来提供更详细的特征描述来记录特征。

特征可以是：

- 标量值：`tf.bool`、`tf.string`、`tf.float32`… 当您想记录特征时，也可以使用 `tfds.features.Scalar(tf.int64, doc='description')`。
- `tfds.features.Audio`、`tfds.features.Video`…（请参阅可用特征[列表](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?version=nightly)）
- 特征的嵌套 `dict`：`{'metadata': {'image': Image(), 'description': tf.string}}`…
- 嵌套 `tfds.features.Sequence`：`Sequence({'image': ..., 'id': ...})`、`Sequence(Sequence(tf.int64))`…

在生成过程中，样本将由 `FeatureConnector.encode_example` 自动序列化为适合磁盘的格式（当前为 `tf.train.Example` 协议缓冲区）：

```python
yield {
    'image': '/path/to/img0.png',  # `np.array`, file bytes,... also accepted
    'label': 'yes',  # int (0-num_classes) also accepted
    'metadata': {
        'id': 43,
        'language': 'en',
    },
}
```

读取数据集时（例如使用 `tfds.load`），数据会使用 `FeatureConnector.decode_example` 自动解码。返回的 `tf.data.Dataset` 将匹配 `tfds.core.DatasetInfo` 中定义的 `dict` 结构：

```python
ds = tfds.load(...)
ds.element_spec == {
    'image': tf.TensorSpec(shape=(28, 28, 1), tf.uint8),
    'label': tf.TensorSpec(shape=(), tf.int64),
    'metadata': {
        'id': tf.TensorSpec(shape=(), tf.int64),
        'language': tf.TensorSpec(shape=(), tf.string),
    },
}
```

## 序列化/反序列化为 proto

TFDS 公开了一个低级 API 以将样本序列化/反序列化为 `tf.train.Example` proto。

要将 `dict[np.ndarray | Path | str | ...]` 序列化为 proto `bytes`，请使用 `features.serialize_example`：

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    f.write(ex_bytes)
```

要将 proto `bytes` 反序列化为 `tf.Tensor`，请使用 `features.deserialize_example`：

```python
ds = tf.data.TFRecordDataset('path/to/file.tfrecord')
ds = ds.map(features.deserialize_example)
```

## 访问元数据

要访问特征元数据（标签名称、形状、数据类型…），请参阅[简介文档](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)。示例：

```python
ds, info = tfds.load(..., with_info=True)

info.features['label'].names  # ['cat', 'dog', ...]
info.features['label'].str2int('cat')  # 0
```

## 创建您自己的 `tfds.features.FeatureConnector`

如果您认为[可用特征](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes)中缺少某个特征，请打开一个[新议题](https://github.com/tensorflow/datasets/issues)。

要创建您自己的特征连接器，需要从  `tfds.features.FeatureConnector` 继承并实现抽象方法。

- 如果您的特征是单个张量值，最好从 `tfds.features.Tensor` 继承并在需要时使用 `super()`。有关示例，请参阅 `tfds.features.BBoxFeature` 源代码。
- 如果您的特征是多个张量的容器，最好从 `tfds.features.FeaturesDict` 继承并使用 `super()` 自动编码子连接器。

`tfds.features.FeatureConnector` 对象可将特征在磁盘中的编码方式从特征如何呈现给用户中抽象出来。下图显示了数据集的抽象层，以及从原始数据集文件到 `tf.data.Dataset` 对象的转换。

<p align="center">   <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/datasets/dataset_layers.png?raw=true" alt="DatasetBuilder abstraction layers" class=""></p>

要创建您自己的特征连接器，请将  `tfds.features.FeatureConnector` 子类化并实现抽象方法：

- `encode_example(data)`：定义如何将在生成器 `_generate_examples()` 中给定的数据编码成兼容 `tf.train.Example` 的数据。可以返回单个值或值的 `dict`。
- `decode_example(data)`：定义如何将从 `tf.train.Example` 读取的张量中的数据解码成 `tf.data.Dataset` 返回的用户张量。
- `get_tensor_info()`：指定 `tf.data.Dataset` 返回的张量的形状/数据类型。如果从另一个 `tfds.features` 继承，则是可选项。
- （可选）`get_serialized_info()`：如果 `get_tensor_info()` 返回的信息与实际将数据写入磁盘的方式不同，那么您需要重写 `get_serialized_info()` 以匹配 `tf.train.Example` 的规范
- `to_json_content`/`from_json_content`：这是允许在没有原始源代码的情况下加载数据集所必需的。有关示例，请参阅[音频特征](https://github.com/tensorflow/datasets/blob/65a76cb53c8ff7f327a3749175bc4f8c12ff465e/tensorflow_datasets/core/features/audio_feature.py#L121)。

注：确保使用 `self.assertFeature` 和 `tfds.testing.FeatureExpectationItem` 测试您的特征连接器。请查看[测试示例](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features/image_feature_test.py)：

如需了解详情，请查看 `tfds.features.FeatureConnector` 文档。最好还是查看一下[真实示例](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features)。

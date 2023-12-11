# 使用 TFDS 加载外部 tfrecord

如果您有一个由第三方工具生成的 `tf.train.Example` proto（在 `.tfrecord`、`.riegeli` 等内），并且您想直接使用 tfds API 进行加载，那么本页面适合您。

要加载您的 `.tfrecord` 文件，只需：

- 遵循 TFDS 命名约定。
- 沿 tfrecord 文件添加元数据文件（`dataset_info.json`、`features.json`）。

限制：

- 不支持 `tf.train.SequenceExample`，仅支持 `tf.train.Example`。
- 您需要能够用 `tfds.features` 来表达 `tf.train.Example`（参阅下面的部分）。

## 文件命名约定

TFDS 支持为文件名定义模板，这为使用不同的文件命名方案提供了灵活性。模板由 `tfds.core.ShardedFileTemplate` 表示，并支持以下变量：`{DATASET}`、`{SPLIT}`、`{ FILEFORMAT}`、`{SHARD_INDEX}`、`{NUM_SHARDS}` 和 `{SHARD_X_OF_Y}`。例如，TFDS 的默认文件命名方案为：`{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}`。对于 MNIST，这意味着[文件名](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1)如下所示：

- `mnist-test.tfrecord-00000-of-00001`
- `mnist-train.tfrecord-00000-of-00001`

## 添加元数据

### 提供特征结构

要使 TFDS 能够解码 `tf.train.Example` proto，您需要提供与您的规范匹配的 `tfds.features` 结构。例如：

```python
features = tfds.features.FeaturesDict({
    'image':
        tfds.features.Image(
            shape=(256, 256, 3),
            doc='Picture taken by smartphone, downscaled.'),
    'label':
        tfds.features.ClassLabel(names=['dog', 'cat']),
    'objects':
        tfds.features.Sequence({
            'camera/K': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
        }),
})
```

对应于以下 `tf.train.Example` 规范：

```python
{
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'objects/camera/K': tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64),
}
```

指定特征允许 TFDS 自动解码图片、视频… 与任何其他 TFDS 数据集一样，特征元数据（例如标签名称…）将公开给用户（例如 `info.features['label'].names` ）。

#### 如果您控制生成流水线

如果您在 TFDS 之外生成数据集但仍控制生成流水线，则可以使用 `tfds.features.FeatureConnector.serialize_example` 将数据从 `dict[np.ndarray]` 编码为 `tf.train.Example` proto `bytes`：

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    writer.write(ex_bytes)
```

这将确保与 TFDS 的特征兼容性。

类似地，存在一个 `feature.deserialize_example` 来解码 proto（[示例](https://www.tensorflow.org/datasets/features#serializedeserialize_to_proto)）

#### 如果您不控制生成流水线

如果您想查看 `tfds.features` 在 `tf.train.Example` 中是如何表示的，可以在 Colab 中进行检查：

- 要将 `tfds.features` 转换为 `tf.train.Example` 的人类可读结构，可以调用 `features.get_serialized_info()`。
- 要获得确切的 `FixedLenFeature`，会将 spec 传递给 `tf.io.parse_single_example`，这样便可使用 `spec = features.tf_example_spec`

注：如果您使用自定义特征连接器，请确保实现 `to_json_content`/`from_json_content` 并使用 `self.assertFeature` 进行测试（请参阅[特征连接器指南](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector)）

### 获取拆分统计信息

TFDS 需要知道每个分片中样本的确切数量。这是 `len(ds)` 等特征或 [subplit API](https://www.tensorflow.org/datasets/splits) 所必需的：`split='train[75%:]'`。

- 如果您有此信息，则可以显式创建 `tfds.core.SplitInfo` 的列表并跳到下一部分：

    ```python
    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=[1024, ...],  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
        tfds.core.SplitInfo(name='test', ...),
    ]
    ```

- 如果您不知道此信息，则可以使用 `compute_split_info.py` 脚本（或在您自己的脚本中使用 `tfds.folder_dataset.compute_split_info`）计算它。这将启动一个 Beam 流水线，该流水线将读取给定目录上的所有分片并计算信息。

### 添加元数据文件

要沿数据集自动添加适当的元数据文件，请使用 `tfds.folder_dataset.write_metadata`：

```python
tfds.folder_dataset.write_metadata(
    data_dir='/path/to/my/dataset/1.0.0/',
    features=features,
    # Pass the `out_dir` argument of compute_split_info (see section above)
    # You can also explicitly pass a list of `tfds.core.SplitInfo`.
    split_infos='/path/to/my/dataset/1.0.0/',
    # Pass a custom file name template or use None for the default TFDS
    # file name template.
    filename_template='{SPLIT}-{SHARD_X_OF_Y}.{FILEFORMAT}',

    # Optionally, additional DatasetInfo metadata can be provided
    # See:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo
    description="""Multi-line description."""
    homepage='http://my-project.org',
    supervised_keys=('image', 'label'),
    citation="""BibTex citation.""",
)
```

在您的数据集目录上调用一次该函数后，元数据文件 (`dataset_info.json`...) 即会添加，并且数据集将准备好使用 TFDS 进行加载（请参阅下一部分）。

## 使用 TFDS 加载数据集

### 直接从文件夹

生成元数据后，可以使用 `tfds.builder_from_directory` 加载数据集，该函数会返回带有标准 TFDS API（如 `tfds.builder`）的 `tfds.core.DatasetBuilder`：

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### 直接从多个文件夹

此外，也可以从多个文件夹加载数据。例如，在强化学习中，当多个代理分别生成一个单独的数据集并且您希望将所有数据集一同加载时，就会发生这种情况。其他用例是定期生成新数据集的情况，例如每天生成一个新数据集，并且您想从某个日期范围加载数据。

要从多个文件夹加载数据，请使用 `tfds.builder_from_directories`，它使用标准 TFDS API（如 `tfds.builder`）返回 `tfds.core.DatasetBuilder`：

```python
builder = tfds.builder_from_directories(builder_dirs=[
    '~/path/my_dataset/agent1/1.0.0/',
    '~/path/my_dataset/agent2/1.0.0/',
    '~/path/my_dataset/agent3/1.0.0/',
])

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

注：每个文件夹必须有自己的元数据，因为它包含有关拆分的信息。

### 文件夹结构（可选）

为了更好地与 TFDS 兼容，您可以将数据组织为 `<data_dir>/<dataset_name>[/<dataset_config>]/<dataset_version>`。例如：

```
data_dir/
    dataset0/
        1.0.0/
        1.0.1/
    dataset1/
        config0/
            2.0.0/
        config1/
            2.0.0/
```

这将使您的数据集与 `tfds.load` / `tfds.builder` API 兼容，只需提供 `data_dir/`：

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```

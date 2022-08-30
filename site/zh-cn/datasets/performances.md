# 性能提示

本文档提供了 TensorFlow Datasets (TFDS) 特定的性能提示。请注意，TFDS 以 `tf.data.Dataset` 对象的形式提供数据集，因此 [`tf.data` 指南](https://www.tensorflow.org/guide/data_performance#optimize_performance)中的建议仍然适用。

## 对数据集进行基准分析

使用 `tfds.benchmark(ds)` 对任何 `tf.data.Dataset` 对象进行基准分析。

确保指示 `batch_size=` 以将结果归一化（例如100 iter/sec -&gt; 3200 ex/sec）。这适用于任何可迭代对象（例如 `tfds.benchmark(tfds.as_numpy(ds))`）。

```python
ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
tfds.benchmark(ds, batch_size=32)
```

## 小型数据集（小于 1 GB）

所有 TFDS 数据集都以 [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord) 格式将数据存储在磁盘上。对于小型数据集（例如 MNIST、CIFAR-10/-100），`.tfrecord` 会显著增加开销。

由于这些数据集能够装入内存，可以通过缓存或预加载数据集来显著提升性能。请注意，TFDS 会自动缓存小型数据集（有关详细信息，请参阅下一部分）。

### 缓存数据集

下面的数据流水线示例在归一化图像后显式缓存数据集。

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


ds, ds_info = tfds.load(
    'mnist',
    split='train',
    as_supervised=True,  # returns `(img, label)` instead of dict(image=, ...)
    with_info=True,
)
# Applying normalization before `ds.cache()` to re-use it.
# Note: Random transformations (e.g. images augmentations) should be applied
# after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
# vectorization [1]).
ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache()
# For true randomness, we set the shuffle buffer to the full dataset size.
ds = ds.shuffle(ds_info.splits['train'].num_examples)
# Batch after shuffling to get unique batches at each epoch.
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
```

- [[1] 向量化映射](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

迭代此数据集时，由于缓存，第二次迭代将比第一次迭代快得多。

### 自动缓存

默认情况下，TFDS 会自动缓存（使用 `ds.cache()`）满足以下约束的数据集：

- 数据集总大小（所有拆分）已定义且 &lt; 250 MiB
- `shuffle_files` 被停用，或仅读取单个分片

可以选择退出自动缓存，方法是在 `tfds.load` 中将 `try_autocaching=False` 传递至 `tfds.ReadConfig`。请参阅数据集目录文档，了解特定数据集是否将使用自动缓存。

### 将全部数据作为单个张量加载

如果您的数据集能够装入内存，您也可以将整个数据集作为单个张量或 NumPy 数组加载。为此，可以通过设置 `batch_size=-1` 来批处理单个 `tf.Tensor` 中的所有样本。然后使用 `tfds.as_numpy` 从 `tf.Tensor` 转换为 `np.array`。

```python
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## 大型数据集

大型数据集需分片（拆分为多个文件），通常无法装入内存，因此不应进行缓存。

### 打乱顺序和训练

在训练过程中，有效地打乱数据顺序非常重要。未能有效地打乱数据顺序会导致训练准确率降低。

除了使用 `ds.shuffle` 来打乱记录顺序，还应设置 `shuffle_files=True` 以使分片成多个文件的大型数据集获得良好的乱序行为。否则，周期将以相同的顺序读取分片，因此无法真正地随机化数据。

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

此外，当 `shuffle_files=True` 时，TFDS 会停用 [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic)，这可能会略微提升性能。要确定性地执行打乱顺序，也可以通过 `tfds.ReadConfig` 来选择退出此功能：方法是设置 `read_config.shuffle_seed` 或重写 `read_config.options.deterministic`。

### 在工作进程之间对数据进行自动分片 (TF)

在多个工作进程上训练时，您可以使用 `tfds.ReadConfig` 的 `input_context` 参数，以便每个工作进程都可以读取一部分数据。

```python
input_context = tf.distribute.InputContext(
    input_pipeline_id=1,  # Worker id
    num_input_pipelines=4,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)
ds = tfds.load('dataset', split='train', read_config=read_config)
```

这与子拆分 API 互补。首先，应用子拆分 API，`train[:50%]` 被转换成要读取的文件列表。随后，在这些文件上应用 `ds.shard()` 运算。例如，结合使用 `train[:50%]` 与 `num_input_pipelines=2` 时，2 个工作进程中的每一个将读取 1/4 的数据。

当 `shuffle_files=True` 时，文件将在一个工作进程内（不是工作进程之间）随机打乱顺序。每个工作进程都将在周期之间读取相同比例的文件。

注：使用 `tf.distribute.Strategy` 时，`input_context` 可以使用 [distribute_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function) 自动创建

### 在工作进程之间对数据进行自动分片 (Jax)

借助 Jax，您可以使用 `tfds.split_for_jax_process` 或 `tfds.even_splits` API 在工作进程之间分布数据。请参阅 [split API 指南](https://www.tensorflow.org/datasets/splits)。

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` 是一个简单的别名：

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### 提高图像解码速度

默认情况下，TFDS 会自动解码图像。但在某些情况下，使用 `tfds.decode.SkipDecoding` 跳过图像解码并手动应用 `tf.io.decode_image` 运算可以提高性能：

- （使用 `tf.data.Dataset.filter`）筛选样本时，在筛选样本后对图像进行解码。
- 裁剪图像时，使用融合的 `tf.image.decode_and_crop_jpeg` 运算。

[解码指南](https://www.tensorflow.org/datasets/decode#usage_examples)中提供了两个示例的代码。

### 跳过未使用的特征

如果您只使用特征的一个子集，则可以完全跳过某些特征。如果您的数据集中有许多未使用的特征，则不解码这些特征可以显著提高性能。请参阅 https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features。

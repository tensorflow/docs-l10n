# 성능 팁

This document provides TensorFlow Datasets (TFDS)-specific performance tips. Note that TFDS provides datasets as `tf.data.Dataset` objects, so the advice from the [`tf.data` guide](https://www.tensorflow.org/guide/data_performance#optimize_performance) still applies.

## 벤치마크 데이터세트

Use `tfds.benchmark(ds)` to benchmark any `tf.data.Dataset` object.

Make sure to indicate the `batch_size=` to normalize the results (e.g. 100 iter/sec -&gt; 3200 ex/sec). This works with any iterable (e.g. `tfds.benchmark(tfds.as_numpy(ds))`).

```python
ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
tfds.benchmark(ds, batch_size=32)
```

## Small datasets (less than 1 GB)

All TFDS datasets store the data on disk in the [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. For small datasets (e.g. MNIST, CIFAR-10/-100), reading from `.tfrecord` can add significant overhead.

As those datasets fit in memory, it is possible to significantly improve the performance by caching or pre-loading the dataset. Note that TFDS automatically caches small datasets (the following section has the details).

### 데이터세트 캐싱하기

다음은 이미지를 정규화한 후 데이터세트를 명시적으로 캐싱하는 데이터 파이프라인의 예입니다.

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

- [[1] Vectorizing mapping](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

이 데이터세트를 반복할 때 두 번째 반복은 캐싱 덕분에 첫 번째 반복보다 훨씬 빠릅니다.

### 자동 캐싱

By default, TFDS auto-caches (with `ds.cache()`) datasets which satisfy the following constraints:

- 총 데이터세트 크기(모든 분할)가 정의되고 &lt; 250MiB
- `shuffle_files`가 비활성화되었거나 단일 샤드만 읽을 때

`tfds.load`에서 `try_autocaching=False`를 `tfds.ReadConfig`에 전달하여 자동 캐싱을 거부할 수 있습니다. 특정 데이터세트가 자동 캐싱을 사용할지 여부를 확인하려면 데이터세트 카탈로그 설명서를 살펴보세요.

### 전체 데이터를 단일 텐서로 로드하기

데이터세트가 메모리에 저장하기 적합한 경우, 전체 데이터세트를 단일 텐서 또는 NumPy 배열로 로드할 수도 있습니다. `batch_size=-1`를 설정하여 모든 예제를 단일 `tf.Tensor`로 배치 처리할 수 ​​있습니다. 그런 다음, `tfds.as_numpy`를 사용하여 `tf.Tensor`에서 `np.array`로 변환합니다.

```python
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## 큰 데이터세트

Large datasets are sharded (split in multiple files) and typically do not fit in memory, so they should not be cached.

### 셔플링 및 훈련

During training, it's important to shuffle the data well - poorly shuffled data can result in lower training accuracy.

`ds.shuffle`을 사용하여 레코드를 셔플하는 것 외에도 `shuffle_files=True`를 설정하여 여러 파일로 샤딩된 큰 데이터세트에 좋은 셔플링 동작을 가져와야 합니다. 그렇지 않으면, epoch는 샤드를 같은 순서로 읽으므로 데이터가 실제로 무작위화되지는 않습니다.

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

Additionally, when `shuffle_files=True`, TFDS disables [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic), which may give a slight performance boost. To get deterministic shuffling, it is possible to opt-out of this feature with `tfds.ReadConfig`: either by setting `read_config.shuffle_seed` or overwriting `read_config.options.deterministic`.

### Auto-shard your data across workers (TF)

여러 사용자에 대해 훈련할 때 `tfds.ReadConfig`의 `input_context` 인수를 사용하면 각 사용자가 데이터의 하위 세트를 읽을 수 있습니다.

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

This is complementary to the subsplit API. First, the subplit API is applied: `train[:50%]` is converted into a list of files to read. Then, a `ds.shard()` op is applied on those files. For example, when using `train[:50%]` with `num_input_pipelines=2`, each of the 2 workers will read 1/4 of the data.

`shuffle_files=True`일 때, 파일은 한 작업자 내에서 셔플되지만 작업자 간에는 셔플되지 않습니다. 각 작업자는 Epoch 간에 파일의 같은 하위 세트를 읽습니다.

Note: When using `tf.distribute.Strategy`, the `input_context` can be automatically created with [distribute_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)

### Auto-shard your data across workers (Jax)

With Jax, you can use the `tfds.split_for_jax_process` or `tfds.even_splits` API to distribute your data across workers. See the [split API guide](https://www.tensorflow.org/datasets/splits).

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` is a simple alias for:

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### 빠른 이미지 디코딩

By default, TFDS automatically decodes images. However, there are cases where it can be more performant to skip the image decoding with `tfds.decode.SkipDecoding` and manually apply the `tf.io.decode_image` op:

- When filtering examples (with `tf.data.Dataset.filter`), to decode images after examples have been filtered.
- 이미지를 자를 때, 융합`tf.image.decode_and_crop_jpeg` op를 사용합니다.

두 예제 모두에 대한 코드는 [디코딩 가이드](https://www.tensorflow.org/datasets/decode#usage_examples)에서 사용할 수 있습니다.

### Skip unused features

If you're only using a subset of the features, it is possible to entirely skip some features. If your dataset has many unused features, not decoding them can significantly improve performances. See https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features.

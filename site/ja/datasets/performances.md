# パフォーマンスに関するヒント

This document provides TensorFlow Datasets (TFDS)-specific performance tips. Note that TFDS provides datasets as `tf.data.Dataset` objects, so the advice from the [`tf.data` guide](https://www.tensorflow.org/guide/data_performance#optimize_performance) still applies.

## データセットのベンチマークを作成する

`tfds.benchmark(ds)` を使用すると、あらゆる `tf.data.Dataset` オブジェクトのベンチマークを作成できます。

結果を正規化するために、`batch_size=` を必ず指定してください（100 iter/sec -&gt; 3200 ex/sec など）。これは、任意のイテラブル（`tfds.benchmark(tfds.as_numpy(ds))` など）で機能します。

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

### データセットのキャッシュ

次は、画像を正規化した後にデータセットを明示的にキャッシュするデータパイプラインの例です。

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

- [[1] マッピングのベクトル化](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

このデータセットをイテレートする際、キャッシュによって、2 回目のイテレーションは最初のイテレーションよりはるかに高速に行われます。

### 自動キャッシュ

By default, TFDS auto-caches (with `ds.cache()`) datasets which satisfy the following constraints:

- 合計データセットサイズ（全分割）が定義されており、250 MiB 未満である
- `shuffle_files` が無効化されているか、単一のシャードのみが読み取られる

`tfds.load` の `tfds.ReadConfig` に`try_autocaching=False` を渡して、自動キャッシュをオプトアウトすることができまます。特定のデータセットで自動キャッシュが使用されるかどうかを確認するには、データセットカタログドキュメントをご覧ください。

### 単一テンソルとしての全データの読み込み

データセットがメモリに収まる場合は、全データセットを単一のテンソルまたは Numpy 配列として読み込むこともできます。これは、`batch_size=-1` を設定して、単一の `tf.Tensor` 内のすべてのサンプルをバッチ処理し、`tfds.as_numpy` を使用して `tf.Tensor` から `np.array` に変換することで行えます。

```python
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## 大規模なデータセット

Large datasets are sharded (split in multiple files) and typically do not fit in memory, so they should not be cached.

### シャッフルとトレーニング

During training, it's important to shuffle the data well - poorly shuffled data can result in lower training accuracy.

`ds.shuffle` を使用してレコードをシャッフルするほかに、`shuffle_files=True` を設定して、複数のファイルシャーディングされている大規模なデータセット向けに、十分なシャッフル動作を得る必要があります。シャッフルが十分でない場合、エポックは、同じ順でシャードを読み取ってしまい、データが実質的にランダム化されません。

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

Additionally, when `shuffle_files=True`, TFDS disables [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic), which may give a slight performance boost. To get deterministic shuffling, it is possible to opt-out of this feature with `tfds.ReadConfig`: either by setting `read_config.shuffle_seed` or overwriting `read_config.options.deterministic`.

### ワーカー間でデータを自動シャーディングする（TF）

複数のワーカーでトレーニングを実施する場合、`tfds.ReadConfig` の `input_context` 引数を使用して、各ワーカーがデータのサブセットを読み取るようにすることができます。

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

`shuffle_files=True` である場合、ファイルは 1 つのワーカー内でシャッフルされますが、ワーカー全体ではシャッフルされません。各ワーカーはエポックごとにファイルの同じサブセットを読み取ります。

Note: When using `tf.distribute.Strategy`, the `input_context` can be automatically created with [distribute_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)

### ワーカー間でデータを自動シャーディングする（Jax）

Jax の場合、`tfds.split_for_jax_process` または `tfds.even_splits` API を使用すると、ワーカー間でデータを分散させることができます。<a>API の分割ガイド</a>をご覧ください。

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` は以下の単純なエイリアスです。

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### 画像のデコードの高速化

By default, TFDS automatically decodes images. However, there are cases where it can be more performant to skip the image decoding with `tfds.decode.SkipDecoding` and manually apply the `tf.io.decode_image` op:

- When filtering examples (with `tf.data.Dataset.filter`), to decode images after examples have been filtered.
- 画像をクロップする場合。結合された `tf.image.decode_and_crop_jpeg` 演算を使用します。

上記の両方の例のコードは、[decode ガイド](https://www.tensorflow.org/datasets/decode#usage_examples)をご覧ください。

### 未使用の特徴量をスキップする

特徴量のサブセットのみを使用している場合は、一部の特徴量を完全にスキップすることが可能です。データセットに未使用の特徴量が多数ある場合は、それらをデコードしないことでパフォーマンスが大幅に改善されます。https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features をご覧ください。

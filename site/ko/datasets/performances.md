# 성능 팁

이 문서는 TensorFlow 데이터세트(TFDS)에 특정한 성능 팁을 제공합니다. TFDS가 `tf.data.Dataset` 개체와 같은 데이터세트를 제공하므로 [`tf.data` 가이드](https://www.tensorflow.org/guide/data_performance#optimize_performance)의 지침이 여전히 적용됩니다.

## 벤치마크 데이터세트

`tfds.benchmark(ds)`를 사용하여 `tf.data.Dataset` 개체를 벤치마킹합니다.

결과를 정규화하기 위해(예: 100 iter/sec -&gt; 3200 ex/sec) `batch_size=`를 나타내야 합니다. 어떤 iterable에서든 작동합니다(예: `tfds.benchmark(tfds.as_numpy(ds))`).

```python
ds = tfds.load('mnist', split='train').batch(32).prefetch()
# Display some benchmark statistics
tfds.benchmark(ds, batch_size=32)
# Second iteration is much faster, due to auto-caching
tfds.benchmark(ds, batch_size=32)
```

## 작은 데이터세트(1GB 미만)

모든 TFDS 데이터세트는 디스크에 데이터를 [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord) 형식으로 저장합니다. 소규모 데이터세트(예: MNIST, CIFAR-10/-100)의 경우, `.tfrecord`에서 읽으면 상당한 오버헤드가 추가될 수 있습니다.

이러한 데이터세트가 메모리에 적합하므로 데이터세트를 캐싱 또는 사전 로드하여 성능을 크게 향상시킬 수 있습니다. TFDS는 작은 데이터세트를 자동으로 캐시합니다(자세한 내용은 다음 섹션 참조).

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

- [[1] 벡터화 매핑](https://www.tensorflow.org/guide/data_performance#vectorizing_mapping)

이 데이터세트를 반복할 때 두 번째 반복은 캐싱 덕분에 첫 번째 반복보다 훨씬 빠릅니다.

### 자동 캐싱

기본적으로 TFDS는 다음 제약 조건을 충족하는 데이터세트를 자동 캐시합니다(`ds.cache()` 이용).

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

큰 데이터세트는 샤딩되고(여러 파일로 분할) 일반적으로 메모리에 맞지 않으므로 캐시하지 않아야 합니다.

### 셔플링 및 훈련

훈련 중에는 데이터를 잘 섞는 것(셔플링)이 중요합니다. 데이터가 잘 섞이지 않으면 훈련 정확성이 떨어질 수 있습니다.

`ds.shuffle`을 사용하여 레코드를 셔플하는 것 외에도 `shuffle_files=True`를 설정하여 여러 파일로 샤딩된 큰 데이터세트에 좋은 셔플링 동작을 가져와야 합니다. 그렇지 않으면, epoch는 샤드를 같은 순서로 읽으므로 데이터가 실제로 무작위화되지는 않습니다.

```python
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

또한, `shuffle_files=True`일 때, TFDS는 [`options.deterministic`](https://www.tensorflow.org/api_docs/python/tf/data/Options#deterministic)를 비활성화하여 성능이 약간 향상될 수 있습니다. 결정론적인 셔플링을 얻으려면, `tfds.ReadConfig`를 사용하여 이 기능을 중지할 수 있습니다. 이를 위해 `read_config.shuffle_seed`를 설정하거나 `read_config.options.deterministic`를 덮어씁니다.

### 작업자 간에 데이터 자동 샤딩하기(TF)

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

이것은 하위 분할 API를 보완합니다. 먼저 하위 분할 API가 적용되고(`train[:50%]`이 읽을 파일의 목록으로 변환됨), `ds.shard()` op가 이러한 파일에 적용됩니다. 예를 들어, `num_input_pipelines=2`와 함께 `train[:50%]`를 사용하는 경우, 두 작업자 각각이 데이터의 1/4을 읽습니다.

`shuffle_files=True`일 때, 파일은 한 작업자 내에서 셔플되지만 작업자 간에는 셔플되지 않습니다. 각 작업자는 Epoch 간에 파일의 같은 하위 세트를 읽습니다.

참고: `tf.distribute.Strategy`를 사용하면 [distribute_datasets_from_function](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#distribute_datasets_from_function)과 함께 `input_context`를 자동으로 생성할 수 있습니다.

### 작업자 간에 데이터 자동 샤딩(Jax)

Jax를 사용하면 `tfds.split_for_jax_process` 또는 `tfds.even_splits` API를 사용하여 작업자 간에 데이터를 배포할 수 있습니다. [분할 API 가이드](https://www.tensorflow.org/datasets/splits)를 참조하세요.

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process`는 다음에 대한 간단한 별칭입니다.

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

### 빠른 이미지 디코딩

기본적으로 TFDS는 이미지를 자동으로 디코딩합니다. 그러나 `tfds.decode.SkipDecoding`으로 이미지 디코딩을 건너뛰고 `tf.io.decode_image` op를 수동으로 적용하는 것이 더 성능이 좋은 경우가 있습니다.

- 예를 필터링할 때(`tf.data.Dataset.filter` 사용) 예제가 필터링된 후 이미지를 디코딩합니다.
- 이미지를 자를 때, 융합`tf.image.decode_and_crop_jpeg` op를 사용합니다.

두 예제 모두에 대한 코드는 [디코딩 가이드](https://www.tensorflow.org/datasets/decode#usage_examples)에서 사용할 수 있습니다.

### 사용하지 않는 요소 건너뛰기

요소의 하위 집합만 사용하는 경우, 일부 요소를 완전히 건너뛸 수 있습니다. 데이터세트에 사용하지 않는 많은 요소가 있는 경우, 이를 디코딩하지 않으면 성능이 크게 향상될 수 있습니다. https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features를 참조하세요.

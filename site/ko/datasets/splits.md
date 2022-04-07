# 분할 및 조각화

모든 TFDS 데이터세트는 <a>카탈로그</a>에서 탐색할 수 있는 다양한 데이터 분할(예: `'train'`, <code>'test'</code>)을 노출합니다.

In addition of the "official" dataset splits, TFDS allow to select slice(s) of split(s) and various combinations.

## 조각화 API

슬라이싱 지침은 `split=` kwarg를 통해 `tfds.load` 또는 `tfds.DatasetBuilder.as_dataset`에 지정됩니다.

```python
ds = tfds.load('my_dataset', split='train[:75%]')
```

```python
builder = tfds.builder('my_dataset')
ds = builder.as_dataset(split='test+train[:75%]')
```

분할은 다음과 같을 수 있습니다.

- **일반 분할**(`'train'`, `'test'`): 분할 내의 모든 예가 선택됩니다.
- **슬라이스**: 슬라이스는 [python 슬라이스 표기](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)와 동일한 의미를 갖습니다. 슬라이스는 다음과 같을 수 있습니다.
    - **Absolute** (`'train[123:450]'`, `train[:4000]`): (see note below for caveat about read order)
    - **백분율**(`'train[:75%]'`, `'train[25%:75%]'`): 전체 데이터를 100개의 균일한 슬라이스로 나눕니다. 데이터를 100으로 나눌 수 없는 경우 일부 백분율에 추가 예가 포함될 수 있습니다.
    - **샤드**(`train[:4shard]`, `train[4shard]`): 요청된 샤드의 모든 예제를 선택합니다. (분할의 샤드 수를 얻으려면 `info.splits['train'].num_shards`을 참조하세요.)
- **분할의 합집합**(`'train+test'`, `'train[:25%]+test'`): 분할이 함께 인터리브됩니다.
- **전체 데이터세트**(`'all'`): `'all'`은 모든 분할의 합집합에 해당하는 특수 분할 이름입니다(`'train+test+...'`).
- **분할 목록**(`['train', 'test']`): 여러 `tf.data.Dataset`가 별도로 반환됩니다.

```python
# Returns both train and test split separately
train_ds, test_ds = tfds.load('mnist', split=['train', 'test[50%]'])
```

참고: 샤드가 [인터리브 처리](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#interleave)되기 때문에 하위 분할 간에 순서가 일치하지 않을 수 있습니다. 즉, `test[0:100]` 다음에 `test[100:200]`를 판독하면 `test[:200]`를 읽는 것과 다른 순서로 예제가 생성될 수 있습니다. TFDS가 예제를 읽는 순서를 이해하려면 [결정론 가이드](https://www.tensorflow.org/datasets/determinism#determinism_when_reading)를 참조하세요.

## `tfds.even_splits` 및 다중 호스트 훈련

`tfds.even_splits` generates a list of non-overlapping sub-splits of the same size.

```python
# Divide the dataset into 3 even parts, each containing 1/3 of the data
split0, split1, split2 = tfds.even_splits('train', n=3)

ds = tfds.load('my_dataset', split=split2)
```

This can be particularly useful when training in a distributed setting, where each host should receive a slice of the original data.

With `Jax`, this can be simplified even further using `tfds.split_for_jax_process`:

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

`tfds.even_splits`, `tfds.split_for_jax_process` accepts on any split value as input (e.g. `'train[75%:]+test'`)

## 슬라이싱 및 메타데이터

[데이터세트 정보](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)를 사용하여 분할/하위 분할(`num_examples`, `file_instructions`,...)에 대한 추가 정보를 얻을 수 있습니다.

```python
builder = tfds.builder('my_dataset')
builder.info.splits['train'].num_examples  # 10_000
builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
builder.info.splits.keys()  # ['train', 'test']
```

## 교차 검증

문자열 API를 사용한 10겹 교차 검증의 예:

```python
vals_ds = tfds.load('mnist', split=[
    f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
])
trains_ds = tfds.load('mnist', split=[
    f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
])
```

검증 데이터세트는 각각 10%가 됩니다: `[0%:10%]`, `[10%:20%]`, ..., `[90%:100%]`. 그리고 훈련 데이터세트는 각각 보완적인 90%가 됩니다: `[10%:100%]`(`[0%:10%]`의 해당 검증 세트에 대해), `[0%:10%]

- [20%:100%]`(for a validation set of `[10%:20%]`),...

## `tfds.core.ReadInstruction` 및 반올림

`str` 대신 `tfds.core.ReadInstruction`으로 분할을 전달할 수 있습니다.

예를 들어 `split = 'train[50%:75%] + test'`는 다음과 같습니다.

```python
split = (
    tfds.core.ReadInstruction(
        'train',
        from_=50,
        to=75,
        unit='%',
    )
    + tfds.core.ReadInstruction('test')
)
ds = tfds.load('my_dataset', split=split)
```

`unit`는 다음과 같을 수 있습니다.

- `abs`: 절대 슬라이싱
- `%`: 백분율 슬라이싱
- `shard`: 샤드 슬라이싱

`tfds.ReadInstruction`에는 반올림 인수도 있습니다. 데이터세트의 예제 수가 `100`으로 균등하게 나눠지지 않는 경우:

- `rounding='closest'`(기본값): 나머지 예제는 백분율로 분산되므로 일부 백분율에는 추가 예제가 포함될 수 있습니다.
- `rounding='pct1_dropremainder'`: 나머지 예제는 삭제되지만 모든 백분율에 정확히 동일한 수의 예제가 포함된다는 보장이 있습니다(예: `len(5%) == 5 * len(1%)`).

### 재현성 및 결정성

생성하는 동안 주어진 데이터세트 버전에 대해 TFDS는 예제가 디스크에서 결정성 있게 셔플되도록 보장합니다. 따라서 데이터세트를 두 번(두 대의 다른 컴퓨터에서) 생성해도 예제 순서는 변경되지 않습니다.

마찬가지로, subsplit API는 플랫폼, 아키텍처 등에 관계없이 항상 동일한 예제 `set`를 선택합니다. 즉, `set('train[:20%]') == set('train[:10%]') + set('train[10%:20%]')`입니다.

However, the order in which examples are read might **not** be deterministic. This depends on other parameters (e.g. whether `shuffle_files=True`).

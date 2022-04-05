# Split とスライス

すべての TFDS データセットは、さまざまなデータの Split（`'train'`、`'test'` など）を公開しており、[カタログ](https://www.tensorflow.org/datasets/catalog/overview)で閲覧することができるようになっています。

In addition of the "official" dataset splits, TFDS allow to select slice(s) of split(s) and various combinations.

## スライス API

スライスの指示は `tfds.load` または `tfds.DatasetBuilder.as_dataset` に `split=` kwarg を介して指定されます。

```python
ds = tfds.load('my_dataset', split='train[:75%]')
```

```python
builder = tfds.builder('my_dataset')
ds = builder.as_dataset(split='test+train[:75%]')
```

Split には次のものがあります。

- **プレーンな Split**（`'train'`、`'test'`）: 選択された Split 内のすべての Example。
- **スライス**: スライスのセマンティックは[Python のスライス表記法](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)と同じです。スライスには次のものがあります。
    - **Absolute** (`'train[123:450]'`, `train[:4000]`): (see note below for caveat about read order)
    - **パーセント率**（`'train[:75%]'`、`'train[25%:75%]'`）: 全データを 100 個の均一なスライスに分けます。データを 100 で割り切れない場合は、一部の 100 分の 1 のスライスに追加の Example が含まれる場合があります。
    - **シャード**（`train[:4shard]`、`train[4shard]`）: リクエストされたシャードのすべての Example を選択します。（Split のシャード数を取得するには、`info.splits['train'].num_shards` を確認します。）
- **Split の和集合**（`'train+test'`、`'train[:25%]+test'`）: Split は共にインターリーブされます。
- **Full dataset** (`'all'`): `'all'` is a special split name corresponding to the union of all splits (equivalent to `'train+test+...'`).
- **Split のリスト**（`['train', 'test']`）: 次のようにして、複数の `tf.data.Dataset` が個別に返されます。

```python
# Returns both train and test split separately
train_ds, test_ds = tfds.load('mnist', split=['train', 'test[50%]'])
```

Note: Due to the shards being [interleaved](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#interleave), order isn't guaranteed to be consistent between sub-splits. In other words reading `test[0:100]` followed by `test[100:200]` may yield examples in a different order than reading `test[:200]`. See [determinism guide](https://www.tensorflow.org/datasets/determinism#determinism_when_reading) to understand the order in which TFDS read examples.

## `tfds.even_splits` とマルチホストトレーニング

`tfds.even_splits` generates a list of non-overlapping sub-splits of the same size.

```python
# Divide the dataset into 3 even parts, each containing 1/3 of the data
split0, split1, split2 = tfds.even_splits('train', n=3)

ds = tfds.load('my_dataset', split=split2)
```

これは特に、ホストごとに元のデータのスライスを受け取る必要のある分散環境でのトレーニングに役立ちます。

`Jax` では、`tfds.split_for_jax_process` を使用してさらにこれを単純化できます。

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

`tfds.even_splits`、`tfds.split_for_jax_process` は任意の Split 値を入力として受け入れます（例: `'train[75%:]+test'`）。

## スライスとメタデータ

Split/サブスプリットに関する追加情報（`num_examples`、`file_instructions` など）は、[データセットの info](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata) を使って取得することができます。

```python
builder = tfds.builder('my_dataset')
builder.info.splits['train'].num_examples  # 10_000
builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
builder.info.splits.keys()  # ['train', 'test']
```

## クロス検証

文字列 API を使った 10 段階クロス検証の例:

```python
vals_ds = tfds.load('mnist', split=[
    f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
])
trains_ds = tfds.load('mnist', split=[
    f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
])
```

検証データセットは、`[0%:10%]`, `[10%:20%]`, ..., `[90%:100%]` というように、それぞれ 10% になります。また、トレーニングデータセットは、`[10%:100%]`（対応する検証セットの `[0%:10%]`）、`[0%:10%] というようにそれぞれ補完する 90% になります。

- [20%:100%]`（検証セットの `[10%:20%]`）,...

## `tfds.core.ReadInstruction` と四捨五入

Split は、`str` ではなく、`tfds.core.ReadInstruction` として渡すことが可能です。

たとえば、`split = 'train[50%:75%] + test'` は次と同等です。

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

`unit` は、次であることができます。

- `abs`: 全体的スライス
- `%`: パーセント率スライス
- `shard`: シャードスライス

`tfds.ReadInstruction` には四捨五入の引数もあります。データセットの Example 数が `100` で均等に割り切れない場合は、次のようになります。

- `rounding='closest'`（デフォルト）: 残りの Example は、パーセント率で配分され、一部のパーセントに追加の Example が含まれることがあります。
- `rounding='pct1_dropremainder'`: 残りの Example はドロップされますが、こうすることですべての 100 分の 1 スライスにまったく同じ数の Example が確実に含まれることになります（`len(5%) == 5 * len(1%)` など）。

### 再現可能性と決定性

生成中、特定のデータセットのバージョンにおいて、TFDS はExample が決定的にディスクでシャッフルされることを保証します。そのため、データセットを 2 回生成しても（2 台のコンピュータで）、Example の順序は変わりません。

同様に、サブスプリット API は必ず Example の同じ `set` を選択し、これにはプラットフォームやアーキテクチャなどは考慮されません。つまり、`set('train[:20%]') == set('train[:10%]') + set('train[10%:20%]')` となります。

ただし、Example が読み取られる順序は**決定的**ではない場合があります。これは他のパラメータ`shuffle_files=True` であるかどうか）に依存しています。

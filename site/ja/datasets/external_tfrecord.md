# TFDS を使って外部 tfrecord を読み込む

サードパーティツールで生成された`tf.train.Example` proto が（`.tfrecord`、`.riegeli` などの中に）あり、それを tfds API を使って直接読み込もうと考えているなら、このページが正解です。

`.tfrecord` ファイルを読み込むには、以下のみを行う必要があります。

- TFDS の命名規則に従うこと。
- tfrecord ファイルと共にメタデータファイル（`dataset_info.json`、`features.json`）を追加すること。

制限事項:

- `tf.train.SequenceExample` はサポートされていません。`tf.train.Example` のみがサポートされています。
- `tf.train.Example` を `tfds.features` に関して表現できる必要があります（以下のセクションをご覧ください）。

## ファイルの命名規則

TFDS は、ファイル名のテンプレートの定義をサポートし、様々なファイル命名スキームを使用できる柔軟性を提供します。テンプレートは `tfds.core.ShardedFileTemplate` で表現されており、`{DATASET}`、`{SPLIT}`、`{FILEFORMAT}`、`{SHARD_INDEX}`、`{NUM_SHARDS}`、および `{SHARD_X_OF_Y}` という変数を使用できるようになっています。たとえば、TFDS のデフォルトのファイル命名スキームは、`{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}` です。MNIST の場合、[ファイル名](https://console.cloud.google.com/storage/browser/tfds-data/datasets/mnist/3.0.1)は以下のようになります。

- `mnist-test.tfrecord-00000-of-00001`
- `mnist-train.tfrecord-00000-of-00001`

## メタデータを追加する

### 特徴量の構造を提供する

TFDS が `tf.train.Example` proto をデコードできるようにするには、仕様に一致する `tfds.features` 構造を指定する必要があります。以下に例を示します。

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

上記は、以下の `tf.train.Example` 仕様に対応しています。

```python
{
    'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    'objects/camera/K': tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64),
}
```

特徴量を指定すると、TFDS は画像や動画などを自動的にデコードできるようになります。その他のあらゆる TFDS データセットと同様に、特徴量メタデータ（ラベル名など）は、ユーザーに公開されます（例: `info.features['label'].names`）。

#### 生成パイプラインを制御している場合

TFDS 外部でデータセットを生成していても、生成パイプラインを制御している場合は、`tfds.features.FeatureConnector.serialize_example` を使用して、`dict[np.ndarray]` から `tf.train.Example` proto `bytes` にデータをエンコードできます。

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    writer.write(ex_bytes)
```

こうすることで、TFDS との互換性が確実にある特徴量を得られます。

同様に、`feature.deserialize_example` は proto をデコードするために存在しています（[例](https://www.tensorflow.org/datasets/features#serializedeserialize_to_proto)）。

#### 生成パイプラインを制御していない場合

`tfds.features` が `tf.train.Example` でどのように表現されているかは、colab で確認できます。

- `tfds.features` を人間が読み取れる `tf.train.Example` の構造に変換するには、`features.get_serialized_info()` を呼び出します。
- `tf.io.parse_single_example` に渡される正確な `FixedLenFeature` などの仕様を取得するには、`spec = features.tf_example_spec` を使用します。

注意: カスタム特徴量コネクタを使用している場合は、必ず `to_json_content`/`from_json_content` を実装し、`self.assertFeature` でテストしてください（[特徴量コネクタのガイド](https://www.tensorflow.org/datasets/features#create_your_own_tfdsfeaturesfeatureconnector)をご覧ください）。

### Split に関する統計を取得する

TFDS は、シャードごとの正確なサンプル数を知る必要があります。これは、`len(ds)` などの特徴量や、[subplit API](https://www.tensorflow.org/datasets/splits)（`split='train[75%:]'` など）で必要となります。

- この情報がある場合は、明示的に `tfds.core.SplitInfo` のリストを作成し、次のセクションに進みます。

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

- この情報がない場合は、`compute_split_info.py` スクリプトを使用して（または独自のスクリプトに `tfds.folder_dataset.compute_split_info` を使用して）その情報を計算できます。特定のディレクトリのすべてのシャードを読み取って情報を計算する beam パイプラインが起動します。

### メタデータファイルを追加する

データセットとともに適切なメタデータファイルを自動的に追加するには、`tfds.folder_dataset.write_metadata` を使用します。

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

データセットディレクトリに一度関数が呼び出されると、メタデータファイル（`dataset_info.json` など）が追加され、データセットを TFDS で読み込む準備が整います（次のセクションをご覧ください）。

## TFDS でデータセットを読み込む

### フォルダから直接読み込む

メタデータが生成されたら、`tfds.builder_from_directory` を使ってデータセットを読み込めます。これにより、標準的な TFDS API（`tfds.builder` など）で、`tfds.core.DatasetBuilder` が返されます。

```python
builder = tfds.builder_from_directory('~/path/to/my_dataset/3.0.0/')

# Metadata are available as usual
builder.info.splits['train'].num_examples

# Construct the tf.data.Dataset pipeline
ds = builder.as_dataset(split='train[75%:]')
for ex in ds:
  ...
```

### 複数のフォルダから直接読み込む

複数のフォルダからデータを読み込むことも可能です。たとえば強化学習において複数のエージェントがそれぞれに個別のデータセットを生成するときに、すべてをまとめて読み込む場合に使用できます。または、新しいデータセットが毎日など定期的に生成されている場合に、日付範囲を決めてデータを読み込むといったユースケースもあります。

複数のフォルダからデータを読み込むには、`tfds.builder_from_directories` を使用します。これは、標準的な TFDS API（`tfds.builder` など）で `tfds.core.DatasetBuilder` を返します。

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

注意: 各フォルダには独自のメタデータが必要です。メタデータには、Split に関すsる情報が含まれているためです。

### フォルダ構造（オプション）

TFDS との互換性を高めるには、`<data_dir>/<dataset_name>[/<dataset_config>]/<dataset_version>` の構造でデータを編成することができます。以下に例を示します。

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

これにより、`data_dir/` を指定するだけで、データセットに `tfds.load` / `tfds.builder` API との互換性を与えることができます。

```python
ds0 = tfds.load('dataset0', data_dir='data_dir/')
ds1 = tfds.load('dataset1/config0', data_dir='data_dir/')
```

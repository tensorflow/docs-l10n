# FeatureConnector

`tfds.features.FeatureConnector` API:

- 最終的な `tf.data.Dataset` の構造、形状、dtypes を定義します。
- ディスクとの間のシリアル化を抽象化します。
- 追加メタデータ（ラベル名、音声サンプルレートなど）を公開します。

## Overview

`tfds.features.FeatureConnector` は、データセットの特徴量の構造を定義します（`tfds.core.DatasetInfo` 内）:

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

特徴量は、テキストによる説明（`doc='description'`）を使用するか、`tfds.features.Documentation` を直接使用してさらに詳細な特徴量の説明を提供することで、文書化できます。

以下のような特徴量を含められます。

- スカラー値: `tf.bool`、`tf.string`、`tf.float32` など。特徴量を文書化する場合、`tfds.features.Scalar(tf.int64, doc='description')` も使用できます。
- `tfds.features.Audio`、`tfds.features.Video` など（使用可能な特徴量の[リスト](https://www.tensorflow.org/datasets/api_docs/python/tfds/features?version=nightly)をご覧ください）
- 特徴量のネストされた `dict`: `{'metadata': {'image': Image(), 'description': tf.string}}` など
- ネストされた `tfds.features.Sequence`: `Sequence({'image': ..., 'id': ...})`、 `Sequence(Sequence(tf.int64))` など

生成中、この例は `FeatureConnector.encode_example` によって自動的にディスクに最適なフォーマットにシリアル化されます（現在は `tf.train.Example` プロトコルバッファ）。

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

データセットを読み取る場合（`tfds.load` などを私用）、データは `FeatureConnector.decode_example` によって自動的にデコードされます。戻される `tf.data.Dataset` は、`tfds.core.DatasetInfo` に定義された `dict` 構造に一致します。

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

## proto のシリアル化と逆シリアル化

TFDS は、例を `tf.train.Example` proto にシリアル化/逆シリアル化するための低レベル API を公開します。

`dict[np.ndarray | Path | str | ...]` を proto `bytes` にシリアル化するには、`features.serialize_example` を使用します。

```python
with tf.io.TFRecordWriter('path/to/file.tfrecord') as writer:
  for ex in all_exs:
    ex_bytes = features.serialize_example(data)
    f.write(ex_bytes)
```

proto `bytes` を `tf.Tensor` に逆シリアル化するには、`features.deserialize_example` を使用します。

```python
ds = tf.data.TFRecordDataset('path/to/file.tfrecord')
ds = ds.map(features.deserialize_example)
```

## メタデータにアクセスする

特徴量メタデータ（ラベル名、形状、dtype など）にアクセスするには、[基礎ドキュメント](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)をご覧ください。以下に例を示します。

```python
ds, info = tfds.load(..., with_info=True)

info.features['label'].names  # ['cat', 'dog', ...]
info.features['label'].str2int('cat')  # 0
```

## 独自の `tfds.features.FeatureConnector` を作成する

[利用可能な特徴量](https://www.tensorflow.org/datasets/api_docs/python/tfds/features#classes)に特徴量が見当たらない場合は、[新しい課題](https://github.com/tensorflow/datasets/issues)を送信してください。

独自の特徴量コネクタを作成するには、`tfds.features.FeatureConnector` から継承し、抽象メソッドを実装する必要があります。

- 特徴量が単一のテンソル値である場合、`tfds.features.Tensor` から継承して、必要に応じて `super()` を使用するのが最善です。例については、`tfds.features.BBoxFeature` のソースコードをご覧ください。
- 特徴量が複数のテンソルのコンテナである場合、`tfds.features.FeaturesDict` から継承して、`super()` を使用して自動的にサブコネクタをエンコードするのが最善です。

The `tfds.features.FeatureConnector` object abstracts away how the feature is encoded on disk from how it is presented to the user. Below is a diagram showing the abstraction layers of the dataset and the transformation from the raw dataset files to the `tf.data.Dataset` object.

<p align="center">   <img src="dataset_layers.png" width="700" alt="DatasetBuilder 抽象化レイヤー"></p>

To create your own feature connector, subclass `tfds.features.FeatureConnector` and implement the abstract methods:

- `encode_example(data)`: ジェネレータ `_generate_examples()` に指定されたデータを `tf.train.Example` 対応データにエンコードする方法を定義します。単一の値、または複数の値の `dict` を返します。
- `decode_example(data)`: `tf.train.Example` から読み取られたテンソルから `tf.data.Dataset` が返すユーザーテンソルにデータをデコードする方法を定義します。
- `get_tensor_info()`: `tf.data.Dataset` によって返されたテンソルの形状/dtype を示します。別の `tfds.features` から継承する場合はオプションの場合があります。
- (optionally) `get_serialized_info()`: If the info returned by `get_tensor_info()` is different from how the data are actually written on disk, then you need to overwrite `get_serialized_info()` to match the specs of the `tf.train.Example`
- `to_json_content`/`from_json_content`: これは、元のソースコードなしでデータセットを読み込む場合に必須です。例については、[音声特徴量](https://github.com/tensorflow/datasets/blob/65a76cb53c8ff7f327a3749175bc4f8c12ff465e/tensorflow_datasets/core/features/audio_feature.py#L121)をご覧ください。

注意: 作成した特徴量コネクタは、`self.assertFeature` と `tfds.testing.FeatureExpectationItem` を使って必ずテストしてください。[テスト例](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features/image_feature_test.py)をご覧ください。

詳細については、`tfds.features.FeatureConnector` ドキュメントをご覧ください。[実際の例](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/core/features)を見ることもお勧めします。

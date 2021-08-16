# Transform TFX パイプラインコンポーネント

The Transform TFX pipeline component performs feature engineering on tf.Examples emitted from an [ExampleGen](examplegen.md) component, using a data schema created by a [SchemaGen](schemagen.md) component, and emits both a SavedModel as well as statistics on both pre-transform and post-transform data.  When executed, the SavedModel will accept tf.Examples emitted from an ExampleGen component and emit the transformed feature data.

- 消費: ExampleGen コンポーネントの tf.Examples、SchemaGen コンポーネントのデータスキーマ
- Emits: A SavedModel to a Trainer component, pre-transform and post-transform statistics.

## Transform コンポーネントを構成する

`preprocessing_fn` の記述が完了すると、Python モジュールで定義され、そのモジュールは入力として Transform コンポーネントに提供されます。このモジュールは Transform によって読み込まれ、Transform は `preprocessing_fn` という関数を検出して、プリプロセッシングパイプラインの構築に使用されます。

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

Additionally, you may wish to provide options to the [TFDV](tfdv.md)-based pre-transform or post-transform statistics computation. To do so, define a `stats_options_updater_fn` within the same module.

## Transform と TensorFlow Transform

Transform は、データセットに特徴量エンジニアリングを実施するために [TensorFlow Transform](tft.md) を多大に使用しています。TensorFlow Transform は特徴量データがトレーニングプロセスの一環としてモデルに送られる前に変換するための最適なツールです。以下に、一般的な特徴量変換の一部を示します。

- **埋め込み**: スパース特徴量（ボキャブラリが生成した整数型 ID など）を高次元空間から低次元空間への有意義なマッピングを見つけ出して密な特徴量に変換します。埋め込みの基礎について、[機械学習における埋め込みユニットのクラッシュコース](https://developers.google.com/machine-learning/crash-course/embedding)をご覧ください。
- **ボキャブラリ生成**: 文字列またはその他の非数値特徴量を、一意の値から ID 番号にマッピングするボキャブラリを作成して整数に変換します。
- **値の正規化**: 数値特徴量を類似する範囲内に収まるように変換します。
- **バケット化**: 連続した値の特徴量を、値を離散バケットに代入してカテゴリ特徴量に変換します。
- **テキスト特徴量の充実化**: トークン、n-gram、エンティティ、センチメントなどの生データから特徴量を生成して特徴量セットを充実化します。

TensorFlow Transform にはこれらのサポートや、その他多くの変換のサポートが備わっています。

- 最新のデータから自動的にボキャブラリを生成する

- データをモデルに送信する前に、任意の変換をデータに実施する。TensorFlow Transform は変換をモデルの TensorFlow グラフに組み込むため、トレーニング時と推論時に同じ変換が行われます。全トレーニングインスタンスの特徴量の最大値など、データのグローバルプロパティを参照する変換を定義することができます。

データは TFX を実行する前に任意に変換することができますが、TensorFlow Transform 内で行う場合、その変換は TensorFlow グラフの一部となります。この方法は、トレーニング/サービングスキューを回避する上で役立ちます。

モデリングコード内での変換には FeatureColumns が使用されます。FeatureColumns を使用すると、バケット化、事前定義済みのボキャブラリを使用する整数化、またはデータを確認せずに定義できるその他の変換を定義できます。

一方で TensorFlow Transform は全データを確認して、あらかじめわかっていない値を計算する必要のある変換を行うように設計されています。たとえば、ボキャブラリを生成するには、全データの確認が必要です。

注意: これらの計算は、内部的に [Apache Beam](https://beam.apache.org/) に実装されています。

Apache Beam を使った値の計算に加え、TensorFlow Transform ではユーザーがこれらの値を TensorFlow グラフに埋め込むことができます。このグラフはその後トレーニンググラフに読み込むことができます。たとえば、特徴量を正規化する際、`tft.scale_to_z_score` 関数によって特徴量の平均と標準偏差が計算され、TensorFlow グラフ内にある平均を減算して標準偏差で除算する関数の表現も計算されます。統計だけでなく TensorFlow グラフも出力することで、TensorFlow Transform はプリプロセシングパイプラインのオーサリングプロセスを単純化しています。

プリプロセッシングはグラフとして表現されているため、サーバーで発生することができ、トレーニングとサービング間の一貫性が保証されます。この一貫性により、トレーニング/サービングスキューの原因の 1 つを消し去られます。

TensorFlow Transform allows users to specify their preprocessing pipeline using TensorFlow code. This means that a pipeline is constructed in the same manner as a TensorFlow graph. If only TensorFlow ops were used in this graph, the pipeline would be a pure map that accepts batches of input and returns batches of output. Such a pipeline would be equivalent to placing this graph inside your `input_fn` when using the `tf.Estimator` API. In order to specify full-pass operations such as computing quantiles, TensorFlow Transform provides special functions called `analyzers` that appear like TensorFlow ops, but in fact specify a deferred computation that will be done by Apache Beam, and the output inserted into the graph as a constant. While an ordinary TensorFlow op will take a single batch as its input, perform some computation on just that batch and emit a batch, an `analyzer` will perform a global reduction (implemented in Apache Beam) over all batches and return the result.

通常の TensorFlow 演算と TensorFlow Transform analyzer を組み合わせると、データを事前処理する複雑なパイプラインを作成することができます。たとえば、`tft.scale_to_z_score` 関数は入力テンソルを取り、平均 `0` と分散 `1` を持つように正規化されたテンソルを返します。内部的には `mean` と `var` analyzer を呼び出してこれを実施しており、こうすることで入力テンソルの平均と分散に等しい定数がグラフ内に効果的に生成されます。その後 TensorFlow 演算を使用して、平均の減算と標準偏差による除算が行われます。

## TensorFlow Transform `preprocessing_fn`

TFX Transform コンポーネントは、データの読み書きに関連する API 呼び出しを処理し、出力の SavedModel をディスクに書き込むことで、Transform の使用方法を単純化しています。TFX ユーザーは、`preprocessing_fn` という 1 つの関数を定義するだけで良いのです。`preprocessing_fn` には、テンソルの入力 dict を操作してテンソルの出力 dict を生成する一連の関数を定義します。[TensorFlow Transform API](/tfx/transform/api_docs/python/tft) には scale_to_0_1 や compute_and_apply_vocabulary などのヘルパー関数がありますが、以下に示されるように通常の TensorFlow 関数を使用することができます。

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
```

### preprocessing_fn への入力を理解する

`preprocessing_fn` はテンソル（`Tensor` または `SparseTensor`）における一連の演算を記述するため、`preprocessing_fn` を正しく記述するにはデータがテンソルとしてどのように表現されているかを理解する必要があります。`preprocessing_fn` への入力はスキーマによって決定されます。`Schema` proto には `Feature` のリストが含まれており、Transform はこれらを「特徴量仕様」（「解析仕様」と呼ばれこともあります）に変換します。これは、特徴量名をキー、その値を `FixedLenFeature` または `VarLenFeature`（または TensorFlow Transform が使用していないその他のオプション）とする dict です。

`Schema` から特徴量仕様を推論する際のルールは次のとおりです。

- `shape` が設定されている `feature` はそれぞれ形状と `default_value=None` を持つ `tf.FixedLenFeature` になります。 `presence.min_fraction` は `1` である必要がありますが、そうでない場合はエラーが発生します。デフォルト値がない場合、`tf.FixedLenFeature` には必ず特徴量が必要であるためです。
- `shape` が設定されていない `feature` は、`VarLenFeature` になります。
- `sparse_feature` は `tf.SparseFeature` になります。その `size` と `is_sorted` は `SparseFeature` メッセージの `fixed_shape` と `is_sorted` フィールドによって決まります。
- `sparse_feature` の `index_feature` または `value_feature` として使用される特徴量には、特徴量仕様に生成される独自のエントリはありません。
- `feature` の `type` フィールド（または `sparse_feature` proto の値特徴量）と特徴量仕様の `dtype` の対応は以下の表のとおりです。

`type` | `dtype`
--- | ---
`schema_pb2.INT` | `tf.int64`
`schema_pb2.FLOAT` | `tf.float32`
`schema_pb2.BYTES` | `tf.string`

## TensorFlow Transform を使用して文字列のラベルを処理する

TensorFlow Transform は通常、ボキャブラリを生成し、そのボキャブラリを適用して文字列を整数に変換するために使用されます。このワークフローに従った場合、モデルに構築される `input_fn` は整数化された文字列を出力することになります。ただし、ラベルについては例外です。モデルが出力（整数）ラベルを文字列にマッピングするには、モデルはラベルの可能な値のリストとともに文字列のラベルを出力する `input_fn` を必要とします。たとえばラベルが `cat` と `dog` である場合、これらの生の文字列が `input_fn` の出力であり、キー `["cat", "dog"]` をパラメーターとして Estimator に渡す必要があります（詳細は以下を参照）。

文字列のラベルと整数のマッピングを処理するには、TensorFlow Transform を使用してボキャブラリを生成する必要があります。以下に、これを実演するコードスニペットを示します。

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

The preprocessing function above takes the raw input feature (which will also be returned as part of the output of the preprocessing function) and calls `tft.vocabulary` on it. This results in a vocabulary being generated for `education` that can be accessed in the model.

この例では、ラベルを変換してから、変換されたラベルのボキャブラリを生成する方法も示しています。具体的には、生のラベル `education` を取り、ラベルを整数に変換せずに、上位 5 つ（頻度別）のラベルを除くすべてのラベルを `UNKNOWN` に変換しています。

In the model code, the classifier must be given the vocabulary generated by `tft.vocabulary` as the `label_vocabulary` argument. This is done by first reading this vocabulary as a list with a helper function. This is shown in the snippet below. Note the example code uses the transformed label discussed above but here we show code for using the raw label.

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## Configuring pre-transform and post-transform statistics

As mentioned above, the Transform component invokes TFDV to compute both pre-transform and post-transform statistics. TFDV takes as input an optional [StatsOptions](https://github.com/tensorflow/datavalidation/blob/master/tensorflow_data_validation/statistics/stats_options.py) object. Users may wish to configure this object to enable certain additonal statistics (e.g. NLP statistics) or to set thresholds that are validated (e.g. min / max token frequency). To do so, define a `stats_options_updater_fn` in the module file.

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

Post-transform statistics often benefit from knowledge of the vocabulary being used for preprocessing a feature. The vocabulary name to path mapping is is provided to StatsOptions (and hence TFDV) for every TFT-generated vocabulary. Additionally, mappings for externally-created vocabularies can be added by either (i) directly modifying the `vocab_paths` dictionary within StatsOptions or by (ii) using `tft.annotate_asset`.

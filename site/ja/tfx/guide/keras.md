# TFX での TensorFlow 2.x

[TensorFlow 2.0 は 2019 年にリリース](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html)され、[Keras との緊密な統合](https://www.tensorflow.org/guide/keras/overview)、デフォルトでの [Eager execution](https://www.tensorflow.org/guide/eager)、[Python 式の関数の実行](https://www.tensorflow.org/guide/function) など、[新しい機能と改善](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes)を導入しました。

このガイドでは、TFX における TF 2.x の技術的な概要を総合的に説明します。

## どのバージョンを使用すればよいのか

TFX は TensorFlow 2.x と互換性があり、TensorFlow 1.x で存在していた高位 API（特に Estimator）も引き続き動作します。

### TensorFlow 2.x で新しいプロジェクトを開始する

TensorFlow 2.x は TensorFlow 1.x の高位機能を保持しているため、新しいプロジェクトで新機能を使用しないにしても古いバージョンを使用するメリットありません。

したがって、新しい TFX プロジェクトを開始する場合は、TensorFlow 2.x を使用することをお勧めします。後になって、Keras のフルサポートや新機能が利用できるようになったときにコードを更新する場合、TensorFlow 2.x で作成しているほうが、TensorFlow 1.x からアップグレードしようとするよりも、将来的に変更の範囲がはるかに狭まります。

### 既存のプロジェクトを TensorFlow 2.x に変換する

TensorFlow 1.x 向けに記述されたコードはほとんど TensorFlow 2.xと互換しており、TFX でも引き続き動作します。

ただし、TF 2.x で改善や新機能が利用できるようになったときにそれらを利用する場合は、[TF 2.x への移行手順](https://www.tensorflow.org/guide/migrate)を実施することができます。

## Estimator

Estimator API は TensorFlow 2.x でも保持されていますが、新しい機能と開発の焦点ではありません。TensorFlow 1.x または 2.x で Estimator を使用して記述されたコードは TFX でも引き続き期待通りに動作します。

[Taxi example (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py) は、純粋な Estimator を使用したエンドツーエンドの TFX の例です。

## `model_to_estimator` を使った Keras

Keras モデルは `tf.keras.estimator.model_to_estimator` 関数で囲むことができます。こうすると、Keras モデルが Estimator であるかのように動作させることができます。この関数を使用するには、次のように行います。

1. Keras モデルを構築します。
2. コンパイルされたモデルを `model_to_estimator` に渡します。
3. 通常の Estimator の使用と同じように、`model_to_estimator` の結果を Trainer に使用します。

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Trainer のユーザーモジュールファイルを除き、残りのパイプラインに変更はありません。

## ネイティブの Keras（`model_to_estimator` を使用しない Keras）

注意: Keras の全機能のフルサポートに現在取り組んでいます。ほとんどの場合、TFX 内の Keras は期待通りに機能しますが、FeatireColumns のスパース特徴量ではまだ動作しません。

### 例と Colab

以下は、ネイティブ Keras を使用したいくつかの例です。

- [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)（[モジュールファイル](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils.py)）: 「Hello world」のエンドツーエンドの例
- [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)（[モジュールファイル](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)）: 画像と TFLite のエンドツーエンドの例
- [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py)（[モジュールファイル](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)）: 高度な Transform を使用したエンドツーエンドの例

また、コンポーネント別の [Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras) もあります。

### TFX コンポーネント

以下のセクションでは、関連する TFX コンポーネントがどのようにネイティブ Keras をサポートしているのかを説明します。

#### Transform

Transform の Keras モデルサポートは、現在実験的です。

Transform コンポーネント自体は変更なしでネイティブ Keras に使用できます。`preprocessing_fn` の定義も同じままで、[TensorFlow](https://www.tensorflow.org/api_docs/python/tf) と [tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) 演算を使用します。

ネイティブ Keras のサービング関数と評価関数は変更されています。詳細は、以下の Trainer と Evaluator のセクションで説明します。

注意: `preprocessing_fn` 内での変換は、training または eval のラベル特徴量に適用することはできません。

#### Trainer

ネイディブ Keras を構成するには、Trainer の `GenericExecutor` を設定し、デフォルトの Estimator に基づく Executor を置き換える必要があります。詳細は[こちら](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor)をご覧ください。

##### Transform を使った Keras モジュールファイル

トレーニングモジュールファイルには `run_fn` が必要であり、これは、`GenericExecutor` によって呼び出されます。一般的な Keras `run_fn` は次のようになります。

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

上記の `run_fn` ではトレーニング済みのモデルをエクスポートする際に予測するための生の Example を取れるよう、サービングシグネチャが必要です。一般的なサービング関数は次のようになります。

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

上記のサービング関数では、[`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md) レイヤーを使用して、tf.Transform 変換を生データに適用して推論する必要があります。Estimator に必要であった以前の `_serving_input_receiver_fn` は Keras では使用する必要がなくなりました。

##### Transform を使用しない Keras モジュールファイル

これは上記に示したモジュールファイルに似ていますが、変換が使用されていません。

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

#####

[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

現時点では、TFX はシングルワーカーストラテジー（[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)、[OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy) など）のみをサポートしています。

分散型ストラテジーを使用するには、適切な tf.distribute.Strategy を作成し、作成したものと Keras モデルのコンパイルをストラテジーのスコープ内に移動します。

たとえば、上記の `model = _build_keras_model()` を次のように置き換えます。

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

`MirroredStrategy` が使用するデバイス（CPU/GPU）を確認するには、info レベルの TensorFlow ログ機能を有効にします。

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

すると、ログに `Using MirroredStrategy with devices (...)` が表示されるようになります。

注意: GPU のメモリ不足の問題では、環境変数の `TF_FORCE_GPU_ALLOW_GROWTH=true` が必要な場合があります。詳細は、[TensorFlow の GPU ガイド](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)をご覧ください。

#### Evaluator

TFMA v0.2x では、ModelValidator と Evaluator は 1 つの[新しい Evaluator コンポーネント](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md)に統合されました。新しい Evaluator コンポーネントは、単一モデルの評価だけでなく、現在のモデルを前のモデルと比較して検証することができます。この変更により、Pusher コンポーネントは ModelValidator ではなく Evaluator の blessing 結果を消費するようになりました。

新しい Evaluator は、Keras モデルだけでなく Estimator モデルもサポートしています。Evaluator がサービングに使用されるものと同じ `SavedModel` に基づくようになったため、以前に必要であった `_eval_input_receiver_fn` と eval Saved Model は Keras では必要でなくなっています。

[詳細は、Evaluator をご覧ください](evaluator.md)。

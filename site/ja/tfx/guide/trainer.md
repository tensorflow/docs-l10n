# Trainer TFX パイプラインコンポーネント

Trainer TFX パイプラインコンポーネントは TensorFlow モデルをトレーニングします。

## Trainer と TensorFlow

Trainer は、モデルのトレーニングに Python [TensorFlow](https://www.tensorflow.org) API を多大に使用しています。

注意: TFX は TensorFlow 1.15 と 2.x をサポートします。

## コンポーネント

Trainer は次を取り込みます。

- training と eval に使用される tf.Examples
- Trainer のロジックを定義するユーザー指定のモジュールファイル
- train args と eval args の [Protobuf](https://developers.google.com/protocol-buffers) 定義
- （オプション）SchemaGen パイプラインコンポーネントが作成し、開発者がオプションとして変更できるデータスキーマ
- （オプション）上流の Transform コンポーネントが生成する transform グラフ
- （オプション）warmstart などのシナリオに使用される事前トレーニング済みのモデル
- （オプション）ユーザーモジュール関数に渡されるハイパーパラメータ。Tuner との統合に関する詳細は、[こちら](tuner.md)をご覧ください。

Trainer の出力: 少なくとも 1 つの推論/サービング用モデル（通常 SavedModel 形式）とオプションとして eval 用のモデル（通常 EvalSavedModel）

[TFLite](https://www.tensorflow.org/lite) などの代替のモデル形式のサポートは [Model Rewriting ライブラリ](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md)を通じて提供しています。Estimator と Keras モデルの両方の変換方法の例については、Model Rewriting ライブラリへのリンクをご覧ください。

## 汎用 Trainer

汎用の Trainer を使用すると、開発者はあらゆる TensorFlow でもる API を Trainer コンポーネントと使用できるようになります。TensorFlow Estimator のほか、Keras モデルやカスタムトレーニングループを使用できます。詳細については、[汎用 Trainer 用の RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md)をご覧ください。

### Trainer コンポーネントを構成する

以下は、汎用 Trainer の一般的なパイプライン DSL コードの例です。

```python
from tfx.components import Trainer

...

trainer = Trainer(
    module_file=module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer は `module_file` パラメーターに指定されているトレーニングモジュールを呼び出します。`custom_executor_spec` に `GenericExecutor` が指定されている場合、モジュールファイルには `trainer_fn` の代わりに `run_fn` が必要です。`trainer_fn` はモデルの作成を行います。そのほか、`run_fn` はトレーニングの部分を処理し、トレーニング済みのモデルを [FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py) で指定された目的の場所に出力する必要もあります。

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

上記の [Example モジュールファイル](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)では `run_fn` を使用しています。

Transform コンポーネントがパイプラインで使用されていない場合、Trainer は直接 ExampleGen の Example を取るところに注意してください。

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

詳細については、[Trainer API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer)をご覧ください。

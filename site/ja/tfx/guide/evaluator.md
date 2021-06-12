# Evaluator TFX パイプラインコンポーネント

Evaluator TFX パイプラインコンポーネントは、モデルのトレーニングの結果を綿密に分析し、データのサブセットでモデルがどのようなパフォーマンスを示すかを理解しやすくするコンポーネントです。Evaluator はエクスポートされたモデルを検証することができるため、本番にプッシュする上で「十分な質」を備えていることを保証することができます。

検証が有効である場合、Evaluator は新しいモデルをベースライン（現在のサービングモデルなど）に比較して、ベースラインに比べて「十分に良い」かどうかを判断します。これには、両方のモデルを eval データセットで評価し、メトリクス（AUC、損失など）でそれらのパフォーマンスを計算するプロセスが伴います。新しいモデルのメトリクスがベースラインモデルと比較して開発者が指定する基準を満たしている場合（AUC がより低くないなど）、そのモデルは「blessed」であるため（良としてマークされるため）、[プッシャー](pusher.md)にモデルを本番にプッシュしてよいことが示されます。

- 入力:
    - [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) の eval split
    - [トレーナー](trainer.md)から得るトレーニング済みのモデル
    - 以前の blessed モデル（検証が実行される場合）
- 出力:
    - [ML メタデータ](mlmd.md)への分析結果
    - [ML メタデータ](mlmd.md)への検証結果（検証が実行される場合）

## Evaluator と TensorFlow Model Analysis

Evaluator は [TensorFlow Model Analysis](tfma.md) ライブラリを利用して分析を実行する代わりに、[Apache Beam](beam.md) を使用してスケーラブルな処理を行います。

## Evaluator コンポーネントを使用する

通常 Evaluator パイプラインコンポーネントは非常にデプロイしやすく、ほとんどカスタマイズする必要がありません。大部分の作業は Evaluator TFX コンポーネントが実行します。

Evaluator をセットアップするには、次の情報が必要となります。

- 構成するメトリクス（モデルと保存されている以外のメトリクスを追加する必要がある場合のみ）。詳細は、[Tensorflow Model Analysis のメトリクス](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md)をご覧ください。
- 構成するスライス（スライスが指定されていない場合は、デフォルトで「overall」スライスが追加されます）。詳細は、[Tensorflow Model Analysis のセットアップ](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md)をご覧ください。

検証が含まれる場合は、次の追加情報が必要になります。

- 比較対象のモデル（latest blessed など）
- 検証するモデル検証（しきい値）。詳細は、[Tensorflow Model Analysis モデル検証](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md)をご覧ください。

検証は有効である場合に定義されているすべてのメトリクスとスライスに対して実施されます。

次は、典型的なコードです。

```python
import tensorflow_model_analysis as tfma
...

# For TFMA evaluation

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name='eval' and
        # remove the label_key. Note, if using a TFLite model, then you must set
        # model_type='tf_lite'.
        tfma.ModelSpec(label_key='<label_key>')
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ])

# The following component is experimental and may change in the future. This is
# required to specify the latest blessed model will be used as the baseline.
model_resolver = Resolver(
      strategy_class=dsl.experimental.LatestBlessedModelStrategy,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

model_analyzer = Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```

Evaluator は [EvalResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult)（および検証が使用されている場合はオプションとして [ValidationResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/ValidationResult)）を生成するため、[TFMA](tfma.md) を使用して読み込むことができます。以下に、結果を Jupyter ノートブックに読み込む例を示します。

```
import tensorflow_model_analysis as tfma

output_path = evaluator.outputs['evaluation'].get()[0].uri

# Load the evaluation results.
eval_result = tfma.load_eval_result(output_path)

# Visualize the metrics and plots using tfma.view.render_slicing_metrics,
# tfma.view.render_plot, etc.
tfma.view.render_slicing_metrics(tfma_result)
...

# Load the validation results
validation_result = tfma.load_validation_result(output_path)
if not validation_result.validation_ok:
  ...
```

詳細については、[Evaluator API リファレンス](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Evaluator)をご覧ください。

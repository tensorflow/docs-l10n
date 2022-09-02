# Evaluator TFX 流水线组件

Evaluator TFX 流水线组件能够对模型的训练结果进行深入分析，以帮助您了解模型在数据子集上的表现。Evaluator 还能帮助您验证导出的模型，确保它们“足够好”，可以推送到生产环境。

启用验证后，Evaluator 会将新模型与基准（如当前应用的模型）进行比较，以确定它们相对于基准是否“足够好”。它通过在评估数据集上评估两个模型并根据指标（例如 AUC、损失）计算其性能来实现这一点。如果新模型的指标满足开发者指定的相对于基准模型的标准（例如，AUC 不低于标准），则模型会被“祝福”（标记为良好），向 [Pusher](pusher.md) 表示可以将该模型推送到生产环境。

- 使用：
    - 来自 [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) 的评估拆分
    - 来自 [Trainer](trainer.md) 的训练模型
    - 先前被祝福的模型（如果要执行验证）
- 发出：
    - [ML Metadata](mlmd.md) 的分析结果
    - [ML Metadata](mlmd.md) 的验证结果（如果要执行验证）

## Evaluator 和 TensorFlow Model Analysis

Evaluator 利用 [TensorFlow Model Analysis](tfma.md) 库执行分析，而分析又使用 [Apache Beam](beam.md) 进行可扩展处理。

## 使用 Evaluator 组件

Evaluator 流水线组件通常非常易于部署，而且几乎不需要自定义，因为所有工作均由 Evaluator TFX 组件完成。

设置 Evaluator 需要以下信息：

- 要配置的指标（仅在与模型一起保存的指标之外添加其他指标时需要）。有关更多信息，请参阅 [Tensorflow Model Analysis 指标](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md)。
- 要配置的切片（如果未提供切片，则会默认添加“整体”切片）。有关详情，请参阅 [Tensorflow Model Analysis 设置](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md)。

如果要包括验证，则需要以下附加信息：

- 要与之比较的模型（最新被祝福的模型等）。
- 要验证的模型验证（阈值）。有关更多信息，请参阅 [Tensorflow Model Analysis 模型验证](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md)。

启用后，将针对定义的所有指标和切片执行验证。

典型的代码如下所示：

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

Evaluator 会生成 [EvalResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult)（如果使用了验证，可以选择生成 [ValidationResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/ValidationResult)），可以使用 [TFMA](tfma.md) 加载它。下面是一个将结果加载到 Jupyter 笔记本中的示例：

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

有关更多详细信息，请参阅 [Evaluator API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Evaluator)。

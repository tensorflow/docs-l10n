# ModelValidator TFX 파이프라인 구성 요소(더 이상 사용되지 않음)

ModelValidator는 모델이 운영 환경에서 사용될 수 있을 만큼 좋은지 확인하는 데 사용되었습니다. 여전히 검증이 유용하다고 여겨지지만, 모델 [Evaluator](evaluator.md)가 이미 검증하려는 모든 메트릭을 계산했으므로 계산을 복제할 필요가 없도록 이 두 가지를 결합하기로 했습니다.

ModelValidator는 더 이상 사용되지 않으며 Modelvalidator를 사용하지 않는 것이 좋지만, 기존 ModelValidator 구성 요소를 유지해야 하는 경우 예제 구성은 다음과 같습니다.

```python
import tfx
import tensorflow_model_analysis as tfma
from tfx.components.model_validator.component import ModelValidator

...

model_validator = ModelValidator(
      examples=example_gen.outputs['output_data'],
      model=trainer.outputs['model'])
```

구성을 Evaluator로 마이그레이션하려는 경우 Evaluator에 대한 유사한 구성은 다음과 같습니다.

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
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

model_resolver = Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)
```

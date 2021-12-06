# Evaluator TFX 파이프라인 구성 요소

Evaluator TFX 파이프라인 구성 요소는 모델의 훈련 결과에 대한 심층 분석을 수행하여 모델이 데이터의 하위 세트에서 어떻게 동작하는지 이해하는 데 도움을 줍니다. 또한, Evaluator는 내보낸 모델을 검증하여 프로덕션 환경으로 푸시하기에 "충분히 적합한지" 확인하는 데도 도움을 줍니다.

검증이 활성화되면 Evaluator는 새 모델을 기준선(예: 현재 제공 중인 모델)과 비교하여 기준선에 비해 "충분히 좋은지" 확인합니다. 이를 위해, 평가 데이터세트에서 두 모델을 모두 평가하고 메트릭(예: AUC, 손실)의 성능을 계산합니다. 새 모델의 메트릭이 기준선 모델과 관련하여 개발자가 지정한 기준을 충족하면(예: AUC가 더 낮지 않음) 모델이 "탄생"(양호로 표시됨)하여 [Pusher](pusher.md)에 모델을 프로덕션 환경으로 푸시해도 괜찮음을 나타냅니다.

- 입력:
    - [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)의 평가 분할
    - [Trainer](trainer.md)의 훈련된 모델
    - 이전에 탄생한 모델(검증을 수행할 경우)
- 출력:
    - [ML 메타데이터](mlmd.md)에 대한 분석 결과
    - [ML 메타데이터](mlmd.md)에 대한 검증 결과(검증을 수행할 경우)

## Evaluator 및 TensorFlow 모델 분석

Evaluator는 [TensorFlow Model Analysis](tfma.md) 라이브러리를 활용하여 분석을 수행하고, 분석 과정에서 확장 가능한 처리를 위해 [Apache Beam](beam.md)이 사용됩니다.

## Evaluator 구성 요소 사용하기

Evaluator 파이프라인 구성 요소는 일반적으로 배포가 매우 쉽고 대부분의 작업이 Evaluator TFX 구성 요소에 의해 수행되므로 사용자 정의가 거의 필요하지 않습니다.

Evaluator를 설정하려면 다음 정보가 필요합니다.

- 구성할 메트릭(모델과 함께 저장된 메트릭 외에 다른 메트릭이 추가되는 경우에만 필요함). 자세한 내용은 [Tensorflow 모델 분석 메트릭](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md)을 참조하세요.
- 구성할 슬라이스(슬라이스가 제공되지 않은 경우, 기본적으로 "전체" 슬라이스가 추가됨). 자세한 내용은 [Tensorflow 모델 분석 설정](https://github.com/tensorflow/model-analysis/blob/master/g3doc/setup.md)을 참조하세요.

검증이 포함되는 경우, 다음 추가 정보가 필요합니다.

- 비교할 모델(최근 탄생 등)
- 확인할 모델 검증(임계값). 자세한 내용은 [Tensorflow 모델 분석 모델 검증](https://github.com/tensorflow/model-analysis/blob/master/g3doc/model_validations.md)를 참조하세요.

검증이 활성화되면 정의된 모든 메트릭과 조각을 대상으로 검증 작업이 수행됩니다.

일반적인 코드는 다음과 같습니다.

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

Evaluator는 [TFMA](tfma.md)를 사용하여 로드할 수 있는 [EvalResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult) (및 검증이 사용된 경우 선택적으로 [ValidationResult](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/ValidationResult))를 생성합니다. 다음은 결과를 Jupyter 노트북에 로드하는 방법을 보여주는 예입니다.

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

자세한 내용은 [Evaluator API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Evaluator)에서 확인할 수 있습니다.

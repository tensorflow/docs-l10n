# BulkInferrer TFX 파이프라인 구성 요소

BulkInferrer TFX 구성 요소는 레이블이 지정되지 않은 데이터에 대해 배치 추론을 수행합니다. 생성된 InferenceResult([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto))에는 원래 특성과 예측 결과가 포함됩니다.

BulkInferrer는 다음을 사용합니다.

- [SavedModel](https://www.tensorflow.org/guide/saved_model.md) 형식의 훈련된 모델
- 특성을 포함하는 레이블이 없는 tf.Examples
- (선택 사항) [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) 구성 요소의 검증 결과

BulkInferrer는 다음을 내보냅니다.

- [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## BulkInferrer 구성 요소 사용하기

BulkInferrer TFX 구성 요소는 레이블이 지정되지 않은 tf.Examples에서 배치 추론을 수행하는 데 사용됩니다. 일반적으로, 검증된 모델로 추론을 수행하기 위해 [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) 구성 요소 다음에 배포되거나, 내보낸 모델에서 직접 추론을 수행하기 위해 [Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) 구성 요소 다음에 배포됩니다.

현재는 메모리 내 모델 추론과 원격 추론을 수행합니다. 원격 추론을 수행하려면 모델을 Cloud AI Platform에서 호스팅해야 합니다.

일반적인 코드는 다음과 같습니다.

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

자세한 내용은 [BulkInferrer API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer)에서 확인할 수 있습니다.

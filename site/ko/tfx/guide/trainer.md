# Trainer TFX 파이프라인 구성 요소

Trainer TFX 파이프라인 구성 요소는 TensorFlow 모델을 훈련합니다.

## Trainer와 TensorFlow

Trainer는 모델 훈련을 위해 Python [TensorFlow](https://www.tensorflow.org) API를 광범위하게 사용합니다.

참고: TFX는 TensorFlow 1.15 및 2.x를 지원합니다.

## 구성 요소

Trainer는 다음을 사용합니다.

- 훈련 및 평가에 사용되는 tf.Examples
- Trainer 로직을 정의하는 사용자 제공 모듈 파일
- 훈련 인수 및 평가 인수의 [Protobuf](https://developers.google.com/protocol-buffers) 정의
- (Optional) A data schema created by a SchemaGen pipeline component and optionally altered by the developer.
- 업스트림 Transform 구성 요소에서 생성한 변환 그래프(선택 사항)
- 웜스타트와 같은 시나리오에 사용되는 사전 훈련된 모델(선택 사항)
- 사용자 모듈 함수에 전달될 하이퍼 매개변수(Tuner와의 통합에 대한 자세한 내용은 [여기](tuner.md)에서 찾을 수 있음)(선택 사항)

Trainer는 추론/서빙을 위한 하나 이상의 모델(일반적으로 SavedModelFormat에 있음)을 내보내거나, 평가를 위한 또 다른 모델(일반적으로 EvalSavedModel)을 선택적으로 내보냅니다.

[모델 재작성 라이브러리](https://www.tensorflow.org/lite)를 통해 [TFLite](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md)와 같은 대체 모델 형식에 대한 지원을 제공합니다. Estimator 및 Keras 모델을 모두 변환하는 방법에 대한 예제는 모델 재작성 라이브러리 링크를 참조하세요.

## 제네릭 Trainer

제네릭 Trainer를 사용하면 개발자가 Trainer 구성 요소와 함께 모든 TensorFlow 모델 API를 사용할 수 있습니다. TensorFlow Estimator 외에도 개발자는 Keras 모델 또는 사용자 정의 훈련 루프를 사용할 수 있습니다. 자세한 내용은 [제네릭 Trainer용 RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md)를 참조하세요.

### Trainer 구성 요소 구성하기

제네릭 Trainer의 일반적인 파이프라인 DSL 코드는 다음과 같습니다.

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

Trainer는 `module_file` 매개변수에 지정된 훈련 모듈을 호출합니다. `custom_executor_spec`에 `GenericExecutor`가 지정된 경우 모듈에 `trainer_fn` 대신 `run_fn`이 필요합니다. `trainer_fn`은 모델 생성을 담당했습니다. 이와 함께 `run_fn`은 훈련 부분을 처리하고 훈련된 모델을 [FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py)에서 지정한 원하는 위치로 출력해야 합니다.

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

Here is an [example module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py) with `run_fn`.

Transform 구성 요소가 파이프라인에서 사용되지 않으면 Trainer는 ExampleGen에서 직접 예제를 가져옵니다.

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

자세한 내용은 [Trainer API 참조](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer)에서 확인할 수 있습니다.

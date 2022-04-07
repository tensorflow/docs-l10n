# InfraValidator TFX 파이프라인 구성 요소

InfraValidator는 모델을 프로덕션 환경으로 푸시하기 전에 조기 경고 레이어로 사용되는 TFX 구성 요소입니다. "인프라 검증기(Infra Validator)"라는 이름은 "인프라"를 제공하는 실제 모델에서 모델을 검증한다는 사실에서 비롯되었습니다. [Evaluator](evaluator.md)가 모델의 성능을 보장한다면, InfraValidator는 모델이 기계적으로 정상인지 확인하고 잘못된 모델이 배포되는 것을 방지합니다.

## 동작하는 방식은?

InfraValidator는 모델을 가져와 모델과 함께 샌드박스 모델 서버를 시작한 다음, 모델이 성공적으로 로드되고 필요에 따라 쿼리될 수 있는지 확인합니다. 인프라 검증 결과는 [Evaluator](evaluator.md)의 경우와 마찬가지로 `blessing` 출력으로 생성됩니다.

InfraValidator는 모델 서버 바이너리(예: [TensorFlow Serving](serving.md))과 배포할 모델 간의 호환성에 중점을 둡니다. "인프라" 검증기라는 이름에도 불구하고 환경을 올바르게 구성하는 것은 **사용자의 책임**이며 인프라 검증기는 사용자가 구성한 환경의 모델 서버와 상호 작용하여 모델이 제대로 동작하는지만 확인합니다. 이 환경을 올바르게 구성하면 인프라 검증의 합격 또는 불합격 결과에 따라 실제 서비스 환경에서 모델이 제 기능을 수행할 것인지 여부를 판단할 수 있습니다. 구체적으로 이것은 다음을 의미합니다(모든 사항을 포함하지는 않음).

1. InfraValidator는 프로덕션 환경에서 사용되는 것과 같은 모델 서버 바이너리를 사용합니다. 이것은 인프라 검증 환경이 수렴해야 하는 최소 수준입니다.
2. InfraValidator는 실제 환경에 사용되는 것과 같은 리소스(예: 할당 수량 및 CPU 유형, 메모리 및 가속기)를 사용합니다.
3. InfraValidator는 실제 환경에 사용되는 것과 같은 모델 서버 구성을 사용합니다.

상황에 따라 사용자는 InfraValidator가 실제 환경과 동일해야 하는 정도를 선택할 수 있습니다. 기술적으로, 로컬 Docker 환경에서 인프라 검증을 마친 모델은 완전히 다른 환경(예: Kubernetes 클러스터)에서 문제 없이 제공될 수 있습니다. 그러나 InfraValidator는 이 차이를 확인하지 않습니다.

### 동작 모드

구성에 따라 인프라 검증은 다음 중 하나의 모드에서 수행됩니다.

- `LOAD_ONLY` 모드: 모델이 제공 인프라에 성공적으로 로드되었는지 여부를 확인합니다. **또는**
- `LOAD_AND_QUERY` 모드: `LOAD_ONLY` 모드에 더해 모델이 추론을 제공할 수 있는지 확인하기 위해 일부 샘플 요청을 보냅니다. InfraValidator는 예측이 정확했는지 여부는 따지지 않으며 요청의 성공 여부만 확인합니다.

## 사용하는 방법은?

일반적으로, InfraValidator는 Evaluator 구성 요소 옆에 정의되며 해당 출력은 Pusher에 공급됩니다. InfraValidator가 실패하면 모델이 푸시되지 않습니다.

```python
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=tfx.proto.EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

### InfraValidator 구성 요소 구성하기

InfraValidator를 구성하는 프로토타입에는 세 가지가 있습니다.

#### `ServingSpec`

`ServingSpec`은 InfraValidator에 가장 중요한 구성으로, 다음을 정의합니다.

- 실행할 모델 서버의 <u>유형</u>
- 실행 <u>위치</u>

우리가 지원하는 모델 서버 유형(제공 바이너리라고 함)의 경우

- [TensorFlow Serving](serving.md)

참고: InfraValidator에서는 모델 호환성에 영향을 주지 않고 모델 서버의 버전을 업그레이드하기 위해 동일한 모델 서버 유형을 여러 버전으로 지정할 수 있습니다. 예를 들어, 사용자는 `2.1.0` 및 `latest` 버전 모두에서 `tensorflow/serving` 이미지를 테스트하여 모델이 최신 `tensorflow/serving` 버전과도 호환되는지 확인할 수 있습니다.

현재, 지원되는 제공 플랫폼은 다음과 같습니다.

- 로컬 Docker(Docker가 미리 설치되어 있어야 함)
- Kubernetes(KubflowDagRunner만 제한적으로 지원)

제공 바이너리 및 제공 플랫폼에 대한 선택은 `ServingSpec`의 [`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof) 블록을 지정하여 이루어집니다. 예를 들어, Kubernetes 클러스터에서 실행되는 TensorFlow Serving 바이너리를 사용하려면 `tensorflow_serving` 및 `kubernetes` 필드를 설정해야 합니다.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(
        tensorflow_serving=tfx.proto.TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=tfx.proto.KubernetesConfig()
    )
)
```

`ServingSpec`을 추가로 구성하려면 [protobuf 정의](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)를 확인하세요.

#### `ValidationSpec`

인프라 검증 기준 또는 워크플로를 조정하기 위한 선택적 구성입니다.

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...),
    validation_spec=tfx.proto.ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

모든 ValidationSpec 필드에는 믿을 만한 기본값이 있습니다. [protobuf 정의](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)에서 자세한 내용을 확인하세요.

#### `RequestSpec`

`LOAD_AND_QUERY` 모드에서 인프라 검증을 실행할 때 샘플 요청을 빌드하는 방법을 지정하는 선택적 구성입니다. `LOAD_AND_QUERY` 모드를 사용하려면 구성 요소 정의에서 `request_spec` 실행 속성과 `examples` 입력 채널을 모두 지정해야 합니다.

```python
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=tfx.proto.ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
        local_docker=tfx.proto.LocalDockerConfig(),
    ),
    request_spec=tfx.proto.RequestSpec(
        # InfraValidator will look at how "classification" signature is defined
        # in the model, and automatically convert some samples from `examples`
        # artifact to prediction RPC requests.
        tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

### 워밍업이 있는 SavedModel 생성

(버전 0.30.0부터)

InfraValidator는 실제 요청으로 모델을 검증하기 때문에 이러한 검증 요청을 SavedModel의 [워밍업 요청](https://www.tensorflow.org/tfx/serving/saved_model_warmup)으로 쉽게 재사용할 수 있습니다. InfraValidator는 워밍업과 함께 SavedModel을 내보내기 위한 옵션(`RequestSpec.make_warmup`)을 제공합니다.

```python
infra_validator = InfraValidator(
    ...,
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)
```

그러면 출력 `InfraBlessing` 아티팩트에 워밍업이 있는 SavedModel이 포함되며 `Model` 아티팩트와 마찬가지로 <a>푸셔</a>에 의해 푸시될 수도 있습니다.

## 한계

현재 InfraValidator는 아직 완전하지 않으며 몇 가지 제한 사항이 있습니다.

- TensorFlow [SavedModel](/guide/saved_model) 모델 형식만 검증할 수 있습니다.

- TFX를 Kubernetes에서 실행하는 경우, Kubeflow Pipelines 내의 `KubeflowDagRunner`에서 파이프라인을 실행해야 합니다. 모델 서버는 Kubeflow가 사용하는 것과 같은 Kubernetes 클러스터 및 네임스페이스에서 시작됩니다.

- InfraValidator는 주로 [TensorFlow Serving](serving.md)에 대한 배포에 초점을 맞추고 있고 [TensorFlow Lite](/lite) 및 [TensorFlow.js](/js) 또는 기타 추론 프레임워크에 대한 배포에도 유용하기는 하지만 정확성이 떨어집니다.

- [Predict](/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def) 메서드 서명(TensorFlow 2에서 유일하게 내보낼 수 있는 메서드)에 대한 `LOAD_AND_QUERY` 모드는 제한적으로 지원됩니다. InfraValidator는 직렬화된 [`tf.Example`](/tutorials/load_data/tfrecord#tfexample)을 유일한 입력으로 사용하기 위해 Predict 서명이 필요합니다.

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    - [ Penguin example 예제](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local_infraval.py) 샘플 코드를 확인하여 이 서명이 TFX의 다른 구성 요소와 어떻게 상호 작용하는지 확인하세요.

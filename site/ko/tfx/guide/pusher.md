# Pusher TFX 파이프라인 구성 요소

Pusher 구성 요소는 검증된 모델을 모델 훈련 또는 재훈련 중에 [배포 대상](index.md#deployment_targets)에 푸시하는 데 사용됩니다. 배포 전에 Pusher는 다른 검증 구성 요소로부터 기준을 충족하는 하나 이상의 구성 요소를 사용하여 모델 푸시 여부를 결정합니다.

- [Evaluator](evaluator)는 훈련된 새로운 모델이 운영 환경에 푸시될 수 있을만큼 '충분한' 경우 모델을 축복합니다.
- [InfraValidator](infra_validator)는 모델이 운영 환경에서 기계적으로 서빙될 수 있는 경우 양호한 모델로 표시합니다(선택 사항이지만 권장됨).

Pusher 구성 요소는 [SavedModel](/guide/saved_model) 형식의 훈련된 모델을 사용하고, 버전 관리 메타데이터와 함께 같은 SavedModel을 생성합니다.

## Pusher 구성 요소 사용하기

Pusher 파이프라인 구성 요소는 일반적으로 배포가 매우 쉽고 모든 작업이 Pusher TFX 구성 요소로 수행되므로 사용자 정의가 거의 필요하지 않습니다. 일반적인 코드는 다음과 같습니다.

```python
from tfx import components

...

pusher = components.Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

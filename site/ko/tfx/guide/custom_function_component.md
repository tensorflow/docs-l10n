# 사용자 정의 Python 함수 구성 요소

참고: TFX 0.22부터 새로운 Python 함수 기반의 구성 요소 정의 스타일에 대한 실험적인 지원이 제공됩니다.

Python 함수 기반 구성 요소 정의를 사용하면 구성 요소 사양 클래스, 실행기 클래스 및 구성 요소 인터페이스 클래스를 정의하는 불편이 해소되어 TFX 사용자 정의 구성 요소를 더 쉽게 만들 수 있습니다. 이 구성 요소 정의 스타일에서 유형 힌트로 주석이 달린 함수를 작성합니다. 유형 힌트는 구성 요소의 입력 아티팩트, 출력 아티팩트 및 매개변수를 설명합니다.

이 스타일로 사용자 정의 구성 요소를 작성하는 작업은 다음 예제와 같이 매우 간단합니다.

```python
@component
def MyValidationComponent(
    model: InputArtifact[Model],
    blessing: OutputArtifact[Model],
    accuracy_threshold: Parameter[int] = 10,
    ) -> OutputDict(accuracy=float):
  '''My simple custom model validation component.'''

  accuracy = evaluate_model(model)
  if accuracy >= accuracy_threshold:
    write_output_blessing(blessing)

  return {
    'accuracy': accuracy
  }
```

TFX 파이프라인을 처음 사용하는 경우, [TFX 파이프라인의 핵심 개념에 대해 자세히 알아보세요](understanding_tfx_pipelines).

## 입력, 출력 및 매개변수

TFX에서 입력 및 출력은 기본 데이터와 관련된 메타데이터 속성 및 위치를 설명하는 아티팩트 객체로 추적됩니다. 이 정보는 ML 메타데이터에 저장됩니다. 아티팩트는 복잡한 데이터 유형 또는 단순 데이터 유형(예: 정수, 부동 소수점, 바이트 또는 유니코드 문자열)을 설명할 수 있습니다.

매개변수는 파이프라인 생성 시 알려진 구성 요소에 대한 인수(정수, 부동 소수점, 바이트 또는 유니코드 문자열)입니다. 매개변수는 훈련 반복 횟수, 드롭아웃 비율 및 기타 구성 요소와 같은 하이퍼 매개변수와 인수를 구성 요소에 지정하는 데 유용합니다. 매개변수는 ML 메타데이터에서 추적될 때 구성 요소 실행의 속성으로 저장됩니다.

참고: 현재, 출력 단순 데이터 유형 값은 실행 시 알 수 없으므로 매개변수로 사용할 수 없습니다. 마찬가지로, 입력 단순 데이터 유형 값은 현재, 파이프라인 구성 시 알려진 구체적인 값을 취할 수 없습니다. 향후 TFX 릴리스에서 이 제한이 없어질 수 있습니다.

## 정의

사용자 정의 구성 요소를 만들려면 사용자 정의 로직을 구현하는 함수를 작성하고 `tfx.dsl.component.experimental.decorators` 모듈의 [`@component` 데코레이터](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py){: .external }로 데코레이트합니다. 구성 요소의 입력 및 출력 스키마를 정의하려면 함수의 인수에 주석을 달고 `tfx.dsl.component.experimental.annotations` 모듈의 주석을 사용하여 값을 반환합니다.

- 각 **아티팩트 입력**에 대해 `InputArtifact[ArtifactType]` 유형 힌트 주석을 적용합니다. `ArtifactType`을 `tfx.types.Artifact`의 서브 클래스인 아티팩트 유형으로 대체합니다. 이러한 입력은 선택적 인수일 수 있습니다.

- 각 **출력 아티팩트**에 대해 `OutputArtifact[ArtifactType]` 유형 힌트 주석을 적용합니다. `ArtifactType`을 `tfx.types.Artifact`의 서브 클래스인 아티팩트 유형으로 대체합니다. 구성 요소 출력 아티팩트는 함수의 입력 인수로 전달되어야 구성 요소가 시스템 관리 위치에 출력을 쓰고 적절한 아티팩트 메타데이터 속성을 설정할 수 있습니다. 각 매개변수에 대해, 유형 힌트 주석 `Parameter[T]`을 사용합니다. `T`를 `int`, `float`, `str` 또는 `bytes`와 같은 매개변수 유형으로 대체합니다. 이 인수는 선택 사항이거나 기본값으로 정의할 수 있습니다.

- 파이프라인 생성 시 알려지지 않은 각 **단순 데이터 유형 입력**(`int`, `float`, `str` 또는 `bytes`)에 대해 유형 힌트 `T`를 사용합니다. TFX 0.22 릴리스에서는 이 유형의 입력에 대한 파이프라인 구성 시 구체적인 값을 전달할 수 없습니다(이전 섹션의 설명과 같이 `Parameter` 주석을 대신 사용). 이 인수는 선택 사항이거나 기본값으로 정의할 수 있습니다. 구성 요소에 단순 데이터 유형 출력(`int`, `float`, `str` 또는 `bytes`)이 있는 경우, `OutputDict` 인스턴스를 사용하여 이러한 출력을 반환할 수 있습니다. `OutputDict` 유형 힌트를 구성 요소의 반환 값으로 적용합니다.

- 각 **출력**에 대해 `<output_name>=<T>` 인수를 `OutputDict` 생성자에 추가합니다. 여기서 `<output_name>`은 출력 이름이고 `<T>`는 `int`, `float`, `str` 또는 `bytes`와 같은 출력 유형입니다.

함수 본문에서 입력 및 출력 아티팩트는 `tfx.types.Artifact` 객체로 전달됩니다. `.uri`를 검사하여 시스템 관리 위치를 얻고 속성을 읽고 설정할 수 있습니다. 입력 매개변수 및 단순 데이터 유형 입력은 지정된 유형의 객체로 전달됩니다. 단순 데이터 유형 출력은 사전으로 반환되어야 합니다. 여기서 키는 적절한 출력 이름이고 값은 원하는 반환 값입니다.

완성된 함수 구성 요소는 다음과 같습니다.

```python
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import Examples
from tfx.types.standard_artifacts import Model

@component
def MyTrainerComponent(
    training_data: InputArtifact[Examples],
    model: OutputArtifact[Model],
    dropout_hyperparameter: float,
    num_iterations: Parameter[int] = 10
    ) -> OutputDict(loss=float, accuracy=float):
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }

# Example usage in a pipeline graph definition:
# ...
trainer = MyTrainerComponent(
    examples=example_gen.outputs['examples'],
    dropout_hyperparameter=other_component.outputs['dropout'],
    num_iterations=1000)
pusher = Pusher(model=trainer.outputs['model'])
# ...
```

앞의 예에서는 `MyTrainerComponent`를 Python 함수 기반의 사용자 정의 구성 요소로 정의합니다. 이 구성 요소는 `examples` 아티팩트를 입력으로 사용하고 `model` 아티팩트를 출력으로 생성합니다. 이 구성 요소는 `artifact_instance.uri`를 사용하여 시스템 관리 위치에서 아티팩트를 읽거나 씁니다. 이 구성 요소는 `num_iterations` 입력 매개변수와 `dropout_hyperparameter` 단순 데이터 유형 값을 입력으로 받고 `loss` 및 `accuracy` 메트릭을 단순 데이터 유형 출력 값으로 출력합니다. 그러면 출력 `model` 아티팩트가 `Pusher` 구성 요소에서 사용됩니다.

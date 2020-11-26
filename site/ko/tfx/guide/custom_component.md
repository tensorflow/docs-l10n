# 완전 사용자 정의 구성 요소 빌드하기

이 가이드에서는 TFX API를 사용하여 사용자 정의 구성 요소를 원하는 대로 빌드하는 방법을 설명합니다. 완전 사용자 정의 구성 요소를 사용하면 구성 요소 사양, 실행기 및 구성 요소 인터페이스 클래스를 정의하여 구성 요소를 빌드할 수 있습니다. 이 접근 방식을 통해 필요에 맞게 표준 구성 요소를 재사용하고 확장할 수 있습니다.

TFX 파이프라인을 처음 사용하는 경우, [TFX 파이프라인의 핵심 개념에 대해 자세히 알아보세요](understanding_tfx_pipelines).

## 사용자 정의 실행기 또는 사용자 정의 구성 요소

구성 요소의 입력, 출력 및 실행 속성이 기존 구성 요소와 동일하지만 사용자 정의 처리 논리만 필요한 경우, 사용자 정의 실행기로 충분합니다. 입력, 출력 또는 실행 속성이 기존 TFX 구성 요소와 다른 경우에는 완전 사용자 정의 구성 요소가 필요합니다.

## 사용자 정의 구성 요소를 만드는 방법은?

완전 사용자 정의 구성 요소를 개발하려면 다음이 필요합니다.

- 새 구성 요소에 대해 정의된 입력 및 출력 아티팩트 사양 세트. 특히, 입력 아티팩트의 유형은 아티팩트를 생성하는 구성 요소의 출력 아티팩트 유형과 일치해야 하며, 출력 아티팩트의 유형은 아티팩트를 소비하는 구성 요소의 입력 아티팩트 유형과 일치해야 합니다.
- 새 구성 요소에 필요한, 아티팩트가 아닌 실행 매개변수

### ComponentSpec

`ComponentSpec` 클래스는 구성 요소 실행에 사용되는 매개변수뿐만 아니라 구성 요소에 대한 입력 및 출력 아티팩트를 정의하여 구성 요소 계약을 정의합니다. 여기에는 세 부분이 있습니다.

- *INPUTS*: 구성 요소 실행기에 있는 입력 아티팩트에 대한 형식 지정된 매개변수 사전입니다. 일반적으로, 입력 아티팩트는 업스트림 구성 요소의 출력이므로 같은 유형을 공유합니다.
- *OUTPUTS*: 구성 요소가 생성하는 출력 아티팩트에 대한 형식 지정된 매개변수 사전입니다.
- *PARAMETERS*: 구성 요소 실행기에 전달될 추가 [ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274) 항목의 사전입니다. 이들 사전은 파이프라인 DSL에서 유연하게 정의하고 실행으로 전달하려는 아티팩트가 아닌 매개변수입니다.

다음은 ComponentSpec의 예입니다.

```python
class HelloComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=Text),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }
```

### 실행기

다음으로, 새 구성 요소의 실행기 코드를 작성합니다. 기본적으로, `base_executor.BaseExecutor`의 새 서브 클래스는 `Do` 함수를 재정의하여 생성해야 합니다. `Do` 함수에서, 전달되는 `input_dict`, `output_dict` 및 `exec_properties` 함수는 ComponentSpec에서 정의되는 `INPUTS`, `OUTPUTS` 및 `PARAMETERS`에 각각 매핑됩니다. `exec_properties`의 경우, 값은 사전 조회를 통해 직접 가져올 수 있습니다. `input_dict` 및 `output_dict` 아티팩트의 경우, 아티팩트 인스턴스 또는 아티팩트 uri를 가져오는 데 사용할 수 있는 [artifact_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py) 클래스에서 제공되는 편리한 함수들이 있습니다.

```python
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = os.path.join(artifact.uri, split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
```

#### 사용자 정의 실행기 단위 테스트하기

사용자 정의 실행기에 대한 단위 테스트는 [여기서](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py)와 유시하게 만들 수 있습니다.

### 구성 요소 인터페이스

이제 가장 복잡한 부분이 완성되었으므로 다음 단계는 이러한 부분을 구성 요소 인터페이스로 어셈블링하여 구성 요소를 파이프라인에서 사용할 수 있도록 하는 것입니다. 다음 몇 가지 단계를 거칩니다.

- 구성 요소 인터페이스를 `base_component.BaseComponent`의 서브 클래스로 만듭니다.
- 이전에 정의된 `ComponentSpec` 클래스로 클래스 변수 `SPEC_CLASS`를 할당합니다.
- 이전에 정의된 Executor 클래스로 클래스 변수 `EXECUTOR_SPEC`을 할당합니다.
- 함수에 대한 인수를 사용하여 ComponentSpec 클래스의 인스턴스를 생성하고 선택적 이름과 함께 해당 값으로 수퍼 함수를 호출하는 방법으로 `__init__()` 생성자 함수를 정의합니다.

구성 요소의 인스턴스가 생성되면 전달된 인수가 `ComponentSpec` 클래스에 정의된 유형 정보와 호환되는지 확인하기 위해 `base_component.BaseComponent` 클래스의 유형 검사 논리가 호출됩니다.

```python
from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
```

### TFX 파이프라인으로 어셈블링하기

마지막 단계는 새 사용자 정의 구성 요소를 TFX 파이프라인으로 연결하는 것입니다. 새 구성 요소의 인스턴스를 추가하는 것 외에도 다음 사항이 필요합니다.

- 새 구성 요소의 업스트림 및 다운스트림 구성 요소를 올바르게 연결합니다. 이를 위해, 새 구성 요소에서 업스트림 구성 요소의 출력을 참조하고 다운스트림 구성 요소에서 새 구성 요소의 출력을 참조합니다.
- 파이프라인을 생성할 때 구성 요소 목록에 새 구성 요소 인스턴스를 추가합니다.

아래의 예는 앞서 언급한 변경 사항을 잘 보여줍니다. 전체 예제는 [TFX GitHub 리포지토리](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world)에서 찾을 수 있습니다.

```python
def _create_pipeline():
  ...
  example_gen = CsvExampleGen(input_base=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name='HelloWorld')
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])
  ...
  return pipeline.Pipeline(
      ...
      components=[example_gen, hello, statistics_gen, ...],
      ...
  )
```

## 완전 사용자 정의 구성 요소 배포하기

코드 변경 외에도 파이프라인을 올바르게 실행하려면 파이프라인 실행 환경에서 새로 추가된 모든 부분(`ComponentSpec`, `Executor`, 구성 요소 인터페이스)에 액세스할 수 있어야 합니다.

# 컨테이너 기반 구성 요소 빌드하기

컨테이너 기반 구성 요소는 Docker 컨테이너에서 해당 코드를 실행할 수만 있으면 모든 언어로 작성된 코드를 파이프라인에 통합할 수 있는 유연성을 제공합니다.

TFX 파이프라인을 처음 사용하는 경우, [TFX 파이프라인의 핵심 개념에 대해 자세히 알아보세요](understanding_tfx_pipelines).

## 컨테이너 기반 구성 요소 만들기

컨테이너 기반 구성 요소는 컨테이너화된 명령줄 프로그램의 지원을 받습니다. 이미 컨테이너 이미지가 있는 경우에는 [`create_container_component` 함수](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py){: .external}를 사용하여 입력 및 출력을 선언함으로써 TFX를 사용하여 이 이미지로부터 구성 요소를 만들 수 있습니다. 함수 매개변수는 다음과 같습니다.

- **name:** 구성 요소의 이름입니다.
- **inputs:** 입력 이름을 유형에 매핑하는 사전입니다. outputs: 출력 이름을 유형에 매핑하는 사전입니다. parameters: 매개변수 이름을 유형에 매핑하는 사전입니다.
- **image:** 컨테이너 이미지 이름 및 선택적으로 이미지 태그입니다.
- **command:** 컨테이너 진입점 명령줄입니다. 쉘 내에서 실행되지 않습니다. 명령줄은 컴파일 시 입력, 출력 또는 매개변수로 대체되는 자리 표시자 객체를 사용할 수 있습니다. 자리 표시자 객체는 [`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external}에서 가져올 수 있습니다. Jinja 템플릿은 지원되지 않습니다.

**Return value:** 파이프라인 내에서 인스턴스화하고 사용할 수 있는 base_component.BaseComponent에서 상속된 Component 클래스입니다.

### 자리 표시자

입력 또는 출력이 있는 구성 요소의 경우, `command`에 런타임 시 실제 데이터로 대체되는 자리 표시자가 필요한 경우가 많습니다. 이를 위해 몇 가지 자리 표시자가 제공됩니다.

- `InputValuePlaceholder`: 입력 아티팩트의 값에 대한 자리 표시자입니다. 런타임 시 이 자리 표시자는 아티팩트 값의 문자열 표현으로 대체됩니다.

- `InputUriPlaceholder`: 입력 아티팩트 인수의 URI에 대한 자리 표시자입니다. 런타임 시 이 자리 표시자는 입력 아티팩트 데이터의 URI로 대체됩니다.

- `OutputUriPlaceholder`: 출력 아티팩트 인수의 URI에 대한 자리 표시자입니다. 런타임 시 이 자리 표시자는 구성 요소가 출력 아티팩트의 데이터를 저장해야 하는 URI로 대체됩니다.

[TFX 구성 요소 명령줄 자리 표시자](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external}에 대해 자세히 알아보세요.

### 컨테이너 기반 구성 요소의 예

다음은 데이터를 다운로드, 변환 및 업로드하는 비 Python 구성 요소의 예입니다.

```python
import tfx.v1 as tfx

grep_component = tfx.dsl.components.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': tfx.standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': tfx.standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to Google Cloud Storage, so the
    # container image needs to have gsutil installed and configured.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$1"
          text_uri="$3"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          text_path=$(mktemp)
          filtered_text_uri="$5"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        '--pattern', tfx.dsl.placeholders.InputValuePlaceholder('pattern'),
        '--text', tfx.dsl.placeholders.InputUriPlaceholder('text'),
        '--filtered-text', tfx.dsl.placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)
```

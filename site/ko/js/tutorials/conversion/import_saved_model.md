# TensorFlow GraphDef 기반 모델을 TensorFlow.js로 가져오기

TensorFlow GraphDef 기반 모델(일반적으로 Python API를 통해 생성됨)은 다음 형식 중 하나로 저장할 수 있습니다.

1. TensorFlow [저장된 모델](https://www.tensorflow.org/tutorials/keras/save_and_load)
2. 고정 모델
3. [Tensorflow Hub 모듈](https://www.tensorflow.org/hub/)

위의 모든 형식은 [TensorFlow.js 변환기](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)에서 TensorFlow.js로 직접 로드할 수 있는 형식으로 변환하여 추론할 수 있습니다.

(참고: TensorFlow는 세션 번들 형식을 더 이상 사용하지 않습니다. 모델을 저장된 모델 형식으로 마이그레이션하세요.)

## 요구 사항

변환 절차에는 Python 환경이 필요한데 [pipenv](https://github.com/pypa/pipenv) 또는 [virtualenv](https://virtualenv.pypa.io)를 사용하여 격리를 유지해야 합니다. 변환기를 설치하려면 다음 명령을 실행하세요.

```bash
 pip install tensorflowjs
```

TensorFlow 모델을 TensorFlow.js로 가져오는 것은 2단계 프로세스입니다. 먼저 기존 모델을 TensorFlow.js 웹 형식으로 변환한 다음 TensorFlow.js로 로드합니다.

## 1단계: 기존 TensorFlow 모델을 TensorFlow.js 웹 형식으로 변환하기

pip 패키지에서 제공하는 변환기 스크립트를 실행합니다.

사용법: 저장된 모델의 예제는 다음과 같습니다.

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

고정 모델의 예제는 다음과 같습니다.

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Tensorflow Hub 모듈의 예제는 다음과 같습니다.

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

위치 인수 | 설명
--- | ---
`input_path` | 저장된 모델 디렉터리, 세션 번들 디렉터리, 고정 모델 파일, TensorFlow Hub 모듈 핸들 또는 경로의 전체 경로
`output_path` | 모든 출력 아티팩트의 경로

옵션 | 설명
--- | ---
`--input_format` | 입력 모델의 형식은 저장된 모델에 tf_saved_model, 고정 모델에 tf_frozen_model, 세션 번들에 tf_session_bundle, TensorFlow Hub 모듈에 tf_hub, Keras HDF5에 Keras를 사용합니다.
`--output_node_names` | 쉼표로 구분된 출력 노드의 이름입니다.
`--saved_model_tags` | 저장된 모델 변환과 로드할 MetaGraphDef의 태그에만 쉼표로 구분된 형식으로 적용됩니다. 기본적으로 `serve`가 됩니다.
`--signature_name` | TensorFlow Hub 모듈 변환, 로드할 서명에만 적용됩니다. 기본값은 `default`입니다. https://www.tensorflow.org/hub/common_signatures/를 참조하세요.

자세한 도움말 메시지를 보려면 다음 명령을 사용하세요.

```bash
tensorflowjs_converter --help
```

### 변환기 생성 파일

위의 변환 스크립트는 두 가지 유형의 파일을 생성합니다.

- `model.json`(데이터 흐름 그래프 및 가중치 매니페스트)
- `group1-shard\*of\*`(바이너리 가중치 파일 모음)

예를 들어 다음은 MobileNet v2를 변환한 결과입니다.

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## 2단계: 브라우저에서 로드하기 및 실행하기

1. tfjs-converter npm 패키지 설치하기

`yarn add @tensorflow/tfjs` 또는 `npm install @tensorflow/tfjs`

1. [FrozenModel 클래스](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts)를 인스턴스화하고 추론을 실행합니다.

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

[MobileNet 데모](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/demo/mobilenet)를 확인하세요.

`loadGraphModel` API는 요청과 함께 자격 증명 또는 사용자 정의 헤더를 보내는 데 사용할 수 있는 추가 `LoadOptions` 매개변수를 허용합니다. 자세한 내용은 [loadGraphModel() 설명서](https://js.tensorflow.org/api/1.0.0/#loadGraphModel)를 참조하세요.

## 지원되는 연산

현재 TensorFlow.js는 제한된 집합의 TensorFlow 연산만을 지원합니다. 모델이 지원되지 않는 연산을 사용하는 경우 `tensorflowjs_converter` 스크립트가 실패하고 모델에서 지원되지 않는 연산 목록을 출력합니다. 연산 지원이 실패된 경우 각 연산에 대한 [문제](https://github.com/tensorflow/tfjs/issues)를 제출하여 지원이 필요한 연산을 알려주세요.

## 가중치만 로드하기

가중치만 로드하려는 경우 다음 코드 조각을 사용할 수 있습니다.

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

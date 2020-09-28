# Keras 모델을 TensorFlow.js로 가져오기

Keras 모델(일반적으로 Python API를 통해 생성됨)은 [여러 형식 중 하나](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)로 저장할 수 있습니다. '전체 모델' 형식은 추론 또는 추가 훈련을 위해 TensorFlow.js에 직접 로드할 수 있는 TensorFlow.js Layer 형식으로 변환할 수 있습니다.

대상 TensorFlow.js Layer 형식은 `model.json` 파일과 바이너리 형식의 샤딩된 가중치 파일 집합이 포함된 디렉터리입니다. `model.json` 파일에는 모델 토폴로지('아키텍처' 또는 '그래프.' 즉, 레이어에 대한 설명 및 연결 방법)와 가중치 파일의 매니페스트가 모두 포함되어 있습니다.

## 요구 사항

변환 절차에는 Python 환경이 필요합니다. [pipenv](https://github.com/pypa/pipenv) 또는 [virtualenv를](https://virtualenv.pypa.io) 사용하여 격리된 것을 유지할 수 있습니다. 변환기를 설치하려면 <code>pip install tensorflowjs</code>를 사용하세요.

Keras 모델을 TensorFlow.js로 가져오는 것은 2단계 프로세스입니다. 먼저 기존 Keras 모델을 TF.js Layer 형식으로 변환한 다음 TensorFlow.js로 로드합니다.

## 1단계: 기존 Keras 모델을 TF.js Layer 형식으로 변환하기

Keras 모델은 일반적으로 `model.save(filepath)`를 통해 저장되며 모델 토폴로지와 가중치를 모두 포함하는 단일 HDF5(.h5) 파일을 생성합니다. 해당 파일을 TF.js Layer 형식으로 변환하려면 다음 명령을 실행합니다. 여기서 *`path/to/my_model.h5`*는 소스 Keras .h5 파일이고 *`path/to/tfjs_target_dir`*은 TF.js 파일의 대상 출력 디렉터리입니다.

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## 대체: Python API를 사용하여 TF.js Layer 형식으로 직접 내보내기

Python에 Keras 모델이 있는 경우 다음과 같이 TensorFlow.js Layer 형식으로 직접 내보낼 수 있습니다.

```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## 2단계: TensorFlow.js에 모델 로드하기

웹 서버를 사용하여 1단계에서 생성한 변환된 모델 파일을 제공합니다. JavaScript에서 파일 가져오기를 허용하려면 [Cross-Origin Resource Sharing(CORS)](https://enable-cors.org/)을 허용하는 서버를 구성해야 할 수도 있습니다.

그런 다음 model.json 파일에 URL을 제공하여 TensorFlow.js에 모델을 로드합니다.

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

이제 모델을 추론, 평가 또는 재훈련할 준비가 되었습니다. 예를 들어 로드된 모델을 즉시 사용하여 예측값을 낼 수 있습니다.

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

많은 [TensorFlow.js 예제](https://github.com/tensorflow/tfjs-examples)는 Google Cloud Storage에서 변환 및 호스팅하고 사전 훈련된 모델을 사용하는 접근법을 취합니다.

`model.json` 파일 이름을 사용하여 전체 모델을 참조합니다. `loadModel(...)`은 `model.json`를 가져온 다음 추가 HTTP(S) 요청을 수행하여 `model.json` 가중치 매니페스트에서 참조되는 분할된 가중치 파일을 가져옵니다. 이 접근 방식을 사용하면 `model.json` 및 가중치 분할 요소가 각각 일반적인 캐시 파일 크기 제한보다 작아서 모든 파일을 브라우저(및 인터넷의 추가 캐싱 서버)에서 캐시할 수 있습니다. 따라서 모델은 이후에 더 빨리 로드될 수 있습니다.

## 지원되는 특성

TensorFlow.js Layer는 현재 표준 Keras 구성을 사용하는 Keras 모델만 지원합니다. 지원되지 않는 연산 또는 레이어(예: 사용자 정의 레이어, 람다 레이어, 사용자 정의 손실 또는 사용자 정의 메트릭)를 사용하는 모델은 JavaScript로 안정적으로 변환할 수 없는 Python 코드에 의존하기 때문에 자동으로 가져올 수 없습니다.

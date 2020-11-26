# 모델 변환

TensorFlow.js에는 브라우저에서 즉시 사용할 수 있는 사전 훈련된 다양한 모델이 함께 제공됩니다. 이들 모델은 [모델 리포지토리](https://github.com/tensorflow/tfjs-models)에서 찾을 수 있습니다. 그러나 웹 애플리케이션에서 사용하려는 TensorFlow 모델을 다른 위치에서 찾았거나 작성했을 수도 있습니다. TensorFlow.js는 이러한 경우에 사용하기 위한 모델 [변환기](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)를 제공합니다. TensorFlow.js 변환기에는 두 가지 구성 요소가 있습니다.

1. TensorFlow.js에서 사용하기 위한 Keras 및 TensorFlow 모델을 변환하는 명령줄 유틸리티
2. TensorFlow.js를 사용하여 브라우저에서 모델을 로드하고 실행하기 위한 API

## 모델 변환하기

TensorFlow.js 변환기는 다양한 모델 형식에서 동작합니다.

**SavedModel**: TensorFlow 모델이 저장되는 기본 형식입니다. SavedModel 형식에 대한 설명은 [여기](https://www.tensorflow.org/guide/saved_model)에 나와 있습니다.

**Keras 모델**: Keras 모델은 일반적으로 HDF5 파일로 저장됩니다. Keras 모델 저장에 관한 자세한 내용은 [여기](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state)에서 확인할 수 있습니다.

**TensorFlow Hub 모듈**: 모델을 공유하고 찾기 위한 플랫폼인 TensorFlow Hub에서 배포하기 위해 패키지로 구성되는 모델입니다. 모델 라이브러리는 [여기](https://tfhub.dev/)에서 확인할 수 있습니다.

변환하려는 모델의 유형에 따라 변환기로 서로 다른 인수를 전달해야 합니다. 예를 들어, `model.h5`라는 Keras 모델을 `tmp/` 디렉토리에 저장했다고 가정하겠습니다. TensorFlow.js 변환기를 사용하여 모델을 변환하려면 다음 명령을 실행할 수 있습니다.

```
$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

그러면 `/tmp/model.h5`에 있는 모델이 변환되고 이진 가중치 파일과 함께 `model.json` 파일이 `tmp/tfjs_model/` 디렉토리에 출력됩니다.

여러 모델 형식에 해당하는 명령줄 인수에 대한 자세한 내용은 TensorFlow.js 변환기 [README](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)에서 찾을 수 있습니다.

변환 프로세스 중에 모델 그래프를 탐색하면서 각 연산이 TensorFlow.js에서 지원되는지 확인합니다. 지원이 되면 브라우저가 사용할 수 있는 형식으로 그래프를 작성합니다. 가중치를 4MB 파일로 샤딩하여 웹에서 제공할 목적에 맞게 모델을 최적화하려고 합니다. 이렇게 하면 브라우저에서 파일을 캐싱할 수 있습니다. 또한 오픈 소스 [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) 프로젝트를 사용하여 모델 그래프 자체를 단순화하려고 합니다. 그래프 단순화에는 인접한 연산을 함께 축소하거나 공통 하위 그래프를 제거하는 등이 포함됩니다. 이러한 변경은 모델의 출력에 영향을 주지 않습니다. 추가적인 최적화를 위해 모델을 특정 바이트 크기로 양자화하도록 변환기에 지시하는 인수를 전달할 수 있습니다. 양자화는 더 적은 비트로 가중치를 표시하여 모델 크기를 줄이는 기술입니다. 양자화 후 모델이 허용 가능한 정확성을 유지하도록 주의해야 합니다.

변환 중에 지원되지 않는 연산이 발생하면 프로세스가 실패하고 사용자가 알 수 있게 연산 이름을 출력합니다. [GitHub](https://github.com/tensorflow/tfjs/issues)에 문제를 제출하여 알려주세요. 사용자 요구에 따라 새로운 연산을 구현하도록 하겠습니다.

### 모범 사례

변환하는 동안 모델을 최적화하기 위해 최선을 다하고 있지만 리소스가 제한된 환경을 염두에 두고 모델을 빌드하는 것이 모델이 잘 동작하도록 하는 최선의 방법인 경우가 많습니다. 이는 지나치게 복잡한 아키텍처를 피하고 가능한 경우 매개변수(가중치) 수를 최소화하는 것을 의미합니다.

## 모델 실행하기

모델을 성공적으로 변환하면 가중치 파일 세트와 모델 토폴로지 파일이 생성됩니다. TensorFlow.js는 이들 모델 자산을 가져오고 브라우저에서 추론을 실행하는 데 사용할 수 있는 모델 로드 API를 제공합니다.

변환된 TensorFlow SavedModel 또는 TensorFlow Hub 모듈의 API는 다음과 같습니다.

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

변환된 Keras 모델은 다음과 같습니다.

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

`tf.loadGraphModel` API는 `tf.FrozenModel`을 반환합니다. 즉, 매개변수가 고정되어 새 데이터로 모델을 미세 조정할 수 없습니다. `tf.loadLayersModel` API는 훈련할 수 있는 tf.Model을 반환합니다. tf.Model 훈련 방법에 대해서는 [모델 훈련 가이드](train_models.md)를 참조하세요.

변환 후에 추론을 몇 번 실행하고 모델의 속도를 벤치마킹하는 것이 좋습니다. 이 목적으로 사용할 수 있는 독립형 벤치마킹 페이지가 있습니다(https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html). 여기서 초기 워밍업 실행의 측정값이 삭제된다는 사실을 알아챘을 수 있습니다. 이는 일반적으로 모델의 첫 번째 추론이 텍스처를 생성하고 셰이더를 컴파일하는 오버헤드로 인해 후속 추론보다 몇 배 더 느리기 때문입니다.

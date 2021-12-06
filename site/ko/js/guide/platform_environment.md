# 플랫폼과 환경

TensorFlow.js는 브라우저와 Node.js에서 동작하며 두 플랫폼에서 모두 사용 가능한 다양한 구성이 있습니다. 각 플랫폼에는 애플리케이션 개발 방식에 영향을 주는 고유한 고려 사항이 있습니다.

브라우저에서 TensorFlow.js는 모바일 기기와 데스크톱 기기를 지원합니다. 각 기기에는 사용 가능한 WebGL API처럼 자동으로 결정되고 구성되는 특정 제약 조건이 있습니다.

Node.js에서 TensorFlow.js는 TensorFlow API에 직접 바인딩하거나 더 느린 vanilla CPU 구현으로 실행하는 것을 지원합니다.

## Environments

TensorFlow.js 프로그램이 실행될 때 특정 구성을 환경이라고 합니다. 환경은 단일 글로벌 백엔드와 TensorFlow.js의 세분화된 특성을 제어하는 플래그 세트로 구성됩니다.

### Backends

TensorFlow.js는 텐서 저장소 및 수학 연산을 구현하는 여러 다중 백엔드를 지원합니다. 주어진 시간에 하나의 백엔드만 활성화됩니다. 대부분은 TensorFlow.js는 현재 환경에서 가장 적합한 백엔드를 자동으로 선택합니다. 그러나 때때로 사용 중인 백엔드와 해당 백엔드를 전환하는 방법을 아는 것도 중요합니다.

사용 중인 백엔드를 찾으려면 다음을 참고하세요.

```js
console.log(tf.getBackend());
```

백엔드를 수동으로 변경하려면 다음을 참고하세요.

```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```

#### WebGL backend

WebGL 백엔드인 'webgl'은 현재 브라우저용으로 가장 강력한 백엔드입니다. 이 백엔드는 vanilla CPU 백엔드보다 속도가 최대 100배 빠릅니다. 텐서는 WebGL 텍스처로 저장되고 수학 연산은 WebGL 셰이더에서 구현됩니다. 이 백엔드를 사용할 때 알아야 할 몇 가지 유용한 정보는 다음과 같습니다. \

##### UI 스레드 차단 방지하기

tf.matMul (a, b)와 같은 연산이 호출되면 결과 tf.Tensor가 동기식으로 반환되지만, 행렬식 곱셈 계산이 실제로 아직 준비되지 않았을 수 있습니다. 즉, 반환된 tf.Tensor는 계산에 대한 핸들일 뿐입니다. `x.data()` 또는 `x.array()`를 호출하면 계산이 실제로 완료된 후 값이 해결됩니다. 따라서 계산하는 동안 UI 스레드 차단을 방지하려면 동기식 `x.dataSync()`와 `x.arraySync()` 메서드 대신 비 동기 `x.data()`와 `x.array()` 메서드를 사용하는 것이 중요합니다.

##### Memory management

WebGL 백엔드를 사용할 때 한 가지 주의해야 할 점은 명시적인 메모리 관리가 필요하다는 것입니다. 텐서 데이터가 궁극적으로 저장되는 WebGLTextures는 브라우저에서 자동으로 가비지 모음이 되지 않습니다.

`tf.Tensor`의 메모리를 삭제하려면 `dispose()` 메서드를 사용할 수 있습니다.

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

애플리케이션에서 여러 연산을 함께 연결하는 것은 매우 일반적입니다. 이를 처리하기 위해 모든 중간 변수에 관한 참조를 유지하면 코드 가독성이 떨어질 수 있습니다. 해당 문제를 해결하기 위해 TensorFlow.js는 함수 실행 후 함수가 반환하지 않는 모든 `tf.Tensor`를 정리하는 `tf.tidy()` 메서드를 제공합니다. 이는 함수가 실행될 때 지역 변수가 정리되는 방식과 유사합니다.

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> 참고: 자동 가비지 모음 과정이 있는 비 webgl 환경(예: Node.js 또는 CPU 백엔드)에서 ` dispose()` 또는 `tidy()`를 사용하는 데 따르는 단점은 없습니다. 실제로 가비지 모음이 자연스럽게 일어나는 것보다 더 빨리 텐서 메모리를 비우는 것이 성능을 향상시키는 방법일 수 있습니다.

##### 정밀도

모바일 기기에서 WebGL은 16비트 부동 소수점 텍스처만 지원할 수 있습니다. 그러나 대부분의 머신러닝 모델은 32비트 부동 소수점 가중치 및 활성화로 훈련됩니다. 16비트 부동 숫자는 `[0.000000059605, 65504]` 범위의 숫자만 나타낼 수 있으므로 모바일 기기용 모델을 포팅할 때 정밀도 문제가 발생할 수 있습니다. 즉, 모델의 가중치와 활성화가 이 범위를 초과하지 않도록 주의해야 합니다. 기기가 32비트 텍스처를 지원하는지 확인하려면 `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')` 값을 확인하세요. 이 값이 거짓이면 기기는 16비트 부동 소수점 텍스처만 지원합니다. `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')`를 사용하여 TensorFlow.js가 현재 32비트 텍스처를 사용하고 있는지 확인할 수 있습니다.

##### Shader compilation &amp; texture uploads

TensorFlow.js는 WebGL 셰이더 프로그램을 실행하여 GPU에서 연산을 실행합니다. 이 셰이더는 사용자가 연산 실행을 요청할 때 느리게 어셈블되고 컴파일됩니다. 셰이더 컴파일은 메인 스레드의 CPU에서 발생하며 속도가 느릴 수 있습니다. TensorFlow.js는 컴파일된 셰이더를 자동으로 캐시해서 같은 형상의 입력 및 출력 텐서를 사용하여 같은 연산에 대한 두 번째 호출을 훨씬 빠르게 만듭니다. 일반적으로 TensorFlow.js 애플리케이션은 해당 애플리케이션 수명 동안 같은 연산을 여러 번 사용하므로 머신러닝 모델을 통한 두 번째 전달이 훨씬 빠릅니다.

TensorFlow.js는 또한 tf.Tensor 데이터를 WebGLTextures로 저장합니다. `tf.Tensor`가 생성될 때 GPU에 데이터를 즉시 업로드하지 않고 `tf.Tensor`가 연산에 사용될 때까지 데이터를 CPU에 보관합니다. `tf.Tensor`를 두 번째로 사용하는 경우 데이터가 이미 GPU에 있으므로 업로드 비용이 없습니다. 일반적인 머신러닝 모델에서 이는 모델을 통한 첫 번째 예측값 중에 가중치가 업로드되고 해당 모델을 통한 두 번째 전달이 훨씬 빠르다는 것을 의미합니다.

모델이나 TensorFlow.js 코드를 통한 첫 번째 예측의 성능에 관심이 있다면 실제 데이터를 사용하기 전에 같은 형상의 입력 텐서를 전달하여 모델을 워밍업하는 것이 좋습니다.

아래 예제를 봅시다.

```js
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);
```

#### Node.js TensorFlow backend

TensorFlow Node.js 백엔드 '노드'에서 TensorFlow C API는 연산을 가속하는 데 사용됩니다. 이때 가능한 경우 CUDA와 같은 컴퓨터의 사용 가능한 하드웨어 가속을 사용합니다.

이 백엔드에서는 WebGL 백엔드와 마찬가지로 연산이 `tf.Tensor`를 동기식으로 반환합니다. 그러나 WebGL 백엔드와 달리 텐서가 반환되기 전에 연산이 완료됩니다. `tf.matMul(a, b)` 대한 호출이 UI 스레드를 차단한다는 의미입니다.

따라서 운영 애플리케이션에서 사용하려면 작업자 스레드에서 TensorFlow.js를 실행하여 주 스레드를 차단하지 않아야 합니다.

Node.js에 대한 자세한 내용은 이 가이드를 참조하세요.

#### WASM backend

TensorFlow.js는 CPU 가속을 제공하고 vanilla JavaScript CPU(<code>cpu</code>) 및 WebGL 가속(`webgl`) 백엔드의 대안으로 사용할 수 있는 <a>WebAssembly 백엔드</a>(`wasm`)를 제공합니다. 사용 방법은 아래와 같습니다.

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

서버가 다른 경로나 다른 이름에서 `.wasm` 파일을 제공하는 경우 백엔드를 초기화하기 전에 `setWasmPath`를 사용합니다. 자세한 내용은 README의 ['번들러 사용'](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers) 섹션을 참조하세요.

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> 참고: TensorFlow.js는 각 백엔드의 우선순위를 정의하고 주어진 환경에 대해 가장 잘 지원되는 백엔드를 자동으로 선택합니다. WASM 백엔드를 명시적으로 사용하려면 `tf.setBackend('wasm')`을 호출해야 합니다.

##### Why WASM?

[WASM](https://webassembly.org/)은 2015년 새로운 웹 기반 바이너리 형식으로 도입되어 웹에서 실행하기 위한 컴파일 대상인 JavaScript, C, C++ 등으로 작성된 프로그램을 제공합니다. WASM는 Chrome, Safari, Firefox, 및 Edge에서 2017년부터 [지원되고 있으며](https://webassembly.org/roadmap/) [기기의 90%](https://caniuse.com/#feat=wasm)가 전 세계적으로 지원됩니다.

**성능**

WASM 백엔드는 신경망 연산의 최적화된 구현을 위해 [XNNPACK 라이브러리](https://github.com/google/XNNPACK)를 활용합니다.

*JavaScript와 비교* 시: WASM 바이너리는 브라우저가 로드, 구문 분석 및 실행하는 속도가 JavaScript 번들보다 일반적으로 훨씬 빠릅니다. JavaScript는 동적으로 형식화되고 가비지 모음이 되므로 런타임 속도가 저하될 수 있습니다.

*WebGL과 비교* 시: WebGL은 대부분의 모델에서 WASM보다 빠르지만 작은 모델의 경우 WASM은 WebGL 셰이더를 실행하는 고정된 오버헤드 비용으로 인해 WebGL을 능가할 수 있습니다. 아래의 'WASM을 사용해야 하는 경우' 섹션에서는 이러한 결정을 내리기 위한 휴리스틱에 대해 설명합니다.

**이식성과 안정성**

WASM에는 이식성 32비트 부동 산술 연산이 있어 모든 기기에서 정밀한 패리티를 제공합니다. 반면 WebGL은 하드웨어에 따라 다르며 기기마다 정밀도가 다를 수 있습니다(예: iOS 기기에서 16비트 부동 소수점으로 대체).

WebGL과 마찬가지로 WASM은 모든 주요 브라우저에서 공식적으로 지원됩니다. WebGL과 달리 WASM은 Node.js에서 실행될 수 있으며 네이티브 라이브러리를 컴파일할 필요없이 서버 측에서 사용할 수 있습니다.

##### WASM은 언제 사용해야 하나요?

**Model size and computational demand**

일반적으로 WASM은 모델이 작은 편이거나 WebGL 지원(` OES_texture_float` 익스텐션)이 부족하거나 GPU가 덜 강력한 저가형 기기에 관심이 있을 때 좋은 선택입니다. 아래 차트는 WebGL, WASM 및 CPU 백엔드에서 공식적으로 지원되는 5개의 [모델](https://github.com/tensorflow/tfjs-models)에 대한 2018 MacBook Pro의 Chrome에서 추론 시간(TensorFlow.js 1.5.2 기준)을 보여줍니다.

**소규모 모델**

Model | WebGL | WASM | CPU | 메모리
--- | --- | --- | --- | ---
BlazeFace | 22.5 ms | 15.6 ms | 315.2 ms | 0.4MB
FaceMesh | 19.3 ms | 19.2 ms | 335 ms | 2.8MB

**대규모 모델**

Model | WebGL | WASM | CPU | 메모리
--- | --- | --- | --- | ---
PoseNet | 42.5 ms | 173.9 ms | 1514.7 ms | 4.5MB
BodyPix | 77 ms | 188.4 ms | 2683 ms | 4.6MB
MobileNet v2 | 37 ms | 94 ms | 923.6 ms | 13 MB

위의 표는 WASM이 모델 전체에서 일반 JS CPU 백엔드보다 10배에서 30배 더 빠르며, 가볍지만(400KB) 적절한 수(~140)의 연산을 수행하는 [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface)와 같은 작은 모델에 대해 WebGL과 경쟁한다는 것을 보여줍니다. WebGL 프로그램은 연산 실행당 고정된 오버헤드 비용이 발생한다는 점을 고려할 때 BlazeFace와 같은 모델이 WASM에서 더 빠른 이유가 설명됩니다.

**결과는 기기에 따라 다릅니다. WASM이 애플리케이션에 적합한지 확인하는 가장 좋은 방법은 다른 백엔드에서 테스트해보는 것입니다.**

##### Inference vs Training

사전 훈련된 모델 배포의 주요 사용 사례를 다루기 위해 WASM 백엔드 개발은 *훈련* 지원보다 *추론*을 우선시합니다. WASM에서 지원되는 연산의 [최신 목록](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts)을 확인하고 모델에 지원되지 않는 연산이 있는 경우 [팀에게 알려주세요](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc). 훈련 모델의 경우는 Node(TensorFlow C++) 백엔드 또는 WebGL 백엔드를 사용하는 것이 좋습니다.

#### CPU backend

CPU 백엔드 'cpu'는 성능이 가장 떨어지는 백엔드이지만 사용하기가 가장 간단합니다. 연산은 모두 vanilla JavaScript로 구현되어 병렬화 가능성이 낮습니다. 또한 UI 스레드를 차단합니다.

이 백엔드는 테스트 또는 WebGL을 사용할 수 없는 기기에서 매우 유용할 수 있습니다.

### Flags

TensorFlow.js에는 자동으로 평가되고 현재 플랫폼에서 최상의 구성을 결정하는 일련의 환경 플래그가 있습니다. 이러한 플래그는 대부분 내부용이지만 일부 글로벌 플래그는 공용 API로 제어될 수 있습니다.

- `tf.enableProdMode():`을 위해 모델 검증, NaN 검사 및 기타 정확성 검사를 제거하는 운영 모드를 활성화합니다.
- `tf.enableDebugMode()`: 디버그 모드를 활성화하여 실행되는 모든 연산은 물론 메모리 공간 및 총 커널 실행 시간과 같은 런타임 성능 정보를 콘솔에 기록합니다. 이렇게 하면 애플리케이션이 크게 느려지므로 운영 환경에서 사용하지 않는 것이 좋습니다.

> 참고: 이 두 메서드는 캐시될 다른 플래그의 값에 영향을 미치므로 TensorFlow.js 코드를 사용하기 전에 먼저 사용되어야 합니다. 같은 이유로 '비 활성화' 아날로그 함수가 존재하지 않습니다.

> 참고: 콘솔에 `tf.ENV.features`를 로깅하여 평가된 모든 플래그를 볼 수 있습니다. 이는 **공개 API의 일부가 아니므로**(따라서 버전 간 안정성이 보장되지 않음) 플랫폼 및 장치에서 동작을 디버깅하거나 미세 조정하는 데 유용할 수 있습니다. `tf.ENV.set`을 사용하여 플래그 값을 재정의할 수 있습니다.

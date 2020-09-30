# Keras 사용자를 위한 TensorFlow.js Layers API

TensorFlow.js의 Layers API는 Keras를 모델로 하며, JavaScript와 Python의 차이점을 고려하여 합리적 수준에서 최대한 [Layers API](https://js.tensorflow.org/api/latest/)를 Keras와 유사하게 만들려고 노력하고 있습니다. 이를 통해 Python에서 Keras 모델을 개발한 경험이 있는 사용자가 JavaScript의 TensorFlow.js Layers로 보다 쉽게 마이그레이션할 수 있습니다. 예를 들어, 다음 Keras 코드는 아래의 JavaScript로 변환됩니다.

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

그러나 이 문서에서 설명하고자 하는 몇 가지 차이점이 있습니다. 이러한 차이점과 그 이면의 근거를 이해하면 Python에서 JavaScript로의 마이그레이션(또는 그 반대)이 비교적 원활할 것입니다.

## 생성자가 JavaScript 객체를 구성으로 사용

위의 예에서 다음 Python 및 JavaScript 줄을 비교합니다. 둘 모두 [Dense](https://keras.io/layers/core/#dense) 레이어를 만듭니다.

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

JavaScript 함수에는 Python 함수의 키워드 인수에 해당하는 요소가 없습니다. 생성자 옵션을 JavaScript에서 위치 인수로 구현하는 것은 피하는 것이 좋은데, 키워드 인수가 많은 생성자의 경우(예: [LSTM](https://keras.io/layers/recurrent/#lstm))에 이러한 인수를 기억하고 사용하기가 상당히 번거로울 것이기 때문입니다. JavaScript 구성 객체를 사용하는 이유가 여기에 있습니다. 이러한 객체는 Python 키워드 인수와 동일한 수준의 위치 불변성과 유연성을 제공합니다.

Model 클래스의 일부 메서드(예: [`Model.compile()`](https://keras.io/models/model/#model-class-api))도 JavaScript 구성 객체를 입력으로 사용합니다. 그러나, `Model.fit()`, `Model.evaluate()` 및 `Model.predict()`는 약간 다르다는 점을 알고 있어야 합니다. 이러한 메서드는 의무적인 `x`(특성) 및 `y`(레이블 또는 대상) 데이터를 입력으로 사용하기 때문입니다. `x` 및 `y`는 키워드 인수의 역할을 수행하는 후속 구성 객체와는 별도의 위치 인수입니다.

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit()은 비동기로 동작

`Model.fit()`은 사용자가 TensorFlow.js에서 모델 훈련을 수행할 때 이용되는 기본 메서드입니다. 이 메서드는 종종 수 초에서 수 분까지 장시간 실행될 수 있습니다. 따라서 JavaScript 언어의 `async` 기능을 활용하면 브라우저에서 실행할 때 메인 UI 스레드를 차단하지 않는 방식으로 이 함수를 사용할 수 있습니다. `async` [fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)와 같이 JavaScript에서 잠재적으로 오래 실행될 수 있는 다른 함수의 경우에도 마찬가지입니다. `async`는 Python에 존재하지 않는 구조입니다. Keras의 [`fit()`](https://keras.io/models/model/#model-class-api) 메서드가 History 객체를 반환하는 반면, JavaScript에서 그에 해당하는 `fit()` 메서드 부분은 History의 [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)를 반환합니다. 이 응답은 위 예에서와 같이 [await](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await)하거나 then() 메서드와 함께 사용할 수 있습니다.

## TensorFlow.js에 대한 NumPy 없음

Python Keras 사용자는 종종 [NumPy](http://www.numpy.org/)를 사용하여 위의 예에서 2D 텐서를 생성하는 것과 같은 기본적인 숫자 및 배열 연산을 수행합니다.

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

TensorFlow.js에서 이러한 종류의 기본 숫자 연산은 패키지 자체로 수행됩니다. 예를 들면, 다음과 같습니다.

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

`tf.*` 네임스페이스도 행렬 곱셈과 같은 배열 및 선형 대수 연산을 위한 여러 가지 함수를 제공합니다. 자세한 내용은 [TensorFlow.js Core 설명서](https://js.tensorflow.org/api/latest/)를 참조하세요.

## 생성자가 아닌 팩터리 메서드 사용

위의 예에서 이어지는 Python의 다음 줄은 생성자 호출입니다.

```python
# Python:
model = keras.Sequential()
```

엄격하게 JavaScript로 변환한다면 동등한 생성자 호출은 다음과 같을 것입니다.

```js
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```

그러나 1) "new" 키워드는 코드를 비대화시키고 2) "new" 생성자는 JavaScript에서 "나쁜 부분"으로 간주되기 때문에 "new" 생성자를 사용하지 않기로 했습니다. 이 잠재적인 위험은 [*JavaScript: the Good Parts*](http://archive.oreilly.com/pub/a/javascript/excerpts/javascript-good-parts/bad-parts.html)에 설명되어 있습니다. TensorFlow.js에서 모델과 레이어를 만들려면 lowerCamelCase 이름을 가진 팩터리 메서드를 호출합니다. 예를 들면, 다음과 같습니다.

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## 옵션 문자열 값은 snake_case가 아니라 lowerCamelCase

JavaScript에서는 스네이크 케이스가 일반적으로 사용되는(예: Keras에서) Python과 비교하여 기호 이름에 카멜 케이스를 사용하는 것이 더 일반적입니다(예: [Google JavaScript 스타일 가이드](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined) 참조). 따라서 다음을 포함한 옵션의 문자열 값에 lowerCamelCase를 사용하기로 했습니다.

- 데이터 형식(예: `channels_first` 대신 **`channelsFirst`**)
- 이니셜라이저(예: `glorot_normal` 대신 **`glorotNormal`**)
- 손실 및 메트릭(예: `mean_squared_error` 대신 **`meanSquaredError`**, `categorical_crossentropy` 대신 **`categoricalCrossentropy`**)

예를 들어, 위의 예는 다음과 같습니다.

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

모델 직렬화 및 역직렬화와 관련해서는 걱정하지 않아도 됩니다. TensorFlow.js의 내부 메커니즘이 예를 들어, Python Keras에서 사전 훈련된 모델을 로드할 때 JSON 객체의 스네이크 케이스가 올바르게 처리되도록 합니다.

## Layer 객체를 함수로 호출하지 않고 apply()로 실행

Keras에서 Layer 객체에는 `__call__` 메서드가 정의되어 있습니다. 따라서 사용자는 다음과 같이 객체를 함수로 호출하여 레이어의 로직을 호출할 수 있습니다.

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

이 Python 구문 슈거는 TensorFlow.js에서 apply() 메서드로 구현됩니다.

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply()는 구체적인 텐서에 대한 명령적 (즉시) 평가를 지원함

현재 Keras에서 **call** 메서드는 기호적 의미만 있고 실제 숫자 값을 보유하지 않는 (Python) TensorFlow의 `tf.Tensor` 객체(TensorFlow 백엔드 가정)에서 작동할 수 있습니다. 이전 섹션의 예에 이 내용이 나와 있습니다. 그러나 TensorFlow.js에서 레이어의 apply() 메서드는 기호 모드와 명령 모드 모두에서 동작할 수 있습니다. `apply()`가 SymbolicTensor(tf.Tensor와 매우 유사)와 함께 호출되면 반환 값은 SymbolicTensor가 됩니다. 이러한 상황은 일반적으로 모델 빌드 중에 발생합니다. 그러나 `apply()`가 실제 구체적인 텐서 값으로 호출되면 구체적인 텐서를 반환합니다. 예를 들면, 다음과 같습니다.

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

이 기능은 (Python) TensorFlow의 [즉시 실행](https://www.tensorflow.org/guide/eager)을 연상시킵니다. 이를 통해 동적 신경망을 구성할 수 있을 뿐만 아니라 모델 개발 시 상호 작용과 디버깅도 향상됩니다.

## 옵티마이저는 *optimizers*가 아니라 train. 아래에 있음

Keras에서 Optimizer 객체의 생성자는 `keras.optimizers.*` 네임스페이스 아래에 있습니다. TensorFlow.js Layers에서 Opitimzer에 대한 팩터리 메서드는 `tf.train.*` 네임스페이스 아래에 있습니다. 예를 들면, 다음과 같습니다.

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel()은 HDF5 파일이 아닌 URL에서 로드됨

Keras에서 모델은 일반적으로 HDF5(.h5) 파일로 [저장](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)되며 나중에 `keras.models.load_model()` 메서드를 사용하여 로드할 수 있습니다. 이 메서드는 .h5 파일에 대한 경로를 사용합니다. TensorFlow.js에서 `load_model()`에 해당하는 부분은 [`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel)입니다. HDF5는 브라우저 친화적인 파일 형식이 아니기 때문에 `tf.loadLayersModel()`은 TensorFlow.js 특정 형식을 사용합니다. `tf.loadLayersModel()`은 model.json 파일을 입력 인수로 받습니다. model.json은 tensorflowjs pip 패키지를 사용하여 Keras HDF5 파일에서 변환할 수 있습니다.

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

또한 `tf.loadLayersModel()`은 [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model)의 [`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)를 반환합니다.

일반적으로, TensorFlow.js에서 `tf.Model`을 저장하고 로드하는 작업은 각각 `tf.Model.save` 및 `tf.loadLayersModel` 메서드를 사용하여 수행됩니다. 이러한 API는 Keras의 [save 및 load_model API](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)와 유사하게 설계되었습니다. 그러나 브라우저 환경은 Keras와 같은 주요 딥 러닝 프레임워크가 실행되는 백엔드 환경, 특히 데이터 유지 및 변환을 위한 경로 배열과는 상당히 다릅니다. 따라서 TensorFlow.js와 Keras에서 save/load API 간에는 몇 가지 흥미로운 차이점이 있습니다. 자세한 내용은 [tf.Model 저장 및 로드](./save_load.md)에 대한 튜토리얼을 참조하세요.

## `fitDataset()`를 사용하여 `tf.data.Dataset` 객체를 사용하는 모델 훈련

Python TensorFlow의 tf.keras에서 모델은 [Dataset](https://www.tensorflow.org/guide/datasets) 객체를 사용하여 훈련할 수 있습니다. 모델의 `fit()` 메서드는 이러한 객체를 직접적으로 허용합니다. TensorFlow.js 모델은 Dataset 객체에 해당하는 JavaScript를 사용하여 훈련할 수 있습니다([TensorFlow.js의 tf.data API 설명서](https://js.tensorflow.org/api/latest/#Data) 참조). 그러나 Python과 달리 데이터세트 기반 훈련은 [fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset)라는 전용 메서드를 통해 수행됩니다. [fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) 메서드는 텐서 기반 모델 훈련에만 이용됩니다.

## Layer 및 Model 객체의 메모리 관리

TensorFlow.js는 브라우저의 WebGL에서 실행되며, Layer 및 Model 객체의 가중치는 WebGL 텍스처에서 지원됩니다. 그러나 WebGL에는 내장된 가비지 수집 지원이 없습니다. Layer 및 Model 객체는 추론 및 훈련 호출 중에 사용자를 위해 텐서 메모리를 내부적으로 관리합니다. 그러나 이와 동시에 사용자가 이러한 객체를 삭제하여 객체가 차지하는 WebGL 메모리를 확보할 수도 있습니다. 이는 단일 페이지 로드 내에서 많은 모델 인스턴스가 생성 및 해제되는 경우에 유용합니다. Layer 또는 Model 객체를 삭제하려면 `dispose()` 메서드를 사용합니다.

# 모델 및 레이어

머신러닝에서 *모델*은 입력을 출력에 매핑하는 *학습 가능한* [매개변수](https://developers.google.com/machine-learning/glossary/#parameter)가 있는 함수입니다. 최적의 매개변수는 데이터에서 모델을 훈련하여 얻습니다. 잘 훈련된 모델은 입력에서 원하는 출력으로 정확한 매핑을 제공합니다.

TensorFlow.js에는 기계 학습 모델을 만드는 두 가지 방법이 있습니다.

1. *레이어*를 사용하여 모델을 빌드하는 Layers API를 사용하기
2. `tf.matMul()`나 `tf.add()` 등과 같은 하위 수준 연산과 함께 Core API 사용하기

먼저 모델 빌드를위한 상위 수준 API 인 Layers API를 살펴 보겠습니다. 그런 다음 Core API를 사용하여 동일한 모델을 빌드하는 방법을 보여줍니다.

## Layers API로 모델 생성

Layers API를 사용하여 모델을 생성하는 방법에는 *순차* 모델과 *기능* 모델의 두 가지가 있습니다. 다음 두 섹션에서는 각 유형을 더 자세히 살펴 봅니다.

### 순차형 모델

가장 일반적인 모델 유형은 레이어의 선형 스택인 <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#class:Sequential" data-md-type="link"&gt;Sequential&lt;/a&gt;</code> 모델입니다. 레이어 목록을 <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#sequential" data-md-type="link"&gt;sequential()&lt;/a&gt;</code> 함수에 전달하여 <code>Sequential</code> 모델을 만들 수 있습니다.

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

또는 `add()` 메서드를 통해 :

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> 중요: 모델의 첫 번째 레이어에는 `inputShape`이 필요합니다. `inputShape`을 제공할 때는 배치 크기를 제외해야 합니다. 예를 들어, 형상 `[B, 784]`의 모델 텐서를 공급하려는 경우 `B`의 배치 크기는 임의의 값일 수 있습니다. 따라서 모델을 만들 때 `inputShape`을 `[784]`로 지정합니다.

`model.layers` ,보다 구체적으로 `model.inputLayers` 및 `model.outputLayers` 를 통해 모델의 레이어에 액세스 할 수 있습니다.

### 기능적 모델

`LayersModel`를 만들 수 있는 또 다른 방법은 `tf.model()` 함수를 통해서입니다. `tf.model()`과 `tf.sequential()`의 주요 차이는 주기가 없는 한`tf.model()`로 레이어의 임의 그래프를 만들 수 있다는 점입니다.

다음은 `tf.model()` API를 사용하여 위와 동일한 모델을 정의하는 코드 스 니펫입니다.

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

다른 레이어의 출력에 연결하기 위해 각 레이어에 `apply()`를 호출합니다. 이 경우 `apply()`의 결과는`Tensor`처럼 동작하지만 구체적인 값이 없는 `SymbolicTensor`입니다.

순차 모델과 달리 첫 번째 레이어에 `inputShape` 를 제공하는 대신 `tf.input()` 통해 `SymbolicTensor` 를 만듭니다.

또한 `apply()`는 구체적인 `Tensor`를 전달하면 구체적인 `Tensor`를 제공할 수 있습니다.

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

이는 레이어를 분리하여 테스트하고 출력을 볼 때 유용 할 수 있습니다.

순차 모델과 마찬가지로 `model.layers` ,보다 구체적으로 `model.inputLayers` 및 `model.outputLayers` 를 통해 모델의 레이어에 액세스 할 수 있습니다.

## 확인

순차형 모델과 함수형 모델은 모두 `LayersModel` 클래스의 인스턴스입니다. `LayersModel`을 사용하는 주요 이점 중 하나는 검증입니다. 입력 형상을 지정하여 후에 이 값을 입력을 검증하는 데 사용합니다. `LayersModel`은 데이터가 레이어를 통과할 때 자동 형상 추론을 수행합니다. 형상을 미리 알면 모델이 자동으로 매개변수를 생성할 수 있으며 두 개의 연속된 레이어가 서로 호환되지 않는지 여부를 알 수 있습니다.

## 모델 요약

`model.summary()` 를 호출하여 다음을 포함하는 유용한 모델 요약을 인쇄합니다.

- 모델에있는 모든 레이어의 이름과 유형입니다.
- 각 레이어의 출력 모양.
- 각 레이어의 가중치 매개 변수 수입니다.
- 모델에 일반 토폴로지가있는 경우 (아래에서 설명) 각 레이어가 수신하는 입력
- 모델의 학습 가능 및 학습 불가능 매개 변수의 총 수입니다.

위에서 정의한 모델의 경우 콘솔에 다음과 같은 출력이 표시됩니다.

<table>
  <tr>
   <td>레이어 (유형)</td>
   <td>출력 형태</td>
   <td>Param #</td>
  </tr>
  <tr>
   <td>density_Dense1 (밀도)</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>density_Dense2(밀도)</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">총 매개 변수 : 25450 <br> 학습 가능한 매개 변수 : 25450 <br> 훈련 할 수없는 매개 변수 : 0</td>
  </tr>
</table>

레이어의 출력 형상에 있는 `null` 값에 유의하세요. 모델은 해당 입력에 가장 바깥쪽 차원으로 배치의 크기를 가질 것으로 예상하며, 이 경우 `null` 값으로 인해 유연할 수 있습니다.

## 직렬화

하위 수준 API에 `LayersModel`을 사용할 때 얻는 주요 이점 중 하나는 모델을 저장하고 로드하는 기능입니다. `LayersModel`은 다음에 대해 알고 있습니다.

- 모델의 아키텍처를 통해 모델을 다시 만들 수 있습니다.
- 모델의 무게
- 교육 구성 (손실, 최적화 도구, 측정 항목).
- 학습을 재개 할 수 있도록 최적화 프로그램의 상태.

모델을 저장하거나로드하려면 코드 한 줄만 있으면됩니다.

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

위의 예는 모델을 브라우저의 로컬 저장소에 저장합니다. <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.save" data-md-type="link"&gt;model.save() documentation</code> 및 다른 매체에 저장하는 방법에 대한 [저장 및 로드](save_load.md) 가이드 (예: 파일 저장소, <code>IndexedDB{/ code2}, 브라우저 다운로드 트리거 등)을 참조하세요.</code>

## 커스텀 레이어

레이어는 모델의 구성 요소입니다. 모델이 사용자 지정 계산을 수행하는 경우 나머지 계층과 잘 상호 작용하는 사용자 지정 계층을 정의 할 수 있습니다. 아래에서는 제곱합을 계산하는 사용자 지정 레이어를 정의합니다.

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

이를 테스트하기 위해 구체적인 텐서로 `apply()` 메서드를 호출 할 수 있습니다.

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> 중요 : 사용자 지정 계층을 추가하면 모델을 직렬화 할 수 없습니다.

## Core API로 모델 만들기

가이드의 시작 부분에서 TensorFlow.js에서 머신러닝 모델을 만드는 두 가지 방법이 있다고 언급했습니다.

Layers API는 [모범 사례를 따르고 인지 부하를 줄이는](https://keras.io/why-use-keras/) Keras API를 모델로 하므로 일반적으로 먼저 Layers API를 사용하도록 권장됩니다. Layers API는 가중치 초기화, 모델 직렬화, 모니터링 훈련, 이식성 및 보안 검사와 같은 다양한 기성 솔루션도 제공합니다.

다음과 같은 경우 Core API를 사용할 수 있습니다.

- 최대한의 유연성 또는 제어가 필요합니다.
- 직렬화가 필요하지 않거나 자체적으로 직렬화 논리를 구현할 수 있습니다.

Core API의 모델은 하나 이상의 `Tensor`를 사용하고 `Tensor`를 반환하는 함수입니다. Core API를 사용하여 작성된 위와 같은 모델은 다음과 같습니다.

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

Core API에서 우리는 모델의 가중치를 만들고 초기화 할 책임이 있습니다. 모든 가중치는 TensorFlow.js에 이러한 텐서가 학습 가능하다는 신호를 보내는 `Variable` 의해 뒷받침됩니다. <a>tf.variable ()을</a> 사용하고 기존 <code>Tensor</code> 전달하여 `Variable` 를 만들 수 있습니다.

이 가이드를 통해 Layers와 Core API를 사용하여 모델을 만드는 다양한 방법을 알아보았습니다. 모델 훈련 방법을 보려면 [모델 훈련](train_models.md) 가이드를 참조하세요.

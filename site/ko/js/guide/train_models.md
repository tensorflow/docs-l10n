# 훈련 모델

이 가이드는 이미 [모델 및 레이어](models_and_layers.md) 가이드를 읽었다는 가정하에 쓰였습니다.

TensorFlow.js에는 머신러닝 모델을 훈련하는 두 가지 방법이 있습니다.

1. <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fit" data-md-type="link"&gt;LayersModel.fit()&lt;/a&gt;</code> 또는 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;LayersModel.fitDataset()&lt;/a&gt;</code>와 함께 Layers API 사용하기
2. <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;Optimizer.minimize()&lt;/a&gt;</code>와 함께 Core API 사용하기

먼저 모델 빌드 및 훈련을 위한 상위 수준 API 인 Layers API를 살펴보겠습니다. 그런 다음 Core API를 사용하여 같은 모델을 훈련하는 방법을 보겠습니다.

## 소개

머신러닝 *모델*은 입력을 원하는 출력에 매핑하는 학습 가능한 매개변수가 있는 함수입니다. 최적의 매개변수는 데이터에서 모델을 훈련하여 얻습니다.

훈련에는 여러 단계가 포함됩니다.

- 모델에 데이터 [배치](https://developers.google.com/machine-learning/glossary/#batch) 가져오기
- 모델에 예측값 요청하기
- 해당 예측값을 '참'값과 비교하기
- 모델이 향후 해당 배치에 대해 더 나은 예측값을 내도록 각 매개변수의 변경 범위 결정하기

잘 훈련된 모델은 입력에서 원하는 출력으로 정확한 매핑을 제공합니다.

## 모델 매개변수

Layers API를 사용하여 간단한 2레이어 모델을 정의해보겠습니다.

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

내부적으로 모델에는 데이터 학습을 통해 훈련할 수 있는 매개변수(종종 *가중치* 라고 함)가 있습니다. 이 모델 및 형상과 관련된 가중치의 이름을 출력해봅니다.

```js
model.weights.forEach(w => {
 console.log(w.name, w.shape);
});
```

다음과 같은 출력이 표시됩니다.

```
> dense_Dense1/kernel [784, 32]
> dense_Dense1/bias [32]
> dense_Dense2/kernel [32, 10]
> dense_Dense2/bias [10]
```

총 4개의 가중치가 있으며 밀집 레이어당 2개입니다. 이러한 결과가 예상되는 이유는 밀집 레이어는 수식 `y = Ax + b`를 통해 입력 텐서 `x`를 출력 텐서 `y`로 매핑하는 함수를 나타내기 때문입니다. 여기서 `A`(커널) 및 `b`(바이어스)는 밀집 레이어의 매개변수입니다.

> 참고: 기본적으로 밀집 레이어에는 바이어스가 포함되지만, 밀집 레이어를 만들 때 옵션에서 `{useBias: false}`를 지정하여 바이어스를 제외할 수 있습니다.

`model.summary()`는 모델의 개요와 총 매개변수 수를 확인하려는 경우 유용한 메서드입니다.

<table>
  <tr>
   <td>레이어(유형)</td>
   <td>출력 형상</td>
   <td>매개변수 번호</td>
  </tr>
  <tr>
   <td>density_Dense1(밀도)</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>density_Dense2(밀도)</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">총 매개변수: 25450 <br> 훈련 가능한 매개변수: 25450 <br> 훈련할 수 없는 매개변수: 0</td>
  </tr>
</table>

모델의 각 가중치는 <code>&lt;a href="https://js.tensorflow.org/api/0.14.2/#class:Variable" data-md-type="link"&gt;Variable&lt;/a&gt;</code> 객체의 백엔드입니다. TensorFlow.js에서 <code>Variable</code>은 값을 업데이트하는 데 사용되는 하나의 추가 메서드 <code>assign()</code>이 있는 부동 소수점 <code>Tensor</code>입니다. Layers API는 모범 사례를 사용하여 가중치를 자동으로 초기화합니다. 데모를 위해 기본 변수에 대해 <code>assign()</code>을 호출하여 가중치를 덮어쓸 수 있습니다.

```js
model.weights.forEach(w => {
  const newVals = tf.randomNormal(w.shape);
  // w.val is an instance of tf.Variable
  w.val.assign(newVals);
});
```

## 옵티마이저, 손실 및 메트릭

훈련을 시작하기 전에 다음 세 가지를 결정해야 합니다.

1. **옵티마이저**: 옵티마이저는 현재 모델의 예측값이 주어졌을 때 모델의 각 매개변수를 얼마나 변경할 것인지 결정하는 역할을 합니다. Layers API를 사용할 때 기존 옵티마이저의 문자열 식별자인(예: `'sgd'` 또는 `'adam'` ) 또는 <code>&lt;a href="https://js.tensorflow.org/api/latest/#Training-Optimizers" data-md-type="link"&gt;Optimizer&lt;/a&gt;</code> 클래스의 인스턴스를 제공할 수 있습니다.
2. <strong>손실 함수</strong>: 모델은 최소화를 목표로 합니다. 모델의 예측값이 '얼마나 잘못되었는지'에 대한 단일 숫자를 제공하는 것입니다. 손실은 모델이 가중치를 업데이트할 수 있도록 모든 데이터 배치에서 계산됩니다. Layers API를 사용할 때 기존 손실 함수의 문자열 식별자(예: <code>'categoricalCrossentropy'</code>) 또는 예측값과 참값을 가져와 손실을 반환하는 모든 함수를 제공할 수 있습니다. API 설명서에서 [사용 가능한 손실 목록](https://js.tensorflow.org/api/latest/#Training-Losses)을 참조하세요.
3. <strong>메트릭 목록</strong>: 손실과 유사하게 메트릭은 단일 숫자를 계산하여 모델의 성능을 요약합니다. 메트릭은 일반적으로 각 epoch가 끝날 때 전체 데이터에 대해 계산됩니다. 최소한 시간이 지남에 따라 손실이 감소하고 있는지 모니터링해야 합니다만, 정확성과 같은 보다 인간 친화적인 메트릭을 원하는 경우가 많습니다. Layers API를 사용하는 경우 기존 메트릭의 문자열 식별자(예: <code>'accuracy'</code>) 또는 예측값 및 참값을 가져와 점수를 반환하는 모든 함수를 제공할 수 있습니다. API 설명서에서 [사용 가능한 메트릭 목록](https://js.tensorflow.org/api/latest/#Metrics)을 참조하세요.

결정했으면 제공된 옵션으로 <code>model.compile()</code>을 호출하여 <code>LayersModel</code>을 컴파일합니다.

```js
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});
```

컴파일하는 동안 모델은 선택한 옵션이 서로 호환되는지를 확인하는 몇 가지 검증을 수행합니다.

## 훈련

`LayersModel`을 훈련하는 방법에는 두 가지가 있습니다.

- `model.fit()`를 사용하고 데이터를 하나의 큰 텐서로 제공하기
- `model.fitDataset()` 및 `Dataset` 객체를 통해 데이터 제공하기

### model.fit()

데이터세트가 주 메모리에 적합하게 맞고 단일 텐서로 사용할 수있는 경우 `fit()` 메서드를 호출하여 모델을 훈련할 수 있습니다.

```js
// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
model.fit(data, labels, {
   epochs: 5,
   batchSize: 32,
   callbacks: {onBatchEnd}
 }).then(info => {
   console.log('Final accuracy', info.history.acc);
 });
```

내부적으로 `model.fit()`는 많은 일을 할 수 있습니다.

- 데이터를 훈련 및 검증 세트로 분할하고 검증 세트를 사용하여 훈련 중 진행 상황을 측정합니다.
- 분할 후에 데이터를 셔플합니다. 안전을 위해 데이터를 `fit()`로 전달하기 전에 미리 셔플해야 합니다.
- 큰 데이터 텐서를 `batchSize.` 크기의 더 작은 텐서로 분할합니다.
- 데이터 배치와 관련하여 모델 손실을 계산하는 동안 `optimizer.minimize()`를 호출합니다.
- 각 epoch 또는 배치의 시작과 끝을 알려줄 수 있습니다. 이 경우에는 모든 배치가 끝날 때 `callbacks.onBatchEnd` 옵션을 사용하여 알림을 받습니다. 다른 옵션으로는 `onTrainBegin` , `onTrainEnd`, `onEpochBegin` , `onEpochEnd` 및 `onBatchBegin`이 있습니다.
- JS 이벤트 루프에 대기 중인 작업을 적시에 처리할 수 있도록 주 스레드에 양보합니다.

자세한 내용은 `fit()` [설명서](https://js.tensorflow.org/api/latest/#tf.Sequential.fit)를 참조하세요. Core API를 사용하기로 선택한 경우 이 로직을 직접 구현해야 합니다.

### model.fitDataset()

데이터가 메모리에 완전히 맞지 않거나 스트리밍되는 경우 `Dataset` 객체를 사용하는 `fitDataset()`를 호출하여 모델을 훈련할 수 있습니다. 다음은 같은 훈련 코드이지만 생성기 함수를 래핑하는 데이터세트가 있습니다.

```js
function* data() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomNormal([784]);
 }
}

function* labels() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomUniform([10]);
 }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// Train the model for 5 epochs.
model.fitDataset(ds, {epochs: 5}).then(info => {
 console.log('Accuracy', info.history.acc);
});
```

데이터세트에 대한 자세한 내용은 `model.fitDataset()` [설명서](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset)를 참조하세요.

## 새로운 데이터 예측하기

한번 훈련을 거치면 모델이 `model.predict()`를 호출하여 보이지 않는 데이터의 예측값을 낼 수 있습니다.

```js
// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();
```

참고: [모델 및 레이어](models_and_layers) 가이드에서 언급되었듯이`LayersModel`은 입력의 가장 바깥쪽 차원이 배치 크기일 것으로 예상합니다. 위의 예에서 배치 크기는 3입니다.

## Core API

앞서 TensorFlow.js에서 머신러닝 모델을 훈련하는 방법에는 두 가지가 있다고 언급했습니다.

일반적인 방법으로는 먼저 잘 채택된 Keras API를 모델로 하는 Layers API를 사용하는 것입니다. Layers API는 가중치 초기화, 모델 직렬화, 모니터링 훈련, 이식성 및 안전 검사와 같은 다양한 기성 솔루션도 제공합니다.

다음과 같은 경우 Core API를 사용할 수 있습니다.

- 최대한의 유연성 또는 제어가 필요합니다.
- 그리고 직렬화가 필요하지 않거나 자체적으로 직렬화 논리를 구현할 수 있습니다.

자세한 내용은 [모델 및 레이어](models_and_layers.md) 가이드의 'Core API'섹션을 참조하세요.

Core API를 사용하여 작성된 위와 동일한 모델은 다음과 같습니다.

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}
```

Layers API 외에도 Data API는 Core API와 원활하게 동작합니다. 셔플 및 일괄 처리를 수행하는 [model.fitDataset ()](#model.fitDataset()) 섹션에서 이전에 정의한 데이터세트를 다시 사용해보겠습니다.

```js
const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
```

모델을 훈련해보겠습니다.

```js
const optimizer = tf.train.sgd(0.1 /* learningRate */);
// Train for 5 epochs.
for (let epoch = 0; epoch < 5; epoch++) {
  await ds.forEachAsync(({xs, ys}) => {
    optimizer.minimize(() => {
      const predYs = model(xs);
      const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
      loss.data().then(l => console.log('Loss', l));
      return loss;
    });
  });
  console.log('Epoch', epoch);
}
```

위의 코드는 Core API로 모델을 훈련할 때 쓰이는 표준 레시피입니다.

- epoch 수를 반복합니다.
- 각 epoch 내에서 데이터 배치를 반복합니다. `Dataset`를 사용할 때 <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#tf.data.Dataset.forEachAsync" data-md-type="link"&gt;dataset.forEachAsync()&lt;/a&gt;</code>는 배치를 반복하는 편리한 방법입니다.
- 각 배치에 대해 <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;optimizer.minimize(f)&lt;/a&gt;</code>를 호출하면 <code>f</code>를 실행하고 앞서 정의한 4개의 변수에 대한 그래디언트를 계산하여 출력을 최소화합니다.
- <code>f</code>는 손실을 계산합니다. 모델의 예측값과 실제값을 사용하여 미리 정의된 손실 함수 중 하나를 호출합니다.

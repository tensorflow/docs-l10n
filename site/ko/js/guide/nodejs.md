# 노드의 TensorFlow.js

## TensorFlow CPU

TensorFlow CPU 패키지는 다음과 같이 가져올 수 있습니다.

```js
import * as tf from '@tensorflow/tfjs-node'
```

이 패키지에서 TensorFlow.js를 가져올 때 얻은 모듈은 TensorFlow C 바이너리에 의해 가속화되고 CPU에서 실행됩니다. CPU의 TensorFlow는 하드웨어 가속을 사용하여 진행되는 선형 대수 계산을 가속화합니다.

이 패키지는 TensorFlow가 지원되는 Linux, Windows 및 Mac 플랫폼에서 동작합니다.

> 참고: '@tensorflow/tfjs'를 가져오거나 package.json에 추가할 필요가 없이 노드 라이브러리에서 간접적으로 가져옵니다.

## TensorFlow GPU

TensorFlow GPU 패키지는 다음과 같이 가져올 수 있습니다.

```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

CPU 패키지와 마찬가지로 모듈은 TensorFlow C 바이너리로 가속화되지만 CUDA를 사용하여 GPU에서 텐서 연산을 실행하므로 Linux만 실행됩니다. 이 바인딩은 다른 바인딩 옵션보다 적어도 10배 더 빠를 수 있습니다.

> 참고: 이 패키지는 현재 CUDA에서만 동작합니다. 이 경로를 진행하기 전에 NVIDIA 그래픽 카드가 있는 컴퓨터에 CUDA를 설치해야합니다.

> 참고: '@tensorflow/tfjs'를 가져오거나 package.json에 추가할 필요가 없이 노드 라이브러리에서 간접적으로 가져옵니다.

## Vanilla CPU

vanilia CPU 연산으로 실행되는 TensorFlow.js 버전은 다음과 같이 가져올 수 있습니다.

```js
import * as tf from '@tensorflow/tfjs'
```

이 패키지는 브라우저에서 사용하는 것과 같은 패키지입니다. 이 패키지에서 연산은 CPU에서 vanilla JavaScript로 실행됩니다. 해당 패키지는 TensorFlow 바이너리가 필요하지 않기 때문에 다른 패키지보다 규모는 훨씬 작지만, 속도는 훨씬 느립니다.

이 패키지는 TensorFlow에 의존하지 않기 때문에 Linux, Windows 및 Mac뿐 아니라 Node.js를 지원하는 더 많은 기기에서 사용할 수 있습니다.

## 운영 고려 사항

Node.js 바인딩은 연산을 동기식으로 구현하는 TensorFlow.js용 백엔드를 제공합니다. 즉, `tf.matMul(a, b)`와 같은 연산을 호출하면 연산이 완료될 때까지 주 스레드가 차단됩니다.

이러한 이유로 현재 바인딩은 스크립트 및 오프라인 작업에 적합합니다. 웹 서버와 같은 운영 애플리케이션에서 Node.js 바인딩을 사용하려면 TensorFlow.js 코드가 주 스레드를 차단하지 않도록 작업 큐를 설정하거나 작업자 스레드를 설정해야 합니다.

## API

위의 옵션 중 하나에서 패키지를 tf로 가져오면 모든 일반 TensorFlow.js 기호가 가져온 모듈에 나타납니다.

### tf.browser

일반 TensorFlow.js 패키지에서 `tf.browser.*` 네임스페이스의 기호는 브라우저별 API를 사용하므로 Node.js에서 사용할 수 없습니다.

기호는 다음과 같습니다.

- tf.browser.fromPixels
- tf.browser.toPixels

### tf.node

두 개의 Node.js 패키지는 노드별 API를 포함하는 네임스페이스 `tf.node`도 제공합니다.

TensorBoard는 Node.js 관련 API 예제로 주목할 만합니다.

요약을 Node.js의 TensorBoard로 내보내는 예제는 다음과 같습니다.

```js
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [200] }));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['MAE']
});


// Generate some random fake data for demo purpose.
const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);


// Start model training process.
async function train() {
  await model.fit(xs, ys, {
    epochs: 100,
    validationData: [valXs, valYs],
    // Add the tensorBoard callback here.
    callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
  });
}
train();
```

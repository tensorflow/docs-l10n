# 설정

## 브라우저 설정

브라우저 기반 프로젝트에서 TensorFlow.js를 가져오는 두 가지 주요 방법이 있습니다.

- [스크립트 태그](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage) 사용하기

- [NPM](https://www.npmjs.com)에서 설치하고 [Parcel](https://parceljs.org/) , [WebPack](https://webpack.js.org/) 또는 [Rollup](https://rollupjs.org/guide/en)과 같은 빌드 도구를 사용하기

웹 개발이 처음이거나 webpack 또는 parcel과 같은 도구에 대해 들어 본 적이 없는 경우 *스크립트 태그 접근 방식을 사용하는 것이 좋습니다*. 이미 웹 개발 경험이 있거나 더 큰 규모의 프로그램을 작성하려는 경우 빌드 도구를 사용하여 탐색하는 것이 좋습니다.

### 스크립트 태그를 통한 사용법

기본 HTML 파일에 다음 스크립트 태그를 추가합니다.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
```

<section class="expandable">
  <h4 class="showalways">스크립트 태그 설정 코드 샘플 보기</h4>
  <pre class="prettyprint">
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
  </pre>
</section>

### NPM에서 설치

[npm cli](https://docs.npmjs.com/cli/npm) 도구 또는 [yarn](https://yarnpkg.com/en/)을 사용하여 TensorFlow.js를 설치할 수 있습니다.

```
yarn add @tensorflow/tfjs
```

*또는*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">NPM을 통한 설치 샘플 코드 보기</h4>
  <pre class="prettyprint">
import * as tf from '@tensorflow/tfjs';

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
  </pre>
</section>

## Node.js 설정

[npm cli](https://docs.npmjs.com/cli/npm) 도구 또는 [yarn](https://yarnpkg.com/en/)을 사용하여 TensorFlow.js를 설치할 수 있습니다.

**옵션 1:** 네이티브 C++ 바인딩으로 TensorFlow.js를 설치합니다.

```
yarn add @tensorflow/tfjs-node
```

*또는*

```
npm install @tensorflow/tfjs-node
```

**옵션 2:** (Linux 전용) 시스템에 [CUDA를 지원](https://www.tensorflow.org/install/install_linux#NVIDIARequirements)하는 NVIDIA® GPU가 있는 경우 GPU 패키지를 사용하여 성능을 끌어올릴 수 있습니다.

```
yarn add @tensorflow/tfjs-node-gpu
```

*또는*

```
npm install @tensorflow/tfjs-node-gpu
```

**옵션 3:** pure JavaScript 버전을 설치합니다. 이 버전은 성능면에서 가장 느린 옵션입니다.

```
yarn add @tensorflow/tfjs
```

*또는*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">Node.js 사용법에 대한 샘플 코드 보기</h4>
  <pre class="prettyprint">
const tf = require('@tensorflow/tfjs');

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require('@tensorflow/tfjs-node');

// Train a simple model:
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});
  </pre>
</section>

### TypeScript

TypeScript를 사용할 때 프로젝트에서 엄격한 null 검사를 사용하거나 컴파일 중에 오류가 발생하는 경우 `tsconfig.json` 파일에서 `skipLibCheck: true`를 설정해야 할 수 있습니다.

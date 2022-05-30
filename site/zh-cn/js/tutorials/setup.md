# 设置

## 浏览器设置

您可以通过两种主要方式在浏览器项目中获取 TensorFlow.js：

- 使用[脚本代码](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage)。

- 从 [NPM](https://www.npmjs.com) 安装并使用诸如 [Parcel](https://parceljs.org/)、[WebPack](https://webpack.js.org/) 或 [Rollup](https://rollupjs.org/guide/en) 的构建工具。

如果您是 Web 开发新手，或者从未听说过诸如 Webpack 或 Parcel 的工具，*建议您使用脚本代码*。如果您比较有经验或想编写更大的程序，则最好使用构建工具进行探索。

### 使用脚本代码

将以下脚本代码添加到您的主 HTML 文件中。

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
```

<section class="expandable">
  <h4 class="showalways">请参见代码示例了解脚本代码设置</h4>
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

### 从 NPM 安装

您可以使用 [npm cli](https://docs.npmjs.com/cli/npm) 工具或 [yarn](https://yarnpkg.com/en/) 安装 TensorFlow.js。

```
yarn add @tensorflow/tfjs
```

*或*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">请参见示例代码以通过 NPM 安装</h4>
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

## Node.js 设置

您可以使用 [npm cli](https://docs.npmjs.com/cli/npm) 工具或 [yarn](https://yarnpkg.com/en/) 安装 TensorFlow.js。

**选项 1**：使用原生 C++ 绑定安装 TensorFlow.js。

```
yarn add @tensorflow/tfjs-node
```

*或*

```
npm install @tensorflow/tfjs-node
```

**选项 2**：（仅限 Linux）如果您的系统搭载具备 [CUDA 支持](https://www.tensorflow.org/install/install_linux#NVIDIARequirements)的 NVIDIA® GPU，请使用 GPU 软件包以获得更高性能。

```
yarn add @tensorflow/tfjs-node-gpu
```

*或*

```
npm install @tensorflow/tfjs-node-gpu
```

**选项 3**：安装纯净版 JavaScript。这是性能最慢的选项。

```
yarn add @tensorflow/tfjs
```

*或*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">请参见示例代码以了解 Node.js 用法</h4>
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

使用 TypeScript 时，如果您的项目使用严格的 null 检查，则可能需要在 `tsconfig.json` 文件中设置 `skipLibCheck: true`，否则在编译过程中会遇到错误。

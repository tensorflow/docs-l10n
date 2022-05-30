# セットアップ

## ブラウザのセットアップ

ブラウザベースのプロジェクトで TensorFlow.js を取得するには、主に次の 2 つの方法があります。

- [script タグ](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage)を使用する。

- [NPM](https://www.npmjs.com) からインストールし、[Parcel](https://parceljs.org/)、[WebPack](https://webpack.js.org/)、[Rollup](https://rollupjs.org/guide/en) などのビルドツールを使用する。

ウェブ開発の経験が浅い方、または webpack や parcel などのツールを聞いたことがない方は、*script タグのアプローチを使うことをお勧めします*。経験の豊富な方や、より大型のプログラムを作成しようと考えている場合は、ビルドツールを使用するアプローチを探る価値があります。

### scriptタグから使用する

メインの HTML ファイルに次の script タグを追加します。

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
```

<section class="expandable">
  <h4 class="showalways">コードサンプルの script タグのセットアップを参照する</h4>
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

### NPM からインストール

[npm cli](https://docs.npmjs.com/cli/npm) ツールまたは [yarn](https://yarnpkg.com/en/) を使用して、TensorFlow.js をインストールできます。

```
yarn add @tensorflow/tfjs
```

*または*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">NPM 経由でインストールするためのサンプルコードを参照する</h4>
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

## Node.js のセットアップ

[npm cli](https://docs.npmjs.com/cli/npm) ツールまたは [yarn](https://yarnpkg.com/en/) を使用して、TensorFlow.js をインストールできます。

**オプション 1:** ネイティブの C++  バインディングで TensorFlow.js をインストールします。

```
yarn add @tensorflow/tfjs-node
```

*または*

```
npm install @tensorflow/tfjs-node
```

**オプション 2:**（Linux のみ）システムに [CUDA サポート](https://www.tensorflow.org/install/install_linux#NVIDIARequirements)付きの NVIDIA® GPU が備わっている場合は、より高いパフォーマンスを得るために GPU パッケージを使用します。

```
yarn add @tensorflow/tfjs-node-gpu
```

*または*

```
npm install @tensorflow/tfjs-node-gpu
```

**オプション 3:** ピュア JavaScript バージョンをインストールします。パフォーマンスの観点では、これが速度が最も遅いオプションです。

```
yarn add @tensorflow/tfjs
```

*または*

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">Node.js 使用のサンプルコードを参照する</h4>
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

TypeScript を使用する際、プロジェクトで厳格な null チェックを行っている場合は、コンパイル中にエラーが発生する可能性があるため、`skipLibCheck: true` を `tsconfig.json` ファイルに設定する必要があるかもしれません。

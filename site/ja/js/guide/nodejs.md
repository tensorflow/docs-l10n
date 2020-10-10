# Node での TensorFlow.js

## TensorFlow CPU

TensorFlow CPU パッケージは次のようにインポートできます。

```js
import * as tf from '@tensorflow/tfjs-node'
```

このパッケージから TensorFlow.js をインポートすると、取得したモジュールは TensorFlow C バイナリによって加速され、CPU で実行されます。CPU 上の TensorFlow は、ハードウェアアクセラレーションを使用して、内部で線形代数計算を高速化します。

このパッケージは、TensorFlow がサポートされている Linux、Windows、Mac プラットフォームで動作します。

> 注: 「@tensorflow/tfjs」をインポートしたり、package.json に追加したりする必要はありません。これは、ノードライブラリによって間接的にインポートされます。

## TensorFlow GPU

The TensorFlow GPU パッケージは次のようにインポートできます。

```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

CPU パッケージと同様に、取得したモジュールは TensorFlow C バイナリによって加速されますが、CUDA を使用して GPU でテンソル演算を実行するため、Linux のみが実行されます。このバインディングは、他のバインディングオプションよりも少なくとも 1 桁高速である可能性があります。

> 注: このパッケージは現在 CUDA でのみ動作します。このルートに進む前に、NVIDIA グラフィックスカードを搭載したマシンに CUDA をインストールする必要があります。

> 注: 「@tensorflow/tfjs」をインポートしたり、package.json に追加したりする必要はありません。これは、ノードライブラリによって間接的にインポートされます。

## 標準的な CPU

標準的な CPU で実行されている TensorFlow.js のバージョンは、次のようにインポートできます。

```js
import * as tf from '@tensorflow/tfjs'
```

このパッケージは、ブラウザで使用するものと同じパッケージです。このパッケージでは、演算は CPU 上の標準的な JavaScriptで実行されます。このパッケージは TensorFlow バイナリを必要としないため、他のパッケージよりも大幅に小さくなりますが、速度は大幅に低下します。

このパッケージは TensorFlow に依存しないため、Linux、Windows、Mac だけでなく、Node.js をサポートする多くのデバイスで使用できます。

## 本番環境に関する考慮事項

Node.js バインディングは、演算を同期的に実装する TensorFlow.js のバックエンドを提供します。これは、`tf.matMul(a, b)`などの演算を呼び出すとき、演算が完了するまでメインスレッドがブロックされることを意味します。

このため、バインディングは現在、スクリプトとオフラインタスクに適しています。Web サーバーなどの本番アプリケーションで Node.js バインディングを使用する場合は、TensorFlow.js コードがメインスレッドをブロックしないように、ジョブキューを設定するか、ワーカースレッドを設定する必要があります。

## API

上記のオプションのいずれかでパッケージを tf としてインポートすると、通常の TensorFlow.js シンボルがすべてインポートされたモジュールに表示されます。

### tf.browser

通常の TensorFlow.js パッケージでは、`tf.browser.*`名前空間のシンボルは、ブラウザ固有の API を使用するため、Node.js では使用できません。

現在、これらは次のとおりです。

- tf.browser.fromPixels
- tf.browser.toPixels

### tf.node

2 つの Node.js パッケージは、ノード固有の API を含む名前空間`tf.node`も提供します。

TensorBoard は、Node.js 固有の API の注目すべき例です。

以下は、Node.js で TensorBoard に概要をエクスポートする例です。

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

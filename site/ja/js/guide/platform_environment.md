# プラットフォームと環境

TensorFlow.js はブラウザと Node.js で機能し、両方のプラットフォームでさまざまな利用可能な構成があります。各プラットフォームには、アプリケーションの開発方法に影響を与える独自の考慮事項があります。

ブラウザでは、TensorFlow.js はデスクトップデバイスだけでなくモバイルデバイスもサポートします。各デバイスには、利用可能な WebGL API のような一連の特定の制約があり、これらは自動的に決定されて構成されます。

Node.js では、TensorFlow.js は TensorFlow API への直接のバインドやより遅い標準的な CPU 実装での実行をサポートします。

## 環境

TensorFlow.js プログラム実行時の特定の構成は環境と呼ばれます。環境は、単一のグローバルバックエンドと TensorFlow.js の細かい機能を制御する一連のフラグで構成されます。

### バックエンド

TensorFlow.js は、テンソルストレージと数学演算を実装する複数の異なるバックエンドをサポートします。常に、1 つのバックエンドのみがアクティブです。ほとんどの場合、TensorFlow.js は、その時点の環境を考慮して、最適なバックエンドを自動的に選択します。ただし、使用されているバックエンドとその切り替え方法を知ることが重要な場合があります。

使用しているバックエンドを見つけるには、以下を使用します。

```js
console.log(tf.getBackend());
```

バックエンドを手動で変更するには、以下を使用します。

```js
tf.setBackend('cpu');
console.log(tf.getBackend());
```

#### WebGL バックエンド

WebGL バックエンド 「WebGL」は、現在ブラウザの最も強力なバックエンドです。このバックエンドは、標準的な CPU バックエンドよりも最大 100 倍高速です。テンソルは WebGL テクスチャとして保存され、数学演算は WebGL シェーダーで実装されます。このバックエンドを使用するときに知っておくと便利なことがいくつかあります。

##### UI スレッドのブロックを回避する

When an operation is called, like tf.matMul(a, b), the resulting tf.Tensor is synchronously returned, however the computation of the matrix multiplication may not actually be ready yet. This means the tf.Tensor returned is just a handle to the computation. When you call `x.data()` or `x.array()`, the values will resolve when the computation has actually completed. This makes it important to use the asynchronous `x.data()` and `x.array()` methods over their synchronous counterparts `x.dataSync()` and `x.arraySync()` to avoid blocking the UI thread while the computation completes.

##### メモリ管理

WebGL バックエンドを使用する際の注意点の 1 つは、明示的なメモリ管理の必要性です。テンソルデータが最終的に格納される WebGLTextures は、ブラウザによって自動的にガベージコレクションされません。

`tf.Tensor`のメモリを破棄するには、`dispose()`メソッドを使用します。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

It is very common to chain multiple operations together in an application. Holding a reference to all of the intermediate variables to dispose them can reduce code readability. To solve this problem, TensorFlow.js provides a `tf.tidy()` method which cleans up all `tf.Tensor`s that are not returned by a function after executing it, similar to the way local variables are cleaned up when a function is executed:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> Note: there is no downside of using `dispose()` or `tidy()` in non-webgl environments (like Node.js or a CPU backend) that have automatic garbage collection. In fact, it often can be a performance win to free tensor memory faster than would naturally happen with garbage collection.

##### 精度

On mobile devices, WebGL might only support 16 bit floating point textures. However, most machine learning models are trained with 32 bit floating point weights and activations. This can cause precision issues when porting a model for a mobile device as 16 bit floating numbers can only represent numbers in the range `[0.000000059605, 65504]`. This means that you should be careful that weights and activations in your model do not exceed this range. To check whether the device supports 32 bit textures, check the value of `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')`, if this is false then the device only supports 16 bit floating point textures. You can use `tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')` to check if TensorFlow.js is currently using 32 bit textures.

##### シェーダーのコンパイルとテクスチャのアップロード

TensorFlow.js は、WebGL シェーダープログラムを実行することにより、GPU で演算を実行します。これらのシェーダーは、ユーザーが演算の実行を要求すると、レイジーにアセンブルおよびコンパイルされます。シェーダーのコンパイルはメインスレッドの CPU で行われ、遅くなる可能性があります。TensorFlow.js はコンパイルされたシェーダーを自動的にキャッシュし、同じ形状の入力テンソルと出力テンソルで同じ演算を 2 回目に呼び出します。通常、TensorFlow.js アプリケーションは、アプリケーションの存続期間中に同じ演算を複数回使用するため、機械学習モデルの 2 回目のパスははるかに高速です。

TensorFlow.js also stores tf.Tensor data as WebGLTextures. When a `tf.Tensor` is created, we do not immediately upload data to the GPU, rather we keep the data on the CPU until the `tf.Tensor` is used in an operation. If the `tf.Tensor` is used a second time, the data is already on the GPU so there is no upload cost. In a typical machine learning model, this means weights are uploaded during the first prediction through the model and the second pass through the model will be much faster.

モデルまたは TensorFlow.js コードによる最初の予測のパフォーマンスを向上するには、実際のデータを使用する前に、同じ形状の入力テンソルを渡すことでモデルをウォームアップすることをお勧めします。

例:

```js
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);
```

#### Node.js TensorFlow バックエンド

TensorFlow Node.js のバックエンド「ノード」は、TensorFlow C APIを使用して演算を高速化します。これは、CUDA などのマシンで利用可能なハードウェアアクセラレーションを使用します。

In this backend, just like the WebGL backend, operations return `tf.Tensor`s synchronously. However, unlike the WebGL backend, the operation is completed before you get the tensor back. This means that a call to `tf.matMul(a, b)` will block the UI thread.

このため、本番環境アプリケーションでこれを使用する場合は、メインスレッドをブロックしないようにワーカースレッドで TensorFlow.js を実行する必要があります。

Node.js についての詳細はこちらを参照してください。

#### WASM バックエンド

TensorFlow.js provides a [WebAssembly backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md) (`wasm`), which offers CPU acceleration and can be used as an alternative to the vanilla JavaScript CPU (`cpu`) and WebGL accelerated (`webgl`) backends.  To use it:

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

If your server is serving the `.wasm` file on a different path or a different name, use `setWasmPath` before you initialize the backend. See the ["Using Bundlers"](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers) section in the README for more info:

```js
import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
setWasmPath(yourCustomPath);
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

> 注: TensorFlow.js は各バックエンドの優先度を定義し、特定の環境でサポートされている最適なバックエンドを自動的に選択します。WASM バックエンドを明示的に使用するには、`tf.setBackend('wasm')`を呼び出す必要があります。

##### WASM を使用する理由

[WASM](https://webassembly.org/) は 2015 年に新しい Web ベースのバイナリ形式として公開され、JavaScript、C、C ++などで記述されたプログラムに Web 上で実行するためのコンパイルターゲットを提供します。WASM は 2017 年からChrome、Safari、Firefox、Edge で[サポート](https://webassembly.org/roadmap/)されており、世界中の[デバイスの 90% ](https://caniuse.com/#feat=wasm)でサポートされています。

**パフォーマンス**

WASM バックエンドは、ニューラルネットワークオペレーターの最適化された実装のために [XNNPACK ライブラリ](https://github.com/google/XNNPACK)を活用します。

*JavaScript との比較*: WASM バイナリでは、ブラウザの読み込み、解析、実行は JavaScript バンドルよりも一般的に高速です。JavaScript は動的に型指定され、ガベージコレクションが行われるため、実行時に速度が低下する可能性があります。

*WebGL との比較*: WebGL は、ほとんどのモデルで WASM よりも高速ですが、小さなモデルでは、WebGL シェーダーを実行するオーバーヘッドコストが固定されているため WASM は WebGL よりもパフォーマンスが優れています。以下の「WASM を使用すべき場合」セクションでは、この決定を行うためのヒューリスティックについて説明します。

**移植性と安定性**

WASM はポータブルな 32 ビット浮動小数点演算を備えており、すべてのデバイスで高精度のパリティを提供します。一方、WebGL はハードウェア固有であり、デバイスごとに精度が異なる可能性があります（たとえば、iOS デバイスでは 16 ビット浮動小数点にフォールバックします）。

WebGL と同様に、WASM はすべての主要なブラウザで公式にサポートされています。WebGL とは異なり、WASM は Node.js で実行でき、ネイティブライブラリをコンパイルする必要なくサーバー側で使用できます。

##### WASM を使用すべき場合

**モデルのサイズと計算の要件**

In general, WASM is a good choice when models are smaller or you care about lower-end devices that lack WebGL support (`OES_texture_float` extension) or have less powerful GPUs. The chart below shows inference times (as of TensorFlow.js 1.5.2) in Chrome on a 2018 MacBook Pro for 5 of our officially supported [models](https://github.com/tensorflow/tfjs-models) across the WebGL, WASM, and CPU backends:

**小規模モデル**

モデル | WebGL | WASM | CPU | メモリ
--- | --- | --- | --- | ---
BlazeFace | 22.5 ms | 15.6 ms | 315.2 ms | .4 MB
FaceMesh | 19.3 ms | 19.2 ms | 335 ms | 2.8 MB

**大規模モデル**

モデル | WebGL | WASM | CPU | メモリ
--- | --- | --- | --- | ---
PoseNet | 42.5 ms | 173.9 ms | 1514.7 ms | 4.5 MB
BodyPix | 77 ms | 188.4 ms | 2683 ms | 4.6 MB
MobileNet v2 | 37 ms | 94 ms | 923.6 ms | 13 MB

The table above shows that WASM is 10-30x faster than the plain JS CPU backend across models, and competitive with WebGL for smaller models like [BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface), which is lightweight (400KB), yet has a decent number of ops (~140). Given that WebGL programs have a fixed overhead cost per op execution, this explains why models like BlazeFace are faster on WASM.

**これらの結果は、デバイスによって異なります。WASM がアプリケーションに適しているかどうかを判断するするには、さまざまなバックエンドで WASM をテストしてみてください。**

##### 推論とトレーニング

To address the primary use-case for deployment of pre-trained models, the WASM backend development will prioritize *inference* over *training* support. See an [up-to-date list](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts) of supported ops in WASM and [let us know](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) if your model has an unsupported op. For training models, we recommend using the Node (TensorFlow C++) backend or the WebGL backend.

#### CPU バックエンド

CPU バックエンド 「cpu'」はパフォーマンスが最も低いバックエンドですが、最もシンプルです。演算はすべて標準的な JavaScript で実装されているため、並列性が低下しています。また、UI スレッドもブロックします。

このバックエンドは、テストや WebGL が利用できないデバイスで非常に役立ちます。

### フラグ

TensorFlow.js には、自動的に評価し、その時点のプラットフォームで最適な構成を決定する一連の環境フラグがあります。これらのフラグはほとんどが内部フラグですが、いくつかのグローバルフラグはパブリック API で制御できます。

- `tf.enableProdMode():` 本番モードを有効にします。パフォーマンスを優先するために、モデルの検証、NaN チェック、およびその他の正当性チェックが削除されます。
- `tf.enableDebugMode()`: デバッグモードを有効にします。これは、実行されるすべての演算、およびメモリフットプリントや合計カーネル実行時間などのランタイムパフォーマンス情報をコンソールに記録します。これはアプリケーションの速度を大幅に低下させることに注意してください。本番環境では使用しないでください。

> 注: これら2つのメソッドは、キャッシュされる他のフラグの値に影響を与えるため、TensorFlow.js コードを使用する前に使用する必要があります。また、同じ理由で「無効」アナログ機能はありません。

> 注: `tf.ENV.features`をコンソールにログすると、評価されたすべてのフラグを確認できます。これらは**パブリック API の一部ではありません**（したがって、バージョン間の安定性が保証されていません）。ただし、プラットフォームおよびデバイス全体の動作のデバッグまたは微調整に役立つ場合があります。フラグの値を上書きするには、`tf.ENV.set`を使用します。

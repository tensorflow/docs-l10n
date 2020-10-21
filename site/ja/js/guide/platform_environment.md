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

tf.matMul(a, b)のような演算が呼び出されると、結果の tf.Tensor は同期的に返されますが、行列乗算の計算は実際には準備ができていない場合があります。これは、返される tf.Tensor が計算の単なるハンドルであることを意味します。`x.data()`または`x.array()`を呼び出すと、計算が実際に完了したときに値が解決されます。計算が完了するまで UI スレッドをブロックしないように非同期の`x.data()`および`x.array()`メソッドを対応する同期する`x.dataSync()`および`x.arraySync()`に対して使用することが重要になります。

##### メモリ管理

WebGL バックエンドを使用する際の注意点の 1 つは、明示的なメモリ管理の必要性です。テンソルデータが最終的に格納される WebGLTextures は、ブラウザによって自動的にガベージコレクションされません。

`tf.Tensor`のメモリを破棄するには、`dispose()`メソッドを使用します。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose();
```

アプリケーションで複数の演算をチェーン化することは非常に一般的です。それらを破棄する場合、すべての中間変数への参照を保持すると、コードが読みにくくなる可能性があります。この問題を解決するために、TensorFlow.js は`tf.tidy()`メソッドを提供します。これは、関数の実行時にローカル変数がクリーンアップされるように、関数の実行後に関数から返されないすべての`tf.Tensor`をクリーンアップします。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

> 注: 自動ガベージコレクションを持つ非 WebGL 環境 (Node.js や CPU バックエンドなど) で`dispose()`や`tidy()`を使用しても問題はありません。多くの場合、ガベージコレクションは、自動的に発生するガベージコレクションよりも速くテンソルメモリを解放できるのでパフォーマンスが向上します。

##### 精度

モバイルデバイスでは、WebGL は 16 ビットの浮動小数点テクスチャのみをサポートする場合があります。ただし、ほとんどの機械学習モデルは、32 ビット浮動小数点の重みとアクティベーションでトレーニングされています。16 ビットの浮動小数点数が`[0.000000059605, 65504]`の範囲の数値しか表現できないため、モバイルデバイスのモデルを移植するときに精度の問題を引き起こす可能性があります。そのため、モデルの重みとアクティブ化がこの範囲を超えないように注意する必要があります。デバイスが 32 ビットテクスチャをサポートするかどうかを確認するには、`tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE')`の値を確認します。これが false の場合、デバイスは 16 ビット浮動小数点テクスチャのみをサポートします。TensorFlow.js がその時点で 32 ビットテクスチャを使用しているかどうかを確認するには`tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')` を使用します。

##### シェーダーのコンパイルとテクスチャのアップロード

TensorFlow.js は、WebGL シェーダープログラムを実行することにより、GPU で演算を実行します。これらのシェーダーは、ユーザーが演算の実行を要求すると、レイジーにアセンブルおよびコンパイルされます。シェーダーのコンパイルはメインスレッドの CPU で行われ、遅くなる可能性があります。TensorFlow.js はコンパイルされたシェーダーを自動的にキャッシュし、同じ形状の入力テンソルと出力テンソルで同じ演算を 2 回目に呼び出します。通常、TensorFlow.js アプリケーションは、アプリケーションの存続期間中に同じ演算を複数回使用するため、機械学習モデルの 2 回目のパスははるかに高速です。

TensorFlow.js はまた、tf.Tensor データを WebGLTextures として保存します。`tf.Tensor`が作成されると、データをすぐに GPU にアップロードするのではなく、演算で`tf.Tensor`が使用されるまで CPU にデータを保持します。`tf.Tensor`が 2 回目に使用される場合、データはすでに GPU にあるため、アップロードのコストはありません。典型的な機械学習モデルでは、モデルを介して最初の予測時に重みがアップロードされ、モデルの 2回目のパスがはるかに高速になります。

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

このバックエンドでは、WebGL バックエンドと同様に、演算は`tf.Tensor`を同期的に返します。ただし、WebGL バックエンドとは異なり、テンソルを取得する前に演算が完了します。つまり、`tf.matMul(a, b)`を呼び出すと、UI スレッドがブロックされます。

このため、本番環境アプリケーションでこれを使用する場合は、メインスレッドをブロックしないようにワーカースレッドで TensorFlow.js を実行する必要があります。

Node.js についての詳細はこちらを参照してください。

#### WASM バックエンド

TensorFlow.js は [WebAssembly バックエンド](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md) (`wasm`)を提供します。これは、CPU アクセラレーションを提供し、標準的な JavaScript CPU(`cpu`)および WebGL アクセラレーション(`webgl`)バックエンドの代替として使用できます。以下のように使用します。

```js
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm');
tf.ready().then(() => {...});
```

サーバーが`.wasm`ファイルを別のパスまたは別の名前で提供している場合は、バックエンドを初期化する前に`setWasmPath`を使用します。詳細については、README の[「Bundlers の使用」](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers)セクションを参照してください。

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

一般的に、WASM はモデルが小さい場合、または WebGL (`OES_texture_float` 拡張機能) サポートがないか、性能が低い GPU が搭載されたローエンドデバイスがある場合に適しています。以下の表は、WebGL、WASM、および CPU バックエンドで公式にサポートされている 5 つの[モデル](https://github.com/tensorflow/tfjs-models)の 2018 MacBook Pro 上の Chrome における推論時間 (TensorFlow.js 1.5.2) を示しています。

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

上記の表は、WASM がモデル全体で標準的な JS CPU バックエンドよりも 10〜30 倍速く、[BlazeFace](https://github.com/tensorflow/tfjs-models/tree/master/blazeface) のような小規模モデルの WebGL と同等であることを示しています。BlazeFace は小型 (400KB) ですが、演算数 (〜140) は適切です。WebGL プログラムでは演算を実行するたびに固定オーバーヘッドコストが発生するため、BlazeFace などのモデルでは WASM の方が高速になります。

**これらの結果は、デバイスによって異なります。WASM がアプリケーションに適しているかどうかを判断するするには、さまざまなバックエンドで WASM をテストしてみてください。**

##### 推論とトレーニング

事前トレーニング済みモデルのデプロイの主なユースケースに対処するために、WASM バックエンド開発では、*トレーニング*サポートよりも*推論*サポートを優先します。WASM でサポートされている演算の[最新リスト](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/kernels/all_kernels.ts)を参照し、モデルにサポートされていない演算がある場合は[お知らせください](https://github.com/tensorflow/tfjs/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)。モデルのトレーニングには、Node (TensorFlow C++) バックエンドまたは WebGL バックエンドの使用をお勧めします。

#### CPU バックエンド

CPU バックエンド 「cpu'」はパフォーマンスが最も低いバックエンドですが、最もシンプルです。演算はすべて標準的な JavaScript で実装されているため、並列性が低下しています。また、UI スレッドもブロックします。

このバックエンドは、テストや WebGL が利用できないデバイスで非常に役立ちます。

### フラグ

TensorFlow.js には、自動的に評価し、その時点のプラットフォームで最適な構成を決定する一連の環境フラグがあります。これらのフラグはほとんどが内部フラグですが、いくつかのグローバルフラグはパブリック API で制御できます。

- `tf.enableProdMode():` 本番モードを有効にします。パフォーマンスを優先するために、モデルの検証、NaN チェック、およびその他の正当性チェックが削除されます。
- `tf.enableDebugMode()`: デバッグモードを有効にします。これは、実行されるすべての演算、およびメモリフットプリントや合計カーネル実行時間などのランタイムパフォーマンス情報をコンソールに記録します。これはアプリケーションの速度を大幅に低下させることに注意してください。本番環境では使用しないでください。

> 注: これら2つのメソッドは、キャッシュされる他のフラグの値に影響を与えるため、TensorFlow.js コードを使用する前に使用する必要があります。また、同じ理由で「無効」アナログ機能はありません。

> 注: `tf.ENV.features`をコンソールにログすると、評価されたすべてのフラグを確認できます。これらは**パブリック API の一部ではありません**（したがって、バージョン間の安定性が保証されていません）。ただし、プラットフォームおよびデバイス全体の動作のデバッグまたは微調整に役立つ場合があります。フラグの値を上書きするには、`tf.ENV.set`を使用します。

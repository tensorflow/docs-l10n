# TensorFlow.js 3.0 へのアップグレード

## TensorFlow.js 3.0 での変更点

リリースノートは[こちらから入手できます](https://github.com/tensorflow/tfjs/releases)。いくつかの注目すべきユーザー向け機能は次のとおりです。

### カスタムモジュール

サイズが最適化されたブラウザバンドルの作成をサポートするカスタム tfjs モジュールの作成をサポートします。ユーザーに出荷する JavaScript の数を減らします。これについて詳しくは、[こちらのチュートリアルをご覧ください](https://github.com/tensorflow/tfjs-website/blob/master/docs/tutorials/deployment/size_optimized_bundles.md)。

この機能はブラウザでの展開を対象としていますが、この機能を有効にすると、以下で説明するいくつかの変更を実行する動機づけになります。

### ES2017 コード

いくつかのプリコンパイルバンドルのほかに、**コードを [ES2017 構文](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)の [ES モジュール](https://2ality.com/2016/02/ecmascript-2017.html)として NPM に公開できます。**これにより、開発者は[最新の JavaScript 機能](https://web.dev/publish-modern-javascript/)を利用して、エンドユーザーに出荷する内容をより細かく制御できます。

package.json `module` のエントリポイントは、ES2017形式（バンドルではない）の個々のライブラリファイルを指します。これによりツリーシェイキングが可能になり、開発者が下流の変換をより細かく制御できるようになります。

レガシーブラウザやその他のモジュールシステムをサポートするために、コンパイル済みのバンドルとしていくつかの代替フォーマットを提供しています。これらは、以下の表で説明されている命名規則に従っており、JsDelivr や Unpkg などの一般的な CDN からロードできます。

<table>
  <tr>
   <td>ファイル名</td>
   <td>モジュール形式</td>
   <td>言語バージョン</td>
  </tr>
  <tr>
   <td>tf[-package].[min].js*</td>
   <td>UMD</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.[min].js</td>
   <td>UMD</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>tf[-package].node.js**</td>
   <td>CommonJS</td>
   <td>ES5</td>
  </tr>
  <tr>
   <td>tf[-package].es2017.fesm.[min].js</td>
   <td>ESM（単一のフラットファイル）</td>
   <td>ES2017</td>
  </tr>
  <tr>
   <td>index.js***</td>
   <td>ESM</td>
   <td>ES2017</td>
  </tr>
</table>

* [package] は、メインの tf.js パッケージのサブパッケージの core/converter/layers などの名前を指します。[min] は、縮小されていないファイルに加えて、縮小されたファイルを提供する場所を示しています。

** package.json `main` エントリはこのファイルを指しています。

*** package.json `module` のエントリはこのファイルを指しています。

npm 経由で tensorflow.js を使用していて、バンドラーを使用している場合は、バンドラー構成を調整して、ES2017 モジュールを使用するか、outpackage.json の別のエントリを指すようにする必要があります。

### デフォルトでよりスリムな @tensorflow/tfjs-core

より効果的な [tree-shaking](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking) を可能にするために、@tensorflow/tfjs-core にデフォルトでテンソルの chaining/fluent API が含まれなくなりました。最小のバンドルを取得するには、演算（ops）を直接使用することをお勧めします。chaining api を復元するインポート `import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';` を提供します。

また、デフォルトでカーネルの勾配を登録しなくなりました。勾配/トレーニングのサポートが必要な場合は、`import '@tensorflow/tfjs-core/dist/register_all_gradients';` を使用することができます。

> 注: @tensorflow/tfjs または @tensorflow/tfjs-layers、あるいはその他の高レベルのパッケージを使用している場合、これは自動的に行われます。

### コードの再編成、カーネルおよび勾配レジストリ

コードを再編成して、演算とカーネルの両方に貢献しやすくするとともに、カスタム演算、カーネル、勾配を実装しやすくしました。[詳細については、こちらガイドをご覧ください](https://www.tensorflow.org/js/guide/custom_ops_kernels_gradients)。

### 重大な変更

重大な変更の完全なリストは[ここ](https://github.com/tensorflow/tfjs/releases)にありますが、mulStrict や addStrict などのすべての *厳密演算の削除が含まれています。

## 2.x からのコードのアップグレード

### @tensorflow/tfjs のユーザー

ここにリストされている重大な変更に対応します（https://github.com/tensorflow/tfjs/releases）

### @tensorflow/tfjs-core のユーザー

ここ（https://github.com/tensorflow/tfjs/releases）にリストされている重大な変更に対応してから、次の手順を実行します。

#### 連鎖した演算 augmentor を追加、または演算を直接使用する

これよりも

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = a.sum(); // this is a 'chained' op.
```

以下を実行する必要があります

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-core/dist/public/chained_ops/sum'; // add the 'sum' chained op to all tensors

const a = tf.tensor([1,2,3,4]);
const b = a.sum();
```

次のインポートを使用して、 chaining/fluent API をインポートすることもできます

```
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
```

または、演算を直接使用することもできます（ここでも名前付きインポートを使用できます）

```
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

const a = tf.tensor([1,2,3,4]);
const b = tf.sum(a);
```

#### 初期化コードをインポートする

（`import * as ...` の代わりに）名前付きインポートのみを使用している場合、場合によっては次を行う必要があります。

```
import @tensorflow/tfjs-core
```

プログラムの上部近くにあるため、積極的な tree-shaker が必要な初期化をドロップするのを防ぎます。

## 1.x からのコードのアップグレード

### @tensorflow/tfjs のユーザー

[ここ](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0)にリストされている重大な変更に対応します。その後、2.x からアップグレードするためのステップに従います。

### @tensorflow/tfjs-core のユーザー

[ここ](https://github.com/tensorflow/tfjs/releases/tag/tfjs-v2.0.0)にリストされている重大な変更に対応し、以下に説明するようにバックエンドを選択して、2.x からアップグレードするためのステップに従います。

#### バックエンドの選択

TensorFlow.js 2.0 では、cpu および webgl バックエンドを独自のパッケージに移動しました。これらのバックエンドを含める方法については、[@tensorflow/tfjs-backend-cpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-cpu)、[@tensorflow/tfjs-backend-webgl](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgl)、[@tensorflow/tfjs-backend-wasm](https://www.npmjs.com/package/@tensorflow/tfjs-backend-wasm)、[@tensorflow/tfjs-backend-webgpu](https://www.npmjs.com/package/@tensorflow/tfjs-backend-webgpu) をご覧ください。

# TensorFlow.js を使用してサイズが最適化されたブラウザバンドルを生成する

## Overview

TensorFlow.js 3.0 は、*サイズが最適化された本番環境向けブラウザバンドル*の構築をサポートします。別の言い方をすれば、ブラウザへのJavaScriptの出荷を簡単にできるようにしたいと考えています。

この機能は、ペイロードからバイトを削ることで特にメリットが得られる（したがって、これを達成するための努力を惜しまない）本番ユースケースのユーザーを対象としています。この機能を使用するには、[ES モジュール](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)、[webpack](https://webpack.js.org/) や [rollup](https://rollupjs.org/guide/en/) などの JavaScript バンドルツール、および [ツリーシェイキング/デッドコード削除](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)などの概念に精通している必要があります。

このチュートリアルでは、バンドラーで使用できるカスタム tensorflow.js モジュールを作成して、tensorflow.js でプログラムのサイズが最適化されたビルドを生成する方法を示します。

### 用語

このドキュメントのコンテキストで使用するいくつかの重要な用語には次があります。

**[ES モジュール](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules)** - **標準の JavaScript モジュールシステム**。 ES6/ES2015 で導入されました。**import** および **export** ステートメントの使用により<br>識別できます。

**バンドル** - 一連の JavaScript アセットを取得し、それらをブラウザで使用可能な 1 つ以上の JavaScript アセットにグループ化/バンドルします。これは通常、ブラウザに提供される最終的なアセットを生成するステップです。***アプリケーションは通常、トランスパイルされたライブラリソースから直接独自のバンドルを実行します*。**一般的な**バンドラー**には、*rollup* と *webpack* があります。バンドルの最終結果は、**バンドル**として知られています（または、複数の部分に分割されている場合は**チャンク**として知られています）

**[ツリーシェイキング / デッドコード削除](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking)** - 最終的に作成されたアプリケーションで使用されていないコードの削除。これは、バンドル中に、*通常*は縮小ステップで行われます。

**演算（Ops）** - 1 つ以上のテンソルを出力として生成する 1 つ以上のテンソルに対する数学演算。演算は「高レベル」コードであり、他の演算を使用してロジックを定義できます。

**カーネル** - 特定のハードウェア機能に関連付けられた演算固有の実装。カーネルは「低レベル」でバックエンド固有です。演算からカーネルへの 1 対 1 のマッピングがある演算もあれば、複数のカーネルを使用する演算もあります。

## 範囲とユースケース

### 推論のみのグラフモデル

これに関連するユーザーから聞いた、このリリースでサポートしている主なユースケースは、***TensorFlow.js グラフモデル*で推論**を行うことです。 *TensorFlow.js レイヤーモデル*を使用している場合は、 [tfjs-converter](https://www.npmjs.com/package/@tensorflow/tfjs-converter) を使用してこれをグラフモデル形式に変換できます。グラフモデル形式は、推論のユースケースに対してより効率的です。

### tfjs-core による低レベルのテンソル操作

サポートされる他のユースケースは、@tensorflow/tjfs-core パッケージを直接使用して低レベルのテンソル操作を行うプログラムです。

## カスタムビルドへのアプローチ

この機能を設計する際の基本原則には、次のものが含まれます。

- JavaScript モジュールシステム（ESM）を最大限に活用し、TensorFlow.js のユーザーが同じことを行えるようにする。
- TensorFlow.js を*既存のバンドラー*（webpack、rollup など）で可能な限りツリーシェイキングを可能にする。これにより、ユーザーはコード分割などの機能を含む、これらのバンドラーのすべての機能を利用できます。
- *バンドルサイズにそれほど敏感でないユーザーのために使いやすさ*を可能な限り維持する。これは、ライブラリのデフォルトの多くがサイズが最適化されたビルドよりも使いやすさをサポートしているため、本番ビルドにはより多くの労力が必要になることを意味します。

ワークフローの主な目標は、最適化しようとしているプログラムに必要な機能のみを含む TensorFlow.js 用のカスタム *JavaScript モジュール*を作成することです。実際の最適化は、既存のバンドラーに依存しています。

主に JavaScript モジュールシステムに依存していますが、ユーザー向けコードでモジュールシステムを介して指定するのが簡単ではない部分を処理するための*カスタム* *CLI ツール*も提供しています。この2つの例は次のとおりです。

- `model.json` ファイルに保存されているモデル仕様
- 使用しているバックエンド固有のカーネルディスパッチングシステムに対する演算。

これにより、カスタム tfjs ビルドの生成は、バンドラーを通常の @tensorflow/tfjs パッケージにただポイントするよりも少し複雑になります。

## サイズが最適化されたカスタムバンドルを作成する方法

### ステップ 1. プログラムが使用しているカーネルを特定する

**この手順により、実行するモデル、または選択したバックエンドで前処理/後処理するコードにより使用されるすべてのカーネルを特定できます。**

tf.profile を使用して、tensorflow.js を使用するアプリケーションの部分を実行し、カーネルを取得します。次のようになります。

```
const profileInfo = await tf.profile(() => {
  // You must profile all uses of tf symbols.
  runAllMyTfjsCode();
});

const kernelNames = profileInfo.kernelNames
console.log(kernelNames);
```

次のステップのために、カーネルのリストをクリップボードにコピーします。

> カスタムバンドルで使用するのと同じバックエンドを使用してコードをプロファイリングする必要があります。

> モデルが変更された場合、または前処理/後処理コードが変更された場合は、このステップを繰り返す必要があります。

### ステップ 2. カスタム tfjs モジュールの構成ファイルを作成する

構成ファイルの例を次に示します。

次のようになります。

```
{
  "kernels": ["Reshape", "_FusedMatMul", "Identity"],
  "backends": [
      "cpu"
  ],
  "models": [
      "./model/model.json"
  ],
  "outputPath": "./custom_tfjs",
  "forwardModeOnly": true
}
```

- kernels: バンドルに含めるカーネルのリスト。ステップ 1 の出力からこれをコピーします。
- backends: 含めるバックエンドのリスト。有効なオプションには、"cpu"、"webgl"、および “wasm” が含まれます。
- models: アプリケーションにロードするモデルの model.json ファイルのリスト。プログラムが tfjs_converter を使用してグラフモデルをロードしない場合は、空にすることができます。
- outputPath: 生成されたモジュールを配置するフォルダーへのパス。
- forwardModeOnly: 前にリストされたカーネルの勾配を含める場合は、これを false に設定します。

### ステップ 3. カスタム tfjs モジュールを生成する

構成ファイルを引数としてカスタムビルドツールを実行します。このツールにアクセスするには、**@tensorflow/tfjs** パッケージをインストールする必要があります。

```
npx tfjs-custom-module  --config custom_tfjs_config.json
```

これにより、`outputPath`にいくつかの新しいファイルを含むフォルダーが作成されます。

### ステップ 4. tfjsを新しいカスタムモジュールにエイリアスするようにバンドラーを構成する。

webpack や rollup などのバンドラーでは、tfjs モジュールへの既存の参照をエイリアスして、新しく生成されたカスタム tfjs モジュールを指すことができます。バンドルサイズを最大限に節約するために、エイリアス化する必要のあるモジュールは 3 つあります。

これが webpack でどのように見えるかのスニペットです（[完全な例はこちらにあります](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/webpack.config.js)）:

```
...

config.resolve = {
  alias: {
    '@tensorflow/tfjs$':
        path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    '@tensorflow/tfjs-core$': path.resolve(
        __dirname, './custom_tfjs/custom_tfjs_core.js'),
    '@tensorflow/tfjs-core/dist/ops/ops_for_converter': path.resolve(
        __dirname, './custom_tfjs/custom_ops_for_converter.js'),
  }
}

...
```

そして、これが rollup の同等のコードスニペットです（[完全な例はこちらにあります](https://github.com/tensorflow/tfjs/blob/master/e2e/custom_module/dense_model/rollup.config.js)）:

```
import alias from '@rollup/plugin-alias';

...

alias({
  entries: [
    {
      find: /@tensorflow\/tfjs$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs.js'),
    },
    {
      find: /@tensorflow\/tfjs-core$/,
      replacement: path.resolve(__dirname, './custom_tfjs/custom_tfjs_core.js'),
    },
    {
      find: '@tensorflow/tfjs-core/dist/ops/ops_for_converter',
      replacement: path.resolve(__dirname, './custom_tfjs/custom_ops_for_converter.js'),
    },
  ],
}));

...
```

> バンドラーがモジュールエイリアシングをサポートしていない場合は、`import` ステートメントを変更して、ステップ 3 で作成した生成された custom_tfjs.js から `custom_tfjs.js` をインポートする必要があります。演算定義はツリーシェイキングにより削除されませんが、カーネルは引き続きツリーシェイキングされます。一般に、カーネルのツリーシェイキングにより、最終的なバンドルサイズを最大に節約することができます。

> @tensoflow/tfjs-core パッケージのみを使用している場合は、その 1 つのパッケージにエイリアスを設定するだけで済みます。

### ステップ 5. バンドルを作成する

バンドルを作成するには、バンドラー（`webpack` や `rollup` など）を実行します。バンドルのサイズは、モジュールのエイリアシングなしでバンドラーを実行する場合よりも小さくする必要があります。[このような](https://www.npmjs.com/package/rollup-plugin-visualizer)ビジュアライザーを使用して、最終的なバンドルを確認することもできます。

### ステップ 6. アプリをテストする

アプリが期待どおりに機能していることを確認してください。

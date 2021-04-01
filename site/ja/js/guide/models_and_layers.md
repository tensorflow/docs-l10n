# モデルとレイヤー

機械学習では、*モデル*は入力を出力にマッピングする*学習可能*[パラメータ](https://developers.google.com/machine-learning/glossary/#parameter)を備えた関数です。モデルをデータでトレーニングすることにより最適なパラメータを取得できます。よくトレーニングされたモデルは、入力から目的とする出力への正確なマッピングを提供します。

TensorFlow.js には、機械学習モデルを作成する 2つ の方法があります。

1. Layers API を使用し、*レイヤー*を使用してモデルを構築する。
2. `tf.matMul()`や`tf.add()`などの下位レベルの演算で Core API を使用する。

まず、モデルを作成するための上位レベルの API である Layers API を見ていきます。次に、Core API を使用して同じモデルを構築する方法を示します。

## Layers API を使用したモデルの作成

Layers API を使用してモデルを作成するには、*Sequential* モデルと *Functional* モデルの 2 つの方法があります。次の 2 セクションでは、各タイプについて詳しく説明します。

### Sequential モデル

最も一般的なモデルのタイプは<code>Sequential</code>で、これはレイヤーの線形スタックです。レイヤーのリストを<code>sequential()</code>関数に渡すことにより、<code>Sequential</code>モデルを作成できます。

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

または、`add()`メソッドを使用します。

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> 重要: モデルの最初のレイヤーには`inputShape`が必要です。 `inputShape`を指定するときは、バッチサイズを除外してください。たとえば、形状`[B, 784]`のモデルテンソルをフィードする場合 (`B`は任意のバッチサイズ)、モデルの作成時に`inputShape` を`[784]`として指定します。

`model.layers`を介してモデルのレイヤーにアクセスできます (より具体的には、`model.inputLayers`および`model.output Layers`)。

### Functional モデル

`LayersModel`を作成する別の方法は、`tf.model()`関数を使用することです。`tf.model()`と`tf.sequential()`の主な違いは、`tf.model()`を使用するとサイクルがない限り、レイヤーの任意のグラフを作成できることです。

以下は、`tf.model()` API を使用して上記と同じモデルを定義するコードスニペットです。

```js
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

別のレイヤーの出力に接続するために、各レイヤーで`apply()`を呼び出します。この場合の`apply()`の結果は`SymbolicTensor`であり、`Tensor`のように機能しますが、具体的な値はありません。

Sequential モデルとは異なり、`input_Shape`を提供する代わりに、`tf.input()`を介して最初のレイヤーに`SymbolicTensor`を作成することに注意してください。

`apply()`は、具体的な`Tensor`を渡すと、具体的な`Tensor`も提供します。

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

これは、レイヤーを個別にテストしてその出力を確認するときに役立ちます。

Sequential モデルと同じように、`model.layers`を介してモデルのレイヤーにアクセスできます (より具体的には、`model.inputLayers`および`model.outputLayers`)。

## 検証

Sequential モデルと Functional モデルはどちらも`LayersModel`クラスのインスタンスです。`LayersModel`を使用する主な利点の 1 つは検証です。これにより、入力形状を指定する必要があり、後でそれを使用して入力を検証します。`LayersModel`は、データがレイヤーを通過するときに自動的に形状を推論します。形状を事前に指定しておくと、モデルはパラメータを自動的に作成し、連続する 2 つのレイヤーに互換性がない場合は判断できます。

## モデルの概要

`model.summary()`を呼び出すと、次のような有用なモデルの概要を出力できます。

- モデル内のすべてのレイヤー名とタイプ。
- 各レイヤーの出力形状。
- 各レイヤーの重みパラメータの数。
- モデルに一般的なトポロジーがある場合（以下で説明）、各レイヤーが受け取る入力。
- モデルのトレーニング可能なパラメータとトレーニング不可能なパラメータの総数。

上記で定義したモデルの場合、コンソールに次の出力が表示されます。

<table>
  <tr>
   <td>レイヤー（タイプ）</td>
   <td>出力形状</td>
   <td>パラメータ #</td>
  </tr>
  <tr>
   <td>dense_Dense1（蜜）</td>
   <td>[null,32]</td>
   <td>25120</td>
  </tr>
  <tr>
   <td>dense_Dense2（蜜）</td>
   <td>[null,10]</td>
   <td>330</td>
  </tr>
  <tr>
   <td colspan="3">パラメータ総数: 25450<br>トレーニング可能なパラメータ: 25450<br> トレーニング不可能なパラメータ: 0</td>
  </tr>
</table>

レイヤーの出力形状の`null`値に注意してください。モデルは入力にバッチサイズがあることを期待しています (最も外側の次元として)。この場合、値は`null`なので、柔軟性があります。

## シリアル化

下位レベルの API と比べて`LayersModel`を使用する主な利点の 1 つは、モデルを保存および読み込めることです。`LayersModel`は、以下を認知しています。

- モデルを再作成するためのモデルのアーキテクチャ。
- モデルの重み
- トレーニング構成（損失、オプティマイザ、メトリック）。
- トレーニングを再開するためのオプティマイザの状態。

わずか 1 行のコードでモデルを保存または読み込むことができます。

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

上記の例では、ブラウザのローカルストレージにモデルを保存します。別のメディア (ファイルストレージ、<code>IndexedDB</code>、ブラウザダウンロードのトリガーなど) に保存する方法については、<code>model.save() ドキュメント</code>および<a>保存と読み込み</a>ガイドを参照してください。

## カスタムレイヤー

レイヤーはモデルの構成要素です。モデルがカスタム計算を実行している場合は、他のレイヤーと適切に相互作用するカスタムレイヤーを定義できます。以下では、平方和を計算するカスタムレイヤーを定義します。

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

これをテストするには、具体的なテンソルを使用して`apply()`メソッドを呼び出します。

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> 重要: カスタムレイヤーを追加すると、モデルをシリアル化できなくなります。

## Core API を使用したモデルの作成

このガイドの冒頭で、TensorFlow.js で機械学習モデルを作成する方法が 2 つあることを説明しました。

一般的な経験則は、[ベストプラクティスと認知負荷の軽減](https://keras.io/why-use-keras/)に従い、レイヤー API を常に最初に使用することです。Layers API は、重みの初期化、モデルのシリアル化、監視トレーニング、移植性、安全性チェックなど、さまざまな既製のソリューションも提供します。

次の場合は Core API を使用することができます。

- 最大限の柔軟性または制御が必要な場合。
- シリアル化が必要ない場合。または、独自のシリアル化ロジックを実装する場合。

Core API のモデルは、1つ以上の`Tensors`を取り、`Tensor`を返す関数です。上記と同じモデルを Core API を使用して記述すると次のようになります。

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

Core API では、モデルの重みを作成および初期化する必要があることに注意してください。すべての重みは、これらのテンソルが学習可能であることを TensorFlow.js に通知する`Variable`によって裏付けられています。<a>tf.variable()</a> を使用して<code>Variable</code>を作成し、既存の`Tensor`に渡すことができます。

このガイドでは、Layers API と Core API を使用してモデルを作成するさまざまな方法を紹介しました。次に、モデルをトレーニングする方法については、[モデルのトレーニング](train_models.md)ガイドをご覧ください。

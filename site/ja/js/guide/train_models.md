# モデルのトレーニング

このガイドは、 [モデルとレイヤーの](models_and_layers.md)ガイドをすでに読んでいることを前提としています。

TensorFlow.js には、機械学習モデルをトレーニングする 2つ の方法があります。

1. Layers API を <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fit" data-md-type="link"&gt;LayersModel.fit()&lt;/a&gt;</code> か <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;LayersModel.fitDataset()&lt;/a&gt;</code> と共に使用する。
2. Core API を <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;Optimizer.minimize()&lt;/a&gt;</code> と共に使用する。

まず、モデルを構築してトレーニングするための上位レベルの API である Layers API を見ていきます。 次に、Core API を使用して同じモデルをトレーニングする方法を示します。

## はじめに

機械学習の*モデル*は、入力を目的の出力にマッピングする学習可能なパラメータを持つ関数です。最適なパラメータは、データでモデルをトレーニングすることによって取得されます。

トレーニングにはいくつかのステップがあります。

- データの[バッチ](https://developers.google.com/machine-learning/glossary/#batch)をモデルに取得します。
- モデルに予測を立てるよう要求します。
- その予測を "true" 値と比較します。
- モデルが今後そのバッチに対してより優れた予測を立てられるよう、各パラメーターをどれだけ変更するのかを決定します。

十分にトレーニングされたモデルは、入力から目的の出力へ正確にマッピングします。

## モデルパラメータ

Layers API を使用し、次のように単純な 2 レイヤーモデルを定義してみましょう。

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

内部的には、モデルはデータによるトレーニングで学習可能なパラメータ（しばしば*重み*と呼ばれる）を保持しています。このモデルに関連する重みの名前と形状を印刷してみましょう。

```js
model.weights.forEach(w => {
 console.log(w.name, w.shape);
});
```

以下の出力が得られます。

```
> dense_Dense1/kernel [784, 32]
> dense_Dense1/bias [32]
> dense_Dense2/kernel [32, 10]
> dense_Dense2/bias [10]
```

重みは合計で 4 つあり、蜜なレイヤーごとに 2 つあります。これは、密なレイヤーが方程式 `y = Ax + b` を介して入力テンソル `x` を出力テンソル `y` にマッピングする関数を表すために期待されます。ここで、 `A`（kernel）および `b`（bias）は密なレイヤーのパラメータです。

> 注意: デフォルトでは蜜なレイヤーに bias が含まれていますが、密なレイヤーを作成するときにオプションで `{useBias: false}` を指定することで bias を除外できます。

`model.summary()` は、モデルの概要を取得してパラメータの総数を確認する場合に便利なメソッドです。

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
   <td colspan="3">パラメータ総数: 25450<br>トレーニング可能なパラメータ: 25450<br>トレーニング不可能なパラメータ: 0</td>
  </tr>
</table>

モデルの各重みは、 <code>&lt;a href="https://js.tensorflow.org/api/0.14.2/#class:Variable" data-md-type="link"&gt;Variable&lt;/a&gt;</code> オブジェクトによるバックエンドです。TensorFlow.js では、<code>Variable</code> はその値を更新するために使用される 1 つの追加メソッド <code>assign()</code> を持つ浮動小数点 <code>Tensor</code> です。Layers API はベストプラクティスを使用して重みを自動的に初期化します。デモ目的のため、基になる変数で <code>assign()</code> を呼び出すことで重みを上書きできます。

```js
model.weights.forEach(w => {
  const newVals = tf.randomNormal(w.shape);
  // w.val is an instance of tf.Variable
  w.val.assign(newVals);
});
```

## オプティマイザ、損失、指標

トレーニングを行う前に、次の3つの項目を決定する必要があります。

1. **オプティマイザ**。オプティマイザの仕事は、現在のモデルの予測を前提として、モデルの各パラメータをどの程度変更するかを決定することです。Layers API を使用する場合、既存オプティマイザの文字列識別子（`'sgd'` または `'adam'` など）か <code>&lt;a href="https://js.tensorflow.org/api/latest/#Training-Optimizers" data-md-type="link"&gt;Optimizer&lt;/a&gt;</code> クラスのインスタンスを指定できます。
2. <strong>損失関数</strong>。モデルが最小化を試みる対象です。その目標は、モデルの予測が「どれほど間違っていたか」を単一の数値で示すことです。モデルの重みを交信できるよう、損失はデータのすべてのバッチで計算されます。Layers API を使用する場合、既存の損失関数の文字列識別子（<code>'categoricalCrossentropy'</code>など）か、予測値と真の値を取り、損失を返す任意の関数を指定できます。API ドキュメントで[利用可能な損失のリスト](https://js.tensorflow.org/api/latest/#Training-Losses)をご覧ください。
3. <strong>指標のリスト</strong>。損失と同様に、 指標は単一の数値を計算し、モデルのパフォーマンスを要約します。指標は通常、各エポックの終わりにデータ全体で計算されます。最低限でも時間の経過と共に損失が減少していることを監視する必要があります。ただし、多くの場合は精度などのより人間に理解しやすい指標が必要です。Layers API を使用する場合、既存指標の文字列識別子（<code>'accuracy'</code> など）か、予測値と真の値を取り、スコアを返す任意の関数を指定できます。API ドキュメントで[利用可能な指標のリスト](https://js.tensorflow.org/api/latest/#Metrics)をご覧ください。

決定したら、次のようにオプションを指定して <code>model.compile()</code> を呼び出し、 <code>LayersModel</code> をコンパイルします。

```js
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});
```

コンパイル中にモデルは検証を行い、選択したオプションが相互に互換性があることを確認します。

## トレーニング

`LayersModel` をトレーニングするには以下の 2 つの方法があります。

- `model.fit()` を使用し、データを 1 つの大きなテンソルとして提供する。
- `model.fitDataset()` を使用し、データを `Dataset` オブジェクト経由で提供する。

### model.fit()

データセットがメインメモリに収まり、単一のテンソルとして使用できる場合は、 `fit()` メソッドを呼び出してモデルをトレーニングできます。

```js
// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
model.fit(data, labels, {
   epochs: 5,
   batchSize: 32,
   callbacks: {onBatchEnd}
 }).then(info => {
   console.log('Final accuracy', info.history.acc);
 });
```

内部的には、`model.fit()` は以下のような多くの処理を実行できます。

- データをトレーニングと検証セットに分割し、検証セットを使用してトレーニング中の進行状況を評価します。
- 分割後にのみ、データをシャッフルします。安全のため、データを `fit()` に渡す前に事前にシャッフルする必要があります。
- 大きなテンソルを `batchSize` サイズのより小さなテンソルに分割します。
- データのバッチに対するモデルの損失を計算しながら、`optimizer.minimize()` を呼び出します。
- 各エポックまたはバッチの開始時と終了時に通知できます。この場合、 `callbacks.onBatchEnd` オプションを使用してすべてのバッチの終了時に通知が届きます。その他にも `onTrainBegin`、`onTrainEnd`、`onEpochBegin`、`onEpochEnd`、`onBatchBegin` のオプションがあります。
- メインスレッドに従い、JS イベントのループでキューに入れられたタスクをタイムリーに処理できるようにします。

詳細については、<code>fit()</code> の<a>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Sequential.fit" data-md-type="link"&gt;ドキュメント&lt;/a&gt;</a>をご覧ください。 Core API を使用する場合は、このロジックを自分で実装する必要があることに注意してください。

### model.fitDataset()

データがメモリに完全に収まらない場合、またはストリーミングされている場合は、 `Dataset` オブジェクトを取る `fitDataset()` を呼び出すことでモデルをトレーニングできます。以下は同じトレーニングコードですが、generator 関数をラップするデータセットを使用しています。

```js
function* data() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomNormal([784]);
 }
}

function* labels() {
 for (let i = 0; i < 100; i++) {
   // Generate one sample at a time.
   yield tf.randomUniform([10]);
 }
}

const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// We zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);

// Train the model for 5 epochs.
model.fitDataset(ds, {epochs: 5}).then(info => {
 console.log('Accuracy', info.history.acc);
});
```

データセットの詳細については、<code>model.fitDataset()</code> の<a>&lt;a href="https://js.tensorflow.org/api/latest/#tf.Model.fitDataset" data-md-type="link"&gt;ドキュメント&lt;/a&gt;</a>をご覧ください。

## 新しいデータの予測

モデルがトレーニングされたら `model.predict()` を呼び出し、未知のデータを予測できます。

```js
// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();
```

注意: [モデルとレイヤー](models_and_layers)のガイドで説明したように、 `LayersModel` では、入力の最も外側の次元がバッチサイズであることが想定されています。上記の例では、バッチサイズは 3 です。

## Core API

以前に、TensorFlow.js で機械学習モデルをトレーニングする方法は 2 つあると述べました。

一般的な経験則は、Layers API を最初に使用することです。なぜなら、よく採用されている Keras API をモデルにして作成されているからです。Layers API は、重みの初期化、モデルのシリアル化、モニタリングトレーニング、移植性、安全性チェックなど、さまざまな既製のソリューションも提供します。

次の場合は Core API を使用することができます。

- 最大限の柔軟性または制御が必要な場合。
- シリアル化が必要ない場合。または、独自のシリアル化ロジックを実装する場合。

この API に関する詳細は、[モデルとレイヤー](models_and_layers.md)のガイドの「Core API」セクションをご覧ください。

Core API を使用して上記と同じモデルを記述すると、次のようになります。

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2);
}
```

Layers API に加えて、Data API も Core API とシームレスに連携します。前に [model.fitDataset()](#model.fitDataset()) セクションで定義した、シャッフルとバッチ処理を行うデータセットを再利用してみましょう。

```js
const xs = tf.data.generator(data);
const ys = tf.data.generator(labels);
// Zip the data and labels together, shuffle and batch 32 samples at a time.
const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
```

モデルをトレーニングしましょう。

```js
const optimizer = tf.train.sgd(0.1 /* learningRate */);
// Train for 5 epochs.
for (let epoch = 0; epoch < 5; epoch++) {
  await ds.forEachAsync(({xs, ys}) => {
    optimizer.minimize(() => {
      const predYs = model(xs);
      const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
      loss.data().then(l => console.log('Loss', l));
      return loss;
    });
  });
  console.log('Epoch', epoch);
}
```

上記のコードは、Core API でモデルをトレーニングするときの標準的なレシピです。

- エポックの数をループします。
- 各エポック内でデータのバッチをループします。`Dataset` を使用する場合、バッチのループには <code>&lt;a href="https://js.tensorflow.org/api/0.15.1/#tf.data.Dataset.forEachAsync" data-md-type="link"&gt;dataset.forEachAsync()&lt;/a&gt;</code> を使用するのが便利です。
- 各バッチについて <code>&lt;a href="https://js.tensorflow.org/api/latest/#tf.train.Optimizer.minimize" data-md-type="link"&gt;optimizer.minimize(f)&lt;/a&gt;</code> を呼び出します。これは <code>f</code> を実行し、事前に定義した 4 つの変数に関して勾配を計算することでその出力を最小化します。
- <code>f</code> は損失を計算します。モデルの予測と真の値を使用し、事前定義した損失関数の 1 つを呼び出します。

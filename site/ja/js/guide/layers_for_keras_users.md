# Keras ユーザー向け TensorFlow.js Layers API

TensorFlow.js の Layers API は Keras に基づいてモデル化されています。[Layers API](https://js.tensorflow.org/api/latest/) は JavaScript と Python の違いを考慮し、 Keras と同様に作成されています。そのため、Python で Keras モデルを開発した経験のあるユーザーは、JavaScript の TensorFlow.js レイヤーに移行しやすくなります。たとえば、次の Keras コードは以下のように JavaScript に変換されます。

```python
# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

ただし、いくつかの違いがあります。このチュートリアルでは、これらについて解説します。これらの違いとその背後にある理論的根拠を理解すれば、Python から JavaScript への移行 (または逆方向の移行) は比較的スムーズになるはずです。

## コンストラクタは JavaScript オブジェクトを構成として受け取る

上記の例の次の Python の行と JavaScript の行を比較します。どちらも [Dense](https://keras.io/layers/core/#dense) レイヤーを作成します。

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

JavaScript 関数には、Python 関数のキーワード引数に相当するものはありません。忘れやすいことですが、JavaScript の位置引数としてコンストラクタオプションを実装することは避けます。また、多数のキーワード引数を持つコンストラクタ([ LSTM ](https://keras.io/layers/recurrent/#lstm)など) での使用を避けます。これが、JavaScript 構成オブジェクトを使用する理由です。<br>このようなオブジェクトは、Python キーワード引数と同じレベルの位置の不変性と柔軟性を提供します。

Model クラスの一部のメソッド( [`Model.compile()`](https://keras.io/models/model/#model-class-api)) も、JavaScript 構成オブジェクトを入力として受け取ります。ただし、`Model.fit()`、 `Model.evaluate()`および`Model.predict()`は少し異なることに注意してください。これらのメソッドは必須の`x` (特徴) および`y` (ラベルまたはターゲット) データを入力として受け取るため、 `x`と`y`は、キーワード引数の役割を果たす後続の構成オブジェクトとは別の位置引数です。以下に例を示します。

```js
// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
```

## Model.fit() は非同期

`Model.fit()`は、ユーザーが TensorFlow.js でモデルトレーニングを実行する主要な方法です。このメソッドは多くの場合、長時間実行され、数秒から数分続くことがあるため、JavaScript 言語の`async`機能を利用して、ブラウザで実行しているときにメイン UI スレッドをブロックしないようにこの関数を使用できるようにします。これは、`async` [fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)など、JavaScript で長時間実行される可能性のある他の関数に似ています。`async`は Python には存在しない構成であることに注意してください。Keras の[`fit()`](https://keras.io/models/model/#model-class-api)メソッドは History オブジェクトを返しますが、JavaScript の`fit()`メソッドに対応するものは History の [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) を返します。これは上記の例のように [await](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await) または then() メソッドで使用できます。

## TensorFlow.js では NumPy を使用しない

Python Keras のユーザーは、頻繁に [NumPy](http://www.numpy.org/) を使用して、上記の例の 2D テンソルの生成などの基本的な数値および配列演算を実行します。

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

TensorFlow.js では、この種の基本的な数値演算はパッケージ自体で行われます。以下に例を示します。

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

`tf.*`名前空間は、行列の乗算などの配列および線形代数演算のための他の多くの関数も提供します。詳細については [TensorFlow.js Core ドキュメント](https://js.tensorflow.org/api/latest/) を参照してください。

## コンストラクタではなく、ファクトリメソッドを使用する

Python の (上記の例からの) この行は、コンストラクタの呼び出しです。

```python
# Python:
model = keras.Sequential()
```

厳密に JavaScript に変換すると、同等のコンストラクタ呼び出しは次のようになります。

```js
// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
```

しかし、1) 「新しい」キーワードはコードを肥大化させ、2) 「新しい」コンストラクタは JavaScript の「悪い部分」と見なされるため ([*JavaScript: 良い部分*](http://archive.oreilly.com/pub/a/javascript/excerpts/javascript-good-parts/bad-parts.html)を参照)「新しい」コンストラクタを使用しないことにしました。TensorFlow.js でモデルとレイヤを作成するには、以下のように、lowerCamelCase 名を持つファクトリメソッドを呼び出します。

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## オプション文字列の値は snake_case ではなく lowerCamelCase

スネークケースが一般的である Python (Keras など) と比較して、JavaScript では、シンボル名にキャメルケースを使用する方が一般的です ([Google JavaScript スタイルガイド](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)を参照)。そのため、以下を含むオプションの文字列値には lowerCamelCase を使用することにしました。

- DataFormat - 例: `channels_first`の代わりに**`channelsFirst`**
- Initializer - 例: `glorot_normal`の代わりに**`glorotNormal`**
- Loss and metrics - 例: `mean_squared_error`の代わりに<strong><code>meanSquaredError</code></strong>、 <code>categorical_crossentropy</code>の代わりに<strong><code>categoricalCrossentropy</code></strong>。

たとえば、上記の例は次のようになります。

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

モデルのシリアル化と逆シリアル化に関しては、ご安心ください。TensorFlow.js の内部メカニズムにより、たとえば Python Keras から事前トレーニング済みのモデルを読み込む場合など、JSON オブジェクトのスネークケースは正しく処理されます。

## 関数として呼び出すのではなく apply() で Layer オブジェクトを実行する

Keras では、Layer オブジェクトに`__call__`メソッドが定義されているので、ユーザーはレイヤーのロジックをオブジェクトを関数として呼び出すことができます。以下に例を示します。

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

この Python 構文シュガーは、TensorFlow.js の apply() メソッドとして実装されています。

```js
// JavaScript:
const myInput = tf.input({shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() は、コンクリートテンソルの命令型 (Eager) 評価をサポートする

現在、Keras では、**call** メソッドは (Python) TensorFlow  `tf.Tensor`オブジェクトでのみ動作します (TensorFlow バックエンドを想定)。これはシンボリックで実際の数値を保持しません。これは、前のセクションの例に示されているものです。ただし、TensorFlow.js では、レイヤーの apply() メソッドはシンボリックモードと命令モードの両方で動作できます。`apply()`が SymbolicTensor( tf.Tensor に非常に類似) で呼び出された場合、戻り値はSymbolicTensor になります。これは通常、モデルの構築中に発生します。ただし、`apply()`が実際の具象テンソル値で呼び出されると、具象テンソルを返します。以下に例を示します。

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

この機能は、(Python) TensorFlow の [Eager Execution](https://www.tensorflow.org/guide/eager)と似ています。動的ニューラルネットワークの構築を可能にするだけでなく、モデル開発時の対話性とデバッグ性も向上します。

## *optimizers* ではなく train の配下にあるオプティマイザ

Keras では、オプティマイザオブジェクトのコンストラクタは`keras.optimizers.*`名前空間の下にあります。TensorFlow.js レイヤーでは、オプティマイザのファクトリメソッドは`tf.train.*`名前空間の下にあります。以下に例を示します。

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## HDF5 ファイルではなく URL から読み込む loadLayersModel()

Keras ではモデルは通常、HDF5 (.h5) ファイルとして[保存](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)され、後で`keras.models.load_model()`メソッドを使用して読み込みます。このメソッドは、.h5 ファイルへのパスを取得します。[`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel)は、TensorFlow.js の`load_model()`に対応します。HDF5 はブラウザフレンドリーなファイル形式ではないため、`tf.loadLayersModel()`は TensorFlow.js 固有の形式を取ります。`tf.loadLayersModel()`は、入力引数として model.json ファイルを取ります。model.json は、tensorflowjs pip パッケージを使用して Keras HDF5 ファイルから変換できます。

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

また、`tf.loadLayersModel()`は[`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model)の[`Promise`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)を返すことに注意してください。

一般的に、TensorFlow.js での`tf.Model`の保存は`tf.Model.save`メソッド、読み込みは、`tf.loadLayersModel`メソッドを使用して実行されます。これらの API は、Keras の [save と load_model API](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) と同様に設計されています。しかし、ブラウザ環境は Keras のような主要なディープラーニングフレームワークが実行されるバックエンド環境とはかなり異なります。特に、データを永続化および変換するための一連のルートが異なります。したがって、TensorFlow.js と Keras の保存/読み込み API の間にはいくつかの興味深い違いがあります。詳細については、「[tf.Model](./save_load.md) の保存と読み込み」チュートリアルを参照してください。

## `fitDataset()`を使用し、`tf.data.Dataset`オブジェクトを使用してモデルをトレーニングする

Python TensorFlow の tf.keras では、[Dataset](https://www.tensorflow.org/guide/datasets) オブジェクトを使用してモデルをトレーニングできます。モデルの`fit()`メソッドは、そのようなオブジェクトを直接受け入れます。TensorFlow.js モデルは、Dataset オブジェクトと同等の JavaScript を使用してトレーニングすることもできます([TensorFlow.js の tf.data API のドキュメント](https://js.tensorflow.org/api/latest/#Data)を参照してください)。ただし、Python とは異なり、Dataset ベースのトレーニングは専用のメソッド、[fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset) を通じて行われます。[fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) メソッドは、テンソルベースのモデルトレーニング専用です。

## Layer および Model オブジェクトのメモリ管理

TensorFlow.js はブラウザの WebGL で実行され Layer オブジェクトと Model オブジェクトの重みは WebGL テクスチャによりサポートされます。ただし、WebGL にはガベージコレクションのサポートが組み込まれていません。Layer および Model オブジェクトは、推論とトレーニングの呼び出し中にユーザーのテンソルメモリを内部的に管理しますが、占有している WebGL メモリを解放する目的でにユーザーがそれらを破棄することも可能です。これは、1 回のページの読み込みで多くのモデルインスタンスが作成および解放される場合に役立ちます。Layer または Model オブジェクトを破棄するには、`dispose()`メソッドを使用します。

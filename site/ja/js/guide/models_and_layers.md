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

This can be useful when testing layers in isolation and seeing their output.

Sequential モデルと同じように、`model.layers`を介してモデルのレイヤーにアクセスできます (より具体的には、`model.inputLayers`および`model.outputLayers`)。

## 検証

Sequential モデルと Functional モデルはどちらも`LayersModel`クラスのインスタンスです。`LayersModel`を使用する主な利点の 1 つは検証です。これにより、入力形状を指定する必要があり、後でそれを使用して入力を検証します。`LayersModel`は、データがレイヤーを通過するときに自動的に形状を推論します。形状を事前に指定しておくと、モデルはパラメータを自動的に作成し、連続する 2 つのレイヤーに互換性がない場合は判断できます。

## モデルの概要

`model.summary()`を呼び出すと、次のような有用なモデルの概要を出力できます。

- Name and type of all layers in the model.
- Output shape for each layer.
- Number of weight parameters of each layer.
- If the model has general topology (discussed below), the inputs each layer receives
- The total number of trainable and non-trainable parameters of the model.

For the model we defined above, we get the following output on the console:

<table>
  <tr>
   <td>Layer (type)    </td>
   <td>Output shape    </td>
   <td>Param #    </td>
  </tr>
  <tr>
   <td>dense_Dense1 (Dense)    </td>
   <td>[null,32]    </td>
   <td>25120    </td>
  </tr>
  <tr>
   <td>dense_Dense2 (Dense)    </td>
   <td>[null,10]    </td>
   <td>330    </td>
  </tr>
  <tr>
   <td colspan="3">Total params: 25450<br>Trainable params: 25450<br> Non-trainable params: 0    </td>
  </tr>
</table>

Note the `null` values in the output shapes of the layers: a reminder that the model expects the input to have a batch size as the outermost dimension, which in this case can be flexible due to the `null` value.

## Serialization

One of the major benefits of using a `LayersModel` over the lower-level API is the ability to save and load a model. A `LayersModel` knows about:

- the architecture of the model, allowing you to re-create the model.
- the weights of the model
- the training configuration (loss, optimizer, metrics).
- the state of the optimizer, allowing you to resume training.

わずか 1 行のコードでモデルを保存または読み込むことができます。

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

The example above saves the model to local storage in the browser. See the <code><a href="https://js.tensorflow.org/api/latest/#tf.Model.save">model.save() documentation</a></code> and the [save and load](save_load.md) guide for how to save to different mediums (e.g. file storage, <code>IndexedDB</code>, trigger a browser download, etc.)

## Custom layers

Layers are the building blocks of a model. If your model is doing a custom computation, you can define a custom layer, which interacts well with the rest of the layers. Below we define a custom layer that computes the sum of squares:

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

To test it, we can call the `apply()` method with a concrete tensor:

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> 重要: カスタムレイヤーを追加すると、モデルをシリアル化できなくなります。

## Creating models with the Core API

このガイドの冒頭で、TensorFlow.js で機械学習モデルを作成する方法が 2 つあることを説明しました。

The general rule of thumb is to always try to use the Layers API first, since it is modeled after the well-adopted Keras API which follows [best practices and reduces cognitive load](https://keras.io/why-use-keras/). The Layers API also offers various off-the-shelf solutions such as weight initialization, model serialization, monitoring training, portability, and safety checking.

You may want to use the Core API whenever:

- You need maximum flexibility or control.
- You don't need serialization, or can implement your own serialization logic.

Models in the Core API are just functions that take one or more `Tensors` and return a `Tensor`. The same model as above written using the Core API looks like this:

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

Note that in the Core API we are responsible for creating and initializing the weights of the model. Every weight is backed by a `Variable `which signals to TensorFlow.js that these tensors are learnable. You can create a `Variable` using [tf.variable()](https://js.tensorflow.org/api/latest/#variable) and passing in an existing `Tensor`.

In this guide you have familiarized yourself with the different ways to create a model using the Layers and the Core API. Next, see the [training models](train_models.md) guide for how to train a model.

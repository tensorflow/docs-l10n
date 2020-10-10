# テンソルと演算

TensorFlow.js は JavaScript でテンソルを使用し、計算を定義および実行するためのフレームワークです。*テンソル*は、ベクトルや行列の高次元への一般化です。

## テンソル

TensorFlow.js のデータの中心的な単位は`tf.Tensor`という、1 つ以上の次元配列の形をとる値のセットです。`tf.Tensor`は多次元配列に非常によく似ています。

また、`tf.Tensor`には、以下のプロパティが含まれています。

- `rank`: テンソルが含む次元の数を定義する
- `shape`: データの各次元の大きさを定義する
- `dtype`: テンソルのデータ型を定義する

注意: 「次元」という用語は、階数と同じ意味で使用しています。機械学習では、テンソルの「次元性」が特定の次元の大きさを指す場合もあります。（例えば形状の行列 [10, 5] は、階数 2 のテンソル、または 2 次元のテンソルです。1 次元目の次元性は 10 になります。混乱させてしまうかもしれませんが、この用語の二重使用には遭遇する可能性が高いので、ここに注意書きとして記しておきます。）

`tf.tensor()`メソッドを使用して、配列から`tf.Tensor`を作成します。

```js
// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();

// Or you can create a tensor from a flat array and specify a shape.
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

`tf.Tensor`のデフォルトは`float32``dtype`です。`tf.Tensor`は、bool、int32、complex64、文字列 dtype を使用しても作成可能です。

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```

TensorFlow.js には、ランダムテンソル、特定の値で埋められたテンソル、`HTMLImageElement`からのテンソルなどを作成できる便利なメソッドのセットをはじめ、その他にも多数が用意されているので、[こちら](https://js.tensorflow.org/api/latest/#Tensors-Creation)からご覧ください。

#### テンソルの形状を変更する

`tf.Tensor`の要素数は、その形状の大きさの積です。同じ大きさの形状が複数存在することがよくあるので、`tf.Tensor`を同じ大きさの別の形状に変更できると便利です。これは`reshape()`メソッドを用いて実現できます。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### テンソルから値を取得する

また、`Tensor.array()`または`Tensor.data()`メソッドを使用して`tf.Tensor`から値を取得することもできます。

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 // Returns the multi dimensional array of values.
 a.array().then(array => console.log(array));
 // Returns the flattened data that backs the tensor.
 a.data().then(data => console.log(data));
```

これらのメソッドには、もっと使いやすい同期バージョンも提供されてはいますが、アプリケーションのパフォーマンスに問題が生じます。本番アプリケーションでは、必ず非同期バージョンのメソッドを使用してください。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
// Returns the multi dimensional array of values.
console.log(a.arraySync());
// Returns the flattened data that backs the tensor.
console.log(a.dataSync());
```

## 演算

テンソルではデータを格納することができますが、演算はそのデータを操作することができます。TensorFlow.js は、テンソルで実行可能な、線形代数や機械学習に適した様々な演算も提供しています。

例: `tf.Tensor`の全要素の x<sup>2</sup> を計算します。

```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // equivalent to tf.square(x)
y.print();
```

例: 2 つの`tf.Tensor`の要素を、要素ごとに追加します。

```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // equivalent to tf.add(a, b)
y.print();
```

テンソルは不変なので、これらの演算がテンソルの値を変更することはありません。その代わりに、ops return で常に新しい`tf.Tensor`を返します。

> 注意: ほとんどの演算は`tf.Tensor`を返しますが、実際には結果の準備がまだできていない場合があります。つまり、取得した`tf.Tensor`は、実際には計算のハンドルであるということです。`Tensor.data()`や`Tensor.array()`を呼び出すと、これらのメソッドは計算が終了した時に値を使用して解決する約束を返します。UI コンテキスト（ブラウザアプリなど）で実行する場合には、計算が完了する前に UI スレッドがブロックされるのを防ぐために、常にこれらのメソッドの同期バージョンではなく、非同期バージョンを使用する必要があります。

TensorFlow.js がサポートする演算のリストは、[こちら](https://js.tensorflow.org/api/latest/#Operations)からご覧ください。

## メモリ

WebGL バックエンドを使用する場合、`tf.Tensor`のメモリを明示的に管理する必要があります。（`tf.Tensor`をスコープ外に出してメモリを解放するのでは**不十分**です。）

tf.Tensor のメモリの破壊には、`dispose() `メソッドまたは`tf.dispose()`を使用します。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // Equivalent to tf.dispose(a)
```

アプリケーション内で複数の演算をチェーン化することは非常に一般的です。すべての中間変数への参照を保持してそれらを破棄すると、コードの可読性を低下させてしまうことがあります。TensorFlow.js はこの問題を解決するために`tf.tidy()`メソッドを提供しています。このメソッドは、関数が実行された時にローカル変数がクリーンアップされる方法に似ており、関数の実行後に関数によって返されないすべての`tf.Tensor`をクリーンアップします。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

この例では、`square()`と`log()`の結果は自動的に破棄されます。`neg()`の結果は tf.tidy() の戻り値なので破棄されません。

また、TensorFlow.js が追跡したテンソルの数も取得することができます。

```js
console.log(tf.memory());
```

`tf.memory()`が出力するオブジェクトには、現在割り当てられているメモリ量の情報が含まれています。詳細の情報に関しては[こちら](https://js.tensorflow.org/api/latest/#memory)をご覧ください。

# 効果的な TensorFlow 2

TensorFlow 2.0 には、TensorFlow ユーザーの生産性を高める複数の変更が適用されています。TensorFlow 2.0 では [冗長性 API](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md) が取り除かれ、API の一貫性の強化（[Unified RNNs](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md)、[Unified Optimizers](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)）と [Eager execution](https://www.tensorflow.org/guide/eager) による Python ランライムとの統合の改善が行われました。

多くの [RFC](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr) では、TensorFlow 2.0 の制作に取り込まれた変更内容が説明されています。このガイドでは、TensorFlow 2.0 での開発がどのようなものかを説明します。TensorFlow 1.x に関するある程度の知識があることを前提としています。

## 主な変更点の簡単な要約

### API のクリーンアップ

TF 2.0 では、多数の API が[取り除かれたか移行](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)されています。主な変更点には、現在ではオープンソースとなった [absl-py](https://github.com/abseil/abseil-py) の導入による `tf.app`、`tf.flags`、および  `tf.logging` の削除、`tf.contrib` にあったプロジェクトの移植、使用頻度の低い関数を `tf.math` などのサブパッケージに移動することによるメインの `tf.*` 名前空間のクリーンアップなどがあります。また、一部の API は、その 2.0 バージョンの `tf.summary`、`tf.keras.metrics`、`tf.keras.optimizers` などに置き換えられています。[v2 アップグレードスクリプト](upgrade.md)を使用すると、こういった名前変更を手っ取り早く自動適用することができます。

### Eager execution

TensorFlow 1.X では、ユーザーは `tf.*` API 呼び出しを行って、手動で[抽象構文木](https://en.wikipedia.org/wiki/Abstract_syntax_tree)（グラフ）を作成する必要がありました。API を呼び出したら、出力テンソルと入力テンソルのセットを `session.run()` 呼び出しに渡して、手動で抽象構文木をコンパイルする必要があったのです。TensorFlow 2.0 はこれを逐次的に実行（Python が通常行うのと同じように）し、グラフとセッションは実装の詳細のような感覚になっています。

Eager execution の副産物として注目しておきたいのは、`tf.control_dependencies()` が不要になったという点です。これは、コードのすべての行が順に実行されるようになったためです（`tf.function` 内では、副次的影響のあるコードは記述された順に実行されます）。

### グローバルの排除

TensorFlow 1.X では、グローバル名前空間に暗黙的に大きく依存していました。`tf.Variable()` を呼び出すと、デフォルトのグラフに配置され、それをポイントする Python 変数を追跡できなくなってもグラフに残されていました。その `tf.Variable` を取り戻せたのは、その作成に使用された名前がわかっている場合のみでした。変数の作成を管理していないユーザーにとっては困難なことだったのです。その結果、変数をもう一度見つけ出すためのさまざまな仕組みが生まれただけでなく、変数スコープ、グローバルコレクション、`tf.get_global_step()` のようなヘルパーメソッド、`tf.global_variables_initializer()`、すべてのトレーニング可能な変数の勾配を暗黙的に計算するオプティマイザーなど、ユーザー作成変数を検索するフレームワークが急増しました。TensorFlow 2.0 は、これらすべての仕組みを排除し（([Variables 2.0 RFC](https://github.com/tensorflow/community/pull/11)）、デフォルトの仕組みを採択しています。自分の変数は自分で追跡！`tf.Variable` を追跡できなくなると、ガベージコレクションによって収集されます。

変数の追跡が必要になったことでユーザーの手間が増えることになりますが、Keras オブジェクト（以下参照）を使用すると、その負荷は最小化されます。

### セッションではなく関数

`session.run()` 呼び出しは、ほぼ関数呼び出しと変わりません。入力と呼び出される関数を指定すれば、一連の出力が返されます。TensorFlow 2.0 では、`tf.function()` を使って Python 関数に飾りつけをし、TensorFlow が単一のグラフとして実行できるように JIT コンパイルのマークを付けます（[Functions 2.0 RFC](https://github.com/tensorflow/community/pull/20)）。この仕組みにより、TensorFlow 2.0 はグラフモードのすべてのメリットを得ることができます。

- パフォーマンス: 関数を最適化できます（ノード枝狩り、カーネル融合など）
- 移植性: 関数をエクスポート/再インポート（[SavedModel 2.0 RFC](https://github.com/tensorflow/community/pull/34)）できるため、ユーザーはモジュール型 TensorFlow 関数を再利用し共有することができます。

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

Python と TensorFlow コードを自由に混在させられるため、Python の表現力を活用することができます。ただし、移植される TensorFlow は Python インタプリタ（モバイル、C++、JavaScript など）を使用しない文脈で実行されます。`@tf.function` を追加する際にコードの書き直しを行わなくてよいように、[AutoGraph](function.ipynb) によって、Python 構造体のサブセットを TensorFlow 相当のものに変換することができます。

- `for`/`while` -&gt; `tf.while_loop` (`break` と `continue` はサポートされています)
- `if` -&gt; `tf.cond`
- `for _ in dataset` -&gt; `dataset.reduce`

AutoGraph では制御フローを任意にネストできるため、シーケンスモデル、強化学習、カスタムトレーニングループなど、多くの複雑な ML プログラムを効率的かつ簡潔に実装することができます。

## 慣用的な TensorFlow 2.0の 推奨事項

### コードを小さな関数にリファクタリングする

TensorFlow 1.X の一般的な使用パターンは「キッチンシンク」戦略でした。可能なすべての計算の和集合を先制的にレイアウトし、` session.run()` を使って一部のテンソルを評価する方法です。TensorFlow 2.0 では、ユーザーは必要に応じて呼び出す小さな関数にコードをリファクタリングする必要があります。一般に、これらの小さな関数に ` tf.function ` を追加する必要はありません。` tf.function ` は、トレーニングの 1 ステップやモデルのフォワードパスといった高レベルの計算にのみ使用してください。

### Keras レイヤーとモデルを使用して変数を管理する

Keras モデルとレイヤーは、すべての従属変数を帰属的に収集する便利な `variables` と `trainable_variables` プロパティを提供しています。このため、変数が使用されている場所での変数の管理を簡単に行うことができます。

以下の 2 つのコードを比較してみましょう。

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# You still have to manage w_i and b_i, and their shapes are defined far away from the code.
```

上記のコードを次の Keras バージョンと比べます。

```python
# Each layer can be called, with a signature equivalent to linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

Keras レイヤー/モデルは `tf.train.Checkpointable` から継承し、`@tf.function` と統合されています。このため、Keras オブジェクトから直接チェックポイントするか SavedModels をエクスポートすることができます。この統合を利用するために、Keras の `.fit()` API を必ずしも使用する必要はありません。

次は、関連する変数のサブセットを Keras で簡単に収集できる様子を示す転移学習の例です。共有トランクを持つマルチヘッドのモデルをトレーニングしているとしましょう。

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# Train on primary dataset
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path1(x, training=True)
    loss = loss_fn_head1(prediction, y)
  # Simultaneously optimize trunk and head1 weights.
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# Fine-tune second head, reusing the trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path2(x, training=True)
    loss = loss_fn_head2(prediction, y)
  # Only optimize head2 weights, not trunk weights
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# You can publish just the trunk computation for other people to reuse.
tf.saved_model.save(trunk, output_path)
```

### tf.data.Datasets と @tf.function を組み合わせる

メモリに収まるトレーニングデータをイテレートする際は、通常の Python イテレーションを使用できますが、ディスクのトレーニングデータをストリーミングするには、`tf.data.Dataset` が最適です。データセットは [イテラブル（イテレータではない）](https://docs.python.org/3/glossary.html#term-iterable)であり、Eager モードの Python イテラブルとまったく同様に機能します。コードを `tf.function()` でラップすることで、データセットの非同期プリフェッチ/ストリーム機能をそのまま利用することができます。この方法は、Python イテレーションを、同等の、AutoGraph を使用した演算に置き換えます。

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      prediction = model(x, training=True)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

Keras の `.fit()` API を使用する場合、データセットのイテレーションを気にする必要はありません。

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### Python 制御フローで AutoGraph を活用する

AutoGraph は、データ依存の制御フローを `tf.cond` や `tf.while_loop` といったグラフモード相当のフローに変換する方法を提供しています。

データ依存の制御フローがよく見られる場所に、シーケンスモデルが挙げられます。`tf.keras.layers.RNN` は RNN セルをラップするため、静的または動的にリカレンスを展開することができます。これを示すために、動的な展開を次のように実装しなおすことができます。

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

AutoGraph の機能の詳しい概要については、[ガイド](./function.ipynb)を参照してください。

### tf.metrics でデータを集計し、tf.summary でログ記録する

要約をログに記録するには、`tf.summary.(scalar|histogram|...)` を使用して、コンテキストマネージャを使ってライターにリダイレクトします。コンテキストマネージャを省略すると、何も起こりません。TF 1.x とは異なり、要約はライターに直接送信されるため、マージ演算や `add_summary()` 呼び出しを別途行う必要がありません。つまり、`step` 値を呼び出しサイトで提供する必要があります。

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

要約としてデータをログに記録する前にデータを集計するには、`tf.metrics` を使用します。メトリックはステートフルです。つまり、値を蓄積し、`.result()` が呼び出されたときに累積結果を返します。累積された値は、 `.reset_states()` を使用すると消去されます。

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  loss = loss_fn(model(test_x, training=False), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

TensorBoard を要約のログディレクトリにポイントし、生成された要約を視覚化します。

```
tensorboard --logdir /tmp/summaries
```

### デバッグ時に tf.config.experimental_run_functions_eagerly() を使用する

TensorFlow 2.0 の Eager execution では、コードをステップごとに実行し、形状、データ型、および値を検査することができます。`tf.function` や `tf.keras` などの特定の API は、パフォーマンスや移植性の目的で、Graph execution を使用するように設計されていますが、デバッグの際は、`tf.config.experimental_run_functions_eagerly(True)` を使って、このコード内で Eager execution を使用することができます。

次に例を示します。

```python
@tf.function
def f(x):
  if x > 0:
    import pdb
    pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)
f(tf.constant(1))
```

```
>>> f()
-> x = x + 1
(Pdb) l
  6  	@tf.function
  7  	def f(x):
  8  	  if x > 0:
  9  	    import pdb
 10  	    pdb.set_trace()
 11  ->	    x = x + 1
 12  	  return x
 13
 14  	tf.config.experimental_run_functions_eagerly(True)
 15  	f(tf.constant(1))
[EOF]
```

これは Keras モデルや、Eager execution をサポートするほかの API 内でも機能します。

```
class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      import pdb
      pdb.set_trace()
      return input_data // 2


tf.config.experimental_run_functions_eagerly(True)
model = CustomModel()
model(tf.constant([-2, -4]))
```

```
>>> call()
-> return input_data // 2
(Pdb) l
 10  	    if tf.reduce_mean(input_data) > 0:
 11  	      return input_data
 12  	    else:
 13  	      import pdb
 14  	      pdb.set_trace()
 15  ->	      return input_data // 2
 16
 17
 18  	tf.config.experimental_run_functions_eagerly(True)
 19  	model = CustomModel()
 20  	model(tf.constant([-2, -4]))
```

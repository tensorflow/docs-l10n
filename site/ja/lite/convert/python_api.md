# コンバータ Python API ガイド

Note: これらのドキュメントは私たちTensorFlowコミュニティが翻訳したものです。コミュニティによる
翻訳は**ベストエフォート**であるため、この翻訳が正確であることや[英語の公式ドキュメント](https://www.tensorflow.org/?hl=en)の
最新の状態を反映したものであることを保証することはできません。
この翻訳の品質を向上させるためのご意見をお持ちの方は、GitHubリポジトリ[tensorflow/docs](https://github.com/tensorflow/docs)にプルリクエストをお送りください。
コミュニティによる翻訳やレビューに参加していただける方は、
[docs-ja@tensorflow.org メーリングリスト](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)にご連絡ください。

このページでは、TensorFlow 2.0 の Python API による [TensorFlow Lite コンバータ](index.md) の使用例を説明します。

Note: このドキュメントでは TensorFlow 2 の Python API についてのみ記述します。
TensorFlow 1 の Python API についてのドキュメントは [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md) にあります。

[TOC]

## Python API

TensorFlow 2.0 において、TensorFlow モデルを TensorFlow Lite に変換する Python API は `tf.lite.TFLiteConverter` です。
 `TFLiteConverter` には、元のモデルフォーマットに基づいてモデルを変換する以下のクラスメソッドがあります：

*   `TFLiteConverter.from_saved_model()`:
    [SavedModel ディレクトリ](https://www.tensorflow.org/guide/saved_model) を変換します。
*   `TFLiteConverter.from_keras_model()`:
    [`tf.keras` モデル](https://www.tensorflow.org/guide/keras/overview) を変換します。
*   `TFLiteConverter.from_concrete_functions()`:
    [具象関数](https://tensorflow.org/guide/concrete_function) を変換します。


このドキュメントでは API の [使用例](＃examples) 、異なるバージョンの TensorFlow で実行する [方法](#versioning) を含みます。

## 例 <a name="examples"></a>

### SavedModel を変換する <a name="saved_model"></a>

以下の例は [SavedModel](https://www.tensorflow.org/guide/saved_model) を TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/) に変換する方法を示しています。

```python
import tensorflow as tf

# 基本的なモデルを構築
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# モデルを保存
export_dir = "/tmp/test_saved_model"
input_data = tf.constant(1., shape=[1, 1])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)

# モデルを変換
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
```

この API は入力となる任意の配列について、shape を指定するオプションを持ちません。
モデルの入力の shape を指定する必要がある場合には、[`from_concrete_functions`](#concrete_function) クラスメソッドを利用して下さい。
コードは次のようになるでしょう。

```python
model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 256, 256, 3])
converter = TFLiteConverter.from_concrete_functions([concrete_func])
```

### Keras モデルを変換する <a name="keras"></a>

以下の例は [`tf.keras` モデル](https://www.tensorflow.org/guide/keras/overview) を TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/) に変換する方法を示しています.


```python
import tensorflow as tf

# シンプルな Keras モデルを構築
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=50)

# モデルを変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 具象関数を変換する <a name="concrete_function"></a>

以下の例は TensorFlow の[具象関数](https://tensorflow.org/guide/concrete_function)を
TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/)
に変換する方法を示しています。

```python
import tensorflow as tf

# 基本的な関数を構築
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# 具象関数を生成
input_data = tf.constant(1., shape=[1, 1])
concrete_func = root.f.get_concrete_function(input_data)

# モデルを変換
#
# `from_concrete_function` は具象関数のリストを引数に取りますが、
# 現在のところは1つの関数のみをサポートしています。 複数関数の変換は開発中です
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
```

### End-to-end な MobileNet の変換 <a name="mobilenet"></a>

以下の例は、訓練済みの `tf.keras` MobileNet モデルを TensorFlow Lite に変換して実行する方法を示しています。
また、 元の TensorFlow モデルと TensorFlow Lite モデルの結果をランダムデータで比較しています。
モデルをファイルからロードするために、 `model_content` の代わりに ` model_path` を使用します。


```python
import numpy as np
import tensorflow as tf

# MobileNet tf.keras モデルをロード
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3))

# モデルを変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite モデルを変換し、テンソルを割当てる
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 入出力テンソルを取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# TensorFlow Lite モデルをランダムな入力データでテスト
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# `get_tensor()` はテンソルのコピーを返す
# テンソルのポインタを取得したい場合は `tensor()` を使う 
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# 元の TensorFlow モデルをランダムな入力データでテスト
tf_results = model(tf.constant(input_data))

# 結果を比較
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
```

#### TensorFlow Lite Metadata

Note: TensorFlow Lite Metadata は experimental (beta) フェーズにあります。

TensorFlow Lite Metadata はモデルの記述についての標準を提供します。
メタデータはモデルが何を行うのか、何を入力 / 出力にするのかについて知るための重要な情報源です。
これは開発者がベストプラクティスを理解したり、コードジェネレーターがプラットフォーム固有のラッパーとなるコードを生成するのを手助けします。より詳細については [TensorFlow Lite Metadata](metadata.md) を参照してください。

## TensorFlow のインストール <a name="versioning"></a>

### TensorFlow nightly のインストール <a name="2.0-nightly"></a>

TensorFlow nightly は次のコマンドでインストールできます。

```
pip install tf-nightly
```

### Custom ops in the experimental new converter

[カスタムの演算](https://www.tensorflow.org/lite/guide/ops_custom) を利用しているモデルが [新しいコンバーター](https://github.com/tensorflow/tensorflow/blob/917ebfe5fc1dfacf8eedcc746b7989bafc9588ef/tensorflow/lite/python/lite.py#L81) でどう扱われるかについて、振る舞いの変更があります。(以前に allow\_custom\_ops をセットしていたユーザー向けです)

**組み込みの TensorFlow の演算**

組み込みの TensorFlow の演算で TensorFlow Lite に存在しないものを利用しているモデルをコンバートする場合、(以前と同様に) allow\_custom\_ops 属性をセットしてください。詳細は[こちら](https://www.tensorflow.org/lite/guide/ops_custom)にあります。

**TensorFlow のカスタムの演算**

カスタムの TensorFlow の演算を用いたモデルをコンバートする場合、[TensorFlow カーネル](https://www.tensorflow.org/guide/create_op)と [TensorFlow Lite カーネル](https://www.tensorflow.org/lite/guide/ops_custom)を記述することを推奨します。これによりモデルが最初から最後まで、TensorFlow でも TensorFlow Lite でも動くことが保証されます。これも allow\_custom\_ops 属性をセットすることが要求されます。

**応用的なカスタム演算の利用 (非推奨)**

上記の対応が不可能な場合であっても、関連するカーネルがなくてもカスタムの演算を含む TensorFlow のモデルをコンバートできます。この場合、TensorFlow のカスタム演算の [OpDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto) を --custom\_opdefs フラグで指定する必要があります。ただし、これは関連する OpDef が TensorFlow のグローバルレジストリに登録されている場合に限ります。これにより TensorFlow のモデルは検証済みである (つまり、 TensorFlow ランタイムで読み込める) ことを保証できます。

カスタム演算が TensorFlow の演算のグローバルレジストリに登録されていない場合、関連する OpDef を --custom\_opdefs フラグで明示する必要があります。これは追加で登録が必要な OpDef の protocol buffer を文字列形式にしたもののリストです。次は TFLiteAwesomeCustomOp という、アウトプットが1つ、インプットが2つ、属性が2つのカスタム演算の場合の例です。

```
converter.custom\_opdefs="name: 'TFLiteAwesomeCustomOp' input\_arg: { name: 'InputA'
type: DT\_FLOAT } input\_arg: { name: ‘InputB' type: DT\_FLOAT }
output\_arg: { name: 'Output' type: DT\_FLOAT } attr : { name: 'Attr1' type:
'float'} attr : { name: 'Attr2' type: 'list(float)'}"
```
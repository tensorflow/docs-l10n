# コンバータ Python API ガイド

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

# 基本的な関数を構築
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

This API does not have the option of specifying the input shape of any input
arrays. If your model requires specifying the input shape, use the
[`from_concrete_functions`](#concrete_function) classmethod instead. The code
looks similar to the following:

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

Note: TensorFlow Lite Metadata is in experimental (beta) phase.

TensorFlow Lite metadata provides a standard for model descriptions. The
metadata is an important source of knowledge about what the model does and its
input / output information. This makes it easier for other developers to
understand the best practices and for code generators to create platform
specific wrapper code. For more information, please refer to the
[TensorFlow Lite Metadata](metadata.md) section.

## Installing TensorFlow <a name="versioning"></a>

### Installing the TensorFlow nightly <a name="2.0-nightly"></a>

The TensorFlow nightly can be installed using the following command:

```
pip install tf-nightly
```

### Custom ops in the experimental new converter

There is a behavior change in how models containing
[custom ops](https://www.tensorflow.org/lite/guide/ops_custom) (those for which
users use to set allow\_custom\_ops before) are handled in the
[new converter](https://github.com/tensorflow/tensorflow/blob/917ebfe5fc1dfacf8eedcc746b7989bafc9588ef/tensorflow/lite/python/lite.py#L81).

**Built-in TensorFlow op**

If you are converting a model with a built-in TensorFlow op that does not exist
in TensorFlow Lite, you should set allow\_custom\_ops attribute (same as
before), explained [here](https://www.tensorflow.org/lite/guide/ops_custom).

**Custom op in TensorFlow**

If you are converting a model with a custom TensorFlow op, it is recommended
that you write a [TensorFlow kernel](https://www.tensorflow.org/guide/create_op)
and [TensorFlow Lite kernel](https://www.tensorflow.org/lite/guide/ops_custom).
This ensures that the model is working end-to-end, from TensorFlow and
TensorFlow Lite. This also requires setting the allow\_custom\_ops attribute.

**Advanced custom op usage (not recommended)**

If the above is not possible, you can still convert a TensorFlow model
containing a custom op without a corresponding kernel. You will need to pass the
[OpDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
of the custom op in TensorFlow using --custom\_opdefs flag, as long as you have
the corresponding OpDef registered in the TensorFlow global op registry. This
ensures that the TensorFlow model is valid (i.e. loadable by the TensorFlow
runtime).

If the custom op is not part of the global TensorFlow op registry, then the
corresponding OpDef needs to be specified via the --custom\_opdefs flag. This is
a list of an OpDef proto in string that needs to be additionally registered.
Below is an example of an TFLiteAwesomeCustomOp with 2 inputs, 1 output, and 2
attributes:

```
converter.custom\_opdefs="name: 'TFLiteAwesomeCustomOp' input\_arg: { name: 'InputA'
type: DT\_FLOAT } input\_arg: { name: ‘InputB' type: DT\_FLOAT }
output\_arg: { name: 'Output' type: DT\_FLOAT } attr : { name: 'Attr1' type:
'float'} attr : { name: 'Attr2' type: 'list(float)'}"
```
# TensorFlow モデルの変換

このページでは、TensorFlow Lite コンバータを使用して、TensorFlow モデルを TensorFlow Lite モデル (`.tflite` ファイル拡張子の最適化された [FlatBuffer](https://google.github.io/flatbuffers/) 形式) を変換する方法について説明します。

注意: このガイドでは、[インストールされた TensorFlow 2.x](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available) と TensorFlow 2.x でトレーニングされたモデルの両方があることを前提とします。TensorFlow 1.x でモデルがトレーニングされる場合、[TensorFlow 2.x に移行](https://www.tensorflow.org/guide/migrate/tflite)することを検討してください。インストールされた TensorFlow バージョンを指定するには、`print(tf.__version__)` を実行します。

## 変換ワークフロー

次の図は、モデルを変換するワークフローの概要を示します。

![TFLite コンバータワークフロー](../../images/convert/convert.png)

**図 1.** コンバータワークフロー。

次のオプションのいずれかを使用して、モデルを変換できます。

1. [Python API](#python_api) (***推奨***): これにより、変換を開発パイプラインに統合し、最適化を適用して、メタデータや、変換プロセスを簡素化する他の多数のタスクを追加できます。
2. [コマンドライン](#cmdline): 基本モデル変換のみがサポートされます。

注意: モデル変換中に問題が発生する場合は、[GitHub 問題](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)を作成してください。

## Python API <a name="python_api"></a>

*ヘルパーコード: TensorFlow Lite converter API の詳細を表示するには、`print(help(tf.lite.TFLiteConverter))` を実行します。*

[`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter) を使用して、TensorFlow モデルを変換します。TensorFlow モデルは SavedModel 形式で保存され、上位の `tf.keras.*` API (Keras モデル) または (具象関数を生成する基になる) 下位の `tf.*` API を使用して生成されます。結果として、次の 3 つオプションを使用できます (例については、次のセクションを参照)。

- `tf.lite.TFLiteConverter.from_saved_model()` (**recommended**): [SavedModel](https://www.tensorflow.org/guide/saved_model) を変換します。
- `tf.lite.TFLiteConverter.from_keras_model()`: [Keras](https://www.tensorflow.org/guide/keras/overview) モデルを変換します。
- `tf.lite.TFLiteConverter.from_concrete_functions()`: [具象関数](https://www.tensorflow.org/guide/intro_to_graphs)を変換します。

### SavedModel の変換 (推奨) <a name="saved_model"></a>

次の例は、[SavedModel](https://www.tensorflow.org/guide/saved_model) を TensorFlow Lite モデルに変換する方法を示します。

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Keras モデルの変換 <a name="keras"></a>

次の例は、[Keras](https://www.tensorflow.org/guide/keras/overview) モデルを TensorFlow Lite モデルに変換する方法を示します。

```python
import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### 具象関数の変換 <a name="concrete_function"></a>

次の例は、[具象関数](https://www.tensorflow.org/guide/intro_to_graphs)を TensorFlow Lite モデルに変換する方法を示します。

```python
import tensorflow as tf

# Create a model using low-level tf.* APIs
class Squared(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### その他の特徴

- [最適化](../../performance/model_optimization.md)を適用します。使用される一般的な最適化は、[トレーニング後の量子化](../../performance/post_training_quantization.md)です。これにより、精度の損失を最小限に抑えながら、モデルの遅延とサイズを減らすことができます。

- [メタデータ](metadata.md)を追加します。これにより、モデルをデバイスにデプロイするときに、プラットフォーム固有のラッパーコードを簡単に作成できます。

### 変換エラー

次に、一般的な変換エラーと解決策を示します。

- エラー: `Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., .., ...`

    解決策: このエラーは、対応する TFLite 実装のない TF がモデルに存在するときに発生します。この問題を解決するには、[TFLite モデルで TF 演算を使用](../../guide/ops_select.md)します (推奨)。TFLite 演算のみのモデルを生成する場合は、[Github 問題 #21526](https://github.com/tensorflow/tensorflow/issues/21526) で不足している TFLite 演算の要求を追加する (要求がメンションされていない場合はコメントを残す) か、自分で [TFLite 演算](../../guide/ops_custom#create_and_register_the_operator)を作成します。

- エラー: `.. is neither a custom op nor a flex op`

    解決策: この TF 演算のサポート状況によって異なります。

    - TF でサポート: [許可リスト](../../guide/op_select_allowlist.md) (TFLite でサポートされる TF 演算の網羅的なリスト) に TF 処理がないため、このエラーが発生します。次の方法で解決できます。

        1. [不足している演算を許可リストに追加](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list)する。
        2. [TF モデルを TFLite モデルに変換し、推論を実行](../../guide/ops_select.md)する。

    - TF でサポートされていない: TFLite が定義されたカスタム TF 演算を認識していないため、このエラーが発生します。次の方法で解決できます。

        1. [TF 演算を作成](https://www.tensorflow.org/guide/create_op)する。
        2. [TF モデルを TFLite モデルに変換](../../guide/op_select_allowlist.md#users_defined_operators)する。
        3. [TFLite 演算](../../guide/ops_custom.md#create_and_register_the_operator)を作成し、TFLite ランタイムに関連付けて推論を実行する。

## コマンドラインツール <a name="cmdline"></a>

**注意:** 可能なかぎり、上記の [Python API](#python_api) を使用することを強くお勧めします。

[pip から TensorFlow 2.x をインストールした](https://www.tensorflow.org/install/pip)場合は、`tflite_convert` コマンドを使用します。使用可能なフラグをすべて表示するには、次のコマンドを使用します。

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

[TensorFlow 2.x ソース](https://www.tensorflow.org/install/source)をダウンロードし、パッケージを構築してインストールせずに、そのソースからコンバータを実行する場合は、コマンドの '`tflite_convert`' を '`bazel run tensorflow/lite/python:tflite_convert --`' で置き換えます。

### SavedModel の変換<a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Keras H5 モデルの変換 <a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## 次のステップ

[TensorFlow Lite インタープリタ](../../guide/inference.md) を使用して、クライアントデバイス (例: モバイル、組み込み) で推論を実行します。

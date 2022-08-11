# 转换 TensorFlow 模型

本页介绍如何使用 TensorFlow Lite 转换器将 TensorFlow 模型转换为 TensorFlow Lite 模型（由文件扩展名 `.tflite` 标识的经过优化的 [FlatBuffer](https://google.github.io/flatbuffers/) 格式）。

注：本指南假设您[已安装 TensorFlow 2.x](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available)，并在 TensorFlow 2.x 中训练了模型。如果您的模型在 TensorFlow 1.x 中进行了训练，请考虑[迁移到 TensorFlow 2.x](https://www.tensorflow.org/guide/migrate/tflite)。要识别已安装的 TensorFlow 版本，请运行 `print(tf.__version__)`。

## 转换工作流

下图说明了转换模型的高级工作流：

![TFLite 转换器工作流](../../images/convert/convert.png)

**图 1.** 转换器工作流。

可以使用下面的其中一个选项来转换模型：

1. [Python API](#python_api)（***推荐***）：这允许您将转换集成到开发流水线中，应用优化，添加元数据，以及许多其他简化转换过程的任务。
2. [命令行](#cmdline)：这仅支持基础模型转换。

注：如果您在模型转换过程中遇到任何问题，请创建 [GitHub 议题](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)。

## Python API <a name="python_api"></a>

*辅助代码：要了解有关 TensorFlow Lite Converter API 的更多信息，请运行 `print(help(tf.lite.TFLiteConverter))`。*

使用 [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter) 转换 TensorFlow 模型。TensorFlow 模型使用 SavedModel 格式存储，并使用高级 `tf.keras.*` API（Keras 模型）或低级 `tf.*` API（从中生成具体函数）。因此，您有以下三个选项（以下几个部分提供了示例）：

- `tf.lite.TFLiteConverter.from_saved_model()`（**推荐**）：转换 [SavedModel](https://www.tensorflow.org/guide/saved_model)。
- `tf.lite.TFLiteConverter.from_keras_model()`：转换 [Keras](https://www.tensorflow.org/guide/keras/overview) 模型。
- `tf.lite.TFLiteConverter.from_concrete_functions()`：转换[具体函数](https://www.tensorflow.org/guide/intro_to_graphs)。

### 转换 SavedModel（推荐）<a name="saved_model"></a>

以下示例展示了如何将 [SavedModel](https://www.tensorflow.org/guide/saved_model) 转换为 TensorFlow Lite 模型。

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### 转换 Keras 模型 <a name="keras"></a>

以下示例展示了如何将 [Keras](https://www.tensorflow.org/guide/keras/overview) 模型转换为 TensorFlow Lite 模型。

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

### 转换具体函数 <a name="concrete_function"></a>

以下示例展示了如何将[个具体函数](https://www.tensorflow.org/guide/intro_to_graphs)转换为 TensorFlow Lite 模型。

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

### 其他功能

- 应用[优化](../../performance/model_optimization.md)。常用的优化是[训练后量化](../../performance/post_training_quantization.md)，它可以在最小准确率损失的情况下进一步减少模型延迟和大小。

- 添加[元数据](metadata.md)，在设备端部署模型时，可以更轻松地创建特定于平台的封装器代码。

### 转换错误

以下是常见的转换错误及其解决方案：

- 错误：`Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., .., ...`

    解决方案：出现该错误是因为您的模型具有没有对应的 TFLite 实现的 TF 算子。您可以通过[使用 TFLite 模型中的 TF 算子](../../guide/ops_select.md)来解决此问题（推荐）。如果您只想生成具有 TFLite 算子的模型，您可以在 [Github 议题 #21526](https://github.com/tensorflow/tensorflow/issues/21526)（如果您的请求尚未被提及，请留下评论）中添加对缺失的 TFlite 算子的请求，或者[自己创建 TFlite 算子](../../guide/ops_custom#create_and_register_the_operator)。

- 错误：`.. is neither a custom op nor a flex op`

    解决方案：如果此 TF 算子：

    - 在 TF 中受支持：出现此错误是因为[允许列表](../../guide/op_select_allowlist.md)（TFLite 支持的 TF 算子的详尽列表）中缺少 TF 算子。您可以按如下方式解决此问题：

        1. [将缺少的算子添加到允许列表](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list)。
        2. [将 TF 模型转换为 TFLite 模型并运行推断](../../guide/ops_select.md)。

    - 在 TF 中不受支持：出现此错误是因为 TFLite 无法识别您定义的自定义 TF 算子。<br>您可以按如下方式解决此问题：

        1. [创建 TF 算子](https://www.tensorflow.org/guide/create_op)。
        2. [将 TF 模型转换为 TFLite 模型](../../guide/op_select_allowlist.md#users_defined_operators)。
        3. [创建 TFLite 算子](../../guide/ops_custom.md#create_and_register_the_operator)，并通过将其链接到 TFLite 运行时来运行推断。

## 命令行工具<a name="cmdline"></a>

**注**：如果可能，强烈建议您使用上面列出的 [Python API](#python_api)。

如果您已[从 pip 安装了 TensorFlow 2.x](https://www.tensorflow.org/install/pip)，请使用 `tflite_convert` 命令。要查看所有可用标志，请使用以下命令：

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

如果您已下载 [TensorFlow 2.x 源文件](https://www.tensorflow.org/install/source)，并且希望在不构建和安装软件包的情况下从该源文件运行转换器，您可以在命令中将 '`tflite_convert`' 替换为 '`bazel run tensorflow/lite/python:tflite_convert --`'。

### 转换 SavedModel <a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### 转换 Keras H5 模型 <a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## 后续步骤

使用 [TensorFlow Lite 解释器](../../guide/inference.md)在客户端设备（例如移动设备、嵌入式设备）上运行推断。

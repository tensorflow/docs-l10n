# 使用 Python 快速入门基于 Linux 的设备

对于基于 Linux 的嵌入式设备（例如 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 和[使用 Edge TPU 的 Coral 设备](https://coral.withgoogle.com/){:.external}），非常适合将 TensorFlow Lite 与 Python 结合使用。

本页介绍如何在几分钟内学会开始使用 Python 运行 TensorFlow Lite 模型。您只需要一个[已转换为 TensorFlow Lite](../models/convert/) 的 TensorFlow 模型。（如果还没有转换的模型，您可以使用随下面链接的示例提供的模型进行实验。）

## 关于 TensorFlow Lite 运行时软件包

为了快速开始使用 Python 执行 TensorFlow Lite 模型，您可以仅安装 TensorFlow Lite 解释器，无需安装所有 TensorFlow 软件包。我们将这种简化的 Python 软件包称为 `tflite_runtime`。

`tflite_runtime` 软件包是整个 `tensorflow` 软件包的一小部分，并且包括使用 TensorFlow Lite 运行推断所需的最少代码（主要是 [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 类）。当您只想执行 `.tflite` 模型并避免因使用大型 TensorFlow 库而浪费磁盘空间时，这种小型软件包是理想选择。

注：如果您需要访问其他 Python API（如 [TensorFlow Lite Converter](../models/convert/)），则必须安装[完整的 TensorFlow 软件包](https://www.tensorflow.org/install/)。例如，`tflite_runtime` 软件包中不包括 [Select TF 算子] (https://www.tensorflow.org/lite/guide/ops_select)。如果您的模型与 Select TF 算子有任何依赖关系，则需要改用完整的 TensorFlow 软件包。

## 安装适用于 Python 的 TensorFlow Lite

您可以使用 pip 在 Linux 上安装：

<pre class="devsite-terminal devsite-click-to-copy">python3 -m pip install tflite-runtime
</pre>

## 受支持的平台

`tflite-runtime` Python wheel 针对以下平台进行预构建并提供给这些平台：

- Linux armv7l（例如运行 32 位 Raspberry Pi OS 的 Raspberry Pi 2、3、4 和 Zero 2）
- Linux aarch64（例如运行 Debian ARM64 的 Raspberry Pi 3、4）
- Linux x86_64

如果您想在其他平台上运行 TensorFlow Lite 模型，则应使用[完整 TensorFlow 软件包](https://www.tensorflow.org/install/)，或者[从源代码构建 tflite-runtime 软件包](build_cmake_pip.md)。

如果您将 TensorFlow 与 Coral Edge TPU 结合使用，则应遵循相应的 [Coral 设置文档](https://coral.ai/docs/setup)。

注：我们已不再更新 Debian 软件包 `python3-tflite-runtime`。最新版 Debian 软件包适用于 TF 2.5 版本，您可以按照[这些早期说明](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python)进行安装。

注：我们已不再发布适用于 Windows 和 macOS 的预构建 `tflite-runtime` wheel。对于这些平台，您应使用[完整 TensorFlow 软件包](https://www.tensorflow.org/install/)，或者[从源代码构建 tflite-runtime 软件包](build_cmake_pip.md)。

## 使用 tflite_runtime 运行推断

现在，您需要从 `tflite_runtime` 导入 `Interpreter`，而不是从 `tensorflow` 模块导入。

例如，安装上述软件包后，如果复制并运行 [`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/) 文件，（可能）会失败，因为您没有安装 `tensorflow` 库。要解决此问题，请编辑该文件中的下面一行：

```python
import tensorflow as tf
```

将其改成：

```python
import tflite_runtime.interpreter as tflite
```

然后更改下面一行：

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

将其改成：

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

现在，重新运行 `label_image.py`。就是这样！您现在执行的正是 TensorFlow Lite 模型。

## 了解详情

- 有关 `Interpreter` API 的更多详细信息，请阅读[在 Python 中加载和运行模型](inference.md#load-and-run-a-model-in-python)。

- 如果您拥有 Raspberry Pi，请查看有关如何使用 TensorFlow Lite 在 Raspberry Pi 上运行目标检测的[视频系列](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe)。

- 如果您使用 Coral ML 加速器，请查看 [GitHub 上的 Coral 示例](https://github.com/google-coral/tflite/tree/master/python/examples)。

- 要将其他 TensorFlow 模型转换为 TensorFlow Lite，请阅读有关 [TensorFlow Lite Converter](../models/convert/) 的内容。

- 如果您想构建 `tflite_runtime` wheel，请阅读[构建 TensorFlow Lite Python Wheel 软件包](build_cmake_pip.md)

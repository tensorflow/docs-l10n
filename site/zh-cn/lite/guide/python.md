# Python 快速入门

对于基于 Linux 的嵌入式设备（例如 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 和[使用 Edge TPU 的 Coral 设备](https://coral.withgoogle.com/){:.external}），非常适合将 TensorFlow Lite 与 Python 结合使用。

本页介绍如何在几分钟内学会开始使用 Python 运行 TensorFlow Lite 模型。您只需要一个[已转换为 TensorFlow Lite](../convert/) 的 TensorFlow 模型。（如果还没有转换的模型，您可以使用随下面链接的示例提供的模型进行实验。）

## 关于 TensorFlow Lite 运行时软件包

为了快速开始使用 Python 执行 TensorFlow Lite 模型，您可以仅安装 TensorFlow Lite 解释器，无需安装所有 TensorFlow 软件包。我们将这种简化的 Python 软件包称为 `tflite_runtime`。

`tflite_runtime` 软件包是整个 `tensorflow` 软件包的一小部分，并且包括使用 TensorFlow Lite 运行推断所需的最少代码（主要是 [`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 类）。当您只想执行 `.tflite` 模型并避免因使用大型 TensorFlow 库而浪费磁盘空间时，这种小型软件包是理想选择。

注：如果您需要访问其他 Python API（如 [TensorFlow Lite 转换器](../convert/python_api.md)），则必须安装[完整 TensorFlow 软件包](https://www.tensorflow.org/install/)。

## 安装适用于 Python 的 TensorFlow Lite

如果您正在运行 Debian Linux 或 Debian 的衍生版本（包括 Raspberry Pi OS），则应从我们的 Debian 软件包仓库中进行安装。这要求您向系统添加新的仓库列表和密钥，随后按以下方式安装：

<pre class="devsite-terminal">echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
&lt;code class="devsite-terminal"
&gt;GL_CODE_5&lt;/code&gt;&lt;code class="devsite-terminal"
&gt;GL_CODE_6&lt;/code&gt;&lt;code class="devsite-terminal"
&gt;GL_CODE_7&lt;/code&gt;
</pre>

对于所有其他系统，可以使用 pip 进行安装：

<pre class="devsite-terminal devsite-click-to-copy">pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

如果您想手动安装 Python wheel，则可以从[所有 `tflite_runtime` wheel](https://github.com/google-coral/pycoral/releases/) 中选择一个。

注：如果您使用的是 Debian Linux，并且使用 pip 安装 `tflite_runtime`，则在使用以 Debian 软件包形式安装且依赖于 TF Lite 的其他软件（例如 [Coral 库](https://coral.ai/software/)）时，它可能会导致运行时失败。如果使用 pip 卸载 `tflite_runtime`，随后使用上面的 `apt-get` 命令重新安装，可以修正此问题。

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

有关 `Interpreter` API 的更多详细信息，请阅读[在 Python 中加载和运行模型](inference.md#load-and-run-a-model-in-python)。

如果您有 Raspberry Pi，请尝试运行 [classify_picamera.py 示例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)，使用 Pi Camera 和 TensorFlow Lite 执行图像分类。

如果您使用 Coral ML 加速器，请查看 [GitHub 上的 Coral 示例](https://github.com/google-coral/tflite/tree/master/python/examples)。

要将其他 TensorFlow 模型转换为 TensorFlow Lite，请阅读 [TensorFlow Lite 转换器](../convert/)。

如果您想构建 `tflite_runtime` wheel，请阅读[构建 TensorFlow Lite Python Wheel 软件包](build_cmake_pip.md)

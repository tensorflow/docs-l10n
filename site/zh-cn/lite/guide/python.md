# Python 快速入门

对于基于 Linux 的嵌入式设备（例如 [Raspberry Pi](https://www.raspberrypi.org/){:.external} 和[使用 Edge TPU 的 Coral 设备](https://coral.withgoogle.com/){:.external}），非常适合将 TensorFlow Lite 与 Python 结合使用。

本页介绍如何在几分钟内学会开始使用 Python 运行 TensorFlow Lite 模型。您只需要一个[已转换为 TensorFlow Lite](../convert/) 的 TensorFlow 模型。（如果还没有转换的模型，您可以使用随下面链接的示例提供的模型进行实验。）

## 安装 TensorFlow Lite 解释器

要使用 Python 快速运行 TensorFlow Lite 模型，您只需安装 TensorFlow Lite 解释器，而不需要安装所有 TensorFlow 软件包。

只包含解释器的软件包是完整 TensorFlow 软件包的一小部分，其中只包含使用 TensorFlow Lite 运行推断所需要的最少代码——仅包含 [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python 类。如果您只想执行 `.tflite` 模型，而不希望庞大的 TensorFlow 库占用磁盘空间，那么这个小软件包是最理想的选择。

注：如果您需要访问其他 Python API（如 [TensorFlow Lite 转换器](../convert/python_api.md)），则必须安装[完整 TensorFlow 软件包](https://www.tensorflow.org/install/)。

要安装，请运行 `pip3 install`，并向其传递下表中适当的 Python wheel 网址。

例如，如果是运行 Raspbian Buster（具有 Python 3.7）的 Raspberry Pi，请使用以下命令安装 Python wheel：

<pre class="devsite-terminal devsite-click-to-copy">pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</pre>

<table>
<tr>
<th>平台</th>
<th>Python</th>
<th>网址</th>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (ARM 32)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_armv7l.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (ARM 64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux (x86-64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">macOS 10.14</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Windows 10</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl</td>
</tr>
</table>

## 使用 tflite_runtime 运行推理

为了将只包含解释器的软件包与完整 TensorFlow 软件包区分开（如果您愿意，可以同时安装两者）, Python 模块在上述 wheel 中提供了命名的 `tflite_runtime`。

因此，不要从 `tensorflow` 模块导入 `Interpreter` 模块，您需要从 `tflite_runtime` 导入。

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

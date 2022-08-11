# 目标检测

在给定图像或视频流的情况下，目标检测模型可以识别可能存在已知目标集合中的哪些目标，并提供关于它们在图像中的位置的信息。

例如，此<a href="#get_started">示例应用程序</a>的屏幕截图显示了如何识别两个目标并注解它们的位置：


<img src="../images/detection.png" class="attempt-right">

注：(1) 要集成现有模型，请尝试 [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector)。(2) 要自定义模型，请尝试 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)。

## 开始使用

要了解如何在移动应用中使用目标检测，推荐查看我们的<a href="https://tensorflow.google.cn/api_docs/python/tf/lite">示例应用和指南</a>。

如果您使用的不是 Android 或 iOS 平台，或者您已经熟悉 TensorFlow Lite API，则可以下载我们的入门目标检测模型和附带的标签。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">下载包含元数据的入门模型</a>

有关元数据和相关字段（例如：`labels.txt`）的详细信息，请参阅<a href="../../models/convert/metadata#read_the_metadata_from_models">从模型读取元数据</a>。

如果要针对您自己的任务训练自定义检测模型，请参阅<a href="#model-customization">模型自定义</a>。

针对以下用例，您应该使用不同的模型：

<ul>
  <li>预测图像最可能代表的单个标签（请参阅<a href="../image_classification/overview.md">图像分类</a>）</li>
  <li>预测图像的组成，例如主体与背景（请参阅<a href="../segmentation/overview.md">分割</a>）</li>
</ul>

### 示例应用和指南

想象一下一个模块被训练用于检测苹果，香蕉和草莓。当我们输入一幅图片后，模块将会返回给我们一组本示例的检测结果。

#### Android

我们使用信任分数和所检测到目标的坐标来表示检测结果。分数反应了被检测到物体的可信度，范围在 0 和 1 之间。最大值为1，数值越大可信度越高。

下面的 Android 示例分别以 [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_task_api) 和 [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter) 形式演示了两个方法的实现。

使用裁切功能必须基于误判位置（物体识别错误或物体位置识别有误）或错误（由于可信度过低导致的物体未被捕捉）。 如下图所示，梨子（未被模块训练检测的物体）被误判为“人”。实例中的误判可以通过适当的图片裁切来忽略。在此示例中，裁剪 0.6 （或 60% ）可以适当的排除误判。

#### iOS

针对每个被检测的物体，模块将会返回一个由四个数字组成的数组，该四个数字代表了围绕物体的一个矩形框。在我们提供的示例模块中，返回的数组中的元素按照如下顺序：

top 的值代表了矩形框的顶部距离图片上部的距离，单位为像素。 left 的值代表了矩形框的左边距离图片左边的距离。bottom 和 right 值的表示方法同理。 注意：目标检测模块接受特定尺寸的模型作为输入。这很有可能与您的图像设备生成的原始图片尺寸不同，所以您需要编写代码将原始图片缩放至模型可接受的尺寸(我们提供了 <a href="#get_started">示范程序</a>)。 <br><br>模块输出的像素值表示在缩放后的图片中的位置，所以您需要调整调整原始图片等尺寸来保证正确。

## 何为物体检测?

本部分介绍了从 [TensorFlow 目标检测 API](https://github.com/tensorflow/models/blob/master/research/object_detection/) 转换为 TensorFlow Lite 的[单样本检测器](https://arxiv.org/abs/1512.02325)模型的签名。

对目标检测模型进行训练以检测多类目标的存在和位置。例如，可以使用包含各种水果的图像以及指定它们所代表的水果类别（例如苹果、香蕉或草莓）的*标签*以及指定每个目标在图像中出现的位置的数据来训练模型。

当图像随后被提供给模型时，它将输出它检测到的目标的列表、包含每个目标的边界框的位置以及指示检测正确的置信度的分数。

### 输入签名

模型接收图像作为输入。

我们假设预期的图像是 300x300 像素，每个像素有三个通道（红、蓝和绿）。这应该作为 270,000 字节值 (300x300x3) 的扁平缓冲区提供给模型。如果模型是<a href="../../performance/post_training_quantization.md">量化模型</a>，则每个值都应该是表示 0 到 255 之间的值的单个字节。

您可以查看我们的[示例应用代码](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)，以了解如何在 Android 上执行此预处理。

### 输出签名

该模型会输出四个数组，映射到索引 0-4。数组 0、1 和 2 描述了 `N` 个检测到的目标，每个数组中的一个元素对应于每个目标。

<table>
  <thead>
    <tr>
      <th>索引</th>
      <th>名称</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>坐标</td>
      <td>介于 0 和 1 之间的 [N][4] 浮点值的多维数组，内部数组以 [上，左，下，右] 的形式表示边界框</td>
    </tr>
    <tr>
      <td>1</td>
      <td>类</td>
      <td>由 N 个整数组成的数组（输出为浮点值），每个整数表示标签文件中的类标签的索引</td>
    </tr>
    <tr>
      <td>2</td>
      <td>分数</td>
      <td>由 0 到 1 之间的 N 个浮点值组成的数组，表示检测到类的概率</td>
    </tr>
    <tr>
      <td>3</td>
      <td>检测次数</td>
      <td>N 的整数值</td>
    </tr>
  </tbody>
</table>

注：结果数（上例中为 10）是将检测模型导出到 TensorFlow Lite 时设置的参数。有关更多详细信息，请参阅<a href="#model-customization">模型自定义</a>。

例如，假设一个模特已经过检测苹果、香蕉和草莓的训练。当提供图像时，它将输出设定数量的检测结果（在本例中为 5）。

<table style="width: 60%;">
  <thead>
    <tr>
      <th>类别</th>
      <th>分数</th>
      <th>坐标</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>苹果</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>香蕉</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>草莓</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td>香蕉</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>苹果</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### 信任分数

为了解释这些结果，我们可以查看每个检测到的目标的分数和坐标。该分数是介于 0 和 1 之间的数字，表示对该目标被真正检测到的置信度。数字越接近 1，模型的置信度就越高。

根据您的应用，您可以决定一个截止阈值，低于该阈值将丢弃检测结果。对于当前示例，合理的分界值为 0.5（意味着检测有效的概率为 50%）。在这种情况下，数组中的最后两个目标将被忽略，因为这些置信度分数低于 0.5：

<table style="width: 60%;">
  <thead>
    <tr>
      <th>类型</th>
      <th>分数</th>
      <th>坐标</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>苹果</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>香蕉</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>草莓</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">香蕉</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">苹果</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

您可以检测到裁切的极限值并据此放弃检测结果，这取决于您的应用。 在我们的示例中，我们检测到裁切的极限值为 0.5 （这意味50%的检测是可信的）。在此示例中，我们将会忽略数组中最后两个目标，因为他们的信任分数低于了 0.5 。

例如，在下面的图像中，一个梨（它不是模型被训练来检测的目标）被错误地识别为了“人”。这是一个可以通过选择适当的截止值来忽略的假正例的例子。在这种情况下，0.6（或 60%）的临界值将轻松排除假正例。

<img src="images/android_apple_banana.png" width="30%" alt="Screenshot of Android example">

#### 坐标

对于每个检测到的目标，模型将返回一个由四个数字组成的数组，该数组表示围绕其位置的一个边界矩形。对于提供的入门模型，编号顺序如下：

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>top,</td>
      <td>left,</td>
      <td>bottom,</td>
      <td>right</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

top 值表示矩形的顶边距图像顶部的距离，以像素为单位。left 值表示左侧边缘与输入图像左侧之间的距离。其他值以类似的方式表示下边缘和右边缘。

<a class="button button-primary" href="http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip">下载初始模型和标签</a>

## 初始模型

我们的<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">入门模型</a>的性能基准数值使用<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">此处所述</a>的工具生成。

<table>
  <thead>
    <tr>
      <th>模型名称</th>
      <th>模型大小</th>
      <th>设备</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
</td>
    <td rowspan="3">       27 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>22ms</td>
    <td>46ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20ms</td>
    <td>29ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>7.6ms</td>
    <td>11ms**</td>
  </tr>
</table>

物体检测模块最多能够在一张图中识别和定位10个物体。目前支持80种物体的识别，详细列表如下： <a href="http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip">model zip</a>.

如果您需为识别新类型而训练模型，请参考 <a href="#customize_model">自定义模块</a>.

## 模型自定义

### 用法和限制

在 [Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models) 中可以找到具有多种延迟和精度特性的移动端优化检测模型。其中的每个模型都遵循以下部分中描述的输入和输出签名。

大多数下载压缩包都包含一个 `model.tflite` 文件。如果没有，则可以使用[这些指令](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)生成 TensorFlow Lite 平面缓冲区。也可以根据[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)的说明，将来自 [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 的 SSD 模型转换为 TensorFlow Lite。需要注意的是，检测模型不能使用 [TensorFlow Lite Converter](../../models/convert) 直接转换，因为它们需要一个中间步骤来生成对移动设备友好的源模型。上面链接的脚本会执行此步骤。

[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) 和 [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) 导出脚本都具有可启用更多输出对象或更慢、更准确的后处理的参数。请将 `--help` 与脚本一起使用，以查看支持的参数的详尽列表。

> 目前，设备端推断仅通过 SSD 模型进行优化。我们正在研究如何更好地支持其他架构，如 CenterNet 和 EfficientDet。

### 如何选择要自定义的模型？

每种模型都有自己的精度（通过 mAP 值量化）和延迟特性。您应该选择最适合您的用例和目标硬件的模型。例如，[Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) 模型非常适合在 Pixel 4 上推断 Google 的 Edge TPU。

您可以使用我们的[基准测试工具](https://www.tensorflow.org/lite/performance/measurement)评估模型并选择最高效的选项。

## 使用自定义数据对模型进行微调

我们提供的预训练模型在训练后可以检测 90 类目标。有关完整的类列表，请参阅<a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">模型元数据</a>中的标签文件。

您可以使用一种称为迁移学习的技术来重新训练模型，以识别不在原始集合中的类。例如，您可以重新训练模型以检测多种蔬菜。为此，您需要为您希望训练的每个新标签提供一组训练图像。推荐的方式是使用 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) 库，该库只需几行代码即可简化使用自定义数据集训练 TensorFlow Lite 模型的过程。它使用迁移学习来减少所需的训练数据量和时间。您还可以学习[少样本检测 Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb)，作为微调只有几个样本的预训练模型的示例。

要对更大的数据集进行微调，请查看以下使用 TensorFlow 目标检测 API 训练您自己的模型的指南：[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md)，[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)。经过训练后，可以按照以下说明将其转换为 TFLite 友好格式：[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)，[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)

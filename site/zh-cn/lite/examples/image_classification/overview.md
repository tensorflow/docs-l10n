# 图像分类


<img src="../images/image.png" class="attempt-right">

识别图像所表示内容的任务称为*图像分类*。我们可以对图像分类模型进行训练以识别各类图像。例如，您可以训练模型来识别表示三种不同类型动物的照片：兔子、仓鼠和狗。TensorFlow Lite 提供经过优化的预训练模型，您可以将其部署在您的移动应用中。请点击[这里](https://www.tensorflow.org/tutorials/images/classification)了解有关使用 TensorFlow 进行图像分类的更多信息。

下图展示了图像分类模型在 Android 上的输出。

<img src="images/android_banana.png" width="30%" alt="Screenshot of Android example">

注：(1) 要集成现有模型，请尝试 [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)。(2) 要自定义模型，请尝试 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification)。

## 开始

如果你使用 Android 和 iOS 之外的平台，或者你已经熟悉了 TensorFlow Lite 接口，你可以直接下载我们的新手图像分类模型及其附带的标签。

利用 TensorFlow Lite Task Library 中开箱即用的 API，只需几行代码即可<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">集成图像分类模型</a>。您也可以使用 TensorFlow Lite Support Library [构建自己的自定义推断流水线](../../inference_with_metadata/lite_support)。

下面的 Android 示例分别以 <a href="#choose_a_different_model">lib_task_api</a> 和 [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support) 形式演示了两个方法的实现。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">查看 Android 示例</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">查看 iOS 示例</a>

如果您使用的平台不是 Android/iOS，或者您已经熟悉 [TensorFlow Lite API](android.md)，请下载入门模型和支持文件（如果适用）。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">下载入门模型</a>

## 模型说明

### 工作原理

在训练中，向图像分类模型馈送图像及其关联*标签*。每个标签是不同概念或类的名称，模型将学习识别这些标签。

在给定足够的训练数据（通常是每个标签数百或数千个图像）的情况下，图像分类模型可以学习预测新图像是否属于它所训练的任何类。这个预测过程称为*推断*。请注意，您还可以使用[迁移学习](https://www.tensorflow.org/tutorials/images/transfer_learning)通过使用预先存在的模型来识别新的图像类。迁移学习不需要很大的训练数据集。

当您随后将新图像作为输入提供给模型时，它将输出表示它所训练的每种动物类型的图像的概率。示例输出可能如下所示：

<table style="width: 40%;">
  <thead>
    <tr>
      <th>动物类型</th>
      <th>概率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>兔子</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>仓鼠</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">狗</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

输出中的每个数字都对应训练数据中的一个标签。将输出和模型所训练的三个标签关联，您可以看到模型预测这个图像有很大概率表示一条狗。

您可能注意到（兔子、仓鼠和狗的）概率的总和是 1。这是多类模型的常见输出。（请参阅 <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a> 了解更多信息）。

注：图像分类只能告诉您图像表示训练模型所用的一个或多个类的概率。它无法告诉您目标的身份或在图像中的位置。如果您需要识别目标及其在图像中的位置，应使用<a href="../object_detection/overview">目标检测</a>模型。

<h4>模糊不清的结果</h4>

由于概率的总和始终等于 1，如果图像没有被确信地识别为属于模型训练模型的任何类，您可能会发现分布到各个标签的概率都不会特别大。

例如，下表可能表示一个模棱两可的结果：


<table style="width: 40%;">   <thead>     <tr>       <th>标签</th>       <th>概率</th>     </tr>   </thead>   <tbody>     <tr>       <td>兔子</td>       <td>0.31</td>     </tr>     <tr>       <td>仓鼠</td>       <td>0.35</td>     </tr>     <tr>       <td>狗</td>       <td>0.34</td>     </tr>   </tbody> </table> 如果您的模型经常返回不明确的结果，则可能需要不同的、更准确的模型。

<h3>选择模型架构</h3>

TensorFlow Lite 为您提供了各种图像分类模型，这些模型都在原始数据集上进行训练。<a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a> 上提供了 MobileNet、Inception 和 NASNet 等模型架构。要为您的用例选择最佳模型，您需要考虑各个架构以及各种模型之间的一些权衡。其中一些模型权衡是基于性能、准确率和模型大小等指标。例如，您可能需要一个更快的模型来构建条码扫描器，而可能更喜欢一个更慢、更准确的模型来构建医学成像应用。

请注意，我们提供的<a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md">图像分类模型</a>接受的输入大小各不相同。对于某些模型，大小标注在文件名中。例如，Mobilenet_V1_1.0_224 模型接受 224x224 像素的输入。所有模型都要求每个像素有三个颜色通道（红、绿、蓝）。经过量化的模型要求每个通道 1 个字节，浮点模型要求每个通道 4 个字节。我们的 <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">Android</a> 和 <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">iOS</a> 代码示例展示了如何将全尺寸相机图像处理为每个模型需要的格式。

<h3>使用和限制</h3>

TensorFlow Lite 图像分类模型对于单标签分类很有用；即，预测图像最可能表示哪个标签。这些模型在训练后可以识别 1000 类图像。有关完整的类列表，请参阅<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">模型 zip 文件</a>中的标签文件。

如果您想要训练模型来识别新类，请参阅<a href="#customize_model">自定义模型</a>。

针对以下用例，您应该使用不同的模型：

<ul>
  <li>预测图像中的一个或多个目标的类型和位置（请参阅<a href="../object_detection/overview">目标检测</a>）</li>
  <li>预测图像的组成，例如主体与背景（请参阅<a href="../segmentation/overview">分割</a>）</li>
</ul>

在您的目标设备上运行入门模型后，您可以尝试其他模型，在性能、准确率和模型大小之间找到最佳平衡。

<h3>自定义模型</h3>

我们提供的预训练模型在训练后可以识别 1000 类图像。有关完整的类列表，请参阅<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">模型 zip 文件</a>中的标签文件。

您也可以使用迁移学习重新训练一个模型来识别原始集中不存在的类。例如，您可以重新训练一个模型来区分不同品种的树，尽管原始训练数据中并没有树。为此，您的每个要训练的新标签都需要一组训练图像。

了解如何使用 <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a> 执行迁移学习，或<a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">使用 TensorFlow Codelab 识别花卉</a>中执行迁移学习。

<h2>选择不同模型</h2>

模型性能是根据模型在给定硬件上运行推断所需的时间量来衡量的。时间越短，模型的速度就越快。

您需要的性能取决于您的应用。对实时视频等应用，性能可能非常重要，因为需要在下一帧绘制完之前及时分析每一帧（例如，推断用时必须少于 33 ms 才能实时推断 30fps 视频流）。

经过量化的 TensorFlow Lite MobileNet 模型的性能范围为 3.7ms 至 80.3 ms。

性能基准数值使用<a href="https://www.tensorflow.org/lite/performance/benchmarks">基准测试工具</a>生成。

<table>
  <thead>
    <tr>
      <th>模型名称</th>
      <th>模型大小</th>
      <th>设备</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
</td>
    <td rowspan="3">       4.3 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>6ms</td>
    <td>13ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>3.3ms</td>
    <td>5ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td></td>
    <td>11ms**</td>
  </tr>
</table>

* 使用 4 个线程。

** 为了获得最佳性能结果，在 iPhone 上使用 2 个线程。

### 模型准确率

我们根据模型正确分类图像的频率来衡量准确率。例如，一个准确率为 60% 的模型平均有 60% 的时间能对图像进行正确分类。

最相关的准确率指标是 Top-1 和 Top-5。Top-1 是指模型输出中正确标签作为概率最高的标签出现的频率。Top-5 是指模型输出中正确标签出现在概率最高前五名的频率。

经过量化的 TensorFlow Lite MobileNet 模型的 Top-5 准确率范围为 64.4 至 89.9%。

### 模型大小

磁盘上模型的大小因其性能和准确率而异。大小对移动开发（可能影响应用的下载大小）或在使用硬件时（可用存储空间可能有限）很重要。

经过量化的 TensorFlow Lite MobileNet 模型的大小范围为 0.5 至 3.4 MB。

## 补充阅读和资源

请使用以下资源了解有关图像分类相关概念的更多信息：

- [使用 TensorFlow 进行图像分类](https://www.tensorflow.org/tutorials/images/classification)
- [使用 CNN 进行图像分类](https://www.tensorflow.org/tutorials/images/cnn)
- [迁移学习](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [数据增强](https://www.tensorflow.org/tutorials/images/data_augmentation)

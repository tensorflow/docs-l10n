# 图像分类


<img src="../images/image.png" class="attempt-right">

The task of identifying what an image represents is called *image classification*. An image classification model is trained to recognize various classes of images. For example, you may train a model to recognize photos representing three different types of animals: rabbits, hamsters, and dogs. TensorFlow Lite provides optimized pre-trained models that you can deploy in your mobile applications. Learn more about image classification using TensorFlow [here](https://www.tensorflow.org/tutorials/images/classification).

如果你对图像分类的概念不熟悉，你应该先阅读 <a href="#what_is_image_classification">什么是图像分类？</a>


<img src="images/android_banana.png" alt="Screenshot of Android example" width="30%">

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

## 开始

如果你使用 Android 和 iOS 之外的平台，或者你已经熟悉了 TensorFlow Lite 接口，你可以直接下载我们的新手图像分类模型及其附带的标签。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">下载新手图像分类及标签</a>

当新手模型在你的目标设备运行起来之后，你可以尝试其他模型，在性能、准确率以及模型体积间找到最佳的平衡点。详见 <a href="#choose_a_different_model">选择不同模型</a>。

我们在 Android 和 iOS 平台上都有图像分类的示例应用，并解释了它们的工作原理。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">查看 iOS 示例</a>

阅读 [Android example guide](android.md) 以了解应用工作原理。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Download starter model</a>

## 什么是图像分类？

### 示例应用和指导

在训练中，向图像分类模型馈送图像及其关联*标签*。每个标签是不同概念或类的名称，模型将学习识别这些标签。

下面的截屏为 Android 图像分类示例应用。

机器学习的一个常见应用是图像识别。比如，我们可能想要知道下图中出现了哪类动物。

<table style="width: 40%;">
  <thead>
    <tr>
      <th>动物种类</th>
      <th>Probability</th>
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

预测图像类别的任务被称为 *图像分类* 。训练图像分类模型的目的是识别各类图像。比如，一个模型可能被训练用于识别三种动物的特征：兔子、仓鼠和狗。

当我们提供一张新的图片给模型时，它会输出这张图片含有这三种动物的概率。以下是一个输出示例：

Note: Image classification can only tell you the probability that an image represents one or more of the classes that the model was trained on. It cannot tell you the position or identity of objects within the image. If you need to identify objects and their positions within images, you should use an <a href="../object_detection/overview">object detection</a> model.

<h4>模糊不清的结果</h4>

注意：图像分类只能告诉你图片里出现的类别及其概率，并且只能是被训练过的类别。它不能告诉你图片里对象的位置或者名称。 如果你需要识别图片里对象的名称及位置，你应该使用 <a href="../object_detection/overview.md">物体检测</a> 模型。

在训练中，用图像和其对应的 *标签* 投喂一个图像分类模型。每个标签是一个概念或种类的名字。这个模型就要学会去识别这些标签。


<table style="width: 40%;">   <thead>     <tr>       <th>Label</th>       <th>Probability</th>     </tr>   </thead>   <tbody>     <tr>       <td>rabbit</td>       <td>0.31</td>     </tr>     <tr>       <td>hamster</td>       <td>0.35</td>     </tr>     <tr>       <td>dog</td>       <td>0.34</td>     </tr>   </tbody> </table> If your model frequently returns ambiguous results, you may need a different, more accurate model.

<h3>Choosing a model architecture</h3>

TensorFlow Lite provides you with a variety of image classification models which are all trained on the original dataset. Model architectures like MobileNet, Inception, and NASNet are available on <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>. To choose the best model for your use case, you need to consider the individual architectures as well as some of the tradeoffs between various models. Some of these model tradeoffs are based on metrics such as performance, accuracy, and model size. For example, you might need a faster model for building a bar code scanner while you might prefer a slower, more accurate model for a medical imaging app.

为了执行推断，一张图片被输入进模型中。接着，模型将输出一串代表概率的数组，元素大小介于 0 和 1 之间。结合我们的示例模型，这个过程可能如下所示：

<h3>使用和限制</h3>

输出中的每个数字都对应训练数据中的一个标签。将我们的输出和这三个训练标签关联，我们能够看出，这个模型预测了这张图片中的对象有很大概率是一条狗。

如果您想要训练模型来识别新类，请参阅<a href="#customize_model">自定义模型</a>。

既然概率的总和总是等于 1，那么如果这张图片没有被模型识别出来，也就是不属于被训练的种类，你可能会发现它的几个标签都没有特别大的概率。

<ul>
  <li>Predicting the type and position of one or more objects within an image (see <a href="../object_detection/overview">Object detection</a>)</li>
  <li>Predicting the composition of an image, for example subject versus background (see <a href="../segmentation/overview">Segmentation</a>)</li>
</ul>

Once you have the starter model running on your target device, you can experiment with different models to find the optimal balance between performance, accuracy, and model size.

<h3>Customize model</h3>

我们提供的这些图形分类模型对单标签分类很有用。单标签分类是指预测图像最有可能表示的某一个标签。这些模型被训练用于识别 1000 类图像。完整的标签列表：<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">模型压缩包</a>

You can also use transfer learning to re-train a model to recognize classes not in the original set. For example, you could re-train the model to distinguish between different species of tree, despite there being no trees in the original training data. To do this, you will need a set of training images for each of the new labels you wish to train.

Learn how to perform transfer learning with the <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a>, or in the <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Recognize flowers with TensorFlow</a> codelab.

<h2>选择不同模型</h2>

Model performance is measured in terms of the amount of time it takes for a model to run inference on a given piece of hardware. The lower the time, the faster the model.

您需要的性能取决于您的应用。对实时视频等应用，性能可能非常重要，因为需要在下一帧绘制完之前及时分析每一帧（例如，推断用时必须少于 33 ms 才能实时推断 30fps 视频流）。

The TensorFlow Lite quantized MobileNet models' performance range from 3.7ms to 80.3 ms.

Performance benchmark numbers are generated with the <a href="https://www.tensorflow.org/lite/performance/benchmarks">benchmarking tool</a>.

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>模型大小</th>
      <th>Device </th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
    </td>
    <td rowspan="3">       4.3 Mb     </td>
    <td>Pixel 3 (Android 10) </td>
    <td>6ms</td>
    <td>13ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>3.3ms</td>
    <td>5ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
     <td></td>
    <td>11ms** </td>
  </tr>
</table>

我们根据在同样的硬件条件下，一个模型执行推断所花费的时间来衡量性能。时间越短，模型越快。

你需要的性能取决于你的应用。对实时视频这类应用来说，性能可能非常重要。因为需要在下一帧绘制完之前及时分析每一帧（例如：推断用时必须少于 33 ms 才能实时推断 30 fps 的视频流）。

### 模型准确率

Accuracy is measured in terms of how often the model correctly classifies an image. For example, a model with a stated accuracy of 60% can be expected to classify an image correctly an average of 60% of the time.

The most relevant accuracy metrics are Top-1 and Top-5. Top-1 refers to how often the correct label appears as the label with the highest probability in the model’s output. Top-5 refers to how often the correct label appears in the 5 highest probabilities in the model’s output.

The TensorFlow Lite quantized MobileNet models’ Top-5 accuracy range from 64.4 to 89.9%.

### Model size

磁盘上模型的大小因其性能和准确率而异。大小对移动开发（可能影响应用的下载大小）或在使用硬件时（可用存储空间可能有限）很重要。

The TensorFlow Lite quantized MobileNet models' sizes range from 0.5 to 3.4 MB.

## Further reading and resources

Use the following resources to learn more about concepts related to image classification:

- [Image classification using TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Image classification with CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [Transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [数据增强](https://www.tensorflow.org/tutorials/images/data_augmentation)

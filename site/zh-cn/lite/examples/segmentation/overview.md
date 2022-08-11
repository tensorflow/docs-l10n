# Segmentation

图像分割是将数字图像分割成多个子区域（像素的集合，也称为图像对象）的过程。分割的目标是将图像的表示简化和/或更改为更有意义和更容易分析的内容。

DeepLab 是用于语义图像分割的最先进的深度学习模型，其目标是为图像中的每个像素分配语义标签(例如人，狗，猫)。

<img src="images/segmentation.gif" class="attempt-right">

注：要集成现有模型，请尝试 [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter)。

## 开始

语义图像分割预测图像的每个像素是否与某个类相关联。这与检测矩形区域中目标[目标检测]的任务(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/object_detection/overview.md)和对整个图像进行分类[图像分类]的任务(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/image_classification/overview.md)形成对照。

您可以利用来自 [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_segmenter) 的开箱即用 API，将图像分割模型集成到几行代码中。您还可以使用 [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java) 集成模型。

下面的 Android 示例分别以 [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_task_api) 和 [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_interpreter) 形式演示了两个方法的实现。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">View Android example</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">View iOS example</a>

如果您使用的不是 Android 或 iOS 平台，或者您已经熟悉 TensorFlow Lite API，则可以下载我们的入门图像分割模型。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">下载入门模型</a>

## 模型说明

![segmentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/images/segmentation.png)

### How it works

语义图像分割会预测图像的每个像素是否与某个类相关联。这与<a href="../object_detection/overview.md">目标检测</a>（检测矩形区域中的目标）和<a href="../image_classification/overview.md">图像分类</a>（对整个图像进行分类）不同。

当前实现包括以下功能：

<ol>
  <li>DeepLabv1：我们使用空洞卷积来显式控制深度卷积神经网络中计算特征响应的分辨率。</li>
  <li>DeepLabv2：我们使用空洞空间金字塔池化 (ASPP) 通过多采样率和有效视场下的筛选器在多个尺度上稳健地分割目标。</li>
  <li>DeepLabv3：我们在 ASPP 模块上增加了图像级特征 [5, 6]，以捕获更远距离的信息。我们还包括批次归一化 [7] 参数，以便于训练。特别是在训练和评估过程中，我们使用空洞卷积来提取不同输出步幅的输出特征，从而有效地实现了在输出步幅 = 16 的情况下训练 BN，并在评估时获得了输出步幅 = 8 的高性能。</li>
  <li>DeepLabv3+：我们对 DeepLabv3 进行了扩展，包括一个简单但有效的解码模块，以优化分割结果，特别是沿着对象边界。此外，在这种编解码器结构中，人们可以通过空洞卷积来任意控制所提取的编码器特征的分辨率，以权衡精度和运行时间。</li>
</ol>

## Example output

Performance benchmark numbers are generated with the tool [described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>设备</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Deeplab v3</a>
</td>
    <td rowspan="3">       2.7 Mb</td>
    <td>Pixel 3 (Android 10) </td>
    <td>16ms</td>
    <td>37ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>20ms</td>
    <td>23ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1) </td>
     <td>16ms</td>
    <td>25ms**</td>
  </tr>
</table>

* 4 threads used.

** 2 threads used on iPhone for the best performance result.

## 补充阅读和资源

<ul>
  <li><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">TensorFlow 中基于 DeepLab 的语义图像分割</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">使用移动端 GPU 的 TensorFlow Lite 变得更快了（开发者预览）</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab：语义图像分割的深度标注</a></li>
</ul>

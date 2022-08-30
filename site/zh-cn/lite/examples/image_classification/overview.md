# 图像分类


<img src="../images/image.png" class="attempt-right">

The task of identifying what an image represents is called *image classification*. An image classification model is trained to recognize various classes of images. For example, you may train a model to recognize photos representing three different types of animals: rabbits, hamsters, and dogs. TensorFlow Lite provides optimized pre-trained models that you can deploy in your mobile applications. Learn more about image classification using TensorFlow [here](https://www.tensorflow.org/tutorials/images/classification).

The following image shows the output of the image classification model on Android.


<img src="images/android_banana.png" alt="Screenshot of Android example" width="30%">

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

## 开始

If you are new to TensorFlow Lite and are working with Android or iOS, it is recommended you explore the following example applications that can help you get started.

You can leverage the out-of-box API from [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_classifier) to integrate image classification models in just a few lines of code. You can also build your own custom inference pipeline using the [TensorFlow Lite Support Library](../../inference_with_metadata/lite_support).

The Android example below demonstrates the implementation for both methods as [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) and [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support), respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">View Android example</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">查看 iOS 示例</a>

If you are using a platform other than Android/iOS, or if you are already familiar with the [TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite), download the starter model and supporting files (if applicable).

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Download starter model</a>

## Model description

### How it works

在训练中，向图像分类模型馈送图像及其关联*标签*。每个标签是不同概念或类的名称，模型将学习识别这些标签。

Given sufficient training data (often hundreds or thousands of images per label), an image classification model can learn to predict whether new images belong to any of the classes it has been trained on. This process of prediction is called *inference*. Note that you can also use [transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning) to identify new classes of images by using a pre-existing model. Transfer learning does not require a very large training dataset.

When you subsequently provide a new image as input to the model, it will output the probabilities of the image representing each of the types of animal it was trained on. An example output might be as follows:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Animal type</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rabbit</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Hamster</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Dog</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

Each number in the output corresponds to a label in the training data. Associating the output with the three labels the model was trained on, you can see that the model has predicted a high probability that the image represents a dog.

You might notice that the sum of all the probabilities (for rabbit, hamster, and dog) is equal to 1. This is a common type of output for models with multiple classes (see <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a> for more information).

Note: Image classification can only tell you the probability that an image represents one or more of the classes that the model was trained on. It cannot tell you the position or identity of objects within the image. If you need to identify objects and their positions within images, you should use an <a href="../object_detection/overview">object detection</a> model.

<h4>Ambiguous results</h4>

Since the output probabilities will always sum to 1, if an image is not confidently recognized as belonging to any of the classes the model was trained on you may see the probability distributed throughout the labels without any one value being significantly larger.

For example, the following might indicate an ambiguous result:


<table style="width: 40%;">   <thead>     <tr>       <th>Label</th>       <th>Probability</th>     </tr>   </thead>   <tbody>     <tr>       <td>rabbit</td>       <td>0.31</td>     </tr>     <tr>       <td>hamster</td>       <td>0.35</td>     </tr>     <tr>       <td>dog</td>       <td>0.34</td>     </tr>   </tbody> </table> If your model frequently returns ambiguous results, you may need a different, more accurate model.

<h3>Choosing a model architecture</h3>

TensorFlow Lite provides you with a variety of image classification models which are all trained on the original dataset. Model architectures like MobileNet, Inception, and NASNet are available on <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>. To choose the best model for your use case, you need to consider the individual architectures as well as some of the tradeoffs between various models. Some of these model tradeoffs are based on metrics such as performance, accuracy, and model size. For example, you might need a faster model for building a bar code scanner while you might prefer a slower, more accurate model for a medical imaging app.

Note that the <a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">image classification models</a> provided accept varying sizes of input. For some models, this is indicated in the filename. For example, the Mobilenet_V1_1.0_224 model accepts an input of 224x224 pixels. All of the models require three color channels per pixel (red, green, and blue). Quantized models require 1 byte per channel, and float models require 4 bytes per channel. The <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md">Android</a> and <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">iOS</a> code samples demonstrate how to process full-sized camera images into the required format for each model.

<h3>使用和局限性</h3>

The TensorFlow Lite image classification models are useful for single-label classification; that is, predicting which single label the image is most likely to represent. They are trained to recognize 1000 image classes. For a full list of classes, see the labels file in the <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model zip</a>.

如果您想要训练模型来识别新类，请参阅<a href="#customize_model">自定义模型</a>。

For the following use cases, you should use a different type of model:

<ul>
  <li>Predicting the type and position of one or more objects within an image (see <a href="../object_detection/overview">Object detection</a>)</li>
  <li>Predicting the composition of an image, for example subject versus background (see <a href="../segmentation/overview">Segmentation</a>)</li>
</ul>

Once you have the starter model running on your target device, you can experiment with different models to find the optimal balance between performance, accuracy, and model size.

<h3>Customize model</h3>

The pre-trained models provided are trained to recognize 1000 classes of images. For a full list of classes, see the labels file in the <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model zip</a>.

You can also use transfer learning to re-train a model to recognize classes not in the original set. For example, you could re-train the model to distinguish between different species of tree, despite there being no trees in the original training data. To do this, you will need a set of training images for each of the new labels you wish to train.

Learn how to perform transfer learning with the <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a>, or in the <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Recognize flowers with TensorFlow</a> codelab.

<h2>Performance benchmarks</h2>

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

使用 4 个线程。

** 2 threads used on iPhone for the best performance result.

### 模型准确率

Accuracy is measured in terms of how often the model correctly classifies an image. For example, a model with a stated accuracy of 60% can be expected to classify an image correctly an average of 60% of the time.

The most relevant accuracy metrics are Top-1 and Top-5. Top-1 refers to how often the correct label appears as the label with the highest probability in the model’s output. Top-5 refers to how often the correct label appears in the 5 highest probabilities in the model’s output.

The TensorFlow Lite quantized MobileNet models’ Top-5 accuracy range from 64.4 to 89.9%.

### 模型规模

磁盘上模型的大小因其性能和准确率而异。大小对移动开发（可能影响应用的下载大小）或在使用硬件时（可用存储空间可能有限）很重要。

The TensorFlow Lite quantized MobileNet models' sizes range from 0.5 to 3.4 MB.

## Further reading and resources

Use the following resources to learn more about concepts related to image classification:

- [Image classification using TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Image classification with CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [Transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- 数据增强

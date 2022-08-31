# 이미지 분류

<img src="../images/image.png" class="attempt-right">

The task of identifying what an image represents is called *image classification*. An image classification model is trained to recognize various classes of images. For example, you may train a model to recognize photos representing three different types of animals: rabbits, hamsters, and dogs. TensorFlow Lite provides optimized pre-trained models that you can deploy in your mobile applications. Learn more about image classification using TensorFlow [here](https://www.tensorflow.org/tutorials/images/classification).

The following image shows the output of the image classification model on Android.

<img src="images/android_banana.png" alt="Android 예시 스크린 샷" width="30%">

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

## 시작하기

If you are new to TensorFlow Lite and are working with Android or iOS, it is recommended you explore the following example applications that can help you get started.

You can leverage the out-of-box API from [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_classifier) to integrate image classification models in just a few lines of code. You can also build your own custom inference pipeline using the [TensorFlow Lite Support Library](../../inference_with_metadata/lite_support).

The Android example below demonstrates the implementation for both methods as [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) and [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support), respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">View Android example</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">View iOS example</a>

If you are using a platform other than Android/iOS, or if you are already familiar with the [TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite), download the starter model and supporting files (if applicable).

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Download starter model</a>

## Model description

### How it works

During training, an image classification model is fed images and their associated *labels*. Each label is the name of a distinct concept, or class, that the model will learn to recognize.

Given sufficient training data (often hundreds or thousands of images per label), an image classification model can learn to predict whether new images belong to any of the classes it has been trained on. This process of prediction is called *inference*. Note that you can also use [transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning) to identify new classes of images by using a pre-existing model. Transfer learning does not require a very large training dataset.

When you subsequently provide a new image as input to the model, it will output the probabilities of the image representing each of the types of animal it was trained on. An example output might be as follows:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>동물 유형</th>
      <th>확률</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>토끼</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>햄스터</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">개</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

Each number in the output corresponds to a label in the training data. Associating the output with the three labels the model was trained on, you can see that the model has predicted a high probability that the image represents a dog.

You might notice that the sum of all the probabilities (for rabbit, hamster, and dog) is equal to 1. This is a common type of output for models with multiple classes (see <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a> for more information).

Note: Image classification can only tell you the probability that an image represents one or more of the classes that the model was trained on. It cannot tell you the position or identity of objects within the image. If you need to identify objects and their positions within images, you should use an <a href="../object_detection/overview">object detection</a> model.

<h4>모호한 결과</h4>

Since the output probabilities will always sum to 1, if an image is not confidently recognized as belonging to any of the classes the model was trained on you may see the probability distributed throughout the labels without any one value being significantly larger.

예를 들어 다음은 모호한 결과를 나타낼 수 있습니다.


<table style="width: 40%;">   <thead>     <tr>       <th>Label</th>       <th>Probability</th>     </tr>   </thead>   <tbody>     <tr>       <td>rabbit</td>       <td>0.31</td>     </tr>     <tr>       <td>hamster</td>       <td>0.35</td>     </tr>     <tr>       <td>dog</td>       <td>0.34</td>     </tr>   </tbody> </table> If your model frequently returns ambiguous results, you may need a different, more accurate model.

<h3>Choosing a model architecture</h3>

TensorFlow Lite provides you with a variety of image classification models which are all trained on the original dataset. Model architectures like MobileNet, Inception, and NASNet are available on <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>. To choose the best model for your use case, you need to consider the individual architectures as well as some of the tradeoffs between various models. Some of these model tradeoffs are based on metrics such as performance, accuracy, and model size. For example, you might need a faster model for building a bar code scanner while you might prefer a slower, more accurate model for a medical imaging app.

Note that the <a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">image classification models</a> provided accept varying sizes of input. For some models, this is indicated in the filename. For example, the Mobilenet_V1_1.0_224 model accepts an input of 224x224 pixels. All of the models require three color channels per pixel (red, green, and blue). Quantized models require 1 byte per channel, and float models require 4 bytes per channel. The <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md">Android</a> and <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">iOS</a> code samples demonstrate how to process full-sized camera images into the required format for each model.

<h3>용도 및 제한</h3>

The TensorFlow Lite image classification models are useful for single-label classification; that is, predicting which single label the image is most likely to represent. They are trained to recognize 1000 image classes. For a full list of classes, see the labels file in the <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model zip</a>.

If you want to train a model to recognize new classes, see <a href="#customize_model">Customize model</a>.

다음과 같은 사용 사례의 경우 다른 유형의 모델을 사용해야 합니다.

<ul>
  <li>Predicting the type and position of one or more objects within an image (see <a href="../object_detection/overview">Object detection</a>)</li>
  <li>Predicting the composition of an image, for example subject versus background (see <a href="../segmentation/overview">Segmentation</a>)</li>
</ul>

Once you have the starter model running on your target device, you can experiment with different models to find the optimal balance between performance, accuracy, and model size.

<h3>모델을 사용자 정의하기</h3>

The pre-trained models provided are trained to recognize 1000 classes of images. For a full list of classes, see the labels file in the <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model zip</a>.

You can also use transfer learning to re-train a model to recognize classes not in the original set. For example, you could re-train the model to distinguish between different species of tree, despite there being no trees in the original training data. To do this, you will need a set of training images for each of the new labels you wish to train.

Learn how to perform transfer learning with the <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a>, or in the <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Recognize flowers with TensorFlow</a> codelab.

<h2>성능 벤치마크</h2>

Model performance is measured in terms of the amount of time it takes for a model to run inference on a given piece of hardware. The lower the time, the faster the model.

필요한 성능은 애플리케이션에 따라 다릅니다. 다음 프레임이 그려지기 전에 각 프레임을 분석하는 것이 중요할 수 있는 실시간 비디오와 같은 애플리케이션의 경우, 성능이 중요할 수 있습니다(예: 30fps 비디오 스트림에서 실시간 추론을 수행하려면 추론이 33ms보다 빨라야 함).

The TensorFlow Lite quantized MobileNet models' performance range from 3.7ms to 80.3 ms.

Performance benchmark numbers are generated with the <a href="https://www.tensorflow.org/lite/performance/benchmarks">benchmarking tool</a>.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>Model size </th>
      <th>기기</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
    </td>
    <td rowspan="3">       4.3 Mb     </td>
    <td>Pixel 3(Android 10)</td>
    <td>6ms</td>
    <td>13ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>3.3ms</td>
    <td>5ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
     <td></td>
    <td>11ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

### Model accuracy

Accuracy is measured in terms of how often the model correctly classifies an image. For example, a model with a stated accuracy of 60% can be expected to classify an image correctly an average of 60% of the time.

The most relevant accuracy metrics are Top-1 and Top-5. Top-1 refers to how often the correct label appears as the label with the highest probability in the model’s output. Top-5 refers to how often the correct label appears in the 5 highest probabilities in the model’s output.

The TensorFlow Lite quantized MobileNet models’ Top-5 accuracy range from 64.4 to 89.9%.

### 모델 크기

The size of a model on-disk varies with its performance and accuracy. Size may be important for mobile development (where it might impact app download sizes) or when working with hardware (where available storage might be limited).

The TensorFlow Lite quantized MobileNet models' sizes range from 0.5 to 3.4 MB.

## Further reading and resources

Use the following resources to learn more about concepts related to image classification:

- [Image classification using TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Image classification with CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [Transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

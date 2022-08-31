# 객체 감지

Given an image or a video stream, an object detection model can identify which of a known set of objects might be present and provide information about their positions within the image.

For example, this screenshot of the <a href="#get_started">example application</a> shows how two objects have been recognized and their positions annotated:


<img src="images/android_apple_banana.png" alt="Screenshot of Android example" width="30%">

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker).

## 시작하기

To learn how to use object detection in a mobile app, explore the <a href="#example_applications_and_guides">Example applications and guides</a>.

If you are using a platform other than Android or iOS, or if you are already familiar with the <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite APIs</a>, you can download our starter object detection model and the accompanying labels.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">Download starter model with Metadata</a>

For more information about Metadata and associated fields (eg: `labels.txt`) see <a href="../../models/convert/metadata#read_the_metadata_from_models">Read the metadata from models</a>

If you want to train a custom detection model for your own task, see <a href="#model-customization">Model customization</a>.

다음 사용 사례의 경우 다른 유형의 모델을 사용해야 합니다.

<ul>
  <li>이미지가 나타낼 가능성이 가장 높은 단일 레이블 예측하기(<a href="../image_classification/overview.md">이미지 분류</a> 참조)</li>
  <li>이미지 구성 예측하기(예: 피사체 대 배경)(<a href="../segmentation/overview.md">세분화</a> 참조)</li>
</ul>

### Example applications and guides

If you are new to TensorFlow Lite and are working with Android or iOS, we recommend exploring the following example applications that can help you get started.

#### Android

You can leverage the out-of-box API from [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/object_detector) to integrate object detection models in just a few lines of code. You can also build your own custom inference pipeline using the [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

The Android example below demonstrates the implementation for both methods as [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_task_api) and [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter), respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">View Android example</a>

#### iOS

You can integrate the model using the [TensorFlow Lite Interpreter Swift API](../../guide/inference#load_and_run_a_model_in_swift). See the iOS example below.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">View iOS example</a>

## Model description

This section describes the signature for [Single-Shot Detector](https://arxiv.org/abs/1512.02325) models converted to TensorFlow Lite from the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/).

객체 감지 모델은 여러 클래스의 객체 존재와 위치를 감지하도록 훈련되었습니다. 예를 들어, 다양한 과일 조각이 포함된 이미지, 과일이 나타내는 과일의 종류(예: 사과, 바나나 또는 딸기)를 지정하는 *레이블* 및 각 객체가 이미지에서 나타나는 위치를 지정하는 데이터로 모델을 훈련할 수 있습니다.

When an image is subsequently provided to the model, it will output a list of the objects it detects, the location of a bounding box that contains each object, and a score that indicates the confidence that detection was correct.

### Input Signature

The model takes an image as input.

Lets assume the expected image is 300x300 pixels, with three channels (red, blue, and green) per pixel. This should be fed to the model as a flattened buffer of 270,000 byte values (300x300x3). If the model is <a href="../../performance/post_training_quantization.md">quantized</a>, each value should be a single byte representing a value between 0 and 255.

You can take a look at our [example app code](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) to understand how to do this pre-processing on Android.

### Output Signature

The model outputs four arrays, mapped to the indices 0-4. Arrays 0, 1, and 2 describe `N` detected objects, with one element in each array corresponding to each object.

<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Name</th>
      <th>설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Locations</td>
      <td>Multidimensional array of [N][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Classes</td>
      <td>Array of N integers (output as floating point values) each indicating the index of a class label from the labels file</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Scores</td>
      <td>Array of N floating point values between 0 and 1 representing probability that a class was detected</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Number of detections</td>
      <td>Integer value of N</td>
    </tr>
  </tbody>
</table>

NOTE: The number of results (10 in the above case) is a parameter set while exporting the detection model to TensorFlow Lite. See <a href="#model-customization">Model customization</a> for more details.

For example, imagine a model has been trained to detect apples, bananas, and strawberries. When provided an image, it will output a set number of detection results - in this example, 5.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>클래스</th>
      <th>Score</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Strawberry</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>Apple</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### Confidence score

To interpret these results, we can look at the score and the location for each detected object. The score is a number between 0 and 1 that indicates confidence that the object was genuinely detected. The closer the number is to 1, the more confident the model is.

Depending on your application, you can decide a cut-off threshold below which you will discard detection results. For the current example, a sensible cut-off is a score of 0.5 (meaning a 50% probability that the detection is valid). In that case, the last two objects in the array would be ignored because those confidence scores are below 0.5:

<table style="width: 60%;">
  <thead>
    <tr>
      <th>클래스</th>
      <th>Score</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Strawberry</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163] </td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Banana</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Apple</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

The cut-off you use should be based on whether you are more comfortable with false positives (objects that are wrongly identified, or areas of the image that are erroneously identified as objects when they are not), or false negatives (genuine objects that are missed because their confidence was low).

예를 들어, 다음 이미지에서 배(모델이 감지하도록 훈련된 객체가 아님)는 '사람'으로 잘못 식별되었습니다. 이는 적절한 컷오프를 선택하여 무시할 수 있는 거짓양성의 예입니다. 이 경우 0.6(또는 60%)의 컷오프는 거짓양성을 수월하게 배제합니다.


<img src="images/false_positive.png" alt="Screenshot of Android example showing a false positive" width="30%">

#### Location

For each detected object, the model will return an array of four numbers representing a bounding rectangle that surrounds its position. For the starter model provided, the numbers are ordered as follows:

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>top,</td>
      <td>좌측,</td>
      <td>하단,</td>
      <td>우측</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

상단 값은 이미지 상단에서 직사각형 상단 가장자리까지의 거리를 픽셀 단위로 나타냅니다. 좌측 값은 입력 이미지의 왼쪽에서 왼쪽 가장자리까지의 거리를 나타냅니다. 다른 값은 유사한 방식으로 하단 및 오른쪽 가장자리를 나타냅니다.

참고: 객체 감지 모델은 특정 크기의 입력 이미지를 허용합니다. 해당 이미지 크기는 기기의 카메라로 캡처한 원시 이미지의 크기와 다를 수 있으며 모델의 입력 크기에 맞게 원시 이미지를 자르고 크기를 조정하는 코드를 작성해야 합니다(관련 예제는 <a href="#get_started">예제 애플리케이션</a>)에서 찾아볼 수 있습니다). <br><br>모델이 출력하는 픽셀값은 잘리고 크기가 조정된 이미지의 위치를 참조하므로 올바르게 해석하려면 원시 이미지에 맞게 크기를 조정해야 합니다.

## 성능 벤치마크

Performance benchmark numbers for our <a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">starter model</a> are generated with the tool [described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">
      <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
    </td>
    <td rowspan="3">       27 Mb     </td>
    <td>Pixel 3(Android 10)</td>
    <td>22ms</td>
    <td>46ms*</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>20ms</td>
    <td>29ms*</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
     <td>7.6ms</td>
    <td>11ms** </td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## Model Customization

### Pre-trained models

Mobile-optimized detection models with a variety of latency and precision characteristics can be found in the [Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models). Each one of them follows the input and output signatures described in the following sections.

Most of the download zips contain a `model.tflite` file. If there isn't one, a TensorFlow Lite flatbuffer can be generated using [these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). SSD models from the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) can also be converted to TensorFlow Lite using the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md). It is important to note that detection models cannot be converted directly using the [TensorFlow Lite Converter](../../models/convert), since they require an intermediate step of generating a mobile-friendly source model. The scripts linked above perform this step.

Both the [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) &amp; [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) exporting scripts have parameters that can enable a larger number of output objects or slower, more-accurate post processing. Please use `--help` with the scripts to see an exhaustive list of supported arguments.

> Currently, on-device inference is only optimized with SSD models. Better support for other architectures like CenterNet and EfficientDet is being investigated.

### How to choose a model to customize?

Each model comes with its own precision (quantified by mAP value) and latency characteristics. You should choose a model that works the best for your use-case and intended hardware. For example, the [Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) models are ideal for inference on Google's Edge TPU on Pixel 4.

You can use our [benchmark tool](https://www.tensorflow.org/lite/performance/measurement) to evaluate models and choose the most efficient option available.

## Fine-tuning models on custom data

The pre-trained models we provide are trained to detect 90 classes of objects. For a full list of classes, see the labels file in the <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">model metadata</a>.

You can use a technique known as transfer learning to re-train a model to recognize classes not in the original set. For example, you could re-train the model to detect multiple types of vegetable, despite there only being one vegetable in the original training data. To do this, you will need a set of training images for each of the new labels you wish to train. The recommended way is to use [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) library which simplifies the process of training a TensorFlow Lite model using custom dataset, with a few lines of codes. It uses transfer learning to reduce the amount of required training data and time. You can also learn from [Few-shot detection Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb) as an example of fine-tuning a pre-trained model with few examples.

For fine-tuning with larger datasets, take a look at the these guides for training your own models with the TensorFlow Object Detection API: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md). Once trained, they can be converted to a TFLite-friendly format with the instructions here: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)

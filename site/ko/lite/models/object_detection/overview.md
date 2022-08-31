# 객체 감지

이미지 또는 비디오 스트림이 주어지면 객체 감지 모델은 알려진 객체 세트 중 어떤 것이 존재할 수 있는지 식별하고 이미지 내 위치에 대한 정보를 제공할 수 있습니다.

예를 들어, 이 <a href="#get_started">예제 애플리케이션</a> 스크린샷은 두 객체가 인식되고 해당 위치에 주석이 추가되는 방식을 보여줍니다.

<img src="../images/detection.png" class="attempt-right">

참고: (1) 기존 모델을 통합하려면 [TensorFlow Lite 작업 라이브러리](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector)를 사용해 보세요. (2) 모델을 사용자 지정하려면 [TensorFlow Lite 모델 제작기](https://www.tensorflow.org/lite/guide/model_maker)를 사용해 보세요.

## 시작하기

모바일 앱에서 객체 감지를 사용하는 방법을 알아보려면 <a href="#example_applications_and_guides">예제 애플리케이션 및 가이드</a>를 살펴보세요.

Android 또는 iOS 이외의 플랫폼을 사용 중이거나 <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite API</a>에 이미 익숙한 경우 스타터 객체 감지 모델과 함께 제공되는 레이블을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">메타데이터가 있는 스타터 모델 다운로드</a>

메타데이터 및 관련 필드(예: `labels.txt`)에 대한 자세한 내용은 <a href="../../models/convert/metadata#read_the_metadata_from_models">모델에서 메타데이터 읽기</a>를 참조하세요.

자신의 작업에 대한 사용자 지정 감지 모델을 훈련하려면 <a href="#model-customization">모델 사용자 지정</a>을 참조하세요.

다음 사용 사례의 경우 다른 유형의 모델을 사용해야 합니다.

<ul>
  <li>이미지가 나타낼 가능성이 가장 높은 단일 레이블 예측하기(<a href="../image_classification/overview.md">이미지 분류</a> 참조)</li>
  <li>이미지 구성 예측하기(예: 피사체 대 배경)(<a href="../segmentation/overview.md">세분화</a> 참조)</li>
</ul>

### 예제 애플리케이션 및 가이드

TensorFlow Lite를 처음 사용하고 Android 또는 iOS로 작업하는 경우, 다음 예제 애플리케이션을 탐색하면 시작하는 데 도움이 됩니다.

#### Android

[TensorFlow Lite 작업 라이브러리](../../inference_with_metadata/task_library/object_detector)의 기본 API를 활용하여 몇 줄의 코드로 객체 감지 모델을 통합할 수 있습니다. [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java)를 사용하여 사용자 지정 추론 파이프라인을 구축할 수도 있습니다.

아래 Android 예제는 각각 [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_task_api) 및 [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter)로 두 메서드를 구현한 내용을 보여줍니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Android 예제 보기</a>

#### iOS

[TensorFlow Lite Interpreter Swift API](../../guide/inference#load_and_run_a_model_in_swift)를 사용하여 모델을 통합할 수 있습니다. 아래의 iOS 예를 참조하세요.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">iOS 예제 보기</a>

## 모델 설명

이 섹션에서는 [TensorFlow Object Detection API](https://arxiv.org/abs/1512.02325)에서 TensorFlow Lite로 변환된 [Single-Shot Detector](https://github.com/tensorflow/models/blob/master/research/object_detection/) 모델의 서명을 설명합니다.

객체 감지 모델은 여러 클래스의 객체 존재와 위치를 감지하도록 훈련되었습니다. 예를 들어, 다양한 과일 조각이 포함된 이미지, 과일이 나타내는 과일의 종류(예: 사과, 바나나 또는 딸기)를 지정하는 *레이블* 및 각 객체가 이미지에서 나타나는 위치를 지정하는 데이터로 모델을 훈련할 수 있습니다.

이후에 모델에 이미지를 제공하면 감지된 객체 목록, 각 객체가 포함된 경계 상자의 위치 및 감지가 정확하다는 확신을 나타내는 점수가 출력됩니다.

### 입력 서명

모델은 이미지를 입력으로 사용합니다.

예상 이미지는 300x300픽셀이며 픽셀당 3개의 채널(빨간색, 파란색, 녹색)이 있습니다. 이는 270,000바이트 값(300x300x3)의 평면화된 버퍼로 모델에 제공되어야 합니다. 모델이 <a href="../../performance/post_training_quantization.md">양자화</a>된 경우, 각 값은 0에서 255 사이의 값을 나타내는 단일 바이트여야 합니다.

Android에서 이 사전 처리를 수행하는 방법을 이해하려면 [예제 앱 코드](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)를 살펴보세요.

### 출력 서명

모델은 인덱스 0-4에 매핑된 4개의 배열을 출력합니다. 배열 0, 1 및 2는 `N`개의 감지된 객체를 설명하며 각 배열의 요소 하나가 각 객체에 해당합니다.

<table>
  <thead>
    <tr>
      <th>인덱스</th>
      <th>이름</th>
      <th>설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>위치</td>
      <td>0과 1 사이의 [N][4] 부동 소수점 값의 다차원 배열, [상단, 좌측, 하단, 우측] 형식의 경계 상자를 나타내는 내부 배열</td>
    </tr>
    <tr>
      <td>1</td>
      <td>클래스</td>
      <td>레이블 파일에서 클래스 레이블의 인덱스를 각각 나타내는 N개의 정수 배열(부동 소수점 값으로 출력)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>점수</td>
      <td>클래스가 감지될 확률을 나타내는 0과 1 사이의 N개의 부동 소수점 값 배열</td>
    </tr>
    <tr>
      <td>3</td>
      <td>감지 수</td>
      <td>N의 정수 값</td>
    </tr>
  </tbody>
</table>

참고: 결과 수(위의 경우 10)는 감지 모델을 TensorFlow Lite로 내보낼 때 설정되는 매개변수입니다. 자세한 내용은 <a href="#model-customization">모델 사용자 지정</a>을 참조하세요.

예를 들어, 모델이 사과, 바나나, 딸기를 감지하도록 훈련되었다고 생각해 보겠습니다. 이미지가 제공되면 설정된 수의 감지 결과(이 예에서는 5)를 출력합니다.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>클래스</th>
      <th>점수</th>
      <th>위치</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>사과</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>바나나</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>딸기</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td>바나나</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>사과</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### 신뢰도 점수

출력된 결과를 해석하기 위해 감지된 각 객체의 점수와 위치를 볼 수 있습니다. 점수는 0에서 1 사이의 숫자로 객체가 실제로 감지되었다는 확신을 나타냅니다. 숫자가 1에 가까울수록 모델의 신뢰도가 높아집니다.

애플리케이션에 따라 감지 결과를 버릴 컷오프 임계값을 결정할 수 있습니다. 현재 예에서는 합리적인 컷오프가 0.5점(감지가 유효할 확률이 50%임을 의미)이라고 결정할 수 있습니다. 이 경우 배열의 마지막 두 객체를 무시하게 되는데, 이에 대한 신뢰도 점수가 0.5 미만이기 때문입니다.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>클래스</th>
      <th>점수</th>
      <th>위치</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>사과</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>바나나</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>딸기</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">바나나</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">사과</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

사용하는 컷오프는 거짓양성(잘못 식별된 객체 또는 그렇지 않은 경우 객체로 잘못 식별된 이미지 영역) 또는 거짓음성(신뢰도 점수가 낮기 때문에 놓친 실제 객체)에 더 익숙한지 여부에 근거해야 합니다.

예를 들어, 다음 이미지에서 배(모델이 감지하도록 훈련된 객체가 아님)는 '사람'으로 잘못 식별되었습니다. 이는 적절한 컷오프를 선택하여 무시할 수 있는 거짓양성의 예입니다. 이 경우 0.6(또는 60%)의 컷오프는 거짓양성을 수월하게 배제합니다.

<img src="images/android_apple_banana.png" alt="Android 예시 스크린 샷" width="30%">

#### 위치

감지된 각 객체에 대해 모델은 위치를 둘러싸는 경계 사각형을 나타내는 4개의 숫자 배열을 반환합니다. 제공된 스타터 모델의 경우 숫자는 다음과 같이 주문됩니다.

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>상단,</td>
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

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">스타터 모델</a>에 대한 성능 벤치마크 수치는 [여기에 설명](https://www.tensorflow.org/lite/performance/benchmarks)된 도구로 생성됩니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
</td>
    <td rowspan="3">       27Mb</td>
    <td>Pixel 3(Android 10)</td>
    <td>22ms</td>
    <td>46ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>20ms</td>
    <td>29ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
     <td>7.6ms</td>
    <td>11ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## 모델 사용자 지정

### 사전 훈련된 모델

다양한 지연 및 정밀도 특성을 가진 모바일에 최적화된 감지 모델은 [감지 동물원](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models)에서 찾을 수 있습니다. 각각은 다음 섹션에서 설명하는 입력 및 출력 서명을 따릅니다.

대부분의 다운로드 zip에는 `model.tflite` 파일이 포함되어 있습니다. 없는 경우 [이 지침](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)에 따라 TensorFlow Lite 플랫 버퍼를 생성할 수 있습니다. [TF2 객체 감지 동물원](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)의 SSD 모델은 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md)의 지침을 사용하여 TensorFlow Lite로 변환할 수도 있습니다. 감지 모델은 모바일 친화적인 소스 모델을 생성하는 중간 단계가 필요하기 때문에 [TensorFlow Lite Converter](../../models/convert)를 사용하여 직접 변환할 수 없다는 점에 유의해야 합니다. 위에 링크된 스크립트가 이 단계를 수행합니다.

[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) 및 [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) 내보내기 스크립트에는 더 많은 수의 출력 객체 또는 더 느리고 정확한 사후 처리를 활성화할 수 있는 매개변수가 있습니다. 지원되는 인수의 전체 목록을 보려면 스크립트와 함께 `--help`를 사용하세요.

> 현재 온디바이스 추론은 SSD 모델에만 최적화되어 있습니다. CenterNet 및 EfficientDet과 같은 다른 아키텍처의 지원을 개선하기 위한 기회를 찾고 있습니다.

### 사용자 지정할 모델을 선택하는 방법은?

각 모델에는 고유한 정밀도(mAP 값으로 정량화됨) 및 대기 시간 특성이 있습니다. 사용 사례와 의도한 하드웨어에 가장 적합한 모델을 선택해야 합니다. 예를 들어 [Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) 모델은 Pixel 4에서 Google의 Edge TPU에 대한 추론에 이상적입니다.

[벤치마크 도구](https://www.tensorflow.org/lite/performance/measurement)를 사용하여 모델을 평가하고 사용 가능한 가장 효율적인 옵션을 선택할 수 있습니다.

## 사용자 지정 데이터에 대한 모델 미세 조정

제공되고 있는 사전 훈련된 모델은 90개의 객체 클래스를 감지하도록 훈련되었습니다. 전체 클래스 목록은 <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">모델 메타데이터</a>의 레이블 파일을 참조하세요.

전이 학습이라고 하는 기술을 사용하여 원래 세트에 없는 클래스를 인식하도록 모델을 다시 훈련할 수 있습니다. 예를 들어, 원래 훈련 데이터에 야채가 하나만 있음에도 불구하고 여러 유형의 야채를 감지하도록 모델을 다시 훈련할 수 있습니다. 이를 위해 훈련하려는 새 레이블 각각에 대한 훈련 이미지 세트가 필요합니다. 권장되는 방법은 몇 줄의 코드로 사용자 지정 데이터세트를 사용하여 TensorFlow Lite 모델 훈련 프로세스를 단순화하는 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) 라이브러리를 사용하는 것입니다. 필요한 훈련 데이터의 양과 시간을 줄이기 위해 전이 학습이 사용됩니다. 몇 가지 예를 통해 사전 훈련된 모델을 미세 조정하는 예로서 [Few-shot detection Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb)을 배울 수도 있습니다.

더 큰 데이터세트로 미세 조정하려면 TensorFlow Object Detection API를 사용하여 자신의 모델을 훈련하기 위한 다음 가이드를 살펴보세요: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md). 훈련 후에는 다음 지침에 따라 TFLite 친화적인 형식으로 변환할 수 있습니다: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)

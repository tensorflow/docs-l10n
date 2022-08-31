# 이미지 분류

<img src="../images/image.png" class="attempt-right">

이미지가 나타내는 내용을 식별하는 작업을 *이미지 분류*라고 합니다. 이미지 분류 모델은 다양한 이미지 클래스를 인식하도록 훈련됩니다. 예를 들어 토끼, 햄스터, 개 등 세 가지 유형의 동물을 나타내는 사진을 인식하도록 모델을 훈련시킬 수 있습니다. TensorFlow Lite는 모바일 애플리케이션에 배포할 수 있는 최적화된 사전 학습된 모델을 제공합니다. [여기](https://www.tensorflow.org/tutorials/images/classification)에서 TensorFlow를 사용한 이미지 분류에 대해 자세히 알아보세요.

다음 이미지는 Android에서 이미지 분류 모델의 출력을 보여줍니다.

<img src="images/android_banana.png" alt="Android 예시 스크린 샷" width="30%">

참고: (1) 기존 모델을 통합하려면 [TensorFlow Lite 작업 라이브러리](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)를 사용해 보세요. (2) 모델을 사용자 지정하려면 [TensorFlow Lite 모델 제작기](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification)를 사용해 보세요.

## 시작하기

TensorFlow Lite를 처음 사용하고 Android 또는 iOS로 작업하는 경우, 다음 예제 애플리케이션을 탐색하면 시작하는 데 도움이 됩니다.

[TensorFlow Lite 작업 라이브러리](../../inference_with_metadata/task_library/image_classifier)의 기본 API를 활용하여 몇 줄의 코드만으로 오디오 분류 모델을 통합할 수 있습니다. [TensorFlow Lite 지원 라이브러리](../../inference_with_metadata/lite_support)를 사용하여 사용자 지정 추론 파이프라인을 구축할 수도 있습니다.

아래 Android 예제는 각각 [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) 및 [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support)로 두 메서드를 구현한 내용을 보여줍니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android 예제 보기</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS 예제 보기</a>

Android/iOS 이외의 플랫폼을 사용 중이거나 이미 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 이미 익숙하다면 스타터 모델과 지원 파일(해당되는 경우)을 다운로드하세요.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">스타터 모델 다운로드</a>

## 모델 설명

### 동작 원리

훈련 중에 이미지 분류 모델에는 이미지 및 관련 *레이블*이 제공됩니다. 각 레이블은 모델이 인식하는 방법을 배울 수 있는 고유한 개념 또는 클래스의 이름입니다.

충분한 훈련 데이터(종종 레이블당 수백 또는 수천 개의 이미지)가 주어지면 이미지 분류 모델은 새 이미지가 훈련된 클래스에 속하는지 예측하는 방법을 학습할 수 있습니다. 이 예측 프로세스를 *추론*이라고 합니다. [전이 학습](https://www.tensorflow.org/tutorials/images/transfer_learning)을 사용하여 기존 모델을 사용하는 식으로 이미지의 새 클래스를 식별할 수도 있습니다. 전이 학습에는 매우 큰 훈련 데이터세트가 필요하지 않습니다.

이후에 모델에 대한 입력으로 새 이미지를 제공하면 훈련된 각 동물 유형을 나타내는 이미지의 확률이 출력됩니다. 예제 출력은 다음과 같습니다.

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

출력의 각 숫자는 훈련 데이터의 레이블에 해당합니다. 해당 출력을 모델이 훈련된 3개의 레이블과 연결하면 모델이 이미지가 개를 나타낼 확률이 높은 것으로 예측한다는 사실을 알 수 있습니다.

모든 확률의 합(토끼, 햄스터 및 개)이 1이라는 것을 알 수 있습니다. 이것은 여러 클래스가 있는 모델에 대한 일반적인 출력 유형입니다(자세한 내용은 <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">소프트맥스</a>를 참조).

참고: 이미지 분류는 이미지가 모델이 훈련된 클래스 중 하나 이상을 나타낼 확률만 알려줍니다. 이미지 내에서 객체의 위치나 정체성을 알 수 없습니다. 이미지 내에서 객체와 위치를 식별해야 하는 경우 <a href="../object_detection/overview">객체 감지</a> 모델을 사용해야 합니다.

<h4>모호한 결과</h4>

출력 확률의 합은 항상 1이기 때문에 이미지가 모델이 훈련된 클래스에 속하는 것으로 확실하게 인식되지 않으면 어떤 하나의 값이 유의할 정도로 커지지 않으면서 확률이 레이블 전체에 걸쳐 분포되는 것을 볼 수 있습니다.

예를 들어 다음은 모호한 결과를 나타낼 수 있습니다.


<table style="width: 40%;">   <thead>     <tr>       <th>레이블</th>       <th>확률</th>     </tr>   </thead>   <tbody>     <tr>       <td>토끼</td>       <td>0.31</td>     </tr>     <tr>       <td>햄스터</td>       <td>0.35</td>     </tr>     <tr>       <td>개</td>       <td>0.34</td>     </tr>   </tbody> </table> 모델이 모호한 결과를 자주 반환하는 경우 더 정확한 다른 모델이 필요할 수 있습니다.

<h3>모델 아키텍처 선택하기</h3>

TensorFlow Lite는 원본 데이터세트에서 모두 훈련된 다양한 이미지 분류 모델을 제공합니다. MobileNet, Inception 및 NASNet과 같은 모델 아키텍처를 <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>에서 사용할 수 있습니다. 사용 사례에 가장 적합한 모델을 선택하려면 개별 아키텍처와 다양한 모델 간의 일부 절충점을 고려해야 합니다. 이러한 모델 간의 일부 절충점은 성능, 정확도 및 모델 크기와 같은 메트릭을 기반으로 합니다. 예를 들어 바코드 스캐너를 구축하기 위해 더 빠른 모델이 필요할 수 있고 의료 영상 앱을 위해 더 느리고 더 정확한 모델이 선호될 수도 있습니다.

참고: 제공되는 <a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">이미지 분류 모델</a>은 다양한 크기의 입력을 허용합니다. 일부 모델의 경우 크기가 파일 이름에 표시됩니다. 예를 들어 Mobilenet_V1_1.0_224 모델은 224x224픽셀의 입력을 허용합니다. 모든 모델에는 픽셀당 3개의 색상 채널(빨간색, 녹색 및 파란색)이 필요합니다. 양자화된 모델에는 채널당 1바이트가 필요하고 부동 모델에는 채널당 4바이트가 필요합니다. <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md">Android</a> 및 <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">iOS</a> 코드 샘플은 전체 크기 카메라 이미지를 각 모델에 필요한 형식으로 처리하는 방법을 보여줍니다.

<h3>용도 및 제한</h3>

TensorFlow Lite 이미지 분류 모델은 단일 레이블 분류에 유용합니다. 즉, 이미지가 나타낼 가능성이 가장 높은 단일 레이블을 예측합니다. 해당 모델은 1000개의 이미지 클래스를 인식하도록 훈련되었습니다. 전체 클래스 목록은 <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">모델 zip</a>의 레이블 파일을 참조하세요.

새 클래스를 인식하도록 모델을 훈련하려면 <a href="#customize_model">모델 사용자 정의</a>를 참조하세요.

다음과 같은 사용 사례의 경우 다른 유형의 모델을 사용해야 합니다.

<ul>
  <li>이미지 내 하나 이상의 객체 유형 및 위치 예측하기(<a href="../object_detection/overview">객체 감지</a> 참조)</li>
  <li>이미지의 구성 예측하기(예: 피사체 대 배경)(<a href="../segmentation/overview">세분화</a> 참조)</li>
</ul>

대상 장치에서 스타터 모델을 실행하면 다양한 모델을 실험하여 성능, 정확도 및 모델 크기 간의 최적 균형을 찾을 수 있습니다.

<h3>모델을 사용자 정의하기</h3>

제공된 사전 훈련된 모델은 1,000개의 이미지 클래스를 인식하도록 훈련되었습니다. 전체 클래스 목록은 <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">모델 zip</a>의 레이블 파일을 참조하세요.

전이 학습을 사용하여 원본 세트에 없는 클래스를 인식하도록 모델을 다시 훈련할 수도 있습니다. 예를 들어, 원본 훈련 데이터에 나무가 없더라도 서로 다른 나무 종을 구별하도록 모델을 다시 훈련할 수 있습니다. 이를 위해 훈련하려는 각 새 레이블에 대한 훈련 이미지 세트가 필요합니다.

<a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a> 또는 <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">TensorFlow를 이용한 꽃 인식</a> codelab에서 전이 학습을 수행하는 방법을 알아보세요.

<h2>성능 벤치마크</h2>

모델 성능은 모델이 주어진 하드웨어에서 추론을 실행하는 데 걸리는 시간으로 측정됩니다. 시간이 낮을수록 모델이 더 빠릅니다.

필요한 성능은 애플리케이션에 따라 다릅니다. 다음 프레임이 그려지기 전에 각 프레임을 분석하는 것이 중요할 수 있는 실시간 비디오와 같은 애플리케이션의 경우, 성능이 중요할 수 있습니다(예: 30fps 비디오 스트림에서 실시간 추론을 수행하려면 추론이 33ms보다 빨라야 함).

TensorFlow Lite는 MobileNet 모델의 성능 범위를 3.7ms에서 80.3ms까지 양자화했습니다.

성능 벤치마크 번호는 <a href="https://www.tensorflow.org/lite/performance/benchmarks">벤치마킹 도구</a>로 생성됩니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
</td>
    <td rowspan="3">       4.3Mb</td>
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

### 모델 정확성

모델이 이미지를 올바르게 분류하는 빈도로 정확도를 측정합니다. 예를 들어, 명시된 정확성이 60%인 모델은 평균 60%의 시간 동안 이미지를 올바르게 분류할 것으로 기대할 수 있습니다.

가장 관련성이 높은 정확도 메트릭은 Top-1 및 Top-5입니다. Top-1은 모델의 출력에서 가장 높은 확률을 가진 레이블로 올바른 레이블이 나타나는 빈도를 나타냅니다. Top-5는 모델의 출력에서 가장 높은 확률 5개에 올바른 레이블이 나타나는 빈도를 나타냅니다.

TensorFlow Lite는 MobileNet 모델의 Top-5 정확도 범위를 64.4~89.9%로 양자화했습니다.

### 모델 크기

디스크에 있는 모델의 크기는 성능과 정확성에 따라 다릅니다. 크기는 모바일 개발(앱 다운로드 크기에 영향을 줄 수 있음) 또는 하드웨어 작업(사용 가능한 저장 용량이 제한될 수 있음)에 중요할 수 있습니다.

TensorFlow Lite는 MobileNet 모델의 크기 범위를 0.5~3.4MB로 양자화했습니다.

## 추가 자료 및 리소스

다음 리소스를 사용하여 오디오 분류와 관련된 개념에 대해 자세히 알아보세요.

- [TensorFlow를 사용한 이미지 분류](https://www.tensorflow.org/tutorials/images/classification)
- [CNN을 사용한 이미지 분류](https://www.tensorflow.org/tutorials/images/cnn)
- [전이 학습](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [데이터 증강](https://www.tensorflow.org/tutorials/images/data_augmentation)

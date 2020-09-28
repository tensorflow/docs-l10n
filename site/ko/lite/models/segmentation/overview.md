# 세분화

<img src="../images/segmentation.png" class="attempt-right">

## 시작하기

*DeepLab*은 시맨틱 이미지 세분화를 위한 최첨단 딥 러닝 모델로, 목표는 입력 이미지의 모든 픽셀에 시맨틱 레이블(예: 사람, 개, 고양이)을 할당하는 것입니다.

TensorFlow Lite를 처음 사용하고 Android 또는 iOS로 작업하는 경우, 다음 예제 애플리케이션을 탐색하면 시작하는 데 도움이 됩니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Android 예제</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">iOS 예제</a>

Android 또는 iOS 이외의 플랫폼을 사용 중이거나 <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite API에</a> 이미 익숙한 경우 스타터 이미지 세분화 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">스타터 모델 다운로드하기</a>

## 동작 원리

시맨틱 이미지 세분화는 이미지의 각 픽셀이 특정 클래스와 연관되어 있는지를 예측합니다. 이는 직사각형 영역에서 객체를 감지하는 <a href="../object_detection/overview.md">객체 감지</a> 및 전체 이미지를 분류하는 <a href="../image_classification/overview.md">이미지 분류</a>와 대조적입니다.

현재 구현에는 다음 특성이 포함됩니다.

<ol>
  <li>DeepLabv1: 심층 컨볼루셔널 신경망 내에서 특성 응답이 계산되는 해상도를 명시적으로 제어하기 위해 atrous 컨볼루션을 사용합니다.</li>
  <li>DeepLabv2: Atrous Spatial Pyramid Pooling(ASPP)을 사용하여 여러 샘플링 속도와 효과적인 시야각에서 필터를 사용하여 여러 스케일로 객체를 강력하게 세분화합니다.</li>
  <li>DeepLabv3: 더 긴 범위의 정보를 캡처하기 위해 이미지 레벨 특성 [5, 6]으로 ASPP 모듈을 확장합니다. 또한 훈련을 용이하게 하기 위해 배치 정규화 [7] 매개변수를 포함합니다. 특히 훈련 및 평가 중에 서로 다른 출력 보폭에서 출력 특성을 추출하기 위해 atrous 컨볼루션을 적용하여 출력 보폭 = 16에서 BN을 효율적으로 훈련할 수 있고, 평가 중에 출력 보폭 = 8에서 높은 성능을 달성합니다.</li>
  <li>DeepLabv3+: DeepLabv3를 확장하여 특히 객체 경계를 따라 세분화 결과를 개선하기 위해 간단하면서도 효과적인 디코더 모듈을 포함합니다. 또한 이 인코더-디코더 구조에서 추출된 인코더 특성의 해상도를 임의로 제어하여 atrous 컨볼루션으로 정밀도와 런타임 간에 상충 관계를 이룰 수 있습니다.</li>
</ol>

## 성능 벤치마크

성능 벤치마크 수치는 [여기에 설명된](https://www.tensorflow.org/lite/performance/benchmarks) 도구를 사용하여 생성됩니다.

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
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite">Deeplab v3</a></td>
    <td rowspan="3">       2.7 Mb</td>
    <td>Pixel 3(Android 10)</td>
    <td>16ms</td>
    <td>37ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>20ms</td>
    <td>23ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
     <td>16ms</td>
    <td>25ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## 예제 출력

모델은 높은 정확성으로 대상 객체 위에 마스크를 만듭니다.

<img alt="이미지 분할을 보여주는 애니메이션" src="images/segmentation.gif">

## 세분화에 대해 자세히 알아보기

<ul>
  <li><p data-md-type="paragraph"><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Semantic Image Segmentation with DeepLab in TensorFlow</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">TensorFlow Lite Now Faster with Mobile GPUs (Developer Preview)</a></p></li>
  <li><p data-md-type="paragraph"><a>DeepLab: Deep Labelling for Semantic Image Segmentation</a></p></li>
</ul>

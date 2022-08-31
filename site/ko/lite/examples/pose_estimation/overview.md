# 포즈 추정

<img src="../images/pose.png" class="attempt-right">

포즈 추정은 ML 모델을 사용하여 주요 신체 관절(키포인트)의 공간적 위치를 추정하여 이미지 또는 비디오로부터 사람의 포즈를 추정하는 작업입니다.

## 시작하기

If you are new to TensorFlow Lite and are working with Android or iOS, explore the following example applications that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android">Android 예제</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">iOS 예제</a>

[TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 익숙하다면 스타터 MoveNet 포즈 추정 모델 및 지원 파일을 다운로드하세요.

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet"> 스타터 모델 다운로드</a>

웹 브라우저에서 포즈 추정을 시도하려면 <a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet">TensorFlow JS 데모</a>를 확인하세요.

## 모델 설명

### 동작 원리

포즈 추정은 예를 들어 이미지에서 누군가의 팔꿈치가 나타나는 위치를 결정할 수 있도록 이미지와 비디오에서 인체를 감지하는 컴퓨터 비전 기술을 말합니다. 포즈 추정은 주요 신체 관절의 위치를 추정할 뿐 이미지나 비디오에서 누가 누구인지 인식하지는 못한다는 사실을 알아야 합니다.

포즈 추정 모델은 처리된 카메라 이미지를 입력으로 받아 키포인트에 대한 정보를 출력합니다. 감지된 키포인트는 파트 ID로 인덱싱되며 신뢰도 점수는 0.0에서 1.0 사이입니다. 신뢰도 점수는 키포인트가 해당 위치에 존재할 확률을 나타냅니다.

두 가지 TensorFlow Lite 포즈 추정 모델의 참조 구현이 제공됩니다.

- MoveNet: 조명과 천둥의 두 가지 형태로 제공되는 첨단 포즈 추정 모델입니다. 아래 섹션에서 이 둘의 비교 내용을 참조하세요.
- PoseNet: 2017년에 출시된 이전 세대의 포즈 추정 모델입니다.

포즈 예측은 이미지와 비디오에서 사람의 모습을 감지하는 컴퓨터 비전 기술을 의미하며, 예를 들어 이미지에서 누군가의 팔꿈치가 나타나는 위치를 결정할 수 있습니다.

<table style="width: 30%;">
  <thead>
    <tr>
      <th>ID</th>
      <th>파트</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>코</td>
    </tr>
    <tr>
      <td>1</td>
      <td>왼쪽 눈</td>
    </tr>
    <tr>
      <td>2</td>
      <td>오른쪽 눈</td>
    </tr>
    <tr>
      <td>3</td>
      <td>왼쪽 귀</td>
    </tr>
    <tr>
      <td>4</td>
      <td>오른쪽 귀</td>
    </tr>
    <tr>
      <td>5</td>
      <td>왼쪽 어깨</td>
    </tr>
    <tr>
      <td>6</td>
      <td>오른쪽 어깨</td>
    </tr>
    <tr>
      <td>7</td>
      <td>왼쪽 팔꿈치</td>
    </tr>
    <tr>
      <td>8</td>
      <td>오른쪽 팔꿈치</td>
    </tr>
    <tr>
      <td>9</td>
      <td>왼쪽 손목</td>
    </tr>
    <tr>
      <td>10</td>
      <td>오른쪽 손목</td>
    </tr>
    <tr>
      <td>11</td>
      <td>왼쪽 골반 부위</td>
    </tr>
    <tr>
      <td>12</td>
      <td>오른쪽 골반 부위</td>
    </tr>
    <tr>
      <td>13</td>
      <td>왼쪽 무릎</td>
    </tr>
    <tr>
      <td>14</td>
      <td>오른쪽 무릎</td>
    </tr>
    <tr>
      <td>15</td>
      <td>왼쪽 발목</td>
    </tr>
    <tr>
      <td>16</td>
      <td>오른쪽 발목</td>
    </tr>
  </tbody>
</table>

아래에 출력의 예를 나타내었습니다.


<img alt="포즈 추정을 보여주는 애니메이션" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif">

## 성능 벤치마크

MoveNet은 두 가지 버전으로 제공됩니다.

- MoveNet.Lightning은 Thunder 버전보다 더 작고 빠르지만 덜 정확합니다. 최신 스마트폰에서 실시간으로 실행할 수 있습니다.
- MoveNet.Thunder는 더 정확한 버전이지만 Lightning보다 크고 느립니다. 더 높은 정확도가 필요한 사용 사례에 유용합니다.

MoveNet은 다양한 데이터세트, 특히 피트니스 동작 이미지가 있는 이미지에서 PoseNet을 능가합니다. 따라서 PoseNet보다 MoveNet을 사용하는 것이 좋습니다.

성능 벤치마크 수치는 [여기에 설명](../../performance/measurement)된 도구로 생성됩니다. 정확도(mAP) 수치는 [COCO 데이터세트](https://cocodataset.org/#home)의 일부분에서 측정되며, 여기에서 한 사람만 포함하도록 각 이미지를 필터링하고 자릅니다.

<table>
<thead>
  <tr>
    <th rowspan="2">모델</th>
    <th rowspan="2">크기(MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">대기 시간(ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - CPU 4 스레드</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - CPU 4 스레드</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td><a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder(FP16 양자화)</a></td>
    <td>12.6MB</td>
    <td>72.0</td>
    <td>155ms</td>
    <td>45ms</td>
    <td>594ms</td>
  </tr>
  <tr>
    <td><a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder(INT8 양자화)</a></td>
    <td>7.1MB</td>
    <td>68.9</td>
    <td>100ms</td>
    <td>52ms</td>
    <td>251ms</td>
  </tr>
  <tr>
    <td><a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning(FP16 양자화)</a></td>
    <td>4.8MB</td>
    <td>63.0</td>
    <td>60ms</td>
    <td>25ms</td>
    <td>186ms</td>
  </tr>
  <tr>
    <td><a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning(INT8 양자화)</a></td>
    <td>2.9MB</td>
    <td>57.4</td>
    <td>52ms</td>
    <td>28ms</td>
    <td>95ms</td>
  </tr>
  <tr>
    <td><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 백본, FP32)</a></td>
    <td>13.3MB</td>
    <td>45.6</td>
    <td>80ms</td>
    <td>40ms</td>
    <td>338ms</td>
  </tr>
</tbody>
</table>

## 추가 자료 및 리소스

- MoveNet 및 TensorFlow Lite를 사용한 포즈 추정에 대해 자세히 알아보려면 이 [블로그 게시물](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)을 확인하세요.
- 웹에서 포즈 추정에 대해 자세히 알아보려면 이 [블로그 게시물](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)을 확인하세요.
- TensorFlow Hub의 모델을 사용하여 Python에서 MoveNet을 실행하는 방법에 대해 알아보려면 이 [튜토리얼](https://www.tensorflow.org/hub/tutorials/movenet)을 확인하세요.
- Coral/EdgeTPU는 에지 장치에서 포즈 추정을 훨씬 빠르게 실행할 수 있습니다. 자세한 내용은 [EdgeTPU 최적화 모델](https://coral.ai/models/pose-estimation/)을 참조하세요.
- [여기](https://arxiv.org/abs/1803.08225)에서 PoseNet 논문을 읽어보세요.

이러한 포즈 추정의 사용 사례도 확인하세요.

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom 미러’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">놀라운 예술 설치물이 당신을 새로 만들어줍니다 | Chris Milk "성역의 배반"</a></li>
  <li><a href="https://vimeo.com/34824490">퍼펫 퍼레이드 - 인터랙티브 Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce(공연), 발췌</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">증강 현실</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">대화형 애니메이션</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">보행 분석</a></li>
</ul>

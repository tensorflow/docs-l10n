# 포즈 예측

<img src="../images/pose.png" class="attempt-right">

## 시작하기

*PoseNet*는 주요 신체 관절의 위치를 예측하여 이미지 또는 비디오에서 사람의 포즈를 예측하는 데 사용할 수 있는 비전 모델입니다.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">스타터 모델 다운로드하기</a>

웹 브라우저에서 포즈 예측을 실험하고 싶다면 <a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TensorFlow.js GitHub 리포지토리</a>를 확인하세요.

### 예제 애플리케이션 및 가이드

Android 및 iOS용 PoseNet 모델을 보여주는 TensorFlow Lite 애플리케이션 예제를 제공합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android">Android 예제</a>

## 동작 원리

포즈 예측은 이미지와 비디오에서 사람의 모습을 감지하는 컴퓨터 비전 기술을 의미하며, 예를 들어 이미지에서 누군가의 팔꿈치가 나타나는 위치를 결정할 수 있습니다.

분명히 하자면 이 기술로 이미지에 있는 사람은 인식하지 못합니다. 알고리즘은 단순히 주요 신체 관절의 위치를 예측할 뿐입니다.

감지된 키포인트는 0.0에서 1.0 사이의 신뢰도 점수와 함께 '파트 ID'로 인덱싱되며 1.0이 가장 높습니다.

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
    <td rowspan="3">
      <p data-md-type="paragraph"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">Posenet</a></p>
    </td>
    <td rowspan="3">12.7Mb</td>
    <td>Pixel 3(Android 10)</td>
    <td>12ms</td>
    <td>31ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>12ms</td>
    <td>19ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
     <td>4.8ms</td>
    <td>22ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## 예제 출력

<img alt="포즈 추정을 보여주는 애니메이션" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif">

## 수행 방법

성능은 기기 및 출력 보폭(히트맵 및 오프셋 벡터)에 따라 다릅니다. PoseNet 모델은 이미지 크기가 불변하므로 이미지가 축소되었는지와 관계없이 원본 이미지와 같은 배율로 포즈 위치를 예측할 수 있습니다. 즉, PoseNet은 성능은 희생하면서 더 높은 정확성을 갖도록 구성될 수 있습니다.

출력 보폭은 입력 이미지 크기를 기준으로 출력을 축소 조정하는 정도를 결정합니다. 레이어의 크기와 모델 출력에 영향을 줍니다. 출력 보폭이 높을수록 네트워크 및 출력 레이어의 해상도와 그에 따른 정확성이 낮아집니다. 이 구현에서 출력 보폭은 8, 16 또는 32의 값을 가질 수 있습니다. 즉, 출력 보폭이 32이면 성능은 가장 빠르지만 정확성은 가장 낮고 8은 정확성은 높지만 성능은 가장 느립니다. 따라서 16으로 시작하는 것이 좋습니다.

다음 이미지는 출력 보폭이 입력 이미지 크기를 기준으로 출력을 축소 조정하는 정도를 결정하는 방법을 보여줍니다. 출력 보폭이 높을수록 더 빠르지만 정확성이 떨어집니다.

<img alt="출력 보폭 및 히트 맵 해상도" src="../images/output_stride.png">

## 포즈 예측에 대해 자세히 알아보기

<ul>
  <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5">블로그 게시물: Real-time Human Pose Estimation in the Browser with TensorFlow.js</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://github.com/tensorflow/tfjs-models/tree/master/posenet">TF.js GitHub: Pose Detection in the Browser: Posenet Model</a></p></li>
   <li><p data-md-type="paragraph"><a href="https://medium.com/tensorflow/track-human-poses-in-real-time-on-android-with-tensorflow-lite-e66d0f3e6f9e">블로그 게시물: Track human poses in real-time on Android with TensorFlow Lite</a></p></li>
</ul>

### 사용 사례

<ul>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/128375543">'PomPom Mirror'</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/34824490">Puppet Parade-Interactive Kinect Puppets</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://vimeo.com/2892576">Messa di Voce(퍼포먼스), 발췌</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.instagram.com/p/BbkKLiegrTR/">증강 현실</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.instagram.com/p/Bg1EgOihgyh/">인터랙티브 애니메이션</a></p></li>
  <li><p data-md-type="paragraph"><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">보행 분석</a></p></li>
</ul>

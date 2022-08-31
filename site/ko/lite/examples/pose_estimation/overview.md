# 포즈 추정


<img src="../images/pose.png" class="attempt-right">

*PoseNet*는 주요 신체 관절의 위치를 예측하여 이미지 또는 비디오에서 사람의 포즈를 예측하는 데 사용할 수 있는 비전 모델입니다.

## 시작하기

If you are new to TensorFlow Lite and are working with Android or iOS, explore the following example applications that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android"> Android example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">
iOS example</a>

If you are familiar with the [TensorFlow Lite APIs](https://www.tensorflow.org/api_docs/python/tf/lite), download the starter MoveNet pose estimation model and supporting files.

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet"> Download starter model</a>

If you want to try pose estimation on a web browser, check out the <a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet"> TensorFlow JS Demo</a>.

## Model description

### 동작 원리

Pose estimation refers to computer vision techniques that detect human figures in images and videos, so that one could determine, for example, where someone’s elbow shows up in an image. It is important to be aware of the fact that pose estimation merely estimates where key body joints are and does not recognize who is in an image or video.

The pose estimation models takes a processed camera image as the input and outputs information about keypoints. The keypoints detected are indexed by a part ID, with a confidence score between 0.0 and 1.0. The confidence score indicates the probability that a keypoint exists in that position.

We provides reference implementation of two TensorFlow Lite pose estimation models:

- MoveNet: the state-of-the-art pose estimation model available in two flavors: Lighting and Thunder. See a comparison between these two in the section below.
- PoseNet: the previous generation pose estimation model released in 2017.

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

감지된 키포인트는 0.0에서 1.0 사이의 신뢰도 점수와 함께 '파트 ID'로 인덱싱되며 1.0이 가장 높습니다.


<img alt="포즈 추정을 보여주는 애니메이션" src="https://www.tensorflow.org/images/lite/models/pose_estimation.gif">

## 성능 벤치마크

MoveNet is available in two flavors:

- MoveNet.Lightning is smaller, faster but less accurate than the Thunder version. It can run in realtime on modern smartphones.
- MoveNet.Thunder is the more accurate version but also larger and slower than Lightning. It is useful for the use cases that require higher accuracy.

MoveNet outperforms PoseNet on a variety of datasets, especially in images with fitness action images. Therefore, we recommend using MoveNet over PoseNet.

Performance benchmark numbers are generated with the tool [described here](../../performance/measurement). Accuracy (mAP) numbers are measured on a subset of the [COCO dataset](https://cocodataset.org/#home) in which we filter and crop each image to contain only one person .

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Size (MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">Latency (ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - CPU 4 threads</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - CPU 4 threads</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder (FP16 quantized)</a>
    </td>
    <td>12.6MB</td>
    <td>72.0</td>
    <td>155ms</td>
    <td>45ms</td>
    <td>594ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder (INT8 quantized)</a>
    </td>
    <td>7.1MB</td>
    <td>68.9</td>
    <td>100ms</td>
    <td>52ms</td>
    <td>251ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning (FP16 quantized)</a>
    </td>
    <td>4.8MB</td>
    <td>63.0</td>
    <td>60ms</td>
    <td>25ms</td>
    <td>186ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning (INT8 quantized)</a>
    </td>
    <td>2.9MB</td>
    <td>57.4</td>
    <td>52ms</td>
    <td>28ms</td>
    <td>95ms</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 backbone, FP32)</a>
    </td>
    <td>13.3MB</td>
    <td>45.6</td>
    <td>80ms</td>
    <td>40ms</td>
    <td>338ms</td>
  </tr>
</tbody>
</table>

## 예제 출력

- Check out this [blog post](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) to learn more about pose estimation using MoveNet and TensorFlow Lite.
- Check out this [blog post](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) to learn more about pose estimation on the web.
- Check out this [tutorial](https://www.tensorflow.org/hub/tutorials/movenet) to learn about running MoveNet on Python using a model from TensorFlow Hub.
- Coral/EdgeTPU can make pose estimation run much faster on edge devices. See [EdgeTPU-optimized models](https://coral.ai/models/pose-estimation/) for more details.
- Read the PoseNet paper [here](https://arxiv.org/abs/1803.08225)

Also, check out these use cases of pose estimation.

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Augmented reality</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">Interactive animation</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Gait analysis</a></li>
</ul>

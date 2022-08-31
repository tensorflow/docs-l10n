# 강화 학습

강화 학습을 사용하여 훈련되고 TensorFlow Lite로 배포된 에이전트와 보드 게임을 하세요.

## Get started

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example application that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android">Android 예제</a>

Android 이외의 플랫폼을 사용 중이거나 이미 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 익숙하다면 훈련된 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike_tf.tflite">모델 다운로드</a>

## How it works

이 모델은 게임 에이전트가 'Plane Strike'라는 작은 보드 게임을 할 수 있도록 제작되었습니다. 이 게임과 규칙에 대한 간략한 소개는 이 [README](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android)를 참조하세요.

앱의 UI 아래에 인간 플레이어와 대결하는 에이전트를 만들었습니다. 에이전트는 보드 상태를 입력으로 사용하고 64개의 가능한 보드 셀 각각에 대한 예측 점수를 출력하는 3계층 MLP입니다. 이 모델은 정책 기울기(REINFORCE)를 사용하여 학습되며 [여기](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml)에서 학습 코드를 찾을 수 있습니다. 에이전트를 훈련한 후, 모델을 TFLite로 변환하고 Android 앱에 배포합니다.

Android 앱에서 실제 게임 플레이 중 에이전트가 움직일 차례가 되면 에이전트는 인간 플레이어의 보드 상태(하단 보드)를 확인합니다. 여기에는 이전의 성공 및 실패한 공격(적중 및 실패)에 대한 정보가 포함되며, 훈련된 모델을 사용하여 인간 플레이어보다 먼저 게임을 완료할 수 있도록 다음 공격 위치를 예측합니다.

## Performance benchmarks

Performance benchmark numbers are generated with the tool described [here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>장치</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2"><a href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike.tflite">정책 기울기</a></td>
    <td rowspan="2">       84Kb</td>
    <td>Pixel 3 (Android 10) </td>
    <td>0.01ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10) </td>
    <td>0.01ms*</td>
  </tr>
</table>

* 1개의 스레드가 사용되었습니다.

## 입력

이 모델은 (1, 8, 8)의 3차원 `float32` 텐서를 보드 상태로 받아들입니다.

## 출력

이 모델은 64개의 가능한 공략 위치 각각에 대한 예측 점수로 형상 (1,64)의 2차원 `float32` 텐서를 반환합니다.

## 자신의 모델 훈련하기

<a>훈련 코드</a>에서 <code>BOARD_SIZE</code> 매개변수를 변경하여 더 크거나 작은 보드에 대해 자신의 모델을 훈련할 수 있습니다.

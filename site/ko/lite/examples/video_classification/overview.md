# 비디오 분류

<img src="../images/video.png" class="attempt-right">

*비디오 분류*는 비디오가 나타내는 내용을 식별하는 머신 러닝 작업입니다. 비디오 분류 모델은 다양한 동작이나 움직임과 같은 고유한 클래스 세트가 포함된 비디오 데이터세트에서 학습됩니다. 모델은 비디오 프레임을 입력으로 받고 각 클래스가 비디오에서 표현될 확률을 출력합니다.

비디오 분류 및 이미지 분류 모델은 모두 이미지를 입력으로 사용하여 사전 정의된 클래스에 속하는 이러한 이미지의 확률을 예측합니다. 그러나 비디오 분류 모델은 비디오의 동작을 인식하기 위해 인접 프레임 간의 시공간 관계도 처리합니다.

예를 들어, *비디오 동작 인식* 모델은 달리기, 박수, 손 흔드는 것과 같은 인간의 동작을 식별하도록 훈련될 수 있습니다. 다음 이미지는 Android에서 비디오 분류 모델의 출력을 보여줍니다.

<img alt="Screenshot of Android example" src="https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/push-up-classification.gif" class="">

## 시작하기

Android 또는 Raspberry Pi 이외의 플랫폼을 사용 중이거나 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 이미 익숙한 경우, 스타터 비디오 분류 모델 및 지원 파일을 다운로드하세요. [TensorFlow Lite 지원 라이브러리](../../inference_with_metadata/lite_support)를 사용하여 고유한 사용자 지정 추론 파이프라인을 구축할 수도 있습니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1">메타데이터가 있는 스타터 모델 다운로드</a>

TensorFlow Lite를 처음 사용하고 Android 또는 Raspberry Pi로 작업하는 경우 시작하는 데 도움이 되는 다음 예제 애플리케이션을 살펴보세요.

### Android

Android 애플리케이션은 지속적인 비디오 분류를 위해 기기의 후면 카메라를 사용합니다. 추론은 [TensorFlow Lite Java API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/package-summary)를 사용하여 수행됩니다. 데모 앱은 프레임을 분류하고 예측된 분류를 실시간으로 표시합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/android">Android 예제</a>

### Raspberry Pi

Raspberry Pi 예제에서는 Python과 함께 TensorFlow Lite를 사용하여 연속 비디오 분류를 수행합니다. Raspberry Pi를 Pi Camera와 같은 카메라에 연결하여 실시간 비디오 분류를 수행합니다. 카메라의 결과를 보려면 모니터를 Raspberry Pi에 연결하고 SSH를 사용하여 Pi 셸에 액세스합니다(Pi에 키보드 연결을 피하기 위해).

시작하기 전에 Raspberry Pi OS(Buster로 업데이트하는 것이 좋음)로 Raspberry Pi를 [설정](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up)합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/raspberry_pi%20">Raspberry Pi 예제</a>

## 모델 설명

모바일 비디오 네트워크([MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet))는 모바일 장치에 최적화된 효율적인 비디오 분류 모델 집합입니다. MoViNets는 여러 대규모 비디오 동작 인식 데이터세트에서 첨단 정확도와 효율성을 보여주므로 *비디오 동작 인식* 작업에 적합합니다.

TensorFlow Lite용 [MoviNet](https://tfhub.dev/s?deployment-format=lite&q=movinet) 모델에는 [MoviNet-A0](https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification), [MoviNet-A1](https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification) 및 [MoviNet-A2](https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification)의 세 가지 형태가 있습니다. 이러한 형태들은 [Kinetics-600](https://arxiv.org/abs/1808.01340) 데이터세트로 훈련되어 600가지의 인간 행동을 인식합니다. *MoviNet-A0*는 가장 작고 빠르며 가장 덜 정확합니다. *MoviNet-A2*는 가장 크고 가장 느리며 가장 정확합니다. *MoviNet-A1*은 A0과 A2의 절충점입니다.

### 동작 원리

훈련 중에 비디오 분류 모델에 비디오 및 관련 *레이블*이 제공됩니다. 각 레이블은 모델이 인식하도록 학습할 고유한 개념 또는 클래스의 이름입니다. *비디오 동작 인식*의 경우, 비디오는 사람의 동작이며 레이블은 연결된 동작입니다.

비디오 분류 모델은 새로운 비디오가 훈련 중에 제공된 클래스에 속하는지 여부를 예측하는 방법을 학습할 수 있습니다. 이 과정을 *추론*이라고 합니다. 또한 [전이 학습](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)을 사용하면 기존 모델로 새로운 비디오 클래스를 식별할 수 있습니다.

모델은 연속 비디오를 수신하고 실시간으로 응답하는 스트리밍 모델입니다. 모델은 비디오 스트림을 수신할 때 훈련 데이터세트의 클래스가 비디오에 표시되는지 여부를 식별합니다. 각 프레임에 대해 모델은 비디오가 클래스를 나타낼 확률과 함께 이러한 클래스를 반환합니다. 주어진 시간의 예시 출력은 다음과 같을 수 있습니다.

<table style="width: 40%;">
  <thead>
    <tr>
      <th>동작</th>
      <th>확률</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>스퀘어 댄스</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>실 바늘</td>
      <td>0.08</td>
    </tr>
    <tr>
      <td>손가락을 만지작거리기</td>
      <td>0.23</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">손 흔들기</td>
      <td style="background-color: #fcb66d;">0.67</td>
    </tr>
  </tbody>
</table>

출력의 각 동작은 훈련 데이터의 레이블에 해당합니다. 확률은 동작이 비디오에 표시될 가능성을 나타냅니다.

### 모델 입력

이 모델은 RGB 비디오 프레임 스트림을 입력으로 받아들입니다. 입력 비디오의 크기는 유연하지만 이상적으로는 모델 훈련 해상도 및 프레임 속도와 일치합니다.

- **MoviNet-A0**: 5fps에서 172 x 172
- **MoviNet-A1**: 5fps에서 172 x 172
- **MoviNet-A1**: 5fps에서 224 x 224

입력 비디오는 일반적인 [이미지 입력 규칙](https://www.tensorflow.org/hub/common_signatures/images#input)에 따라 0과 1 범위 내의 색상 값을 가질 것으로 예상됩니다.

내부적으로도 모델은 이전 프레임에서 수집한 정보를 사용하여 각 프레임의 컨텍스트를 분석합니다. 이는 모델 출력에서 내부 상태를 가져와 다음 프레임을 위해 모델에 다시 공급하는 식으로 수행됩니다.

### 모델 출력

모델은 일련의 레이블과 해당 점수를 반환합니다. 점수는 각 클래스에 대한 예측을 나타내는 로짓 값입니다. 이러한 점수는 softmax 함수(`tf.nn.softmax`)를 사용하여 확률로 변환할 수 있습니다.

```python
    exp_logits = np.exp(np.squeeze(logits, axis=0))
    probabilities = exp_logits / np.sum(exp_logits)
```

내부적으로, 모델 출력에는 모델의 내부 상태도 포함되어 있으며 향후 프레임을 위해 모델에 이를 다시 입력합니다.

## 성능 벤치마크

성능 벤치마크 번호는 [벤치마킹 도구](https://www.tensorflow.org/lite/performance/measurement)로 생성됩니다. MoviNets는 CPU만 지원합니다.

모델 성능은 모델이 주어진 하드웨어에서 추론을 실행하는 데 걸리는 시간으로 측정됩니다. 시간이 짧을수록 모델이 더 빠르다는 것을 의미합니다. 정확도는 모델이 비디오에서 클래스를 올바르게 분류하는 빈도로 측정됩니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>크기</th>
      <th>정확도 *</th>
      <th>장치</th>
      <th>CPU **</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">MoviNet-A0(정수 양자화)</td>
    <td rowspan="2">       3.1MB</td>
    <td rowspan="2">65%</td>
    <td>Pixel 4</td>
    <td>5ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>11ms</td>
  </tr>
    <tr>
    <td rowspan="2">MoviNet-A1(정수 양자화)</td>
    <td rowspan="2">       4.5MB</td>
    <td rowspan="2">70%</td>
    <td>Pixel 4</td>
    <td>8ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>19ms</td>
  </tr>
      <tr>
    <td rowspan="2">MoviNet-A2(정수 양자화)</td>
    <td rowspan="2">       5.1MB</td>
    <td rowspan="2">72%</td>
    <td>Pixel 4</td>
    <td>15ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>36ms</td>
  </tr>
</table>

* Top-1 정확도는 [Kinetics-600](https://arxiv.org/abs/1808.01340) 데이터세트에서 측정됩니다.

** 대기 시간은 1-스레드가 있는 CPU에서 실행할 때 측정됩니다.

## 모델 사용자 지정

사전 훈련된 모델은 [Kinetics-600](https://arxiv.org/abs/1808.01340) 데이터세트에서 600개의 인간 행동을 인식하도록 훈련되었습니다. 또한 전이 학습을 사용하여 원래 세트에 없는 인간 행동을 인식하도록 모델을 재훈련할 수 있습니다. 이렇게 하려면 모델에 도입하려는 각각의 새로운 동작에 대한 훈련 비디오 세트가 필요합니다.

사용자 지정 데이터의 모델 미세 조정에 대한 자세한 내용은 [MoViNets 리포지토리](https://github.com/tensorflow/models/tree/master/official/projects/movinet) 및 [MoViNets 튜토리얼](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)을 참조하세요.

## 추가 자료 및 리소스

다음 리소스를 사용하여 이 페이지에서 논의된 개념에 대해 자세히 알아보세요.

- [MoViNets 리포지토리](https://github.com/tensorflow/models/tree/master/official/projects/movinet)
- [MoViNets 논문](https://arxiv.org/abs/2103.11511)
- [사전 훈련된 MoViNet 모델](https://tfhub.dev/s?deployment-format=lite&q=movinet)
- [MoViNets 튜토리얼](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
- [Kinetics 데이터세트](https://deepmind.com/research/open-source/kinetics)

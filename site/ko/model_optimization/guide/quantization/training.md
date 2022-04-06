# 양자화 인식 훈련

<sub>TensorFlow 모델 최적화로 유지</sub>

양자화에는 훈련 후 양자화와 양자화 인식 훈련의 두 가지 형태가 있습니다. 사용하기 쉬운 [훈련 후 양자화](post_training.md)로 시작하세요. 그러나 양자화 인식 훈련이 종종 모델 정확성에 더 좋습니다.

이 페이지는 양자화 인식 훈련에 대한 개요를 제공하여 사용 사례에 얼마나 적합한지를 결정하는 데 도움이 됩니다.

- 엔드 투 엔드 예제로 바로 들어가려면, [양자화 인식 훈련 예제](training_example.md)를 참조하세요.
- 사용 사례에 필요한 API를 빠르게 찾으려면, [양자화 인식 훈련 종합 가이드](training_comprehensive_guide.md)를 참조하세요.

## 개요

양자화 인식 훈련은 추론 시간 양자화를 에뮬레이트하여 다운스트림 도구가 실제로 양자화된 모델을 생성하는 데 사용할 모델을 생성합니다. 양자화된 모델은 낮은 정밀도(예: 32bit 부동 소수점 대신 8bit)를 사용하므로 배포 중에 이점이 있습니다.

### 양자화로 배포하기

양자화는 모델 압축 및 지연 시간 감소를 통해 개선을 제공합니다. API 기본값을 사용하면 모델 크기가 4배 줄어들며 일반적으로 테스트된 백엔드에서 CPU 지연 시간이 1.5~4배 향상됩니다. [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) 및 NNAPI와 같은 호환 가능한 머신러닝 가속기에서 지연 시간의 개선을 볼 수 있습니다.

이 기술은 음성, 시각, 텍스트 및 번역 사용 사례의 운영 환경에 사용됩니다. 이 코드는 현재 [이들 모델의 하위 집합](#general-support-matrix)을 지원합니다.

### 양자화 및 관련 하드웨어로 실험하기

사용자는 양자화 매개변수(예: 비트 수)와 어느 정도 기본 알고리즘을 구성할 수 있습니다. API 기본값의 이러한 변경으로 인해 현재 백엔드 배포에 지원되는 경로가 없습니다. 예를 들어 TFLite 변환 및 커널 구현은 8비트 양자화만 지원합니다.

이 구성과 관련된 API는 실험적이며 이전 버전과의 호환성이 적용되지 않습니다.

### API 호환성

사용자는 다음 API를 사용하여 양자화를 적용할 수 있습니다.

- 모델 구축: 순차 및 함수형 모델만 있는 `tf.keras`
- TensorFlow 버전: tf-nightly용 TF 2.x
    - TF 2.X 패키지가 있는 `tf.compat.v1`은 지원되지 않습니다.
- TensorFlow 실행 모드: 즉시 실행

다음 영역에 대한 지원 추가가 로드맵에 나와 있습니다.

<!-- TODO(tfmot): file Github issues. -->

- 모델 구축: Subclassed Model의 제한된 지원에서 미지원까지 명확히 합니다.
- 분산 훈련: `tf.distribute`

### 일반 지원 행렬

다음 영역에서 지원을 받을 수 있습니다.

- 모델 적용 범위: Conv2D 및 DepthwiseConv2D 레이어를 따르는 경우 [allowlisted 레이어](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py), BatchNormalization을 사용하는 모델을 사용하며 제한된 경우에는 `Concat`을 사용합니다.
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- 하드웨어 가속: API 기본값은 무엇보다도 EdgeTPU, NNAPI 및 TFLite 백엔드의 가속과 호환됩니다. 로드맵에서 주의 사항을 참조하세요.
- 양자화로 배포: 현재 텐서별 양자화가 아닌 컨볼루셔널 레이어에 대한 축별 양자화만 지원됩니다.

다음 영역에 대한 지원 추가가 로드맵에 나와 있습니다.

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- 모델 적용 범위: RNN/LSTM 및 일반 Concat 지원을 포함하도록 확장되었습니다.
- 하드웨어 가속: TFLite 변환기가 전체 정수 모델을 생성할 수 있는지 확인합니다. 자세한 내용은 [이 문제](https://github.com/tensorflow/tensorflow/issues/38285)를 참조하세요.
- 양자화 사용 사례 실험:
    - Keras 레이어에 걸쳐 있거나 훈련 단계가 필요한 양자화 알고리즘을 실험합니다.
    - API를 안정화합니다.

## 결과

### 도구를 사용한 이미지 분류

<figure>
  <table>
    <tr>
      <th>모델</th>
      <th>비 양자화 Top-1 정확성</th>
      <th>8bit 양자화 정확성</th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

모델은 Imagenet에서 테스트되었으며 TensorFlow 및 TFLite에서 평가되었습니다.

### 기술에 대한 이미지 분류

<figure>
  <table>
    <tr>
      <th>모델</th>
      <th>비 양자화 Top-1 정확성</th>
      <th>8bit 양자화 정확성</th>
    </tr>
<tr>
      <td>Nasnet-Mobile</td>
      <td>74 %</td>
      <td>73 %</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6 %</td>
      <td>75 %</td>
    </tr>
 </table>
</figure>

모델은 Imagenet에서 테스트되었으며 TensorFlow 및 TFLite에서 평가되었습니다.

## 예제

[양자화 인식 훈련 예제](training_example.md) 외에도 다음 예제를 참조하세요.

- 양자화를 사용한, 손으로 작성한 MNIST 숫자 분류 작업의 CNN 모델: [코드](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

유사한 주제에 관한 배경 정보는 *효율적인 정수 산술 전용 추론을 위한 신경망의 양자화 및 훈련* [논문](https://arxiv.org/abs/1712.05877)을 참조하세요. 이 논문에서는 이 도구에서 사용되는 몇 가지 개념을 소개합니다. 구현은 정확히 같지 않으며, 이 도구에 사용되는 추가 개념이 있습니다(예: 축별 양자화).

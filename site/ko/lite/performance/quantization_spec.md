# TensorFlow Lite 8bit 양자화 사양

다음 문서는 TensorFlow Lite의 8bit 양자화 체계에 대한 사양을 설명합니다. 이는 하드웨어 개발자가 TensorFlow Lite 양자화 모델로 추론하기 위한 하드웨어 지원을 제공하는 데 도움을 주기 위한 것입니다.

## 사양 요약

당사는 사양을 제공하고 있으며 사양을 따르는 경우에만 동작에 대한 일부 보증을 제공할 수 있습니다. 또한, 다른 하드웨어에는 사양을 구현할 때 약간의 편차가 발생하여 비트가 정확하지 않은 구현이 발생할 수 있는 환경 설정 및 제한이 있을 수 있음을 이해합니다. 대부분의 경우 수용 가능할 수 있지만(그리고 당사가 아는 한 여러 모델에서 수집한 연산별 허용 오차를 포함하는 일련의 테스트를 제공할 것입니다), 머신러닝의 특성상 (및 가장 일반적인 딥 러닝 사례의 경우) 어떠한 엄격한 보증도 제공할 수 없습니다.

8bit 양자화는 다음 공식을 사용하여 부동 소수점 값을 근사화합니다.

$$real_value = (int8_value - zero_point) \times scale$$

축별(Conv ops에서는 채널별이라고도 함) 또는 텐서별 가중치는 zero-point가 0인 범위 `[-127, 127]`에서 `int8` 2의 보수 값으로 표시됩니다. 텐서별 활성화/입력은 zero-point가 범위 `[-128, 127]`인 상태에서 범위 `[-128, 127]`에서 `int8` 2의 보수 값으로 표시됩니다.

특정 연산에 대한 다른 예외 사항이 아래에 나와 있습니다.

참고: 과거의 양자화 도구에서는 텐서별, 비대칭, `uint8` 양자화를 사용했습니다. 8bit 양자화를 위한 새로운 도구, 참조 커널 및 최적화된 커널에서 이 사양을 사용합니다.

## 부호 있는 정수 vs. 부호 없는 정수

TensorFlow Lite 양자화는 주로 8bit용 `int8` 양자화를 위한 도구 및 커널을 우선시합니다. 이는 0과 같은 zero-point로 표현되는 대칭 양자화의 편의를 위한 것입니다. 또한, 많은 백엔드에는 `int8xint8` 누적을 위한 추가 최적화가 있습니다.

## 축별 vs. 텐서별

텐서별 양자화는 전체 텐서별로 하나의 scale 및/또는 zero-point가 있음을 의미합니다. 축별 양자화는 `quantized_dimension`에서 슬라이스별로 하나의 scale 및/또는 `zero_point`가 있음을 의미합니다. 양자화 차원은 scale과 zero-point가 대응하는 텐서 형상의 차원을 지정합니다. 예를 들어, 양자화 매개변수: `scale=[1.0, 2.0, 3.0]`, `zero_point=[1, 2, 3]`, `quantization_dimension=1`일 때 `dims=[4, 3, 2, 1]`인 텐서 `t`는 `t`의 두 번째 차원에서 양자화됩니다.

```
t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3
```

`quantized_dimension`이 컨볼루션 가중치의 `output_channel`일 경우가 종종 있지만, 이론적으로는 커널 구현에서 각 내적에 해당하는 차원이 될 수 있으며 성능에 영향을 주지 않고 더 세분화된 양자화가 가능합니다. 따라서 정확성이 크게 향상됩니다.

TFLite는 점점 더 많은 연산을 수행할 수 있도록 축별로 지원합니다. 이 문서를 작성하는 시점에는 Conv2d 및 DepthwiseConv2d에 대한 지원이 존재합니다.

## 대칭 vs. 비대칭

활성화는 비대칭입니다. 부호 있는 `int8` 범위 `[-128, 127]` 내의 어느 곳에서나 zero-point를 가질 수 있습니다. 많은 활성화가 본질적으로 비대칭이며, zero-point는 추가 이진 비트의 정밀도를 효과적으로 얻을 수 있는 비교적 저렴한 방법입니다. 활성화에는 상수 가중치만 곱하기 때문에 상수 zero-point 값은 상당히 최적화될 수 있습니다.

가중치는 대칭입니다. 0과 같은 zero-point를 갖도록 강제합니다. 가중치 값에 동적 입력 및 활성화 값을 곱합니다. 이는 가중치의 zero-point에 활성화 값을 곱하는 피할 수 없는 런타임 비용이 발생함을 의미합니다. zero-point를 0으로 설정함으로써 이 비용을 피할 수 있습니다.

수학 설명: 축별 scale 값을 허용한다는 차이점을 제외하면, [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)의 섹션 2.3과 유사합니다. 다음과 같이 쉽게 일반화됩니다.

$A$는 양자화 활성화의 $m \times n$ 행렬입니다. <br> $B$는 양자화 가중치의 $n \times p$ 행렬입니다. <br> $A$, $a_j$의 $j$번째 행에 $B$, $b_k$의 $k$번째 열(둘 다 길이 $n$)을 곱하는 것을 고려하세요. 양자화 정수값과 zero-point 값은 각각 $q_a$, $z_a$ 및 $q_b$, $z_b$입니다.

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} = \sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) = \sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b - \sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

<!-- Don't change these `\\(` `\\)` to `$`. mathjax fails here with `$`-->

(\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)}) 항은 입력값과 가중치 값의 내적을 수행하기 때문에 피할 수 없습니다.

$$\sum_{i=0}^{n} q_{b}^{(i)} z_a$$ 및 $$\sum_{i=0}^{n} z_a z_b$$ A 항은 추론 호출마다 동일하게 유지되는 상수로 구성되어 있으므로 미리 계산할 수 있습니다.

활성화로 인해 추론이 매번 변하기 때문에 (\sum_{i=0}^{n} q_{a}^{(i)} z_b) 항은 추론할 때마다 계산해야 합니다. 가중치를 대칭으로 적용함으로써 이 항의 비용을 없앨 수 있습니다.

## int8 양자화 연산자 사양

아래에서는 int8 tflite 커널에 대한 양자화 요구 사항을 설명합니다.

```
ADD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

AVERAGE_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONCATENATION
  Input ...:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 0)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

DEPTHWISE_CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 3)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

FULLY_CONNECTED
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-tensor
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-tensor
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

L2_NORMALIZATION
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

LOGISTIC
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

MAX_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MUL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

RESHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

RESIZE_BILINEAR
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

SPACE_TO_DEPTH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TANH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

PAD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GATHER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

BATCH_TO_SPACE_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SPACE_TO_BATCH_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TRANSPOSE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MEAN
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUB
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SQUEEZE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LOG_SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (16.0 / 256.0, 127)

MAXIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

ARG_MAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

MINIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LESS
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

PADV2
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GREATER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

GREATER_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

LESS_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SLICE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

NOT_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

QUANTIZE (Requantization)
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
```

## 참고 자료

[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

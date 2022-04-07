# 모델 최적화

에지 기기에서는 메모리 또는 계산 능력이 제한된 경우가 많습니다. 이러한 제약 조건 내에서도 실행될 수 있도록 다양한 최적화를 모델에 적용할 수 있습니다. 또한 일부 최적화를 통해 추론 가속화를 위해 특수 하드웨어를 사용할 수 있습니다.

TensorFlow Lite 및 [TensorFlow 모델 최적화 도구 키트](https://www.tensorflow.org/model_optimization)는 추론 최적화의 복잡성을 최소화하는 도구를 제공합니다.

애플리케이션 개발 프로세스 중에 모델 최적화를 고려하는 것이 좋습니다. 이 문서는 에지 하드웨어에 배포하기 위해 TensorFlow 모델을 최적화하려는 몇 가지 모범 사례를 설명합니다.

## 모델을 최적화해야 하는 이유

모델 최적화가 애플리케이션 개발에 도움이 되는 몇 가지 주요 방법이 있습니다.

### 크기 축소

모델의 크기를 줄이기 위해 몇 가지 형태의 최적화를 사용할 수 있습니다. 작은 모델에는 다음과 같은 이점이 있습니다.

- **더 작은 저장 크기:** 작은 모델은 사용자 기기에서 차지하는 저장 공간이 적습니다. 예를 들어 더 작은 모델을 사용하는 Android 앱은 사용자의 모바일 기기에서 저장 공간을 더 적게 차지합니다.
- **더 작은 다운로드 크기:** 더 작은 모델은 사용자의 기기에 다운로드하는 데 필요한 시간과 대역폭이 더 적습니다.
- **메모리 사용량 감소:** 모델이 작을수록 실행 시 더 적은 RAM을 사용하므로 애플리케이션의 다른 부분에서 사용할 수 있는 메모리가 확보되고 성능과 안정성이 향상될 수 있습니다.

양자화를 통해 모든 경우에서 모델의 크기를 줄일 수 있으며 잠재적으로는 정확성이 떨어집니다. 잘라내기 및 클러스터링은 더 쉽게 압축할 수 있도록 만들어 다운로드할 모델의 크기를 줄일 수 있습니다.

### 지연 시간 감소

*지연 시간*은 주어진 모델로 단일 추론을 실행하는 데 걸리는 시간입니다. 일부 최적화 형태는 모델을 사용하여 추론을 실행하는 데 필요한 계산량을 줄여 지연 시간을 줄일 수 있습니다. 지연 시간은 전력 소비에도 영향을 미칠 수 있습니다.

현재 양자화는 추론 중에 발생하는 계산을 단순화하여 잠재적으로 정확성을 떨어뜨리는 방식으로 지연 시간을 줄이는 데 사용될 수 있습니다.

### 가속기 호환성

[에지 TPU](https://cloud.google.com/edge-tpu/) 와 같은 일부 하드웨어 가속기는 올바르게 최적화된 모델로 매우 빠르게 추론을 실행할 수 있습니다.

일반적으로 이러한 유형의 기기는 모델을 특정 방식으로 양자화해야 합니다. 요구 사항에 대해 자세히 알아보려면 각 하드웨어 가속기의 설명서를 참조하세요.

## 상충 관계

최적화는 잠재적으로 모델 정확성에 변화를 초래할 수 있으며 사용 여부는 애플리케이션 개발 프로세스 중에 고려되어야 합니다.

정확성 변경은 최적화되는 개별 모델에 따라 다르며 예측하기 어렵습니다. 일반적으로 크기 또는 지연 시간에 최적화된 모델은 정확성이 약간 낮아집니다. 애플리케이션에 따라 이는 사용자 경험에 영향을 미칠 수도 있고 그렇지 않을 수도 있습니다. 드물지만 특정 모델은 최적화 프로세스로 인해 정확성이 약간 개선될 수도 있습니다.

## 최적화 유형

TensorFlow Lite는 현재 양자화, 잘라내기 및 클러스터링을 통한 최적화를 지원합니다.

이 유형은 TensorFlow Lite와 호환되는 모델 최적화 기술에 대한 리소스를 제공하는 [TensorFlow 모델 최적화 도구 키트](https://www.tensorflow.org/model_optimization)의 일부입니다.

### 양자화

[양자화](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)는 기본적으로 32bit 부동 소수점 숫자인 모델의 매개변수를 나타내는 데 사용되는 숫자의 정밀도를 줄여서 동작합니다. 그 결과 모델 크기가 작아지고 계산 속도가 빨라집니다.

TensorFlow Lite에서는 다음 유형의 양자화를 사용할 수 있습니다.

기술 | 데이터 요구 사항 | 크기 축소 | 정확성 | 지원되는 하드웨어
--- | --- | --- | --- | ---
[Post-training float16 quantization](post_training_float16_quant.ipynb) | 데이터 없음 | 최대 50% | 사소한 정확성 손실 | CPU, GPU
[Post-training dynamic range quantization](post_training_quant.ipynb) | 데이터 없음 | 최대 75% | 최소 정확성 손실 | CPU, GPU(Android)
[Post-training integer quantization](post_training_integer_quant.ipynb) | 레이블이 없는 대표 샘플 | 최대 75% | 적은 정확성 손실 | CPU, GPU(Android), 에지 TPU, Hexagon DSP
[Quantization-aware training](http://www.tensorflow.org/model_optimization/guide/quantization/training) | 레이블이 지정된 훈련 데이터 | 최대 75% | Smallest accuracy loss | CPU, GPU(Android), 에지 TPU, Hexagon DSP

다음 의사 결정 트리는 단순히 예상되는 모델 크기와 정확도만 따져서 모델에 사용해야 할 수 있는 양자화 체계를 선택하는 데 도움이 됩니다.

![양자화 선택 트리](images/quantization_decision_tree.png)

다음은 몇 가지 모델에서 훈련 후 양자화 및 양자화 인식 훈련으로 나온 지연 시간 및 정확성 결과입니다. 모든 지연 시간 수치는 단일 big 코어 CPU를 사용하는 Pixel 2 기기에서 측정됩니다. 도구 키트가 개선됨에 따라 여기의 수치도 향상됩니다.

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> Benefits of model quantization for select CNN models
  </figcaption>
</figure>

### int16 활성화 및 int8 가중치를 사용한 전체 정수 양자화

[int16 활성화를 사용한 양자화](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)는 int16에서 활성화 및 int8에서 가중치를 사용하는 전체 정수 양자화 체계입니다. 이 모드는 int8의 활성화 및 가중치 모두에서 유사한 모델 크기를 유지하면서 전체 정수 양자화 방식과 비교하여 양자화된 모델의 정확도를 향상시킬 수 있습니다. 활성화가 양자화에 민감한 경우 권장됩니다.

<i>참고:</i> 현재 이 양자화 체계를 위해 TFLite에서는 최적화되지 않은 참조 커널 구현만 사용할 수 있으므로, 기본적으로 int8 커널에 비해 성능이 느립니다. 이 모드의 모든 장점은 현재 특수 하드웨어 또는 맞춤형 소프트웨어를 통해 이용할 수 있습니다.

다음은 이 모드를 활용하는 일부 모델의 정확도 결과입니다.

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy metric type </th>
      <th>Accuracy (float32 activations) </th>
      <th>Accuracy (int8 activations) </th>
      <th>Accuracy (int16 activations) </th>
    </tr> <tr><td>Wav2letter</td><td>WER</td><td>6.7%</td><td>7.7%</td>
      <td>7.2%</td></tr>
    <tr><td>DeepSpeech 0.5.1 (unrolled)</td><td>CER</td><td>6.13%</td><td>43.67%</td>
      <td>6.52%</td></tr>
    <tr><td>YoloV3</td><td>mAP(IOU=0.5)</td><td>0.577</td><td>0.563</td>
      <td>0.574</td></tr>
    <tr><td>MobileNetV1</td><td>Top-1 Accuracy</td><td>0.7062</td><td>0.694</td>
      <td>0.6936</td></tr>
    <tr><td>MobileNetV2</td><td>Top-1 Accuracy</td><td>0.718</td><td>0.7126</td>
      <td>0.7137</td></tr>
    <tr><td>MobileBert</td><td>F1(Exact match)</td><td>88.81(81.23)</td><td>2.08(0)</td>
      <td>88.73(81.15)</td></tr>
 </table>
  <figcaption>
    <b>Table 2</b> Benefits of model quantization with int16 activations
  </figcaption>
</figure>

### 잘라내기

[잘라내기](https://www.tensorflow.org/model_optimization/guide/pruning)는 예측에 미미한 영향만 미치는 모델 내 매개변수를 제거하는 방식으로 동작합니다. 잘라낸 모델은 디스크에서 크기가 같고 런타임 지연 시간이 같지만 더 효과적으로 압축할 수 있습니다. 따라서 잘라내기는 모델 다운로드 크기를 줄이는 데 유용한 기술입니다.

앞으로 TensorFlow Lite는 잘라낸 모델에 대한 지연 시간 감소를 제공할 것입니다.

### 클러스터링

[클러스터링](https://www.tensorflow.org/model_optimization/guide/clustering)은 모델에 있는 각 레이어의 가중치를 미리 정의된 수의 클러스터로 그룹화한 다음 각각의 개별 클러스터에 속하는 가중치의 중심 값을 공유하는 방식으로 동작합니다. 그 결과 모델의 고유한 가중치 값의 수가 줄어들어 복잡성이 줄어듭니다.

결과적으로 클러스터링된 모델을보다 효과적으로 압축하여 잘라내기와 유사한 배포 이점을 제공할 수 있습니다.

## 개발 워크플로

As a starting point, check if the models in [hosted models](../guide/hosted_models.md) can work for your application. If not, we recommend that users start with the [post-training quantization tool](post_training_quantization.md) since this is broadly applicable and does not require training data.

정확성 및 지연 시간 목표가 충족되지 않거나 하드웨어 가속기 지원이 중요한 경우 [양자화 인식 훈련](https://www.tensorflow.org/model_optimization/guide/quantization/training){:.external}이 더 나은 옵션입니다. [TensorFlow 모델 최적화 도구 키트](https://www.tensorflow.org/model_optimization)에서 추가 최적화 기술을 참조하세요.

모델 크기를 더 줄이려면 모델을 양자화하기 전에 [잘라내기](#pruning) 및/또는 [클러스터링](#clustering)을 시도할 수 있습니다.

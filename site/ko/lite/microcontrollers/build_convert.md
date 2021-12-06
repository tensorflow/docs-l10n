# 모델 구축 및 변환하기

마이크로컨트롤러는 RAM과 스토리지가 제한되어 있어 머신러닝 모델의 크기에 제약이 따릅니다. 또한 마이크로컨트롤러용 TensorFlow Lite는 현재 제한적인 연산만 지원하므로 모든 모델 아키텍처가 가능한 것은 아닙니다.

이 문서에서는 TensorFlow 모델을 마이크로컨트롤러에서 실행되도록 변환하는 과정을 설명합니다. 또한 지원되는 연산을 간략하게 설명하고 제한된 메모리에 맞게 모델을 설계하고 훈련하는 방법에 대한 지침을 제공합니다.

모델을 빌드하고 변환하는 실행 가능한 엔드 투 엔드 예제는 *Hello World* 예제의 일부인 다음 Colab을 참조하세요.

train_hello_world_model.ipynb

## 모델 변환

훈련된 TensorFlow 모델을 마이크로컨트롤러에서 실행되도록 변환하려면 [TensorFlow Lite 변환기 Python API](https://www.tensorflow.org/lite/convert/)를 사용해야 합니다. 그러면 모델이 [`FlatBuffer`](https://google.github.io/flatbuffers/)로 변환되어 모델 크기가 줄어들고 TensorFlow Lite 연산을 사용하도록 모델이 수정됩니다.

가능한 한 가장 작은 모델 크기를 얻으려면 [훈련 후 양자화](https://www.tensorflow.org/lite/performance/post_training_quantization) 사용을 고려해야 합니다.

### C 배열로 변환하기

많은 마이크로컨트롤러 플랫폼에는 기본 파일 시스템 지원이 없습니다. 프로그램에서 모델을 사용하는 가장 쉬운 방법은 모델을 C 배열로 포함하고 프로그램 내로 컴파일하는 것입니다.

다음 unix 명령은 TensorFlow Lite 모델을 `char` 배열로 포함하는 C 소스 파일을 생성합니다.

```bash
xxd -i converted_model.tflite > model_data.cc
```

다음과 같은 출력이 얻어집니다.

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

파일을 생성한 후에는 프로그램에 포함할 수 있습니다. 임베디드 플랫폼에서 메모리 효율을 개선하기 위해 배열 선언을 `const`로 변경하는 것이 중요합니다.

모델을 프로그램에 포함하고 사용하는 방법을 보여주는 예는 <em>Hello World</em> 예제의 <a><code>model.cc</code></a>를 참조하세요.

## 모델 아키텍처 및 훈련

마이크로컨트롤러에서 사용할 모델을 설계할 때 모델 크기, 워크로드 및 사용되는 연산을 고려하는 것이 중요합니다.

### 모델 크기

모델은 바이너리와 런타임 모두에서 프로그램의 나머지 부분과 함께 대상 기기의 메모리 내에 맞도록 충분히 작아야 합니다.

더 작은 모델을 만들려면 아키텍처에서 더 작은 레이어를 더 적게 사용할 수 있습니다. 그러나 작은 모델은 과소적합의 문제를 유발할 가능성이 높습니다. 따라서 메모리를 넘지 않는 범위에서 가장 큰 모델을 시도하고 사용하는 것이 합리적입니다. 그러나 더 큰 모델을 사용하면 프로세서 워크로드도 증가합니다.

참고: 마이크로컨트롤러용 TensorFlow Lite의 코어 런타임은 Cortex M3에서 16KB에 맞습니다.

### 워크로드

모델의 크기와 복잡성은 워크로드에 영향을 미칩니다. 크고 복잡한 모델은 효율 주기를 높이는 결과를 가져올 수 있습니다. 즉, 기기 프로세서의 작동 시간이 늘어나고 유휴 시간은 줄어듭니다. 이로 인해 전력 소비와 발열량이 증가하여 애플리케이션에 따라 문제가 될 수 있습니다.

### 연산 지원

마이크로컨트롤러용 TensorFlow Lite는 현재 TensorFlow 연산의 일부만 지원하기 때문에 실행 가능한 모델 아키텍처에서 제약이 따릅니다. 특정 아키텍처의 참조 구현과 최적화 측면에서 연산 지원을 확대하기 위해 노력하고 있습니다.

지원되는 연산은 [`all_ops_resolver.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.cc) 파일에서 확인할 수 있습니다.

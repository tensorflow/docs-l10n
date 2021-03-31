# TensorFlow Lite 시작하기

TensorFlow Lite는 모바일, 임베디드 및 IoT 기기에서 TensorFlow 모델을 변환하고 실행하는 데 필요한 모든 도구를 제공합니다. 다음 가이드는 개발자 워크플로의 각 단계를 안내하고 추가 지침에 대한 링크를 제공합니다.

[TOC]

## 1. 모델 선택하기

<a id="1_choose_a_model"></a>

TensorFlow 모델은 특정 문제를 해결하도록 훈련 된 기계 학습 네트워크의 논리와 지식을 포함하는 데이터 구조입니다. 사전 학습 된 모델 사용부터 자신 만의 학습까지 TensorFlow 모델을 얻는 방법은 여러 가지가 있습니다.

TensorFlow Lite에서 모델을 사용하려면 전체 TensorFlow 모델을 TensorFlow Lite 형식으로 변환해야 합니다. TensorFlow Lite를 사용하여 모델을 만들거나 훈련할 수는 없습니다. 따라서 일반 TensorFlow 모델로 시작한 다음 [모델을 변환](#2_convert_the_model_format)해야 합니다.

참고 : TensorFlow Lite는 TensorFlow 작업의 제한된 하위 집합을 지원하므로 모든 모델을 변환 할 수 없습니다. 자세한 내용은 [TensorFlow Lite 연산자 호환성](ops_compatibility.md) 에 대해 읽어보세요.

### 사전 훈련된 모델 사용하기

TensorFlow Lite 팀은 다양한 기계 학습 문제를 해결하는 사전 학습 된 모델 세트를 제공합니다. 이러한 모델은 TensorFlow Lite와 함께 작동하도록 변환되었으며 애플리케이션에서 사용할 준비가되었습니다.

사전 훈련된 모델에는 다음이 포함됩니다.

- [이미지 분류](../models/image_classification/overview.md)
- [ 객체 감지 ](../models/object_detection/overview.md)
- [스마트 답장](../models/smart_reply/overview.md)
- [포즈 추정](../models/pose_estimation/overview.md)
- [ 세분화 ](../models/segmentation/overview.md)

[모델](../models)에서 사전 훈련된 모델의 전체 목록을 참조하세요.

#### 다른 출처의 모델

[TensorFlow Hub](https://www.tensorflow.org/hub)를 포함하여 여러 위치에서 사전 훈련된 TensorFlow 모델을 얻을 수 있습니다. 대부분의 경우, 이러한 모델은 TensorFlow Lite 형식으로 제공되지 않으므로 사용하기 전에 [변환](#2_convert_the_model_format)해야 합니다.

### 모델 재훈련(전이 학습)

전이 학습을 사용하면 훈련 된 모델을 다시 훈련하여 다른 작업을 수행 할 수 있습니다. 예를 들어, [이미지 분류](../models/image_classification/overview.md) 모델은 새로운 카테고리의 이미지를 인식하도록 재 학습 될 수 있습니다. 재 학습은 모델을 처음부터 학습하는 것보다 시간과 데이터가 덜 필요합니다.

전이 학습을 사용하여 사전 훈련된 모델을 애플리케이션에 맞게 사용자 지정할 수 있습니다. <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android">TensorFlow를 이용한 꽃 인식</a> Codelab에서 전이 학습을 수행하는 방법을 알아보세요.

### 사용자 정의 모델 훈련하기

자체 TensorFlow 모델을 설계하고 학습했거나 다른 출처를 통해 얻은 모델을 훈련한 경우, [이 모델을 TensorFlow Lite 형식으로 변환](#2_convert_the_model_format)해야 합니다.

[TensorFlow Lite Model Maker 라이브러리](model_maker.md)를 사용하면 사용자 정의 데이터세트를 사용하여 TensorFlow Lite 모델을 훈련하는 작업이 보다 간편해집니다.

## 2. 모델 변환하기

<a id="2_convert_the_model_format"></a>

TensorFlow Lite는 컴퓨팅 및 메모리 리소스가 제한적인 모바일 및 기타 임베디드 기기에서 모델을 효율적으로 실행하도록 설계되었습니다. 모델 저장에 특수 형식을 사용한 점도 이러한 효율성 개선에 일부 도움을 줍니다. TensorFlow 모델을 TensorFlow Lite에서 사용하려면 먼저 이 형식으로 변환해야 합니다.

모델을 변환하면 파일 크기가 줄어들고 정확도에 영향을 주지 않는 최적화가 가능합니다. TensorFlow Lite 변환기는 일부 상보적 균형 조정을 통해 파일 크기를 더욱 줄이고 실행 속도를 높일 수 있는 선택을 제공합니다.

참고: TensorFlow Lite는 TensorFlow 연산 중 일부만을 지원하므로 모든 모델을 변환할 수 있는 것은 아닙니다. 자세한 내용은 [TensorFlow Lite 연산자 호환성](ops_compatibility.md)을 읽어보세요.

### TensorFlow Lite 변환기

[TensorFlow Lite 변환기](../convert)는 훈련된 TensorFlow 모델을 TensorFlow Lite 형식으로 변환하는 Python API의 개념으로 사용할 수 있는 도구입니다. 또한 섹션 4, [모델 최적화](#4_optimize_your_model_optional)에서 다루는 최적화를 도입할 수도 있습니다.

다음 예는 TensorFlow `SavedModel`을 TensorFlow Lite 형식으로 변환하는 과정을 보여줍니다.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

같은 방식으로 [TensorFlow 2.0 모델도 변환](../convert/index.md)할 수 있습니다.

변환기는 [명령줄](../convert/cmdline.md)에서도 사용할 수 있지만 Python API 사용을 권장합니다.

### 옵션

변환기는 다음과 같은 다양한 입력 유형으로부터 변환할 수 있습니다.

[TensorFlow 1.x 모델을 변환](../convert/python_api.md)할 때 다음 유형이 해당합니다.

- [저장된 모델 디렉토리](https://www.tensorflow.org/guide/saved_model)
- 고정된 GraphDef([freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)에 의해 생성된 모델)
- [Keras](https://keras.io) HDF5 모델
- `tf.Session`에서 가져온 모델

[TensorFlow 2.x 모델을 변환](../convert/python_api.md)할 때는 다음 유형이 해당합니다.

- [저장된 모델 디렉토리](https://www.tensorflow.org/guide/saved_model)
- [`tf.keras` 모델](https://www.tensorflow.org/guide/keras/overview)
- [구체적인 기능](https://tensorflow.org/guide/concrete_function)

성능을 향상하거나 파일 크기를 줄일 수 있는 다양한 최적화를 적용하도록 변환기를 구성할 수 있습니다. 섹션 4, [모델 최적화](#4_optimize_your_model_optional)에서 이 내용을 다룹니다.

### Ops 호환성

TensorFlow Lite는 현재 [TensorFlow 연산의 일부만](ops_compatibility.md) 지원합니다. 장기적으로는 모든 TensorFlow 연산을 지원하는 것이 목표입니다.

변환하려는 모델에 지원되지 않는 연산이 포함된 경우, [TensorFlow Select](ops_select.md)를 사용하여 TensorFlow의 연산을 포함할 수 있습니다. 이로 인해 기기에 큰 용량의 바이너리가 배포됩니다.

## 3. 모델로 추론 실행하기

<a id="3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app"></a>

*추론*은 예측값을 얻기 위해 모델에서 데이터를 실행하는 과정입니다. 추론을 위해서는 모델, 인터프리터 및 입력 데이터가 필요합니다.

### TensorFlow Lite 인터프리터

[TensorFlow Lite 인터프리터](inference.md)는 모델 파일을 가져와서 입력 데이터에 정의한 작업을 실행하고 출력에 대한 액세스를 제공하는 라이브러리입니다.

인터프리터는 여러 플랫폼에서 작동하며 Java, Swift, Objective-C, C++ 및 Python에서 TensorFlow Lite 모델을 실행하기 위한 간단한 API를 제공합니다.

다음 코드는 Java에서 호출되는 인터프리터를 보여줍니다.

```java
try (Interpreter interpreter = new Interpreter(tensorflow_lite_model_file)) {
  interpreter.run(input, output);
}
```

### GPU 가속 및 Delegate

일부 기기는 머신러닝 연산을 위한 하드웨어 가속을 제공합니다. 예를 들어, 대부분의 휴대전화에는 CPU보다 빠르게 부동 소수점 행렬 연산을 수행할 수 있는 GPU가 있습니다.

속도 향상은 상당할 수 있습니다. 예를 들어 MobileNet v1 이미지 분류 모델은 GPU 가속을 사용할 때 Pixel 3 휴대전화에서 5.5배 더 빠르게 실행됩니다.

TensorFlow Lite 인터프리터는 여러 기기에서 하드웨어 가속을 사용하도록 [Delegate](../performance/delegates.md)로 구성할 수 있습니다. [GPU Delegate](../performance/gpu.md)를 사용하면 인터프리터가 기기의 GPU에서 적절한 연산을 실행할 수 있습니다.

다음 코드는 Java에서 사용되는 GPU Delegate를 보여줍니다.

```java
GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
Interpreter interpreter = new Interpreter(tensorflow_lite_model_file, options);
try {
  interpreter.run(input, output);
}
```

새 하드웨어 가속기에 대한 지원을 추가하려면 [고유한 delegate를 정의](../performance/delegates.md#how_to_add_a_delegate)할 수 있습니다.

### Android 및 iOS

TensorFlow Lite 인터프리터는 양대 모바일 플랫폼에서 쉽게 사용할 수 있습니다. 시작하려면 [Android 빠른 시작](android.md) 및 [iOS 빠른 시작](ios.md) 가이드를 살펴보세요. 두 플랫폼 모두에서 [예제 애플리케이션](https://www.tensorflow.org/lite/examples)을 사용할 수 있습니다.

필요한 라이브러리를 얻으려는 Android 개발자는 [TensorFlow Lite AAR](android.md#use_the_tensorflow_lite_aar_from_jcenter)을 사용해야 합니다. iOS 개발자는 [Swift 또는 Objective-C용 CocoaPods](ios.md#add_tensorflow_lite_to_your_swift_or_objective-c_project)를 사용해야 합니다.

### Linux

Embedded Linux는 머신러닝 배포를 위한 중요한 플랫폼입니다. Python을 사용하여 TensorFlow Lite 모델로 추론을 수행하려면 [Python 빠른 시작](python.md)을 따르세요.

대신 C++ 라이브러리를 설치하려면 [Raspberry Pi](build_rpi.md) 또는 [Arm64 기반 보드](build_arm64.md)(Odroid C2, Pine64, NanoPi와 같은 보드의 경우)에 대한 빌드 지침을 참조하세요.

### 마이크로컨트롤러

[마이크로컨트롤러용 TensorFlow Lite](../microcontrollers)는 마이크로컨트롤러 및 수 킬로바이트의 메모리만 있는 기타 기기를 대상으로 하는 TensorFlow Lite의 실험적 버전입니다.

### 연산

모델에 TensorFlow Lite에서 아직 구현되지 않은 TensorFlow 연산이 필요한 경우, [TensorFlow Select](ops_select.md)를 사용하여 모델에 해당 연산을 사용할 수 있습니다. TensorFlow 연산을 포함하는 인터프리터의 사용자 정의 버전을 빌드해야 합니다.

[사용자 정의 연산자](ops_custom.md)를 사용하여 고유한 연산을 작성하거나 새로운 연산을 TensorFlow Lite로 이식할 수 있습니다.

[연산자 버전](ops_version.md)을 사용하면 기존 작업에 새로운 기능과 매개변수를 추가할 수 있습니다.

## 4. 모델 최적화하기

<a id="4_optimize_your_model_optional"></a>

TensorFlow Lite는 종종 정확도에 미치는 영향을 최소화하면서 모델의 크기와 성능을 최적화하는 도구를 제공합니다. 최적화된 모델에는 약간 더 복잡한 훈련, 변환 또는 통합이 필요할 수 있습니다.

머신러닝 최적화는 진화하는 분야이며 TensorFlow Lite의 [모델 최적화 도구 키트](#model-optimization-toolkit)도 새로운 기술이 개발됨에 따라 지속적으로 발전하고 있습니다.

### 성능

모델 최적화의 목표는 주어진 기기에서 성능, 모델 크기 및 정확성의 이상적인 균형을 찾는 것입니다. [성능 모범 사례](../performance/best_practices.md)에서 어떻게 이러한 균형을 찾을 수 있는지 알아볼 수 있습니다.

### 양자화

양자화는 모델 내에서 값과 연산의 정밀도를 줄임으로써 모델의 크기와 추론에 필요한 시간을 모두 줄일 수 있습니다. 많은 모델의 경우, 정확도 손실은 미미한 수준입니다.

TensorFlow Lite 변환기를 사용하면 TensorFlow 모델을 쉽게 양자화할 수 있습니다. 다음 Python 코드는 `SavedModel`을 양자화하고 그 결과를 디스크에 저장합니다.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)
```

TensorFlow Lite는 전체 부동 소수점에서 반정밀도 부동 소수점(float16) 또는 8bit 정수로 값의 정밀도를 줄일 수 있도록 지원합니다. 각 선택에서 모델 크기와 정확도 사이에 절충이 이루어지며, 일부 연산은 이러한 감소된 정밀도에 최적화된 구현을 가지고 있습니다.

양자화에 대한 자세한 내용은 [훈련 후 양자화](../performance/post_training_quantization.md)를 참조하세요.

### 모델 최적화 도구 키트

[모델 최저화 도구 키트](../performance/model_optimization.md)는 개발자가 모델을 쉽게 최적화할 수 있도록 설계된 도구 및 기술 모음입니다. 많은 기술을 모든 TensorFlow 모델에 적용할 수 있으며 TensorFlow Lite에만 국한되지는 않지만 리소스가 제한된 기기에서 추론을 실행할 때 특히 유용합니다.

## 다음 단계

이제 TensorFlow Lite에 익숙해졌으므로 다음 리소스를 살펴보세요.

- 모바일 개발자라면 [Android 빠른 시작](android.md) 또는 [iOS 빠른 시작](ios.md)을 방문합니다.
- Linux 임베디드 기기를 빌드하는 경우 [Raspberry Pi](python.md) 및 [Arm64 기반 보드](build_rpi.md)에 대한 [Python 빠른 시작](build_arm64.md) 또는 C++ 빌드 지침을 참조합니다.
- [사전 훈련된 모델](../models)을 살펴봅니다.
- [예제 앱](https://www.tensorflow.org/lite/examples)을 사용해봅니다.

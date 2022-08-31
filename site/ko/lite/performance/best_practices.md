# 성능 모범 사례

모바일 및 임베디드 기기에는 계산 리소스가 제한되어 있으므로 애플리케이션 리소스를 효율적으로 유지하는 것이 중요합니다. TensorFlow Lite 모델 성능을 개선하는 데 사용할 수 있는 모범 사례 및 전략 목록을 작성했습니다.

## 작업에 가장 적합한 모델 선택

작업에 따라 모델 복잡성과 크기 간에 균형을 맞춰야 합니다. 작업에 높은 정확성이 필요하다면 크고 복잡한 모델이 필요할 수 있습니다. 정밀도가 낮은 작업의 경우 디스크 공간과 메모리를 적게 사용할 뿐만 아니라 일반적으로 더 빠르고 에너지 효율적이기 때문에 더 작은 모델을 사용하는 것이 좋습니다. 예를 들어 아래 그래프는 몇 가지 일반적인 이미지 분류 모델에 대한 정확성과 지연 시간 절충을 보여줍니다.

![Graph of model size vs accuracy](../images/performance/model_size_vs_accuracy.png "모델 크기 대 정확도")

![Graph of accuracy vs latency](../images/performance/accuracy_vs_latency.png "정확도 대 지연")

One example of models optimized for mobile devices are [MobileNets](https://arxiv.org/abs/1704.04861), which are optimized for mobile vision applications. [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) lists several other models that have been optimized specifically for mobile and embedded devices.

You can retrain the listed models on your own dataset by using transfer learning. Check out the transfer learning tutorials using TensorFlow Lite [Model Maker](../models/modify/model_maker/).

## 모델 프로파일링

작업에 적합한 후보 모델을 선택한 후에는 모델을 프로파일링하고 벤치마킹하는 것이 좋습니다. TensorFlow Lite [벤치마킹 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)에는 연산자별 프로파일링 통계를 표시하는 내장 프로파일러가 있습니다. 이는 성능 병목 현상과 계산 시간을 지배하는 연산자를 이해하는 데 도움이 될 수 있습니다.

You can also use [TensorFlow Lite tracing](measurement#trace_tensorflow_lite_internals_in_android) to profile the model in your Android application, using standard Android system tracing, and to visualize the operator invocations by time with GUI based profiling tools.

## 그래프에서 연산자 프로파일링 및 최적화

If a particular operator appears frequently in the model and, based on profiling, you find that the operator consumes the most amount of time, you can look into optimizing that operator. This scenario should be rare as TensorFlow Lite has optimized versions for most operators. However, you may be able to write a faster version of a custom op if you know the constraints in which the operator is executed. Check out the [custom operators guide](../guide/ops_custom).

## 모델 최적화

모델 최적화는 일반적으로 더 빠르고 에너지 효율적인 작은 모델을 만들어 모바일 기기에 배포될 수 있도록 하는 것을 목표로 합니다. TensorFlow Lite는 양자화와 같은 여러 가지 최적화 기술을 지원합니다.

Check out the [model optimization docs](model_optimization) for details.

## 스레드 수 조정

TensorFlow Lite는 많은 연산자를 위한 다중 스레드 커널을 지원합니다. 스레드 수를 늘리고 연산자 실행 속도를 높일 수 있습니다. 그러나 스레드 수를 늘리면 모델이 더 많은 리소스와 성능을 사용하게 됩니다.

일부 애플리케이션의 경우 지연 시간이 에너지 효율성보다 더 중요할 수 있습니다. 인터프리터 [스레드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) 수를 설정하여 스레드 수를 늘릴 수 있습니다. 그러나 다중 스레드 실행은 동시에 실행되는 다른 항목에 따라 성능 변동성이 증가합니다. 특히 모바일 앱의 경우가 해당합니다. 예를 들어 격리된 테스트는 단일 스레드에 비해 2배의 속도 향상을 보여줄 수 있지만 다른 앱이 동시에 실행되는 경우 단일 스레드보다 성능이 저하될 수 있습니다.

## 중복 사본 제거

애플리케이션이 신중하게 설계되지 않은 경우 모델에 입력을 공급하고 모델에서 출력을 읽을 때 중복 사본이 있을 수 있습니다. 중복된 사본은 제거하십시오. Java와 같은 더 높은 수준의 API를 사용하는 경우 설명서에서 성능 주의사항을 확인하세요. 예를 들어 Java API는 `ByteBuffers`가 [입력](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175)으로 사용되는 경우 훨씬 빠릅니다.

## 플랫폼별 도구로 애플리케이션 프로파일링

[Android 프로파일러](https://developer.android.com/studio/profile/android-profiler) 및 [Instrument](https://help.apple.com/instruments/mac/current/)와 같은 플랫폼별 도구는 앱을 디버그하는 데 사용할 수 있는 풍부한 프로파일링 정보를 제공합니다. 때로는 성능 버그가 모델이 아니라 모델과 상호 작용하는 애플리케이션 코드의 일부에 있을 수 있습니다. 플랫폼별 프로파일링 도구 및 플랫폼에 대한 모범 사례를 숙지하세요.

## Evaluate whether your model benefits from using hardware accelerators available on the device

TensorFlow Lite has added new ways to accelerate models with faster hardware like GPUs, DSPs, and neural accelerators. Typically, these accelerators are exposed through [delegate](delegates) submodules that take over parts of the interpreter execution. TensorFlow Lite can use delegates by:

- Using Android's [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/). You can utilize these hardware accelerator backends to improve the speed and efficiency of your model. To enable the Neural Networks API, check out the [NNAPI delegate](https://www.tensorflow.org/lite/android/delegates/nnapi) guide.
- GPU delegate is available on Android and iOS, using OpenGL/OpenCL and Metal, respectively. To try them out, see the [GPU delegate tutorial](gpu) and [documentation](gpu_advanced).
- Hexagon delegate is available on Android. It leverages the Qualcomm Hexagon DSP if it is available on the device. See the [Hexagon delegate tutorial](https://www.tensorflow.org/lite/android/delegates/hexagon) for more information.
- It is possible to create your own delegate if you have access to non-standard hardware. See [TensorFlow Lite delegates](delegates) for more information.

Be aware that some accelerators work better for different types of models. Some delegates only support float models or models optimized in a specific way. It is important to [benchmark](measurement) each delegate to see if it is a good choice for your application. For example, if you have a very small model, it may not be worth delegating the model to either the NN API or the GPU. Conversely, accelerators are a great choice for large models that have high arithmetic intensity.

## 추가적인 도움이 필요한 경우

TensorFlow 팀은 직면할 수 있는 특정 성능 문제를 진단하고 해결하도록 기꺼이 도움을 드립니다. [GitHub](https://github.com/tensorflow/tensorflow/issues)에 대한 문제를 세부 정보와 함께 제출해주십시오.

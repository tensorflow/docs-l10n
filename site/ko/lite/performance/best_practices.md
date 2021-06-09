# 성능 모범 사례

모바일 및 임베디드 기기에는 계산 리소스가 제한되어 있으므로 애플리케이션 리소스를 효율적으로 유지하는 것이 중요합니다. TensorFlow Lite 모델 성능을 개선하는 데 사용할 수 있는 모범 사례 및 전략 목록을 작성했습니다.

## 작업에 가장 적합한 모델 선택

Depending on the task, you will need to make a tradeoff between model complexity and size. If your task requires high accuracy, then you may need a large and complex model. For tasks that require less precision, it is better to use a smaller model because they not only use less disk space and memory, but they are also generally faster and more energy efficient. For example, graphs below show accuracy and latency tradeoffs for some common image classification models.

![Graph of model size vs accuracy](../images/performance/model_size_vs_accuracy.png "모델 크기 대 정확도")

![Graph of accuracy vs latency](../images/performance/accuracy_vs_latency.png "정확도 대 지연")

모바일 기기에 최적화된 모델의 한 가지 예는 모바일 비전 애플리케이션에 최적화된 [MobileNet](https://arxiv.org/abs/1704.04861)입니다. [호스팅된 모델](../guide/hosted_models.md)에는 모바일 및 임베디드 기기에 특별히 최적화된 몇 가지 다른 모델이 나열되어 있습니다.

전이 학습을 통해 자체 데이터세트에서 나열된 모델을 재훈련할 수 있습니다. [이미지 분류](/lite/tutorials/model_maker_image_classification) 및 [객체 감지](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)에 대한 전이 학습 튜토리얼을 확인하세요.

## 모델 프로파일링

작업에 적합한 후보 모델을 선택한 후에는 모델을 프로파일링하고 벤치마킹하는 것이 좋습니다. TensorFlow Lite [벤치마킹 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)에는 연산자별 프로파일링 통계를 표시하는 내장 프로파일러가 있습니다. 이는 성능 병목 현상과 계산 시간을 지배하는 연산자를 이해하는 데 도움이 될 수 있습니다.

You can also use [TensorFlow Lite tracing](measurement.md#trace_tensorflow_lite_internals_in_android) to profile the model in your Android application, using standard Android system tracing, and to visualize the operator invocations by time with GUI based profiling tools.

## Profile and optimize operators in the graph

If a particular operator appears frequently in the model and, based on profiling, you find that the operator consumes the most amount of time, you can look into optimizing that operator. This scenario should be rare as TensorFlow Lite has optimized versions for most operators. However, you may be able to write a faster version of a custom op if you know the constraints in which the operator is executed. Check out our [custom operator documentation](../custom_operators.md).

## 모델 최적화

모델 최적화는 일반적으로 더 빠르고 에너지 효율적인 작은 모델을 만들어 모바일 기기에 배포될 수 있도록 하는 것을 목표로 합니다. TensorFlow Lite는 양자화와 같은 여러 가지 최적화 기술을 지원합니다.

자세한 내용은 [모델 최적화 설명서](model_optimization.md)를 확인하세요.

## 스레드 수 조정

TensorFlow Lite는 많은 연산자를 위한 다중 스레드 커널을 지원합니다. 스레드 수를 늘리고 연산자 실행 속도를 높일 수 있습니다. 그러나 스레드 수를 늘리면 모델이 더 많은 리소스와 성능을 사용하게 됩니다.

일부 애플리케이션의 경우 지연 시간이 에너지 효율성보다 더 중요할 수 있습니다. 인터프리터 [스레드](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) 수를 설정하여 스레드 수를 늘릴 수 있습니다. 그러나 다중 스레드 실행은 동시에 실행되는 다른 항목에 따라 성능 변동성이 증가합니다. 특히 모바일 앱의 경우가 해당합니다. 예를 들어 격리된 테스트는 단일 스레드에 비해 2배의 속도 향상을 보여줄 수 있지만 다른 앱이 동시에 실행되는 경우 단일 스레드보다 성능이 저하될 수 있습니다.

## 중복 사본 제거

If your application is not carefully designed, there can be redundant copies when feeding the input to and reading the output from the model. Make sure to eliminate redundant copies. If you are using higher level APIs, like Java, make sure to carefully check the documentation for performance caveats. For example, the Java API is a lot faster if `ByteBuffers` are used as [inputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175).

## Profile your application with platform specific tools

[Android 프로파일러](https://developer.android.com/studio/profile/android-profiler) 및 [Instrument](https://help.apple.com/instruments/mac/current/)와 같은 플랫폼별 도구는 앱을 디버그하는 데 사용할 수 있는 풍부한 프로파일링 정보를 제공합니다. 때로는 성능 버그가 모델이 아니라 모델과 상호 작용하는 애플리케이션 코드의 일부에 있을 수 있습니다. 플랫폼별 프로파일링 도구 및 플랫폼에 대한 모범 사례를 숙지하세요.

## Evaluate whether your model benefits from using hardware accelerators available on the device

TensorFlow Lite는 GPU, DSP, 신경 가속기와 같은 더 빠른 하드웨어로 모델을 가속화하는 새로운 방법을 추가했습니다. 일반적으로 이러한 가속기는 인터프리터 실행의 일부를 차지하는 [대리자](delegates.md) 서브 모듈을 통해 노출됩니다. TensorFlow Lite는 다음과 같은 방법으로 대리자를 사용할 수 있습니다.

- Android의 [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/) 사용하기. 하드웨어 가속기 백엔드를 활용하여 모델의 속도와 효율성을 개선할 수 있습니다. Neural Networks API를 활성화하려면 [NNAPI 대리자](nnapi.md) 가이드를 확인하세요.
- GPU 대리자는 각각 OpenGL/OpenCL 및 Metal을 사용하여 Android 및 iOS에서 사용할 수 있습니다. 사용해 보려면 [GPU 대리자 튜토리얼](gpu.md) 및 [설명서](gpu_advanced.md)를 참조하세요.
- Hexagon 대리자는 Android에서 사용할 수 있습니다. 기기에서 사용할 수 있는 경우 Qualcomm Hexagon DSP를 활용합니다. 자세한 내용은 [Hexagon 대리자 튜토리얼](hexagon_delegate.md)을 참조하세요.
- It is possible to create your own delegate if you have access to non-standard hardware. See [TensorFlow Lite delegates](delegates.md) for more information.

Be aware that some accelerators work better for different types of models. Some delegates only support float models or models optimized in a specific way. It is important to [benchmark](measurement.md) each delegate to see if it is a good choice for your application. For example, if you have a very small model, it may not be worth delegating the model to either the NN API or the GPU. Conversely, accelerators are a great choice for large models that have high arithmetic intensity.

## Need more help

The TensorFlow team is happy to help diagnose and address specific performance issues you may be facing. Please file an issue on [GitHub](https://github.com/tensorflow/tensorflow/issues) with details of the issue.

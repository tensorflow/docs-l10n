# 성능 모범 사례

Mobile and embedded devices have limited computational resources, so it is important to keep your application resource efficient. We have compiled a list of best practices and strategies that you can use to improve your TensorFlow Lite model performance.

## 작업에 가장 적합한 모델 선택

Depending on the task, you will need to make a tradeoff between model complexity and size. If your task requires high accuracy, then you may need a large and complex model. For tasks that require less precision, it is better to use a smaller model because they not only use less disk space and memory, but they are also generally faster and more energy efficient. For example, graphs below show accuracy and latency tradeoffs for some common image classification models.

![Graph of model size vs accuracy](../images/performance/model_size_vs_accuracy.png "모델 크기 대 정확도")

![Graph of accuracy vs latency](../images/performance/accuracy_vs_latency.png "정확도 대 지연")

모바일 기기에 최적화된 모델의 한 가지 예는 모바일 비전 애플리케이션에 최적화된 [MobileNet](https://arxiv.org/abs/1704.04861)입니다. [호스팅된 모델](../guide/hosted_models.md)에는 모바일 및 임베디드 기기에 특별히 최적화된 몇 가지 다른 모델이 나열되어 있습니다.

You can retrain the listed models on your own dataset by using transfer learning. Check out our transfer learning tutorial for [image classification](/lite/tutorials/model_maker_image_classification) and [object detection](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193).

## Profile your model

Once you have selected a candidate model that is right for your task, it is a good practice to profile and benchmark your model. TensorFlow Lite [benchmarking tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) has a built-in profiler that shows per operator profiling statistics. This can help in understanding performance bottlenecks and which operators dominate the computation time.

You can also use [TensorFlow Lite tracing](measurement.md#trace_tensorflow_lite_internals_in_android) to profile the model in your Android application, using standard Android system tracing, and to visualize the operator invocations by time with GUI based profiling tools.

## Profile and optimize operators in the graph

If a particular operator appears frequently in the model and, based on profiling, you find that the operator consumes the most amount of time, you can look into optimizing that operator. This scenario should be rare as TensorFlow Lite has optimized versions for most operators. However, you may be able to write a faster version of a custom op if you know the constraints in which the operator is executed. Check out our [custom operator documentation](../custom_operators.md).

## 모델 최적화

Model optimization aims to create smaller models that are generally faster and more energy efficient, so that they can be deployed on mobile devices. TensorFlow Lite supports multiple optimization techniques, such as quantization.

Check out our [model optimization docs](model_optimization.md) for details.

## 스레드 수 조정

TensorFlow Lite supports multi-threaded kernels for many operators. You can increase the number of threads and speed up execution of operators. Increasing the number of threads will, however, make your model use more resources and power.

For some applications, latency may be more important than energy efficiency. You can increase the number of threads by setting the number of interpreter [threads](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346). Multi-threaded execution, however, comes at the cost of increased performance variability depending on what else is executed concurrently. This is particularly the case for mobile apps. For example, isolated tests may show 2x speed-up vs single-threaded, but, if another app is executing at the same time, it may result in worse performance than single-threaded.

## 중복 사본 제거

If your application is not carefully designed, there can be redundant copies when feeding the input to and reading the output from the model. Make sure to eliminate redundant copies. If you are using higher level APIs, like Java, make sure to carefully check the documentation for performance caveats. For example, the Java API is a lot faster if `ByteBuffers` are used as [inputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175).

## Profile your application with platform specific tools

Platform specific tools like [Android profiler](https://developer.android.com/studio/profile/android-profiler) and [Instruments](https://help.apple.com/instruments/mac/current/) provide a wealth of profiling information that can be used to debug your app. Sometimes the performance bug may be not in the model but in parts of application code that interact with the model. Make sure to familiarize yourself with platform specific profiling tools and best practices for your platform.

## Evaluate whether your model benefits from using hardware accelerators available on the device

TensorFlow Lite has added new ways to accelerate models with faster hardware like GPUs, DSPs, and neural accelerators. Typically, these accelerators are exposed through [delegate](delegates.md) submodules that take over parts of the interpreter execution. TensorFlow Lite can use delegates by:

- Android의 [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/) 사용하기. 하드웨어 가속기 백엔드를 활용하여 모델의 속도와 효율성을 개선할 수 있습니다. Neural Networks API를 활성화하려면 [NNAPI 대리자](nnapi.md) 가이드를 확인하세요.
- GPU 대리자는 각각 OpenGL/OpenCL 및 Metal을 사용하여 Android 및 iOS에서 사용할 수 있습니다. 사용해 보려면 [GPU 대리자 튜토리얼](gpu.md) 및 [설명서](gpu_advanced.md)를 참조하세요.
- Hexagon 대리자는 Android에서 사용할 수 있습니다. 기기에서 사용할 수 있는 경우 Qualcomm Hexagon DSP를 활용합니다. 자세한 내용은 [Hexagon 대리자 튜토리얼](hexagon_delegate.md)을 참조하세요.
- It is possible to create your own delegate if you have access to non-standard hardware. See [TensorFlow Lite delegates](delegates.md) for more information.

Be aware that some accelerators work better for different types of models. Some delegates only support float models or models optimized in a specific way. It is important to [benchmark](measurement.md) each delegate to see if it is a good choice for your application. For example, if you have a very small model, it may not be worth delegating the model to either the NN API or the GPU. Conversely, accelerators are a great choice for large models that have high arithmetic intensity.

## Need more help

The TensorFlow team is happy to help diagnose and address specific performance issues you may be facing. Please file an issue on [GitHub](https://github.com/tensorflow/tensorflow/issues) with details of the issue.

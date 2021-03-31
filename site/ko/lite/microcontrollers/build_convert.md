# 모델 구축 및 변환하기

마이크로컨트롤러는 RAM과 스토리지가 제한되어 있어 머신러닝 모델의 크기에 제약이 따릅니다. 또한 마이크로컨트롤러용 TensorFlow Lite는 현재 제한적인 연산만 지원하므로 모든 모델 아키텍처가 가능한 것은 아닙니다.

이 문서에서는 TensorFlow 모델을 마이크로컨트롤러에서 실행되도록 변환하는 과정을 설명합니다. 또한 지원되는 연산을 간략하게 설명하고 제한된 메모리에 맞게 모델을 설계하고 훈련하는 방법에 대한 지침을 제공합니다.

For an end-to-end, runnable example of building and converting a model, see the following Colab which is part of the *Hello World* example:

<a class="button button-primary" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb">train_hello_world_model.ipynb</a>

## Model conversion

훈련된 TensorFlow 모델을 마이크로컨트롤러에서 실행되도록 변환하려면 [TensorFlow Lite 변환기 Python API](https://www.tensorflow.org/lite/convert/)를 사용해야 합니다. 그러면 모델이 [`FlatBuffer`](https://google.github.io/flatbuffers/)로 변환되어 모델 크기가 줄어들고 TensorFlow Lite 연산을 사용하도록 모델이 수정됩니다.

가능한 한 가장 작은 모델 크기를 얻으려면 [훈련 후 양자화](https://www.tensorflow.org/lite/performance/post_training_quantization) 사용을 고려해야 합니다.

### Convert to a C array

Many microcontroller platforms do not have native filesystem support. The easiest way to use a model from your program is to include it as a C array and compile it into your program.

The following unix command will generate a C source file that contains the TensorFlow Lite model as a `char` array:

```bash
xxd -i converted_model.tflite > model_data.cc
```

The output will look similar to the following:

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

Once you have generated the file, you can include it in your program. It is important to change the array declaration to `const` for better memory efficiency on embedded platforms.

For an example of how to include and use a model in your program, see [`model.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/model.cc) in the *Hello World* example.

## Model architecture and training

When designing a model for use on microcontrollers, it is important to consider the model size, workload, and the operations that are used.

### Model size

A model must be small enough to fit within your target device's memory alongside the rest of your program, both as a binary and at runtime.

To create a smaller model, you can use fewer and smaller layers in your architecture. However, small models are more likely to suffer from underfitting. This means for many problems, it makes sense to try and use the largest model that will fit in memory. However, using larger models will also lead to increased processor workload.

Note: The core runtime for TensorFlow Lite for Microcontrollers fits in 16KB on a Cortex M3.

### Workload

The size and complexity of the model has an impact on workload. Large, complex models might result in a higher duty cycle, which means your device's processor is spending more time working and less time idle. This will increase power consumption and heat output, which might be an issue depending on your application.

### Operation support

TensorFlow Lite for Microcontrollers currently supports a limited subset of TensorFlow operations, which impacts the model architectures that it is possible to run. We are working on expanding operation support, both in terms of reference implementations and optimizations for specific architectures.

지원되는 연산은 [`all_ops_resolver.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/all_ops_resolver.cc) 파일에서 확인할 수 있습니다.

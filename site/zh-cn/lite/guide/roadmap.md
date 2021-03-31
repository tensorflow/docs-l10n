# TensorFlow Lite Roadmap

**Updated: April 18, 2020**

下面提供了我们 2020 年计划的简要概览。请注意，该路线图随时可能发生变化，以下顺序也不代表任何类型的优先级。原则上，我们通常根据受影响的用户数量确定问题优先级。

We break our roadmap into four key segments: usability, performance, optimization and portability. We strongly encourage you to comment on our roadmap and provide us feedback in the [TF Lite discussion group](https://groups.google.com/a/tensorflow.org/g/tflite).

## Usability

- **Expanded ops coverage**
    - Prioritized op additions based on user feedback
- **Improvements to using TensorFlow ops in TensorFlow Lite**
    - Pre-built libraries available via Bintray (Android) and Cocoapods (iOS)
    - Smaller binary size when using select TF ops via op stripping
- **LSTM / RNN support**
    - Full LSTM and RNN conversion support, including support in Keras
- **Pre-and-post processing support libraries and codegen tool**
    - Ready-to-use API building blocks for common ML tasks
    - Support more models (e.g. NLP) and more platforms (e.g. iOS)
- **Android Studio Integration**
    - Drag &amp; drop TFLite models into Android Studio to generate model binding classes
- **Control Flow &amp; Training on-device**
    - Support for training on-device, focused on personalization and transfer learning
- **Visualization tooling with TensorBoard**
    - Provide enhanced tooling with TensorBoard
- **Model Maker**
    - Support more tasks, including object detection and BERT-based NLP tasks
- **More models and examples**
    - More examples to demonstrate model usage as well as new features and APIs, covering different platforms.
- **Task Library**
    - Improve the usability of the C++ Task Library, such as providing prebuilt binaries and creating user-friendly workflows for users who want to build from source code.
    - Release reference examples of using the Task Library.
    - Enable more task types.
    - Improve cross-platform support and enable more tasks for iOS.

## Performance

- **Better tooling**
    - Public dashboard for tracking performance gains with each release
- **Improved CPU performance**
    - New highly optimized floating-point kernel library for convolutional models
    - First-class x86 support
- **Updated NN API support**
    - Full support for new Android R NN API features, ops and types
- **GPU backend optimizations**
    - Vulkan support on Android
    - Support integer quantized models
- **Hexagon DSP backend**
    - Per-channel quantization support for all models created through post-training quantization
    - Dynamic input batch size support
    - Better op coverage, including LSTM
- **Core ML backend**
    - Optimizing start-up time
    - Dynamic quantized models support
    - Float16 quantized models support
    - Better op coverage

## Optimization

- **Quantization**

    - Post-training quantization for (8b) fixed-point RNNs
    - During-training quantization for (8b) fixed-point RNNs
    - Quality and performance improvements for post-training dynamic-range quantization

- **Pruning / sparsity**

    - Sparse model execution support in TensorFlow Lite - [WIP](https://github.com/tensorflow/model-optimization/issues/173)
    - Weight clustering API

## Portability

- **Microcontroller Support**
    - Add support for a range of 32-bit MCU architecture use cases for speech and image classification
    - Sample code and models for vision and audio data
    - Full TF Lite op support on microcontrollers
    - Support for more platforms, including CircuitPython support

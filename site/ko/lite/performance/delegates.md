# TensorFlow Lite 대리자

## Introduction

**대리자**는 GPU 및 [DSP(디지털 신호 프로세서)](https://en.wikipedia.org/wiki/Digital_signal_processor)와 같은 온 디바이스 가속기를 활용하여 TensorFlow Lite 모델의 하드웨어 가속을 지원합니다.

기본적으로 TensorFlow Lite는 [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) 명령어 세트에 최적화된 CPU 커널을 사용합니다. 그러나 CPU는 머신러닝 모델에서 일반적으로 발견되는 무거운 산술(예: 컨볼루션 및 밀집 레이어와 관련된 행렬 수학)에 반드시 최적화되었다고 할 수 없는 다목적 프로세서입니다.

On the other hand, most modern mobile phones contain chips that are better at handling these heavy operations. Utilizing them for neural network operations provides huge benefits in terms of latency and power efficiency. For example, GPUs can provide upto a [5x speedup](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html) in latency, while the [Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor) has shown to reduce power consumption upto 75% in our experiments.

Each of these accelerators have associated APIs that enable custom computations, such as [OpenCL](https://www.khronos.org/opencl/) or [OpenGL ES](https://www.khronos.org/opengles/) for mobile GPU and the [Qualcomm® Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk) for DSP. Typically, you would have to write a lot of custom code to run a neural network through these interfaces. Things get even more complicated when you consider that each accelerator has its pros &amp; cons and cannot execute every operation in a neural network. TensorFlow Lite's Delegate API solves this problem by acting as a bridge between the TFLite runtime and these lower-level APIs.

![Original graph](../images/performance/tflite_delegate_graph_1.png "원본 그래프")

## 대리자 선택

TensorFlow Lite는 여러 대리자를 지원하며, 각 대리자는 특정 플랫폼 및 특정 유형의 모델에 최적화되어 있습니다. 일반적으로 타겟으로 삼은 *플랫폼*(Android 또는 iOS?)과 가속화하려는 *모델 유형*(부동 소수점 또는 양자화?)의 두 가지 주요 기준에 따라 사용 사례에 적용할 수 있는 여러 대리자가 있습니다.

### Delegates by Platform

#### 교차 플랫폼(Android 및 iOS)

- **GPU 대리자** - GPU 대리자는 Android와 iOS 모두에서 사용할 수 있으며, GPU를 사용할 수 있는 32bit 및 16bit 부동 기반 모델을 실행하도록 최적화되어 있습니다. 또한, 8bit 양자화 모델을 지원하고 부동 버전과 동등한 GPU 성능을 제공합니다. GPU 대리자에 대한 자세한 내용은 [GPU 기반 TensorFlow Lite](gpu_advanced.md)를 참조하세요. Android 및 iOS에서 GPU 대리자를 사용하는 방법에 대한 단계별 튜토리얼은 [TensorFlow Lite GPU 대리자 튜토리얼](gpu.md)을 참조하세요.

#### Android

- **최신 Android 기기용 NNAPI 대리자** - NNAPI 대리자를 사용하여 GPU, DSP 및/또는 NPU를 사용할 수 있는 Android 기기에서 모델을 가속화할 수 있습니다. Android 8.1(API 27+) 이상에서 사용할 수 있습니다. NNAPI 대리자 개요, 단계별 지침 및 모범 사례는 [TensorFlow Lite NNAPI 대리자](nnapi.md)를 참조하세요.
- **Hexagon delegate for older Android devices** - The Hexagon delegate can be used to accelerate models on Android devices with Qualcomm Hexagon DSP. It can be used on devices running older versions of Android that do not support NNAPI. See [TensorFlow Lite Hexagon delegate](hexagon_delegate.md) for more detail.

#### iOS

- **최신 iPhone 및 iPad용 Core ML 대리자** - Neural Engine을 사용할 수 있는 최신 iPhone 및 iPad의 경우 Core ML 대리자를 사용하여 32bit 또는 16bit 부동점 모델에 대한 추론을 가속화할 수 있습니다. Neural Engine은 A12 SoC 이상의 Apple 모바일 기기를 사용할 수 있습니다. Core ML 대리자에 대한 개요 및 단계별 지침은 [TensorFlow Lite Core ML 대리자](coreml_delegate.md)를 참조하세요.

### Delegates by model type

각 가속기는 특정 비트 폭의 데이터를 염두에 두고 설계되었습니다. 8bit 양자화된 연산(예: [Hexagon delegate](hexagon_delegate.md))만 지원하는 대리자에 부동 소수점 모델을 제공하는 경우 모든 연산이 거부되고 모델은 전적으로 CPU에서 실행됩니다. 이러한 뜻밖의 상황을 방지하기 위해 아래의 표를 보면 모델 유형에 따른 대리자 지원의 개요가 나와있습니다.

**모델 유형** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
부동점 (32bit) | 예 | 예 | 아니요 | 예
[훈련 후 float16 양자화](post_training_float16_quant.ipynb) | 예 | 아니요 | 아니요 | 예
[훈련 후 동적 범위 양자화](post_training_quant.ipynb) | 예 | 예 | 아니요 | 아니요
[훈련 후 정수 양자화](post_training_integer_quant.ipynb) | 예 | 예 | 예 | 아니요
[양자화 인식 훈련](http://www.tensorflow.org/model_optimization/guide/quantization/training) | 예 | 예 | 예 | 아니요

### Validating performance

이 섹션의 정보는 애플리케이션을 개선할 수 있는 대리자를 선정하기 위한 대략적인 가이드라인 역할을 합니다. 그러나 각 대리자가 지원하는 사전 정의된 연산 세트가 있으며 모델 및 기기에 따라 다르게 수행될 수 있다는 점에 유의하는 것이 중요합니다. 예를 들어 [NNAPI 대리자](nnapi.md)는 Pixel 휴대폰에서 Google의 Edge-TPU를 사용하도록 선택할 수 있지만 다른 기기에서는 DSP를 사용할 수 있습니다. 따라서 일반적으로 몇 가지 벤치마킹을 수행하여 대리자가 자신의 필요성에 얼마나 유용한지 평가하는 것이 좋습니다. 이는 또한 대리자를 TensorFlow Lite 런타임에 연결하는 것과 관련된 바이너리 크기 증가를 정당화하는 데 도움이 됩니다.

TensorFlow Lite has extensive performance and accuracy-evaluation tooling that can empower developers to be confident in using delegates in their application. These tools are discussed in the next section.

## 평가 도구

### Latency &amp; memory footprint

TensorFlow Lite’s [benchmark tool](https://www.tensorflow.org/lite/performance/measurement) can be used with suitable parameters to estimate model performance, including average inference latency, initialization overhead, memory footprint, etc. This tool supports multiple flags to figure out the best delegate configuration for your model. For instance, `--gpu_backend=gl` can be specified with `--use_gpu` to measure GPU execution with OpenGL. The complete list of supported delegate parameters is defined in the [detailed documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar).

Here’s an example run for a quantized model with GPU via `adb`:

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

이 도구의 Android, 64bit ARM 아키텍처용 사전 빌드 버전을 [여기](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)([보다 상세한 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android))에서 다운로드할 수 있습니다.

### Accuracy &amp; correctness

대리자는 일반적으로 CPU와 다른 정밀도로 계산을 수행합니다. 결과적으로 하드웨어 가속을 위해 대리자를 사용하는 것과 관련된(보통 사소한) 정확도 절충이 있습니다. 이것이 *항상* 그런 것은 아닙니다. 예를 들어, GPU는 부동 소수점 정밀도를 사용하여 양자화된 모델을 실행하기 때문에 약간의 정밀도 향상(예: ILSVRC 이미지 분류에서 &lt;1% Top-5 향상)이 있을 수 있습니다.

TensorFlow Lite has two types of tooling to measure how accurately a delegate behaves for a given model: *Task-Based* and *Task-Agnostic*. All the tools described in this section support the [advanced delegation parameters](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar) used by the benchmarking tool from the previous section. Note that the sub-sections below focus on *delegate evaluation* (Does the delegate perform the same as the CPU?) rather than model evaluation (Is the model itself good for the task?).

#### Task-Based Evaluation

TensorFlow Lite has tools to evaluate correctness on two image-based tasks:

- [top-K 정확도](http://image-net.org/challenges/LSVRC/2012/)의 [ILSVRC 2012](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K) (이미지 분류)

- [mean 평균 정밀도 (mAP)](https://cocodataset.org/#detection-2020)를 사용하는 [COCO 객체 감지 (경계 상자 포함)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

이들 도구(Android, 64bit ARM 아키텍처)의 미리 빌드된 바이너리와 설명서는 여기에서 찾을 수 있습니다.

- [ImageNet 이미지 분류](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification) ([상세 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification))
- [COCO 객체 감지](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection) ([상세 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection))

아래 예는 Pixel 4에서 Google의 Edge-TPU를 활용하는 NNAPI를 통한 [이미지 분류 평가](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)를 보여줍니다.

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

예상되는 출력은 1에서 10까지의 Top-K 메트릭 목록입니다.

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### Task-Agnostic Evaluation

For tasks where there isn't an established on-device evaluation tool, or if you are experimenting with custom models, TensorFlow Lite has the [Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) tool. (Android, 64-bit ARM binary architecture binary [here](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff))

Inference Diff compares TensorFlow Lite execution (in terms of latency &amp; output-value deviation) in two settings:

- 단일 쓰레드 CPU 추론
- 사용자 정의 추론 - [이들 매개변수](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)에 의해 정의

To do so, the tool generates random Gaussian data and passes it through two TFLite Interpreters - one running single-threaded CPU kernels, and the other parameterized by the user's arguments.

It measures the latency of both, as well as the absolute difference between the output tensors from each Interpreter, on a per-element basis.

For a model with a single output tensor, the output might look like this:

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

What this means is that for the output tensor at index `0`, the elements from the CPU output different from the delegate output by an average of `1.96e-05`.

Note that interpreting these numbers requires deeper knowledge of the model, and what each output tensor signifies. If its a simple regression that determines some sort of score or embedding, the difference should be low (otherwise it's an error with the delegate). However, outputs like the 'detection class' one from SSD models is a little harder to interpret. For example, it might show a difference using this tool, but that may not mean something really wrong with the delegate: consider two (fake) classes: "TV (ID: 10)", "Monitor (ID:20)" - If a delegate is slightly off the golden truth and shows monitor instead of TV, the output diff for this tensor might be something as high as 20-10 = 10.

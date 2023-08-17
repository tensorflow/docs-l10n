# TensorFlow Lite 대리자

## 시작하기

**대리자**는 GPU 및 [DSP(디지털 신호 프로세서)](https://en.wikipedia.org/wiki/Digital_signal_processor)와 같은 온 디바이스 가속기를 활용하여 TensorFlow Lite 모델의 하드웨어 가속을 지원합니다.

기본적으로 TensorFlow Lite는 [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) 명령어 세트에 최적화된 CPU 커널을 사용합니다. 그러나 CPU는 머신러닝 모델에서 일반적으로 발견되는 무거운 산술(예: 컨볼루션 및 밀집 레이어와 관련된 행렬 수학)에 반드시 최적화되었다고 할 수 없는 다목적 프로세서입니다.

반면에 대부분의 최신 휴대폰에는 이러한 무거운 연산을 더 잘 처리하는 칩이 포함되어 있습니다. 신경망 연산을 위해 활용하면 대기 시간 및 전력 효율성 측면에서 큰 이점이 있습니다. 예를 들어 GPU는 대기 시간을 [최대 5배](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)[까지 높일 수 있는 반면 Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor)는 실험에서 전력 소비를 최대 75%까지 줄이는 것으로 나타났습니다.

이러한 각 가속기에는 모바일 GPU용 [OpenCL](https://www.khronos.org/opencl/) 또는 [OpenGL ES](https://www.khronos.org/opengles/) 및 DSP용 [Qualcomm® Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)와 같은 사용자 정의 계산을 가능하게 하는 관련 API가 있습니다. 일반적으로 이러한 인터페이스를 통해 신경망을 실행하려면 많은 사용자 정의 코드를 작성해야 합니다. 각 가속기에 장단점이 있고 신경망에서 모든 연산을 실행할 수 없다는 점을 고려하면 상황이 더욱 복잡해집니다. TensorFlow Lite의 Delegate API는 TFLite 런타임과 이러한 하위 수준 API를 연결하는 다리 역할을 하여 이 문제를 해결합니다.

![Original graph](../images/performance/tflite_delegate_graph_1.png "원본 그래프")

## 대리자 선택

TensorFlow Lite는 여러 대리자를 지원하며, 각 대리자는 특정 플랫폼 및 특정 유형의 모델에 최적화되어 있습니다. 일반적으로 타겟으로 삼은 *플랫폼*(Android 또는 iOS?)과 가속화하려는 *모델 유형*(부동 소수점 또는 양자화?)의 두 가지 주요 기준에 따라 사용 사례에 적용할 수 있는 여러 대리자가 있습니다.

### 플랫폼별 대리자

#### 교차 플랫폼(Android 및 iOS)

- **GPU 대리자** - GPU 대리자는 Android와 iOS 모두에서 사용할 수 있으며, GPU를 사용할 수 있는 32bit 및 16bit 부동 기반 모델을 실행하도록 최적화되어 있습니다. 또한, 8bit 양자화 모델을 지원하고 부동 버전과 동등한 GPU 성능을 제공합니다. GPU 대리자에 대한 자세한 내용은 [GPU 기반 TensorFlow Lite](gpu_advanced.md)를 참조하세요. Android 및 iOS에서 GPU 대리자를 사용하는 방법에 대한 단계별 튜토리얼은 [TensorFlow Lite GPU 대리자 튜토리얼](gpu.md)을 참조하세요.

#### Android

- **최신 Android 기기용 NNAPI 대리자** - NNAPI 대리자를 사용하여 GPU, DSP 및/또는 NPU를 사용할 수 있는 Android 기기에서 모델을 가속화할 수 있습니다. Android 8.1(API 27+) 이상에서 사용할 수 있습니다. NNAPI 대리자 개요, 단계별 지침 및 모범 사례는 [TensorFlow Lite NNAPI 대리자](nnapi.md)를 참조하세요.
- **구형 Android 기기용 Hexagon 대리자** - Qualcomm Hexagon DSP를 사용하는 Android 기기에서 Hexagon 대리자를 사용하여 모델을 가속화할 수 있습니다. NNAPI를 지원하지 않는 이전 버전의 Android 기기에서 사용할 수 있습니다. 자세한 내용은 [TensorFlow Lite Hexagon 대리자](hexagon_delegate.md)를 참조하세요.

#### iOS

- **최신 iPhone 및 iPad용 Core ML 대리자** - Neural Engine을 사용할 수 있는 최신 iPhone 및 iPad의 경우 Core ML 대리자를 사용하여 32bit 또는 16bit 부동점 모델에 대한 추론을 가속화할 수 있습니다. Neural Engine은 A12 SoC 이상의 Apple 모바일 기기를 사용할 수 있습니다. Core ML 대리자에 대한 개요 및 단계별 지침은 [TensorFlow Lite Core ML 대리자](coreml_delegate.md)를 참조하세요.

### 모델 유형별 대리자

각 가속기는 특정 비트 폭의 데이터를 염두에 두고 설계되었습니다. 8bit 양자화된 연산(예: [Hexagon delegate](hexagon_delegate.md))만 지원하는 대리자에 부동 소수점 모델을 제공하는 경우 모든 연산이 거부되고 모델은 전적으로 CPU에서 실행됩니다. 이러한 뜻밖의 상황을 방지하기 위해 아래의 표를 보면 모델 유형에 따른 대리자 지원의 개요가 나와있습니다.

**모델 유형** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
부동점 (32bit) | 예 | 예 | 아니요 | 예
[훈련 후 float16 양자화](post_training_float16_quant.ipynb) | 예 | 아니요 | 아니요 | 예
[훈련 후 동적 범위 양자화](post_training_quant.ipynb) | 예 | 예 | 아니요 | 아니요
[훈련 후 정수 양자화](post_training_integer_quant.ipynb) | 예 | 예 | 예 | 아니요
[양자화 인식 훈련](http://www.tensorflow.org/model_optimization/guide/quantization/training) | 예 | 예 | 예 | 아니요

### 성능 검증

이 섹션의 정보는 애플리케이션을 개선할 수 있는 대리자를 선정하기 위한 대략적인 가이드라인 역할을 합니다. 그러나 각 대리자가 지원하는 사전 정의된 연산 세트가 있으며 모델 및 기기에 따라 다르게 수행될 수 있다는 점에 유의하는 것이 중요합니다. 예를 들어 [NNAPI 대리자](nnapi.md)는 Pixel 휴대폰에서 Google의 Edge-TPU를 사용하도록 선택할 수 있지만 다른 기기에서는 DSP를 사용할 수 있습니다. 따라서 일반적으로 몇 가지 벤치마킹을 수행하여 대리자가 자신의 필요성에 얼마나 유용한지 평가하는 것이 좋습니다. 이는 또한 대리자를 TensorFlow Lite 런타임에 연결하는 것과 관련된 바이너리 크기 증가를 정당화하는 데 도움이 됩니다.

TensorFlow Lite는 개발자가 자신의 애플리케이션에서 대리자를 사용하는 데 확신을 줄 수 있는 광범위한 성능 및 정확도 평가 도구를 갖추고 있습니다. 이들 도구는 다음 섹션에서 다룹니다.

## 평가 도구

### 지연 시간 및 메모리 공간

TensorFlow Lite의 [벤치마크 도구](https://www.tensorflow.org/lite/performance/measurement)는 적절한 매개변수와 함께 사용하여 평균 추론 지연 시간, 초기화 오버헤드, 메모리 공간 등을 포함한 모델 성능을 추정할 수 있습니다. 이 도구는 모델에 가장 적합한 대리자 구성을 파악하기 위해 여러 플래그를 지원합니다. 예를 들어 `--gpu_backend=gl`를 `--use_gpu`와 함께 지정하여 OpenGL로 GPU 실행을 측정할 수 있습니다. 지원되는 대리자 매개변수의 전체 목록은 [상세한 설명서](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)에 정의되어 있습니다.

다음은 `adb`를 통해 GPU로 양자화된 모델을 실행한 예입니다.

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

이 도구의 Android, 64bit ARM 아키텍처용 사전 빌드 버전을 [여기](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)([보다 상세한 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android))에서 다운로드할 수 있습니다.

### 정확도 및 수정

대리자는 일반적으로 CPU와 다른 정밀도로 계산을 수행합니다. 결과적으로 하드웨어 가속을 위해 대리자를 사용하는 것과 관련된(보통 사소한) 정확도 절충이 있습니다. 이것이 *항상* 그런 것은 아닙니다. 예를 들어, GPU는 부동 소수점 정밀도를 사용하여 양자화된 모델을 실행하기 때문에 약간의 정밀도 향상(예: ILSVRC 이미지 분류에서 &lt;1% Top-5 향상)이 있을 수 있습니다.

TensorFlow Lite에는 지정된 모델에 대해 대리자가 얼마나 정확하게 동작하는지 측정하는 두 가지 유형의 도구, 즉 *Task-Based*와 *Task-Agnostic*이 있습니다. 이 섹션에 설명된 모든 도구는 이전 섹션의 벤치마킹 도구에서 사용한 [고급 델리게이션 매개변수](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)를 지원합니다. 아래의 하위 섹션은 모델 평가(모델 자체가 작업에 적합합니까?)보다 *대리자 평가* (대리자가 CPU와 동일한 작업을 수행합니까?)에 중점을 둡니다.

#### 작업 기반 평가

TensorFlow Lite에는 두 개의 이미지 기반 작업의 정확성을 평가하는 도구가 있습니다.

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

#### 작업 불가지론적 평가

설정된 온 디바이스 평가 도구가 없는 작업 또는 사용자 정의 모델을 실험하는 경우 TensorFlow Lite에는 [Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) 도구가 있습니다. (Android, 64bit ARM 바이너리 아키텍처 바이너리에 대해서는 [여기](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff)를 클릭)

Inference Diff는 다음 두 가지 설정에서 TensorFlow Lite 실행 (대기 시간 및 출력 값 편차)을 비교합니다.

- 단일 쓰레드 CPU 추론
- 사용자 정의 추론 - [이들 매개변수](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)에 의해 정의

이를 위해 이 도구는 임의의 가우스 데이터를 생성하고 두 개의 TFLite 인터프리터를 통해 전달합니다. 하나는 단일 스레드 CPU 커널을 실행하고 다른 하나는 사용자 인수에 의해 매개변수화됩니다.

요소별로 각 인터프리터의 출력 텐서 간의 절대 차이뿐만 아니라 두 지연 시간을 측정합니다.

단일 출력 텐서가 있는 모델의 경우 출력은 다음과 같을 수 있습니다.

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

이는 인덱스 `0`의 출력 텐서의 경우 CPU 출력의 요소가 평균 `1.96e-05`만큼 대리자 출력과 다르다는 것을 의미합니다.

이러한 숫자를 해석하려면 모델과 각 출력 텐서가 의미하는 바에 대한 보다 깊은 지식이 필요합니다. 어떤 종류의 점수나 임베딩을 결정하는 단순 회귀인 경우 차이가 낮아야 합니다 (그렇지 않으면 대리자 오류입니다). 그러나 SSD 모델의 '감지 클래스'와 같은 출력은 해석하기가 조금 더 어렵습니다. 예를 들어, 이 도구를 사용하면 차이가 표시될 수 있지만 대리자에게 실제로 문제가 있는 것은 아닙니다. "TV (ID : 10)", "모니터 (ID : 20)"- 대리자는 황금 진실에서 약간 벗어나 TV 대신 모니터를 표시합니다. 이 텐서의 출력 차이는 20-10 = 10만큼 높을 수 있습니다.

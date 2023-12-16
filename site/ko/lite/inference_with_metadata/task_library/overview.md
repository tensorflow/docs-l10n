# TensorFlow Lite Task 라이브러리

TensorFlow Lite Task 라이브러리에는 앱 개발자가 TFLite로 ML 경험을 만들 수 있는 강력하고 사용하기 쉬운 작업별 라이브러리 세트가 포함되어 있습니다. 이미지 분류, 질문 및 답변 등과 같은 주요 머신 러닝 작업에 최적화된 기본 제공 모델 인터페이스가 제공됩니다. 모델 인터페이스는 각 작업에 맞게 특별히 설계되어 최상의 성능과 유용성을 제공합니다. Task 라이브러리는 크로스 플랫폼에서 작동하며 Java, C++ 및 Swift에서 지원됩니다.

## Task 라이브러리에서 기대할 수 있는 사항

- **ML 전문가가 아니더라도 사용할 수 있는 명료하고 잘 정의된 API** <br> 단 5줄의 코드로도 추론을 수행할 수 있습니다. Task 라이브러리의 강력하고 사용하기 쉬운 API를 빌딩 블록으로 사용하여 모바일 기기에서 TFLite로 ML을 쉽게 개발할 수 있습니다.

- **복잡하지만 일반적인 데이터 처리** <br> 공통 비전 및 자연어 처리 논리를 지원하여 데이터와 모델에 필요한 데이터 형식 사이에서 변환할 수 있습니다. 학습 및 추론에 사용할 수 있는 동일하고 공유 가능한 처리 로직을 제공합니다.

- **고성능 게인** <br> 데이터 처리에 수 밀리 초밖에 걸리지 않으므로 TensorFlow Lite를 사용한 빠른 추론 경험이 보장됩니다.

- **확장성 및 사용자 정의 기능**<br> 작업 라이브러리 인프라가 제공하는 모든 이점을 활용하고 자신만의 Android/iOS 추론 API를 쉽게 구축할 수 있습니다.

## 지원되는 작업

다음은 지원되는 작업 유형의 목록입니다. 점차 더 많은 사용 사례가 계속 개발됨에 따라 이 목록은 더 늘어날 것으로 예상됩니다.

- **비전 API**

    - [ImageClassifier](image_classifier.md)
    - [ObjectDetector](object_detector.md)
    - [ImageSegmenter](image_segmenter.md)
    - [ImageSearcher](image_searcher.md)
    - [ImageEmbedder](image_embedder.md)

- **자연어(NL) API**

    - [NLClassifier](nl_classifier.md)
    - [BertNLClassifier](bert_nl_classifier.md)
    - [BertQuestionAnswerer](bert_question_answerer.md)
    - [TextSearcher](text_searcher.md)
    - [TextEmbedder](text_embedder.md)

- **오디오 API**

    - [AudioClassifier](audio_classifier.md)

- **사용자 정의 API**

    - Task API 인프라를 확장하고 [사용자 정의 API](customized_task_api.md)를 구축합니다.

## 대리자로 작업 라이브러리 실행하기

[대리자](https://www.tensorflow.org/lite/performance/delegates)는 [GPU](https://www.tensorflow.org/lite/performance/gpu) 및 [Coral Edge TPU](https://coral.ai/)와 같은 온디바이스 가속기를 활용하여 TensorFlow Lite 모델의 하드웨어 가속을 사용 설정합니다. 이를 신경망 연산에 활용하면 대기 시간과 전력 효율성 측면에서 엄청난 이점을 얻을 수 있습니다. 예를 들어 GPU는 모바일 장치에서 최대 [5배의 속도가 향상](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)된 지연 시간을 제공할 수 있으며 Coral Edge TPU는 데스크톱 CPU보다 [10배 빠른](https://coral.ai/docs/edgetpu/benchmarks/) 추론 기능을 제공합니다.

작업 라이브러리는 대리자를 설정하고 사용하기 위한 간편한 구성 및 대체 옵션을 제공합니다. 이제 Task API에서 다음과 같은 가속기가 지원됩니다.

- Android
    - [GPU](https://www.tensorflow.org/lite/performance/gpu): Java / C++
    - [NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi): Java / C++
    - [Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon): C++
- Linux / Mac
    - [Coral Edge TPU](https://coral.ai/): C++
- iOS
    - [Core ML delegate](https://www.tensorflow.org/lite/performance/coreml_delegate): C++

Task Swift/Web API의 가속화가 곧 지원될 예정입니다.

### Java에서 Android의 GPU 사용의 예

1단계. GPU 대리자 플러그인 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

```java
dependencies {
    // Import Task Library dependency for vision, text, or audio.

    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

참고: NNAPI는 기본적으로 시각, 텍스트 및 오디오에 대한 작업 라이브러리 대상을 함께 제공합니다.

2단계. [BaseOptions](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)를 통해 작업 옵션에서 GPU 대리자를 구성합니다. 예를 들어 다음과 같이 `ObjectDetecor`에서 GPU를 설정할 수 있습니다.

```java
// Turn on GPU delegation.
BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
// Configure other options in ObjectDetector
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(baseOptions)
        .setMaxResults(1)
        .build();

// Create ObjectDetector from options.
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

### C++에서 Android의 GPU 사용의 예

1단계. 다음과 같이 bazel 빌드 대상의 GPU 대리자 플러그인에 종속합니다.

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:gpu_plugin", # for GPU
]
```

참고: `gpu_plugin` 대상은 [GPU 대리자 대상](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu)과 별개입니다. `gpu_plugin`은 GPU 대리자 대상을 래핑하고 안전한 보호를 제공할 수 있습니다. 즉, 대리자 오류 시 TFLite CPU 경로로 대체합니다.

기타 대리자 옵션은 다음과 같습니다.

```
"//tensorflow_lite_support/acceleration/configuration:nnapi_plugin", # for NNAPI
"//tensorflow_lite_support/acceleration/configuration:hexagon_plugin", # for Hexagon
```

2단계. 작업 옵션에서 GPU 대리자를 구성합니다. 예를 들어 다음과 같이 `BertQuestionAnswerer`에서 GPU를 설정할 수 있습니다.

```c++
// Initialization
BertQuestionAnswererOptions options;
// Load the TFLite model.
auto base_options = options.mutable_base_options();
base_options->mutable_model_file()->set_file_name(model_file);
// Turn on GPU delegation.
auto tflite_settings = base_options->mutable_compute_settings()->mutable_tflite_settings();
tflite_settings->set_delegate(Delegate::GPU);
// (optional) Turn on automatical fallback to TFLite CPU path on delegation errors.
tflite_settings->mutable_fallback_settings()->set_allow_automatic_fallback_on_execution_error(true);

// Create QuestionAnswerer from options.
std::unique_ptr<QuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference on GPU.
std::vector<QaAnswer> results = answerer->Answer(context_of_question, question_to_ask);
```

[여기](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/acceleration/configuration/configuration.proto)에서 고급 가속기 설정을 살펴보세요.

### Python에서 Coral Edge TPU 사용의 예

작업 옵션에서 Coral Edge TPU를 구성합니다. 예를 들어 다음과 같이 `ImageClassifier`에서 Coral Edge TPU를 설정할 수 있습니다.

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core

# Initialize options and turn on Coral Edge TPU delegation.
base_options = core.BaseOptions(file_name=model_path, use_coral=True)
options = vision.ImageClassifierOptions(base_options=base_options)

# Create ImageClassifier from options.
classifier = vision.ImageClassifier.create_from_options(options)

# Run inference on Coral Edge TPU.
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

### C++에서 Coral Edge TPU 사용의 예

1단계. 다음과 같이 bazel 빌드 대상의 Coral Edge TPU 대리자 플러그인에 종속합니다.

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin", # for Coral Edge TPU
]
```

2단계. 작업 옵션에서 Coral Edge TPU를 구성합니다. 예를 들어 다음과 같이 `ImageClassifier`에서 Coral Edge TPU를 설정할 수 있습니다.

```c++
// Initialization
ImageClassifierOptions options;
// Load the TFLite model.
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
// Turn on Coral Edge TPU delegation.
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->set_delegate(Delegate::EDGETPU_CORAL);
// Create ImageClassifier from options.
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference on Coral Edge TPU.
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

3단계. 아래와 같이 `libusb-1.0-0-dev` 패키지를 설치합니다. 이미 설치된 경우 다음 단계로 건너뜁니다.

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

4단계. bazel 명령에서 다음 구성으로 컴파일합니다.

```bash
# On the Linux
--define darwinn_portable=1 --linkopt=-lusb-1.0

# On the macOS, add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0

# Windows is not supported yet.
```

Coral Edge TPU 기기에서 [작업 라이브러리 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop)를 사용해 보세요. [사전 훈련된 Edge TPU 모델](https://coral.ai/models/) 및 [고급 Edge TPU 설정](https://github.com/tensorflow/tensorflow/blob/4d999fda8d68adfdfacd4d0098124f1b2ea57927/tensorflow/lite/acceleration/configuration/configuration.proto#L594)에 대해 자세히 알아보세요.

### C++에서 Core ML 대리자 사용 예제

전체 예제는 [Image Classifier Core ML 대리자 테스트](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/test/task/vision/image_classifier/TFLImageClassifierCoreMLDelegateTest.mm)에서 확인할 수 있습니다.

1단계. 다음과 같이 bazel 빌드 대상의 Core ML 대리자 플러그인에 종속합니다.

```
deps = [
  "//tensorflow_lite_support/acceleration/configuration:coreml_plugin", # for Core ML Delegate
]
```

2단계. 작업 옵션에서 Core ML 대리자를 구성합니다. 예를 들어 다음과 같이 `ImageClassifier`에서 Core ML 대리자를 설정할 수 있습니다.

```c++
// Initialization
ImageClassifierOptions options;
// Load the TFLite model.
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
// Turn on Core ML delegation.
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->set_delegate(::tflite::proto::Delegate::CORE_ML);
// Set DEVICES_ALL to enable Core ML delegation on any device (in contrast to
// DEVICES_WITH_NEURAL_ENGINE which creates Core ML delegate only on devices
// with Apple Neural Engine).
options.mutable_base_options()->mutable_compute_settings()->mutable_tflite_settings()->mutable_coreml_settings()->set_enabled_devices(::tflite::proto::CoreMLSettings::DEVICES_ALL);
// Create ImageClassifier from options.
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Run inference on Core ML.
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

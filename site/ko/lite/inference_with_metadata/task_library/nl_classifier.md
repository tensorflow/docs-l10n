# 자연어 분류자 통합하기

Task Library의 `NLClassifier` API는 입력 텍스트를 여러 범주로 분류하는 역할을 하고, 대부분의 텍스트 분류 모델을 처리할 수 있는 활용도 높고 구성 가능한 API입니다.

## NLClassifier API의 주요 특징

- 단일 문자열을 입력으로 받아서 문자열로 분류를 수행하고 분류 결과로 &lt;Label, Score&gt; 쌍을 출력합니다.

- 입력 텍스트에 Regex Tokenization을 선택적으로 사용할 수 있습니다.

- 다양한 분류 모델을 적용하도록 구성할 수 있습니다.

## 지원되는 NLClassifier 모델

다음 모델은 `NLClassifier` API와의 호환성이 보장됩니다.

- The <a href="../../examples/text_classification/overview">movie review sentiment classification</a> model.

- Models with `average_word_vec` spec created by [TensorFlow Lite Model Maker for text Classification](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

Android 앱에서 `NLClassifier`를 사용하는 방법의 예는 [텍스트 분류 참조 앱](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java)을 참조하세요.

### 1단계: Gradle 종속성 및 기타 설정 가져오기

`.tflite` 모델 파일을 모델이 실행될 Android 모듈의 assets 디렉토리에 복사합니다. 파일을 압축하지 않도록 지정하고 TensorFlow Lite 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.3.0'
}
```

참고: Android Gradle 플러그인 버전 4.1부터는 .tflite가 기본적으로 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.

### 2단계: API를 사용하여 추론 실행하기

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options =
    NLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setInputTensorName(INPUT_TENSOR_NAME)
        .setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME)
        .build();
NLClassifier classifier =
    NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

`NLClassifier` 구성을 위한 추가 옵션은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java)를 참조하세요.

## Swift에서 추론 실행하기

### 1단계: CocoaPods 가져오기

Podfile에 TensorFlowLiteTaskText 포드를 추가합니다.

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.2.0'
end
```

### 2단계: API를 사용하여 추론 실행하기

```swift
// Initialization
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLNLClassifier.h)를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
NLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)를 참조하세요.

## 예제 결과

Here is an example of the classification results of the [movie review model](https://www.tensorflow.org/lite/examples/text_classification/overview).

입력: "시간만 낭비했습니다."

출력:

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

자체 모델 및 테스트 데이터로 간단한 [NLClassifier용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier)를 시도해 보세요.

## 모델 호환성 요구 사항

Depending on the use case, the `NLClassifier` API can load a TFLite model with or without [TFLite Model Metadata](../../models/convert/metadata). See examples of creating metadata for natural language classifiers using the [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#nl_classifiers).

호환되는 모델은 다음 요구 사항을 충족해야 합니다.

- 입력 텐서: (kTfLiteString/kTfLiteInt32)

    - 모델의 입력은 kTfLiteString 텐서 원시 입력 문자열이거나 원시 입력 문자열의 토큰화된 regex 인덱스의 kTfLiteInt32 텐서여야 합니다.
    - If input type is kTfLiteString, no [Metadata](../../models/convert/metadata) is required for the model.
    - If input type is kTfLiteInt32, a `RegexTokenizer` needs to be set up in the input tensor's [Metadata](https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#natural_language_classifiers).

- 출력 스코어 텐서: (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    - 분류된 각 범주의 스코어에 대한 필수 출력 텐서

    - 유형이 Int 유형 중 하나이면 해당 플랫폼에 대해 double/float로 역양자화합니다.

    - Can have an optional associated file in the output tensor's corresponding [Metadata](../../models/convert/metadata) for category labels, the file should be a plain text file with one label per line, and the number of labels should match the number of categories as the model outputs. See the [example label file](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/nl_classifier/labels.txt).

- 출력 레이블 텐서: (kTfLiteString/kTfLiteInt32)

    - 각 범주의 레이블에 대한 선택적 출력 텐서는 출력 스코어 텐서와 길이가 같아야 합니다. 이 텐서가 없으면 API는 스코어 인덱스를 클래스 이름으로 사용합니다.

    - 연관된 레이블 파일이 출력 스코어 텐서의 메타데이터에 있는 경우 무시됩니다.

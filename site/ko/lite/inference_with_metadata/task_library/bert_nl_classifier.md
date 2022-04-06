# BERT 자연어 분류자 통합

Task 라이브러리 `BertNLClassifier` API는 입력 텍스트를 여러 범주로 분류하는 `NLClassifier`와 매우 유사하지만, 이 API는 TFLite 모델 밖에서 Wordpiece 및 Sentencepiece 토큰화가 필요한 Bert 관련 모델에 특별히 맞춤화되었습니다.

## BertNLClassifier API의 주요 기능

- 단일 문자열을 입력으로 받아서 문자열로 분류를 수행하고 분류 결과로 &lt;Label, Score&gt; 쌍을 출력합니다.

- 입력 텍스트에서 그래프 외 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 또는 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 토큰화를 수행합니다.

## 지원되는 BertNLClassifier 모델

다음 모델이 `BertNLClassifier` API와 호환됩니다.

- [텍스트 분류용 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification)에 의해 생성된 Bert 모델

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

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

    // Import the Task Text Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'
}
```

참고: Android Gradle 플러그인 버전 4.1부터는 .tflite가 기본적으로 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.

### 2단계: API를 사용하여 추론 실행하기

```java
// Initialization
BertNLClassifierOptions options =
    BertNLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertNLClassifier classifier =
    BertNLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java)를 참조하세요.

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
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h)를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h)를 참조하세요.

## 예제 결과

다음은 Model Maker의 [MobileBert](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) 모델을 사용하여 영화 리뷰를 분류한 결과의 예입니다.

입력: "it's a charming and often affecting journey"

출력:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

자체 모델 및 테스트 데이터로 간단한 [ObjectDetector용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier)를 사용해 보세요.

## 모델 호환성 요구 사항

`BetNLClassifier` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다.

메타데이터는 다음 요구 사항을 충족해야 합니다.

- Wordpiece/Sentencepiece Tokenizer를 위한 input_process_units

- Tokenizer의 출력을 위한 이름이 "ids", "mask" 및 "segment_ids"인 3개의 입력 텐서

- 선택적으로 레이블 파일이 첨부된 float32 유형의 출력 텐서 1개. 레이블 파일이 첨부된 경우 파일은 한 줄에 하나의 레이블이 있는 일반 텍스트 파일이어야 하며 레이블 수는 모델 출력의 범주 수와 일치해야 합니다.

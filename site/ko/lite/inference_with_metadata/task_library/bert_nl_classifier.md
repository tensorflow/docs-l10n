# BERT 자연어 분류자 통합

Task 라이브러리 `BertNLClassifier` API는 입력 텍스트를 여러 범주로 분류하는 `NLClassifier`와 매우 유사하지만, 이 API는 TFLite 모델 밖에서 Wordpiece 및 Sentencepiece 토큰화가 필요한 Bert 관련 모델에 특별히 맞춤화되었습니다.

## BertNLClassifier API의 주요 기능

- Takes a single string as input, performs classification with the string and outputs &lt;Label, Score&gt; pairs as classification results.

- 입력 텍스트에서 그래프 외 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 또는 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 토큰화를 수행합니다.

## 지원되는 BertNLClassifier 모델

The following models are compatible with the `BertNLClassifier` API.

- [텍스트 분류용 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification)에 의해 생성된 Bert 모델

- Custom models that meet the [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module where the model will be run. Specify that the file should not be compressed, and add the TensorFlow Lite library to the module’s `build.gradle` file:

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

Note: starting from version 4.1 of the Android Gradle plugin, .tflite will be added to the noCompress list by default and the aaptOptions above is not needed anymore.

### Step 2: Run inference using the API

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

See the [source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java) for more details.

## Run inference in Swift

### Step 1: Import CocoaPods

Add the TensorFlowLiteTaskText pod in Podfile

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.2.0'
end
```

### Step 2: Run inference using the API

```swift
// Initialization
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

See the [source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h) for more details.

## Run inference in C++

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

See the [source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h) for more details.

## Example results

다음은 Model Maker의 [MobileBert](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification) 모델을 사용하여 영화 리뷰를 분류한 결과의 예입니다.

입력: "it's a charming and often affecting journey"

출력:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

자체 모델 및 테스트 데이터로 간단한 [ObjectDetector용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier)를 사용해 보세요.

## Model compatibility requirements

`BetNLClassifier` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다.

메타데이터는 다음 요구 사항을 충족해야 합니다.

- input_process_units for Wordpiece/Sentencepiece Tokenizer

- 3 input tensors with names "ids", "mask" and "segment_ids" for the output of the tokenizer

- 선택적으로 레이블 파일이 첨부된 float32 유형의 출력 텐서 1개. 레이블 파일이 첨부된 경우 파일은 한 줄에 하나의 레이블이 있는 일반 텍스트 파일이어야 하며 레이블 수는 모델 출력의 범주 수와 일치해야 합니다.

# BERT 질문 응답기 통합하기

Task Library `BertQuestionAnswerer` API는 Bert 모델을 로드하고 주어진 문구의 내용을 바탕으로 질문에 답합니다. 자세한 내용은 <a href="../../models/bert_qa/overview.md">여기</a>에서 질문-답변 모델에 대한 문서를 참조하세요.

## BertQuestionAnswerer API의 주요 특징

- 두 개의 텍스트 입력을 질문 및 컨텍스트로 받아서 가능한 답변 목록을 출력합니다.

- 입력 텍스트에서 그래프 외 Wordpiece 또는 Sentencepiece 토큰화를 수행합니다.

## 지원되는 BertQuestionAnswerer 모델

다음 모델은 `BertNLClassifier` API와 호환됩니다.

- Models created by [TensorFlow Lite Model Maker for BERT Question Answer](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer).

- [TensorFlow Hub에서 사전 훈련된 BERT 모델](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1)

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

    // Import the Task Text Library dependency
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'
}
```

### 2단계: API를 사용하여 추론 실행하기

```java
// Initialization
BertQuestionAnswerer answerer = BertQuestionAnswerer.createFromFile(androidContext, modelFile);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java)를 참조하세요.

## Swift에서 추론 실행하기

### 1단계: CocoaPods 가져오기

Podfile에 TensorFlowLiteTaskText 포드를 추가합니다.

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.0.1-nightly'
end
```

### 2단계: API를 사용하여 추론 실행하기

```swift
// Initialization
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h)를 참조하세요.

## C++에서 추론 실행하기

참고: 사전 빌드된 바이너리를 제공하고 소스 코드에서 빌드할 사용자 친화적인 워크플로를 만드는 등 C++ Task Library의 사용 편리성을 개선하기 위해 노력하고 있습니다. C++ API는 변경될 수 있습니다.

```c++
// Initialization
std::unique_ptr answerer = BertQuestionAnswerer::CreateFromFile(model_file).value();

// Run inference
std::vector positive_results = answerer->Answer(context_of_question, question_to_ask);
```

자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h)를 참조하세요.

## 예제 결과

다음은 [ALBERT 모델](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1)의 답변 결과를 보여주는 예입니다.

컨텍스트: "The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations."

질문: "Where is Amazon rainforest?"

답변:

```
answer[0]:  'South America.'
logit: 1.84847, start_index: 39, end_index: 40
answer[1]:  'most of the Amazon basin of South America.'
logit: 1.2921, start_index: 34, end_index: 40
answer[2]:  'the Amazon basin of South America.'
logit: -0.0959535, start_index: 36, end_index: 40
answer[3]:  'the Amazon biome that covers most of the Amazon basin of South America.'
logit: -0.498558, start_index: 28, end_index: 40
answer[4]:  'Amazon basin of South America.'
logit: -0.774266, start_index: 37, end_index: 40
```

자체 모델 및 테스트 데이터로 간단한 [BertQuestionAnswerer용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer)를 시도해 보세요.

## 모델 호환성 요구 사항

`BertQuestionAnswerer` API는 필수 [TFLite 모델 메타데이터](../../convert/metadata.md)가 있는 TFLite 모델을 예상합니다.

The Metadata should meet the following requirements:

- `input_process_units` for Wordpiece/Sentencepiece Tokenizer

- Tokenizer의 출력을 위한 이름이 "ids", "mask" 및 "segment_ids"인 3개의 입력 텐서

- 2 output tensors with names "end_logits" and "start_logits" to indicate the answer's relative position in the context

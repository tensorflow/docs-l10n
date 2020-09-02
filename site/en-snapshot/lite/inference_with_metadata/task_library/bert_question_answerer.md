# Integrate BERT question answerer

The Task Library `BertQuestionAnswerer` API loads a Bert model and answers
questions based on the content of a given passage. For more information, see the
documentation for the Question-Answer model
<a href="../../models/bert_qa/overview.md">here</a>.

## Key features of the BertQuestionAnswerer API

*   Takes two text inputs as question and context and outputs a list of possible
    answers.

*   Performs out-of-graph Wordpiece or Sentencepiece tokenizations on input
    text.

## Supported BertQuestionAnswerer models

The following models are compatible with the `BertNLClassifier` API.

*   Models created by
    [TensorFlow Lite Model Maker for Question Answer](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer).

*   The
    [pretrained BERT models on TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in Java

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` model file to the assets directory of the Android module
where the model will be run. Specify that the file should not be compressed, and
add the TensorFlow Lite library to the module’s `build.gradle` file:

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
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.0.0-nightly'
}
```

### Step 2: Run inference using the API

```java
// Initialization
BertQuestionAnswerer answerer = BertQuestionAnswerer.createFromFile(androidContext, modelFile);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java)
for more details.

## Run inference in C++

Note: we are working on improving the usability of the C++ Task Library, such as
providing prebuilt binaries and creating user-friendly workflows to build from
source code. The C++ API may be subject to change.

```c++
// Initialization
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromFile(model_file).value();

// Run inference
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h)
for more details.

## Example results

Here is an example of the answer results of
[ALBERT model](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1).

Context: "The Amazon rainforest, alternatively, the Amazon Jungle, also known in
English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon
biome that covers most of the Amazon basin of South America. This basin
encompasses 7,000,000 km2 (2,700,000 sq mi), of which
5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region
includes territory belonging to nine nations."

Question: "Where is Amazon rainforest?"

Answers:

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

Try out the simple
[CLI demo tool for BertQuestionAnswerer](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer)
with your own model and test data.

## Model compatibility requirements

The `BertQuestionAnswerer` API expects a TFLite model with mandatory
[TFLite Model Metadata](../../convert/metadata.md).

The Metadata should meet the following requiresments:

*   `input_process_units` for Wordpiece/Sentencepiece Tokenizer

*   3 input tensors with names "ids", "mask" and "segment_ids" for the output of
    the tokenizer

*   2 output tensors with names "end_logits" and "start_logits" to indicate the
    answer's relative position in the context

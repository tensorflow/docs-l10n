# Integrate Natural language classifier

The Task Library's `NLClassifier` API classifies input text into different
categories, and is a versatile and configurable API that can handle most text
classification models.

## Key features of the NLClassifier API

*   Takes a single string as input, performs classification with the string and
    outputs <Label, Score> pairs as classification results.

*   Optional Regex Tokenization available for input text.

*   Configurable to adapt different classification models.

## Supported NLClassifier models

The following models are guaranteed to be compatible with the `NLClassifier`
API.

*   The <a href="../../models/text_classification/overview.md">movie review
    sentiment classification</a> model.

*   Models with `average_word_vec` spec created by
    [TensorFlow Lite Model Maker for text Classfication](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification).

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
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options = NLClassifierOptions.builder().setInputTensorName(INPUT_TENSOR_NAME).setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME).build();
NLClassifier classifier = NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java)
for more options to configure `NLClassifier`.

## Run inference in C++

Note: We are working on improving the usability of the C++ Task Library, such as
providing prebuilt binaries and creating user-friendly workflows to build from
source code. The C++ API may be subject to change.

```c++
// Initialization
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromFileAndOptions(
    model_path,
    {
      .input_tensor_name=kInputTensorName,
      .output_score_tensor_name=kOutputScoreTensorName,
    }).value();

// Run inference
std::vector<core::Category> categories = classifier->Classify(kInput);
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h)
for more details.

## Example results

Here is an example of the classification results of the
[movie review model](https://www.tensorflow.org/lite/models/text_classification/overview).

Input: "What a waste of my time."

Output:

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

Try out the simple
[CLI demo tool for NLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier)
with your own model and test data.

## Model compatibility requirements

Depending on the use case, the `NLClassifier` API can load a TFLite model with
or without [TFLite Model Metadata](../../convert/metadata.md).

The compatible models should meet the following requirements:

*   Input tensor: (kTfLiteString/kTfLiteInt32)

    -   Input of the model should be either a kTfLiteString tensor raw input
        string or a kTfLiteInt32 tensor for regex tokenized indices of raw input
        string.
    -   If input type is kTfLiteString, no [Metadata](../../convert/metadata.md)
        is required for the model.
    -   If input type is kTfLiteInt32, a `RegexTokenizer` needs to be set up in
        the input tensor's [Metadata](../../convert/metadata.md).

*   Output score tensor:
    (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    -   Mandatory output tensor for the score of each category classified.

    -   If type is one of the Int types, dequantize it to double/float to
        corresponding platforms

    -   Can have an optional associated file in the output tensor's
        corresponding [Metadata](../../convert/metadata.md) for category labels,
        the file should be a plain text file with one label per line, and the
        number of labels should match the number of categories as the model
        outputs.

*   Output label tensor: (kTfLiteString/kTfLiteInt32)

    -   Optional output tensor for the label for each category, should be of the
        same length as the output score tensor. If this tensor is not present,
        the API uses score indices as classnames.

    -   Will be ignored if the associated label file is present in output score
        tensor's Metadata.

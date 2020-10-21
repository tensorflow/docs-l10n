# TensorFlow Lite の推論

*推論*とは、入力データに基づいて予測を立てるために、TensorFlow Lite モデルをオンデバイスで実行するプロセスを指します。TensorFlow Lite モデルで推論を実行するには、*インタプリタ*を使って実行する必要があります。TensorFlow Lite インタプリタは、リーンで高速であるように設計されており、静的なグラフの順序付けとカスタム（あまり動的でない）メモリアロケータを使用して、最小限の読み込み、初期化、および実行遅延を実現しています。

This page describes how to access to the TensorFlow Lite interpreter and perform an inference using C++, Java, and Python, plus links to other resources for each [supported platform](#supported-platforms).

[TOC]

## 重要な概念

TensorFlow Lite の推論は、通常次の手順で行います。

1. **モデルの読み込み**

    You must load the `.tflite` model into memory, which contains the model's execution graph.

2. **データの変換**

    モデルの生の入力データは、モデルが期待する入力データに一致しません。たとえば、画像のサイズを変更したり、画像形式を変更したりすることで、モデルとの互換性を持たせる必要がある場合があります。

3. **推論の実行**

    This step involves using the TensorFlow Lite API to execute the model. It involves a few steps such as building the interpreter, and allocating tensors, as described in the following sections.

4. **出力の解釈**

    モデルの推論から結果を得ると、アプリケーションで役立つ意義のある方法でテンソルを解釈する必要があります。

    For example, a model might return only a list of probabilities. It's up to you to map the probabilities to relevant categories and present it to your end-user.

##  サポートされているプラットフォーム

TensorFlow の推論 API は、Android、iOS、および Linux などの最も一般的なモバイル/組み込みプラットフォーム向けに複数のプログラミング言語で提供されています。

In most cases, the API design reflects a preference for performance over ease of use. TensorFlow Lite is designed for fast inference on small devices, so it should be no surprise that the APIs try to avoid unnecessary copies at the expense of convenience. Similarly, consistency with TensorFlow APIs was not an explicit goal and some variance between languages is to be expected.

すべてのライブラリにおいて、TensorFlow Lite API によって、モデルの読み込み、入力のフィード、および推論出力の取得が可能となります。

### Android プラットフォーム

Android では、TensorFlow Lite の推論は、Java または C++ API のいずれかを使用して実行できます。Java API は、利便性を提供し、Android Activity クラス内で直接使用できます。C++ API は、さらに柔軟性と速度を提供しますが、Java と C++ レイヤー間でデータを移動するには、JNI ラッパーを書く必要がある場合があります。

See below for details about using [C++](#load-and-run-a-model-in-c) and [Java](#load-and-run-a-model-in-java), or follow the [Android quickstart](android.md) for a tutorial and example code.

#### TensorFlow Lite Android ラッパーコードジェネレータ

注意: TensorFlow Lite ラッパーコードジェネレータは実験（ベータ）フェーズにあり、現在 Android のみをサポートしています。

For TensorFlow Lite model enhanced with [metadata](../convert/metadata.md), developers can use the TensorFlow Lite Android wrapper code generator to create platform specific wrapper code. The wrapper code removes the need to interact directly with `ByteBuffer` on Android. Instead, developers can interact with the TensorFlow Lite model with typed objects such as `Bitmap` and `Rect`. For more information, please refer to the [TensorFlow Lite Android wrapper code generator](../inference_with_metadata/codegen.md).

### iOS プラットフォーム

iOS では、TensorFlow Lite は [Swift](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) と [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc) で書かれたネイティブ iOS ライブラリで提供されています。また、直接 Objective-C コードで [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) を使用することもできます。

[Swift](#load-and-run-a-model-in-swift)、[Objective-C](#load-and-run-a-model-in-objective-c)、および [C API](#using-c-api-in-objective-c-code) の使用に関する詳細は以下をご覧ください。または、チュートリアルとサンプルコードについては、[iOS クイックスタート](ios.md)をご覧ください。

### Linux プラットフォーム

On Linux platforms (including [Raspberry Pi](build_rpi.md)), you can run inferences using TensorFlow Lite APIs available in [C++](#load-and-run-a-model-in-c) and [Python](#load-and-run-a-model-in-python), as shown in the following sections.

## モデルを実行する

TensorFlow Lite モデルは、いくつかの単純な手順で実行します。

1. モデルをメモリに読み込みます。
2. 既存のモデルに基づいて、`インタプリタ`を構築します。
3. 入力テンソルの値を設定します（オプションとして、入力テンソルに事前に定義されたサイズが希望するサイズでない場合は、それを変更することができます）。
4. 推論を呼び出します。
5. 出力テンソルの値を読み取ります。

次のセクションでは、上記の手順を各言語で実行する方法を説明します。

## Java でモデルを読み込んで実行する

*プラットフォーム: Android*

TensorFlow Lite で推論を実行するための Java API は主に、Android で使用するように設計されているため、Android ライブラリ依存関係として、`org.tensorflow:tensorflow-lite` のように提供されています。

In Java, you'll use the `Interpreter` class to load a model and drive model inference. In many cases, this may be the only API you need.

`Interpreter` の初期化には、`.tflite` ファイルを使用することができます。

```java
public Interpreter(@NotNull File modelFile);
```

または、`MappedByteBuffer` を使用します。

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

いずれの場合でも、有効な TensorFlow Lite モデルを提供しない場合、API によって `IllegalArgumentException` がスローされてしまいます。`MappedByteBuffer` を使用して `Interpreter` を初期化したら、`Interpreter` が存続する限り、変更してはいけません。

その後でモデルで推論を実行するには、`Interpreter.run()` を実行します。次はその例を示します。

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

`run()` メソッドは 1 つの入力のみを取って 1 つの出力のみを返します。そのためモデルが複数の入力または出力を持つ場合は、次のように行います。

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

この場合、`inputs` の各エントリは入力テンソルに対応し、`map_of_indices_to_outputs` は、出力テンソルのインデックスを対応する出力データにマッピングします。

両方のケースでは、テンソルのインテックスはモデルを作成したときに [TensorFlow Lite コンバータ](../convert/)に指定した値に対応しています。`input` 内のテンソルの順序が TensorFlow Lite コンバータに指定した順序と一致している必要があることに注意してください。

`Interpreter` クラスには、演算名を使用してモデルの入力または出力のインデックスを取得するための便利な関数もあります。

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

モデルの `opName` が有効な演算でない場合、`IllegalArgumentException` がスローされます。

Also beware that `Interpreter` owns resources. To avoid memory leak, the resources must be released after use by:

```java
interpreter.close();
```

Java を使ったサンプルプロジェクトについては、[Android 画像分類サンプル](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)をご覧ください。

### サポートされているデータ型（Java）

TensorFlow Lite を使用するには、入力テンソルと出力テンソルのデータ型が次の原始型のいずれかである必要があります。

- `float`
- `int`
- `long`
- `byte`

`String` 型もサポートされていますが、原始型とは異なってエンコーディングされます。特に、文字列テンソルの形状はテンソル内の文字列の数と配置を示し、各要素そのものが可変長の文字列です。このため、テンソルの（バイト）サイズを形状と型からだけでは計算できず、その結果文字列は単一のフラットな `ByteBuffer` 引数として指定することができません。

`Integer` や `Float` などのボックス型を含むほかのデータ型が使用される場合、`IllegalArgumentException` がスローされます。

#### 入力

各入力は、サポートされている原始型の配列または多次元配列であるか、適切なサイズの生の `ByteBuffer` です。入力が配列または多次元配列である場合、関連する入力テンソルは、推論時の配列の次元に合わせて暗黙的にサイズが変更されます。入力が ByteBuffer である場合は、推論を実行する前に、呼び出し元が関連するテンソルを（`Interpreter.resizeInput()` 経由で）手動でサイズ変更する必要があります。

`ByteBuffer` を使用する際は、ダイレクトバイトバッファを使用する方が好ましいといえます。これは、`Interpreter` が不要なコピーを回避することができるためです。`ByteBuffer` がダイレクトバイトバッファである場合、その順序は `ByteOrder.nativeOrder()` である必要があります。モデルの推論に使用されたら、モデルの推論が完了するまで変更してはいけません。

#### 出力

各出力は、サポートされている原始型の配列または多次元配列であるか、適切なサイズの ByteBuffer です。一部のモデルには動的な出力があり、出力テンソルの形状は入力に応じて異なります。これを既存の Java 推論 API を使った簡単な方法で処理する方法はありませんが、計画的な拡張機能を使って実現することができます。

## Swift でモデルを読み込んで実行する

*プラットフォーム: iOS*

[Swift API](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) は Cocoapods の `TensorFlowLiteSwift` ポッドで提供されています。

まず、`TensorFlowLite` モジュールをインポートする必要があります。

```swift
import TensorFlowLite
```

```swift
// Getting model path
guard
  let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
else {
  // Error handling...
}

do {
  // Initialize an interpreter with the model.
  let interpreter = try Interpreter(modelPath: modelPath)

  // Allocate memory for the model's input `Tensor`s.
  try interpreter.allocateTensors()

  let inputData: Data  // Should be initialized

  // input data preparation...

  // Copy the input data to the input `Tensor`.
  try self.interpreter.copy(inputData, toInputAt: 0)

  // Run inference by invoking the `Interpreter`.
  try self.interpreter.invoke()

  // Get the output `Tensor`
  let outputTensor = try self.interpreter.output(at: 0)

  // Copy output to `Data` to process the inference results.
  let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
  let outputData =
        UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
  outputTensor.data.copyBytes(to: outputData)

  if (error != nil) { /* Error handling... */ }
} catch error {
  // Error handling...
}
```

## Objective-C でモデルを読み込んで実行する

*プラットフォーム: iOS*

[Objective-C API](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc) は Cocoapods の `TensorFlowLiteObjC` ポッドで提供されています。

まず、`TensorFlowLite` モジュールをインポートする必要があります。

```objc
@import TensorFlowLite;
```

```objc
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model"
                                                      ofType:@"tflite"];
NSError *error;

// Initialize an interpreter with the model.
TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                  error:&error];
if (error != nil) { /* Error handling... */ }

// Allocate memory for the model's input `TFLTensor`s.
[interpreter allocateTensorsWithError:&error];
if (error != nil) { /* Error handling... */ }

NSMutableData *inputData;  // Should be initialized
// input data preparation...

// Copy the input data to the input `TFLTensor`.
[interpreter copyData:inputData toInputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Run inference by invoking the `TFLInterpreter`.
[interpreter invokeWithError:&error];
if (error != nil) { /* Error handling... */ }

// Get the output `TFLTensor`
TFLTensor *outputTensor = [interpreter outputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy output to `NSData` to process the inference results.
NSData *outputData = [outputTensor dataWithError:&error];
if (error != nil) { /* Error handling... */ }
```

### Objective-C コードで C API を使用する

現在のところ Objective-C API はデリゲートをサポートしていません。Objective-C コードでデリゲートを使用するには、基盤の [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) を直接呼び出す必要があります。

```c
#include "tensorflow/lite/c/c_api.h"
```

```c
TfLiteModel* model = TfLiteModelCreateFromFile([modelPath UTF8String]);
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

// Create the interpreter.
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

// Allocate tensors and populate the input tensor data.
TfLiteInterpreterAllocateTensors(interpreter);
TfLiteTensor* input_tensor =
    TfLiteInterpreterGetInputTensor(interpreter, 0);
TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                           input.size() * sizeof(float));

// Execute inference.
TfLiteInterpreterInvoke(interpreter);

// Extract the output tensor data.
const TfLiteTensor* output_tensor =
    TfLiteInterpreterGetOutputTensor(interpreter, 0);
TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                         output.size() * sizeof(float));

// Dispose of the model and interpreter objects.
TfLiteInterpreterDelete(interpreter);
TfLiteInterpreterOptionsDelete(options);
TfLiteModelDelete(model);
```

## C++ でモデルを読み込んで実行する

*プラットフォーム: Android、iOS、および Linux*

注意: iOS の C++ API は Bazel を使用している場合にのみ利用できます。

C++ では、モデルは [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html) クラスに格納されます。TensorFlow Lite モデルをカプセル化し、モデルの格納場所に応じていくつかの方法で構築することができます。

```c++
class FlatBufferModel {
  // Build a model based on a file. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter);

  // Build a model based on a pre-loaded flatbuffer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* buffer,
      size_t buffer_size,
      ErrorReporter* error_reporter);
};
```

注意: TensorFlow Lite が [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) の存在を検出すると、`FlatBufferModel` の格納に、自動的に共有メモリを使用しようとします。

`FlatBufferModel` オブジェクトとしてモデルを準備できたので、[`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html) で実行できるようになりました。単一の `FlatBufferModel` を複数の `Interpreter` で同時に使用することができます。

注意: `FlatBufferModel` オブジェクトは、それを使用する `Interpreter` の全インスタンスが破壊されるまで有効な状態を維持する必要があります。

The important parts of the `Interpreter` API are shown in the code snippet below. It should be noted that:

- 文字列比較（および文字列ライブラリへのすべての固定した依存関係）を回避するために、テンソルは整数値で表現されています。
- インタプリタには同時スレッドからアクセスしてはいけません。
- 入力テンソルと出力テンソルのメモリ割り当ては、テンソルのサイズ変更を行った直後に `AllocateTensors()` を呼び出してトリガされる必要があります。

C++ を使った TensorFlow Lite の最も簡単な使用方法を次に示します。

```c++
// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
```

その他のサンプルコードについては、[`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc) と [`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc) をご覧ください。

## Python でモデルを読み込んで実行する

*プラットフォーム: Linux*

推論を実行するための Python API は、`tf.lite` モジュールで提供されています。この API からは主に、モデルを読み込んで推論を実行する [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) のみが必要です。

次の例では、Python インタプリタを使用して `.tflite` ファイルを読み込み、ランダムな入力データで推論を実行する方法を示します。

```python
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

As an alternative to loading the model as a pre-converted `.tflite` file, you can combine your code with the [TensorFlow Lite Converter Python API](https://www.tensorflow.org/lite/convert/python_api) (`tf.lite.TFLiteConverter`), allowing you to convert your TensorFlow model into the TensorFlow Lite format and then run inference:

```python
import numpy as np
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

# Convert to TF Lite format
with tf.Session() as sess:
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Continue to get tensors and so forth, as shown above...
```

その他の Python サンプルコードについては、[`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py) をご覧ください。

ヒント: Python ターミナルで `help(tf.lite.Interpreter)` を実行すると、インタプリタの詳細なドキュメントを閲覧できます。

## サポートされている演算

TensorFlow Lite は、いくつかの制約を伴う TensorFlow のサブセットをサポートしています。演算と制限の全リストについては、[TF Lite 演算のページ](https://www.tensorflow.org/mlir/tfl_ops)をご覧ください。

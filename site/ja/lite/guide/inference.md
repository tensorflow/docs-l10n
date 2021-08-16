# TensorFlow Lite の推論

*推論*とは、入力データに基づいて予測を立てるために、TensorFlow Lite モデルをオンデバイスで実行するプロセスを指します。TensorFlow Lite モデルで推論を実行するには、*インタプリタ*を使う必要があります。TensorFlow Lite インタプリタは、軽量かつ高速であるように設計されており、静的なグラフの順序付けとカスタム（あまり動的でない）メモリアロケータを使用して、最小限の読み込み、初期化、および実行の低レイテンシを実現しています。

このページでは、TesorFlow Lite インタプリタにアクセスして、C++、Java、および Python を使って推論する方法を説明し、各[対応プラットフォーム](#supported-platforms)向けのその他のリソースへのリンクを紹介します。

[TOC]

## 重要な概念

TensorFlow Lite の推論は、通常次の手順で行います。

1. **モデルの読み込み**

    モデルの実行グラフを含む`.tflite`モデルをメモリに読み込む必要があります。

2. **データの変換**

    モデルの生の入力データは、通常モデルが期待する入力データに一致しません。そのため、画像のサイズを変更したり、画像形式を変更したりすることで、モデルとの互換性を持たせる必要がある場合があります。

3. **推論の実行**

    TensorFlow Lite API を使用して、モデルを実行します。インタプリタの構築やテンソルの割り当てなど、次のセクションに説明されるステップがあります。

4. **出力の解釈**

    モデルの推論から結果を取得した後で、アプリケーションで役立つ意義のある方法でテンソルを解釈する必要があります。

    たとえば、モデルは確率のリストのみを返すことがあります。その場合、確率を関連するカテゴリにマッピングし、エンドユーザーに提供できます。

## サポートされているプラットフォーム

TensorFlow の推論 API は、[Android](#android-platform)、[iOS](#ios-platform)、および、[Linux](#linux-platform) などの最も一般的なモバイル/組み込みプラットフォーム向けに複数のプログラミング言語で提供されています。

ほとんどの場合、API の設計は使いやすさよりもパフォーマンスを反映しています。TensorFlow Lite は小型デバイスでの高速推論向けに設計されているため、API が利便性を犠牲にして不要なコピーを回避しようとするのも驚くことではありません。同様に、TensorFlow API との一貫性は、明確な目標ではなく、言語間のバリアンスがあります。

すべてのライブラリにおいて、TensorFlow Lite API により、モデルの読み込み、入力のフィード、および推論出力の取得が可能となります。

### Android プラットフォーム

Android では、TensorFlow Lite の推論は、Java または C++ API のいずれかを使用して実行できます。Java API は、利便性を提供し、Android Activity クラス内で直接使用できます。C++ API は、さらに柔軟性と速度を提供しますが、Java と C++ レイヤー間でデータを移動するには、JNI ラッパーを書く必要がある場合があります。

C++ と Java の使用に関する詳細は以下をご覧ください。または、チュートリアルとサンプルコードについては、[Android クイックスタート](#load-and-run-a-model-in-c)をご覧ください。

#### TensorFlow Lite Android ラッパーコードジェネレータ

注意: TensorFlow Lite ラッパーコードジェネレータは試験（ベータ）フェーズにあり、現在 Android のみをサポートしています。

[メタデータ](../convert/metadata.md)で強化された TensorFlow Lite モデルの場合、開発者は TensorFlow Lite Android ラッパーコードジェネレータを使用して、プラットフォーム固有のラッパーコードを作成できます。ラッパーコードにより、`ByteBuffer`と直接やり取りする必要がなくなり、開発者は`Bitmap`や`Rect`などの型付きオブジェクトを使用して TensorFlow Lite モデルとやり取りできます。詳細は、[TensorFlow Lite Android ラッパーコードジェネレータ](../inference_with_metadata/codegen.md)をご覧ください。

### iOS プラットフォーム

iOS では、TensorFlow Lite は [Swift](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) と [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc) で書かれたネイティブ iOS ライブラリで提供されています。また、直接 Objective-C コードで [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) を使用することもできます。

[Swift](#load-and-run-a-model-in-swift)、[Objective-C](#load-and-run-a-model-in-objective-c)、および [C API](#using-c-api-in-objective-c-code) の使用に関する詳細は以下をご覧ください。または、チュートリアルとサンプルコードについては、[iOS クイックスタート](ios.md)をご覧ください。

### Linux プラットフォーム

Linux プラットフォーム（[Raspberry Pi](build_rpi.md) を含む）では、次のセクションで説明される通り、[C++](#load-and-run-a-model-in-c) と [Python](#load-and-run-a-model-in-python) で提供されている TensorFlow Lite API を使用して推論を実行できます。

## モデルを実行する

TensorFlow Lite モデルは、いくつかの単純な手順で実行します。

1. モデルをメモリに読み込みます。
2. 既存のモデルに基づいて、`Interpreter`を構築します。
3. 入力テンソルの値を設定します（オプションとして、入力テンソルに事前に定義されたサイズが希望するサイズでない場合は、それを変更することができます）。
4. 推論を呼び出します。
5. 出力テンソルの値を読み取ります。

次のセクションでは、上記の手順を各言語で実行する方法を説明します。

## Java でモデルを読み込んで実行する

*プラットフォーム: Android*

TensorFlow Lite で推論を実行するための Java API は主に、Android で使用するように設計されているため、Android ライブラリ依存関係として、`org.tensorflow:tensorflow-lite`と提供されています。

Java では、モデルの読み込みとモデル推論の駆動に、`Interpreter`クラスを使用します。多くの場合、これが唯一必要な API です。

`Interpreter`の初期化には、`.tflite`ファイルを使用することができます。

```java
public Interpreter(@NotNull File modelFile);
```

または、`MappedByteBuffer`を使用します。

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

いずれの場合でも、有効な TensorFlow Lite モデルを提供しない場合、API により<br>`IllegalArgumentException`がスローされてしまいます。`MappedByteBuffer`を使用して`Interpreter`を初期化したら、`Interpreter`が存続する限り、変更してはいけません。

モデルで推論を実行するためには、シグネチャを使用することが推薦されます。これは、Tensorflow 2.5 以降で変換されたモデルで使用可能です。

```Java
try (Interpreter interpreter = new Interpreter(file_of_tensorflowlite_model)) {
  Map<String, Object> inputs = new HashMap<>();
  inputs.put("input_1", input1);
  inputs.put("input_2", input2);
  Map<String, Object> outputs = new HashMap<>();
  outputs.put("output_1", output1);
  interpreter.runSignature(inputs, outputs, "mySignature");
}
```

`runSignature`メソッドは以下の3つの引数を取ります。

- **入力** : シグネチャの入力名から入力オブジェクトへの入力のマップ。

- **出力** : シグネチャの出力名から出力データへの出力マッピングのマップ。の出力名から出力データへの出力マッピングのマップ。

- **シグネチャ名** [オプション]: シグネチャ名（モデルに単一のシグネチャがある場合は空のままにすることができます）。

また、モデルに定義されたシグネチャがない場合に推論を実行するには、`Interpreter.run()`を実行します。次はその例を示します。

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

`run()`メソッドは 1 つの入力のみを取り、 1 つの出力のみを返します。そのためモデルが複数の入力または出力を持つ場合は、次のように行います。

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

この場合、`inputs`の各エントリは入力テンソルに対応し、`map_of_indices_to_outputs`は、出力テンソルのインデックスを対応する出力データにマッピングします。

両方のケースでは、テンソルのインテックスはモデルを作成したときに [TensorFlow Lite コンバータ](../convert/)に指定した値に対応しています。`input`内のテンソルの順序が TensorFlow Lite コンバータに指定した順序と一致している必要があることに注意してください。

`Interpreter`クラスには、演算名を使用してモデルの入力または出力のインデックスを取得するための便利な関数もあります。

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

モデルで`opName`が有効な演算でない場合、`IllegalArgumentException`がスローされます。

また、`Interpreter`はリソースを所有することにも注意してください。メモリリークを回避するには、次のように、使用後にリソースを解放する必要があります。

```java
interpreter.close();
```

Java を使ったサンプルプロジェクトについては、[Android 画像分類サンプル](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)をご覧ください。

### サポートされているデータ型（Java）

TensorFlow Lite を使用するには、入力テンソルと出力テンソルのデータ型が次のプリミティブ型のいずれかである必要があります。

- `float`
- `int`
- `long`
- `byte`

`String`型もサポートされていますが、プリミティブ型とはエンコードが異なります。特に、文字列テンソルの形状はテンソル内の文字列の数と配置を示し、各要素そのものが可変長の文字列です。このため、テンソルの（バイト）サイズを形状と型からだけでは計算できず、その結果文字列は単一のフラットな`ByteBuffer`引数として指定することができません。

`Integer`や`Float`などのボックス型を含むほかのデータ型が使用される場合、`IllegalArgumentException`がスローされます。

#### 入力

各入力は、サポートされているプリミティブ型の配列または多次元配列であるか、適切なサイズの生の`ByteBuffer`です。入力が配列または多次元配列である場合、関連する入力テンソルは、推論時の配列の次元に合わせて暗黙的にサイズが変更されます。入力が ByteBuffer である場合は、推論を実行する前に、呼び出し元が関連するテンソルを（`Interpreter.resizeInput()` 経由で）手動でサイズ変更する必要があります。

`ByteBuffer`を使用する際は、ダイレクトバイトバッファを使用する方が好ましいといえます。これは、`Interpreter`が不要なコピーを回避することができるためです。`ByteBuffer`がダイレクトバイトバッファである場合、その順序は`ByteOrder.nativeOrder()`である必要があります。モデルの推論に使用されたら、モデルの推論が完了するまで変更してはいけません。

#### 出力

各出力は、サポートされているプリミティブ型の配列または多次元配列であるか、適切なサイズの ByteBuffer です。一部のモデルには動的な出力があり、出力テンソルの形状は入力に応じて異なります。これを既存の Java 推論 API では簡単に処理する方法はありませんが、計画的な拡張機能を使って実現することができます。

## Swift でモデルを読み込んで実行する

*プラットフォーム: iOS*

[Swift API](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) は Cocoapods の`TensorFlowLiteSwift`ポッドで提供されています。

まず、`TensorFlowLite`モジュールをインポートする必要があります。

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

[Objective-C API](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc) は Cocoapods の`TensorFlowLiteObjC`ポッドで提供されています。

まず、`TensorFlowLite`モジュールをインポートする必要があります。

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

// Get the input `TFLTensor`
TFLTensor *inputTensor = [interpreter inputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy the input data to the input `TFLTensor`.
[inputTensor copyData:inputData error:&error];
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

C++ では、モデルは[`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html)クラスに格納されます。TensorFlow Lite モデルをカプセル化し、モデルの格納場所に応じていくつかの方法で構築することができます。

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

注意: TensorFlow Lite が [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) の存在を検出すると、`FlatBufferModel`の格納に、自動的に共有メモリを使用しようとします。

`FlatBufferModel`オブジェクトとしてモデルを準備し、[`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html)で実行できるようになりました。単一の`FlatBufferModel`を複数の`Interpreter`で同時に使用することができます。

注意: `FlatBufferModel`オブジェクトは、それを使用する`Interpreter`の全インスタンスが破壊されるまで有効な状態を維持する必要があります。

`Interpreter` API の重要な個所を以下のコードスニペットに示していますが、次のことに注意してください。

- 文字列比較（および文字列ライブラリへのすべての固定した依存関係）を回避するために、テンソルは整数値で表現されています。
- インタプリタには同時スレッドからアクセスしてはいけません。
- 入力テンソルと出力テンソルのメモリ割り当ては、テンソルのサイズ変更を行った直後に`AllocateTensors()`を呼び出してトリガする必要があります。

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

その他のサンプルコードについては、[`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc)と[`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc) をご覧ください。

## Python でモデルを読み込んで実行する

*プラットフォーム: Linux*

推論を実行するための Python API は、`tf.lite`モジュールで提供されています。この API からは主に、モデルを読み込んで推論を実行する[`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)のみが必要です。

次の例では、Python インタプリタを使用して`.tflite`ファイルを読み込み、ランダムな入力データで推論を実行する方法を示します。

この例は、定義された SignatureDef がある SavedModel から変換する場合に推奨されます。これは、TensorFlow 2.5 以降で利用できます。

```python
class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}


SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH,
    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])
```

モデルに定義された SignatureDefs がない場合、

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

事前変換された`.tflite`ファイルとしてモデルを読み込む代わりに、コードを [TensorFlow Lite Converter Python API](https://www.tensorflow.org/lite/convert/python_api)（`tf.lite.TFLiteConverter`）と組み合わせて、TensorFlow モデルを TensorFlow Lite 形式に変換してから推論を実行することができます。

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

その他の Python サンプルコードについては、[`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)を参照してください。

ヒント: Python ターミナルで`help(tf.lite.Interpreter)`を実行すると、インタプリタの詳細なドキュメントを閲覧できます。

## サポートされている演算

TensorFlow Lite は、いくつかの制約を伴う TensorFlow のサブセットをサポートしています。演算と制限の全リストについては、[TF Lite 演算のページ](https://www.tensorflow.org/mlir/tfl_ops)をご覧ください。

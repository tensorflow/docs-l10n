# TensorFlow Lite 推断

*推断*这一术语是指为了根据输入数据进行预测而在设备端执行 TensorFlow Lite 模型的过程。要使用 TensorFlow Lite 模型进行推断，您必须通过*解释器*运行该模型。TensorFlow Lite 解释器旨在实现精益和快速。解释器使用静态计算图排序和自定义（动态程度较低的）内存分配器，来确保最小的负载、初始化和执行延迟。

本页面介绍了如何获得 TensorFlow Lite 解释器、如何使用 C++、Java 和 Python 执行推断，并提供了适用于每个[支持的平台](#supported-platforms)的其他资源的链接。

[TOC]

## 重要概念

TensorFlow Lite 推断通常遵循以下步骤：

1. **加载模型**

    您必须将 `.tflite` 模型加载到内存中，其中包含模型的执行计算图。

2. **转换数据**

    模型的原始输入数据通常与模型期望的输入数据格式不匹配。例如，您可能需要调整图像大小或更改图像格式才能与模型兼容。

3. **运行推断**

    此步骤涉及使用 TensorFlow Lite API 来执行模型。如以下各部分所述，它涉及构建解释器和分配张量等若干步骤。

4. **解释输出**

    当您从模型推断接收到结果后，必须以对您的应用有意义的方式来解释张量。

    例如，模型可能只会返回概率列表。由您来将概率映射到相关类别，并呈现给最终用户。

## 支持的平台

TensorFlow 推断 API 以多种编程语言为大多数常见的移动/嵌入式平台（例如 [Android](#android-platform)、[iOS](#ios-platform) 和 [Linux](#linux-platform)）提供。

在大多数情况下，API 设计反映了对性能而非易用性的偏好。 TensorFlow Lite 专为在小型设备上进行快速推断而设计，因此 API 试图以牺牲便利性为代价来避免不必要的复制也就不足为奇了。同样，与 TensorFlow API 保持一致也非明确目标，而且在不同语言之间还可能会有一些差异。

您可以使用 TensorFlow Lite API 在所有库中加载模型、馈送输入，并检索推断输出。

### Android 平台

在 Android 上，可以使用 Java 或 C++ API 来执行 TensorFlow Lite 推断。Java API 提供了便利性，并且可以直接在 Android Activity 类中使用。C++ API 提供了更好的灵活性和速度，但可能需要编写 JNI 封装容器才能在 Java 和 C++ 层之间移动数据。

有关使用 [C++](#load-and-run-a-model-in-c) 和 [Java](#load-and-run-a-model-in-java) 的详细内容，请参阅下文，或者按照 [Android 快速入门](../android)中的教程和示例代码进行操作。

#### TensorFlow Lite Android 封装容器代码生成器

注：TensorFlow Lite 封装容器代码生成器现处于实验 (Beta) 阶段，目前仅支持 Android。

对于使用[元数据](../inference_with_metadata/overview)增强的 TensorFlow Lite 模型，开发者可以使用 TensorFlow Lite Android 封装容器代码生成器来创建平台特定的封装容器代码。封装容器代码无需在 Android 上直接与 `ByteBuffer` 进行交互。相反，开发者可以使用类型化对象（如 `Bitmap` 和 `Rect`）与 TensorFlow Lite 模型进行交互。如需了解详细信息，请参阅 [TensorFlow Lite Android 封装容器代码生成器](../inference_with_metadata/codegen.md)。

### iOS 平台

在 iOS 上，TensorFlow Lite 适用于以 [Swift](https://www.tensorflow.org/code/tensorflow/lite/swift) 和 [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc) 编写的原生 iOS 库。您也可以直接在 Objective-C 代码中使用 [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h)。

有关使用 Swift、Objective-C 和 C API 的详细信息，请参阅下文，或者按照 [iOS 快速入门](#load-and-run-a-model-in-swift)中的教程和示例代码进行操作。

### Linux 平台

在 Linux 平台（包括 [Raspberry Pi](build_arm)）上，您可以使用以 C++ 和 Python 提供的 TensorFlow Lite API 运行推断，如以下各部分所述。

## 运行模型

运行 TensorFlow Lite 模型涉及几个简单步骤：

1. 将模型加载到内存中。
2. 基于现有模型构建 `Interpreter`。
3. 设置输入张量值。（如果不需要预定义的大小，则可以选择调整输入张量的大小。）
4. 调用推断。
5. 读取输出张量值。

以下各部分描述了在各种语言中完成上述步骤的方式。

## 在 Java 中加载并运行模型

*平台：Android*

使用 TensorFlow Lite 运行推断的 Java API 主要设计用于 Android，因此它可以作为 Android 库依赖项使用：<br><code>org.tensorflow:tensorflow-lite</code>。

在 Java 中，您将使用 `Interpreter` 类加载模型并驱动模型推断。在许多情况下，这可能是您唯一需要的 API。

您可以使用 `.tflite` 文件初始化 `Interpreter`：

```java
public Interpreter(@NotNull File modelFile);
```

或者使用 `MappedByteBuffer`：

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

在这两种情况下，您都必须提供有效的 TensorFlow Lite 模型，否则 API 会引发 `IllegalArgumentException`。如果使用 `MappedByteBuffer` 来初始化 `Interpreter`，则它必须在 `Interpreter` 的整个生命周期内保持不变。

在模型上运行推断的首选方式是使用签名，这适用于从 TensorFlow 2.5 开始转换的模型。

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

`runSignature` 方法需要三个参数：

- **输入**: 从签名中的输入名称到输入对象的输入映射。

- **输出**：从签名中的输出名称到输出数据的输出映射。

- **签名名称** [可选]：签名名称（如果模型具有单个签名，则可以留空）。

当模型没有定义的签名时，另一种运行推断的方式是直接调用 `Interpreter.run()`。例如：

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

`run()` 方法仅接受一个输入，且仅返回一个输出。因此，如果模型具有多个输入或多个输出，请改用：

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

在这种情况下，`inputs` 中的每个条目对应一个输入张量，且 `map_of_indices_to_outputs` 会将输出张量的索引映射到相应的输出数据。

在这两种情况下，张量索引都应与您在创建模型时提供给 [TensorFlow Lite 转换器](../models/convert/)的值相对应。请注意， `input` 中的张量顺序必须与提供给 TensorFlow Lite 转换器的顺序匹配。

`Interpreter` 类还提供了便于使用的函数，您可以通过函数使用运算名称来获取任何模型输入或输出的索引：

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

如果 `opName` 不是模型中的有效运算，它将引发 `IllegalArgumentException`。

还请注意 `Interpreter` 拥有资源。为了避免内存泄漏，资源在使用后必须通过以下方法进行释放：

```java
interpreter.close();
```

有关 Java 的示例项目，请参阅 [Android 图像分类示例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)。

### 支持的数据类型 (Java)

要使用 TensorFlow Lite，输入和输出张量的数据类型必须是以下其中一种基元类型：

- `float`
- `int`
- `long`
- `byte`

`String` 类型也受支持，但它们的编码方式与基元类型不同。特别是，字符串张量的形状决定了张量中字符串的数量和排列，每个元素本身都是可变长度字符串。从这个意义上说，不能仅通过形状和类型计算张量的（字节）大小，因此字符串不能作为单个扁平 `ByteBuffer` 参数提供。

如果使用了其他数据类型（例如 `Integer` 和 `Float` 这样的装箱类型），则会引发 `IllegalArgumentException`。

#### 输入

每个输入应是支持的基元类型的数组或多维数组，或适当大小的原始 `ByteBuffer`。如果输入是数组或多维数组，则在推断时会将关联的输入张量的大小隐式地调整为数组的维数。如果输入是 ByteBuffer，则调用者在运行推断前，应首先手动调整关联的输入张量的大小（通过 `Interpreter.resizeInput()`）。

使用 `ByteBuffer` 时，最好使用直接字节缓冲区，因为这可以使 `Interpreter` 避免不必要的复制。如果 `ByteBuffer` 是直接字节缓冲区，它的顺序必须为 `ByteOrder.nativeOrder()`。在用于模型推断之后，它必须保持不变，直到模型推断完成。

#### 输出

每个输出应是受支持的基元类型的数组或多维数组，或者是适当大小的 ByteBuffer。请注意，某些模型具有动态输出，其中输出张量的形状可能会因输入而异。现有的 Java 推断 API 无法简单地解决这个问题，但计划中的扩展程序将使其成为可能。

## 在 Swift 中加载并运行模型

*平台：iOS*

[Swift API](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) 可从 Cocoapods 的 `TensorFlowLiteSwift` Pod 中获得。

首先，您需要导入 `TensorFlowLite` 模块。

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

## 在 Objective-C 中加载并运行模型

*平台：iOS*

[Objective-C API](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc) 可从 Cocoapods 的 `TensorFlowLiteObjC` Pod 中获得。

首先，您需要导入 `TensorFlowLite` 模块。

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

### 在 Objective-C 代码中使用 C API

Objective-C API 目前不支持委托。为了将委托与 Objective-C 代码一起使用，您需要直接调用底层 [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h)。

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

## 在 C++ 中加载并运行模型

*平台：Android、iOS 和 Linux*

注：iOS 上的 C++ API 仅在使用 Bazel 时可用。

在 C++ 中，模型存储在 [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html) 类中。它封装了 TensorFlow Lite 模型，您可以通过几种不同的方式构建它，具体取决于模型的存储位置：

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

注：如果 TensorFlow Lite 检测到 [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) 的存在，它将自动尝试使用共享内存来存储 `FlatBufferModel`。

现在，您已拥有作为 `FlatBufferModel` 对象的模型，您可以使用 [`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html) 来执行它。单个 `FlatBufferModel` 可供多个 `Interpreter` 同时使用。

小心：`FlatBufferModel` 对象必须保持有效，直到使用它的所有 `Interpreter` 实例都被销毁。

以下代码段展示了 `Interpreter` API 的重要部分。应注意以下几点：

- 用整数来表示张量，以避免字符串比较（以及字符串库上的任何固定依赖项）。
- 不得从并发线程访问解释器。
- 必须在调整张量大小后立即调用 `AllocateTensors()` 来触发输入和输出张量的内存分配。

C++ 中最简单的 TensorFlow Lite 用法如下：

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

有关更多示例代码，请参阅 [`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc) 和 [`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc)。

## 在 Python 中加载并运行模型

*平台：Linux*

`tf.lite` 模块中提供了用于运行推断的 Python API。大多数情况下，您只需 [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) 来加载模型并运行推断。

以下示例展示了如何使用 Python 解释器加载 `.tflite` 文件，以及如何使用随机输入数据运行推断：

如果使用定义的 SignatureDef 从 SavedModel 进行转换，则建议使用此示例。从 TensorFlow 2.5 开始提供：

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

另一个示例（如果模型没有定义 SignatureDefs）：

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

除了将模型作为预转换的 `.tflite` 文件进行加载外，您还可以将代码与 [TensorFlow Lite 转换器 Python API](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter) (`tf.lite.TFLiteConverter`) 组合，进而将 TensorFlow 模型转换为 TensorFlow Lite 格式，然后运行推断：

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

有关更多 Python 示例代码，请参阅 [`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)。

提示：可以在 Python 终端运行 `help(tf.lite.Interpreter)` 获得有关解释器的详细文档。

## 支持的运算

TensorFlow Lite 支持一部分 TensorFlow 运算，但存在一些限制。有关运算和限制的完整列表，请参阅 [TF Lite 运算](https://www.tensorflow.org/mlir/tfl_ops)页面。

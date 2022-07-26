# TensorFlow Lite 추론

*추론*이라는 용어는 입력 데이터를 기반으로 예측을 수행하기 위해 기기에서 TensorFlow Lite 모델을 실행하는 프로세스를 나타냅니다. TensorFlow Lite 모델로 추론을 수행하려면 *인터프리터*를 통해 실행해야 합니다. TensorFlow Lite 인터프리터는 간결하고 빠르게 실행되도록 설계되었습니다. 인터프리터는 정적 그래프 순서 지정 및 사용자 정의(덜 동적임) 메모리 할당자(allocator)를 사용하여 로드, 초기화 및 실행 대기 시간을 최소화합니다.

이 페이지에서는 TensorFlow Lite 인터프리터에 액세스하고 C++, Java, Python을 사용하여 추론을 수행하는 방법을 설명하고 각 [지원 플랫폼](#supported-platforms)의 기타 리소스에 대한 링크를 제공합니다.

[TOC]

## 중요 개념

TensorFlow Lite 추론은 일반적으로 다음 단계를 따릅니다.

1. **모델 로드하기**

    `.tflite` 모델을 모델의 실행 그래프가 포함된 메모리로 로드해야 합니다.

2. **데이터 변환하기**

    모델의 원시 입력 데이터는 일반적으로 모델에서 예상하는 입력 데이터 형식과 일치하지 않습니다. 예를 들어, 모델과 호환되도록 이미지 크기를 조정하거나 이미지 형식을 변경해야 할 수 있습니다.

3. **추론 실행하기**

    이 단계에서는 TensorFlow Lite API를 사용하여 모델을 실행합니다. 이를 위해 다음 섹션의 설명과 같이 인터프리터 빌드 및 텐서 할당과 같은 몇 가지 단계를 진행합니다.

4. **출력 해석하기**

    모델 추론에서 결과를 받으면 애플리케이션에 유용한 의미 있는 방식으로 텐서를 해석해야 합니다.

    예를 들어, 모델은 확률 목록만 반환할 수 있습니다. 확률을 관련 범주에 매핑하고 최종 사용자에게 제시하는 것은 사용자의 몫입니다.

## 지원되는 플랫폼

TensorFlow 추론 API는 [Android](#android-platform), [iOS](#ios-platform) 및 [Linux](#linux-platform)와 같은 가장 일반적인 모바일/임베디드 플랫폼에서 사용할 수 있게 여러 프로그래밍 언어로 제공됩니다.

대부분의 경우, API 설계 시 사용 편의성보다는 성능에 치중합니다. TensorFlow Lite는 소형 기기에서 빠른 추론을 수행하도록 설계되므로, API가 편의성의 손해를 보더라도 불필요한 복사를 피하는 것은 충분히 이해할 수 있습니다. 마찬가지로, TensorFlow API와의 일관성은 명시적인 목표가 아니었으며 언어 간에 약간의 차이가 예상됩니다.

모든 라이브러리에서 TensorFlow Lite API를 사용하여 모델을 로드하고, 입력을 제공하고, 추론 출력을 가져올 수 있습니다.

### Android 플랫폼

Android에서는 Java 또는 C++ API를 사용하여 TensorFlow Lite 추론을 수행할 수 있습니다. Java API는 편의성을 제공하며 Android Activity 클래스 내에서 직접 사용할 수 있습니다. C++ API는 더 많은 유연성과 속도를 제공하지만 Java와 C++ 레이어 간에 데이터를 이동하려면 JNI 래퍼를 작성해야 할 수 있습니다.

[C++](#load-and-run-a-model-in-c) 및 [Java](#load-and-run-a-model-in-java) 사용에 대한 자세한 내용은 아래를 참조하거나 [Android 빠른 시작](#load-and-run-a-model-in-c)의 튜토리얼 및 예제 코드를 따르세요.

#### TensorFlow Lite Android 래퍼 코드 생성기

참고: TensorFlow Lite 래퍼 코드 생성기는 실험(베타) 단계에 있으며 현재 Android만 지원합니다.

[메타데이터](../convert/metadata.md)로 강화된 TensorFlow Lite 모델의 경우, 개발자는 TensorFlow Lite Android 래퍼 코드 생성기를 사용하여 플랫폼별 래퍼 코드를 만들 수 있습니다. 래퍼 코드는 Android에서 `ByteBuffer`와 직접 상호 작용할 필요성을 없애줍니다. 대신, 개발자는 `Bitmap` 및 `Rect`와 같은 형식화된 객체를 사용하여 TensorFlow Lite 모델과 상호 작용할 수 있습니다. 자세한 내용은 [TensorFlow Lite Android 래퍼 코드 생성기](../inference_with_metadata/codegen.md)를 참조하세요.

### iOS 플랫폼

iOS에서 TensorFlow Lite는 [Swift](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift) 및 [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc)로 작성된 기본 iOS 라이브러리와 함께 사용할 수 있습니다. Objective-C 코드에서 직접 [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h)를 사용할 수도 있습니다.

[Swift](#load-and-run-a-model-in-swift), [Objective-C](#load-and-run-a-model-in-objective-c) 및 [C API](#using-c-api-in-objective-c-code) 사용에 대한 자세한 내용은 아래를 참조하거나 [iOS 빠른 시작](ios.md)의 튜토리얼 및 예제 코드를 따르세요.

### Linux 플랫폼

Linux 플랫폼([Raspberry Pi](build_rpi.md) 포함)에서는 다음 섹션과 같이 [C++](#load-and-run-a-model-in-c) 및 [Python](#load-and-run-a-model-in-python)에서 사용할 수 있는 TensorFlow Lite API를 사용하여 추론을 실행할 수 있습니다.

## 모델 실행하기

TensorFlow Lite 모델을 실행하려면 몇 가지 간단한 단계가 필요합니다.

1. 모델을 메모리에 로드합니다.
2. 기존 모델을 기반으로 `Interpreter`를 빌드합니다.
3. 입력 텐서 값을 설정합니다(미리 정의된 크기가 필요하지 않은 경우 선택적으로 입력 텐서의 크기를 조정함).
4. 추론을 호출합니다.
5. 출력 텐서 값을 읽습니다.

다음 섹션에서는 이러한 단계를 각 언어로 수행하는 방법에 대해 설명합니다.

## Java에서 모델 로드 및 실행하기

*플랫폼: Android*

TensorFlow Lite로 추론을 실행하기 위한 Java API는 주로 Android에서 사용하도록 설계되었으므로 Android 라이브러리 종속성으로 사용할 수 있습니다(`org.tensorflow:tensorflow-lite`).

Java에서는 `Interpreter` 클래스를 사용하여 모델을 로드하고 모델 추론을 유도합니다. 많은 경우에 이 API만 있으면 됩니다.

`.tflite` 파일을 사용하여 `Interpreter`를 초기화할 수 있습니다.

```java
public Interpreter(@NotNull File modelFile);
```

또는 `MappedByteBuffer`를 사용합니다.

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

두 경우 모두, 유효한 TensorFlow Lite 모델을 제공해야 합니다. 그렇지 않으면 API에서 `IllegalArgumentException`을 발생시킵니다. `MappedByteBuffer`를 사용하여 `Interpreter`를 초기화하는 경우, `Interpreter`를 사용하는 전체 기간 동안 버퍼가 변경되지 않아야 합니다.

모델에서 추론을 실행하는 선호되는 방법은 서명을 사용하는 것이며, Tensorflow 2.5부터 변환된 모델에 사용 가능합니다.

```Java
try (Interpreter interpreter = new Interpreter(file_of_tensorflowlite_model)) {   Map<String, Object> inputs = new HashMap<>();   inputs.put("input_1", input1);   inputs.put("input_2", input2);   Map<String, Object> outputs = new HashMap<>();   outputs.put("output_1", output1);   interpreter.runSignature(inputs, outputs, "mySignature"); }
```

`runSignature` 메서드는 세 가지 인수를 사용합니다.

- **입력**: 서명의 입력 이름에서 입력 객체에 대한 입력을 매핑합니다.

- **출력**: 서명의 출력 이름에서 출력 데이터로의 출력 매핑을 위한 매핑입니다.

- **서명 이름** [선택 사항]: 서명 이름(모델에 단일 서명이 있는 경우 비워 둘 수 있음).

모델에 정의된 서명이 없을 때 추론을 실행하는 또 다른 방법으로 간단히 `Interpreter.run()`을 호출합니다. 예를 들면 다음과 같습니다.

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

`run()` 메서드는 하나의 입력만 받고 하나의 출력만 반환합니다. 따라서 모델에 여러 입력 또는 여러 출력이 있는 경우 대신 다음을 사용합니다.

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

이 경우, `inputs`의 각 항목은 입력 텐서에 해당하고 `map_of_indices_to_outputs`는 출력 텐서의 인덱스를 해당 출력 데이터에 매핑합니다.

두 경우 모두, 텐서 인덱스는 모델을 생성할 때 [TensorFlow Lite 변환기](../convert/)에 제공한 값과 일치해야 합니다. `input`의 텐서 순서는 TensorFlow Lite 변환기에 지정된 순서와 일치해야 합니다.

`Interpreter` 클래스는 연산 이름을 사용하여 모델 입력 또는 출력의 인덱스를 가져올 수 있는 편리한 함수도 제공합니다.

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

`opName`이 모델에서 유효한 연산이 아니면 `IllegalArgumentException`이 발생합니다.

또한 `Interpreter`가 리소스를 소유한다는 점에 유의하세요. 메모리 누출을 방지하려면 다음을 사용하여 사용 후 리소스를 해제해야 합니다.

```java
interpreter.close();
```

Java를 사용한 예제 프로젝트의 경우, [Android 이미지 분류 샘플](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)을 참조하세요.

### 지원되는 데이터 유형(Java의 경우)

TensorFlow Lite를 사용하려면 입력 및 출력 텐서의 데이터 유형이 다음 기본 유형 중 하나여야 합니다.

- `float`
- `int`
- `long`
- `byte`

`String` 유형도 지원되지만 기본 유형과 다르게 인코딩됩니다. 특히, 문자열 Tensor의 형상에 따라 Tensor에서 문자열의 수와 배열이 결정되며 각 요소 자체는 가변 길이 문자열입니다. 이런 맥락에서 Tensor의 (바이트) 크기는 형상과 유형만으로 계산할 수 없기 때문에 문자열은 단일 플랫 `ByteBuffer` 인수로 제공될 수 없습니다.

`Integer` 및 `Float`와 같은 boxed 유형을 포함한 다른 데이터 유형이 사용되면 `IllegalArgumentException`이 발생합니다.

#### 입력

각 입력은 지원되는 기본 유형의 배열 또는 다차원 배열이거나 적절한 크기의 원시 `ByteBuffer`여야 합니다. 입력이 배열 또는 다차원 배열인 경우, 연결된 입력 텐서는 추론 시에 배열의 차원에 맞게 암시적으로 크기가 조정됩니다. 입력이 ByteBuffer인 경우, 호출자는 추론을 실행하기 전에 먼저 `Interpreter.resizeInput()`을 통해 연결된 입력 텐서의 크기를 수동으로 조정해야 합니다.

`ByteBuffer`를 사용할 때는 `Interpreter`가 불필요한 복사본을 피할 수 있도록 직접 바이트 버퍼를 사용하는 것이 좋습니다. `ByteBuffer`가 직접 바이트 버퍼인 경우, 순서는 `ByteOrder.nativeOrder()`여야 합니다. 모델 추론에 사용된 후에는 모델 추론이 완료될 때까지 버퍼가 변경되지 않아야 합니다.

#### 출력

각 출력은 지원되는 기본 유형의 배열 또는 다차원 배열이거나 적절한 크기의 ByteBuffer여야 합니다. 일부 모델에는 출력 텐서의 형상이 입력에 따라 달라질 수 있는 동적 출력이 있습니다. 기존 Java 추론 API로 이를 처리하는 직접적인 방법은 없지만 앞으로 제공될 확장 기능으로 가능해질 것입니다.

## Swift에서 모델 로드 및 실행하기

*플랫폼: iOS*

[Swift API](https://www.tensorflow.org/code/tensorflow/lite/experimental/swift)는 Cocoapod의 `TensorFlowLiteSwift` Pod에서 사용할 수 있습니다.

먼저, `TensorFlowLite` 모듈을 가져와야 합니다.

```swift
import TensorFlowLite
```

```swift
// Getting model path guard   let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {   // Error handling... }  do {   // Initialize an interpreter with the model.   let interpreter = try Interpreter(modelPath: modelPath)    // Allocate memory for the model's input `Tensor`s.   try interpreter.allocateTensors()    let inputData: Data  // Should be initialized    // input data preparation...    // Copy the input data to the input `Tensor`.   try self.interpreter.copy(inputData, toInputAt: 0)    // Run inference by invoking the `Interpreter`.   try self.interpreter.invoke()    // Get the output `Tensor`   let outputTensor = try self.interpreter.output(at: 0)    // Copy output to `Data` to process the inference results.   let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})   let outputData =         UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)   outputTensor.data.copyBytes(to: outputData)    if (error != nil) { /* Error handling... */ } } catch error {   // Error handling... }
```

## Objective-C에서 모델 로드 및 실행하기

*플랫폼: iOS*

[Objective-C API](https://www.tensorflow.org/code/tensorflow/lite/experimental/objc)는 Cocoapod의 `TensorFlowLiteObjC` Pod에서 사용할 수 있습니다.

먼저, `TensorFlowLite` 모듈을 가져와야 합니다.

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

### Objective-C 코드에서 C API 사용하기

현재, Objective-C API는 대리자(delegate)를 지원하지 않습니다. Objective-C 코드와 함께 대리자를 사용하려면 기본 [C API](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h)를 직접 호출해야 합니다.

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

## C++에서 모델 로드 및 실행하기

*플랫폼: Android, iOS 및 Linux*

참고: iOS에서 C++ API는 베젤을 사용할 때만 사용할 수 있습니다.

C++에서 모델은 [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html) 클래스에 저장됩니다. 이 클래스는 TensorFlow Lite 모델을 캡슐화하며 모델이 저장된 위치에 따라 몇 가지 방법으로 빌드할 수 있습니다.

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

참고: TensorFlow Lite가 [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)의 존재를 감지하면 자동으로 공유 메모리를 사용하여 `FlatBufferModel` 저장을 시도합니다.

모델이 `FlatBufferModel` 객체로 준비되었으므로, 이제 [`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html)로 실행할 수 있습니다. 둘 이상의 `Interpreter`에서 하나의 `FlatBufferModel`을 동시에 사용할 수 있습니다.

주의: `FlatBufferModel` 객체는 이 객체를 사용하는 `Interpreter`의 모든 인스턴스가 소멸될 때까지 유효해야 합니다.

`Interpreter` API의 중요한 부분은 아래 코드 조각에 나와 있습니다. 다음 사항에 유의하세요.

- 문자열 비교(및 문자열 라이브러리에 대한 고정 종속성)를 피하기 위해 텐서는 정수로 표시됩니다.
- 인터프리터는 여러 스레드에서 동시에 액세스할 수 없습니다.
- 입력 및 출력 텐서에 대한 메모리 할당은 텐서 크기를 조정한 직후 `AllocateTensors()`를 호출하여 트리거해야 합니다.

C++에서 TensorFlow Lite의 가장 간단한 사용법은 다음과 같습니다.

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

더 많은 예제 코드는 [`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc) 및 [`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc)를 참조하세요.

## Python에서 모델 로드 및 실행하기

*플랫폼: Linux*

추론 실행을 위한 Python API는 `tf.lite` 모듈에서 제공됩니다. 여기에서 모델을 로드하고 추론을 실행하려면 [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)만 있으면 됩니다.

다음 예제는 Python 인터프리터를 사용하여 `.tflite` 파일을 로드하고 무작위 입력 데이터로 추론을 실행하는 방법을 보여줍니다.

이 예제는 정의된 SignatureDef를 사용하여 SavedModel에서 변환하는 경우에 권장됩니다. TensorFlow 2.5부터 사용 가능합니다.

```python
class TestModel(tf.Module):   def __init__(self):     super(TestModel, self).__init__()    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])   def add(self, x):     '''     Simple method that accepts single input 'x' and returns 'x' + 4.     '''     # Name the output 'result' for convenience.     return {'result' : x + 4}   SAVED_MODEL_PATH = 'content/saved_models/test_variable' TFLITE_FILE_PATH = 'content/test_variable.tflite'  # Save the model module = TestModel() # You can omit the signatures argument and a default signature name will be # created with name 'serving_default'. tf.saved_model.save(     module, SAVED_MODEL_PATH,     signatures={'my_signature':module.add.get_concrete_function()})  # Convert the model using TFLiteConverter converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH) tflite_model = converter.convert() with open(TFLITE_FILE_PATH, 'wb') as f:   f.write(tflite_model)  # Load the TFLite model in TFLite Interpreter interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH) # There is only 1 signature defined in the model, # so it will return it by default. # If there are multiple signatures then we can pass the name. my_signature = interpreter.get_signature_runner()  # my_signature is callable with input as arguments. output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32)) # 'output' is dictionary with all outputs from the inference. # In this case we have single output 'result'. print(output['result'])
```

모델에 SignatureDefs가 정의되지 않은 경우의 또 다른 예입니다.

```python
import numpy as np import tensorflow as tf  # Load the TFLite model and allocate tensors. interpreter = tf.lite.Interpreter(model_path="converted_model.tflite") interpreter.allocate_tensors()  # Get input and output tensors. input_details = interpreter.get_input_details() output_details = interpreter.get_output_details()  # Test the model on random input data. input_shape = input_details[0]['shape'] input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32) interpreter.set_tensor(input_details[0]['index'], input_data)  interpreter.invoke()  # The function `get_tensor()` returns a copy of the tensor data. # Use `tensor()` in order to get a pointer to the tensor. output_data = interpreter.get_tensor(output_details[0]['index']) print(output_data)
```

모델을 미리 변환된 `.tflite` 파일로 로드하는 대신 코드를 [TensorFlow Lite Converter Python API](https://www.tensorflow.org/lite/convert/python_api)( `tf.lite.TFLiteConverter`)와 결합하여 TensorFlow 모델을 TensorFlow Lite 형식으로 변환한 다음 추론을 실행할 수 있습니다.

```python
import numpy as np import tensorflow as tf  img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3)) const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.]) val = img + const out = tf.identity(val, name="out")  # Convert to TF Lite format with tf.Session() as sess:   converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])   tflite_model = converter.convert()  # Load the TFLite model and allocate tensors. interpreter = tf.lite.Interpreter(model_content=tflite_model) interpreter.allocate_tensors()  # Continue to get tensors and so forth, as shown above...
```

Python 샘플 코드는 [`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py)를 참조하세요.

팁: 인터프리터에 대한 자세한 설명을 보려면 Python 단말기에서 `help(tf.lite.Interpreter)`를 실행하세요.

## 지원되는 연산

TensorFlow Lite는 TensorFlow 연산의 일부를 지원하며 몇 가지 제한 사항이 있습니다. 연산 및 제한 사항의 전체 목록은 [TF Lite 연산 페이지](https://www.tensorflow.org/mlir/tfl_ops)를 참조하세요.

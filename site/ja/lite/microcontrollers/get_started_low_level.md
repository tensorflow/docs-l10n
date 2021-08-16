# マイクロコントローラを使ってみる

This document explains how to train a model and run inference using a microcontroller.

## Hello World の例

[Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) の例は、マイクロコントローラ向け TensorFlow Lite を使用するための基本を説明するためのものです。サイン関数を複製するモデルをトレーニングして実行します。つまり、単一の数値を入力として受け取り、その数値の[サイン](https://en.wikipedia.org/wiki/Sine)値を出力します。マイクロコントローラにデプロイされると、その予測は LED を点滅させるか、アニメーションを制御するために使用されます。

エンドツーエンドのワークフローには、次の手順が含まれます。

1. [モデルをトレーニングする](#train_a_model) (Python): デバイス上で使用するためにモデルをトレーニング、変換、最適化するための jupyter ノートブック。
2. [推論を実行する](#run_inference) (C++ 11): [C++ライブラリ](library.md)を使用してモデルで推論を実行するエンドツーエンドの単体テスト。

## Get a supported device

The example application we'll be using has been tested on the following devices:

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers) (using Arduino IDE)
- [SparkFun Edge](https://www.sparkfun.com/products/15170) (building directly from source)
- [STM32F746 Discovery kit](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) (using Mbed)
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400) (using Arduino IDE)
- [Adafruit TensorFlow Lite for Microcontrollers Kit](https://www.adafruit.com/product/4317) (using Arduino IDE)
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all) (Arduino IDE を使用する)
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview) (using ESP IDF)
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview) (using ESP IDF)

Learn more about supported platforms in [TensorFlow Lite for Microcontrollers](index.md).

## モデルをトレーニングする

Note: You can skip this section and use the trained model included in the example code.

Google Colaboratory を使用して、[独自のモデルをトレーニング](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb)します。 詳細については、`README.md`を参照してください。

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello World Training README.md</a>

## Run inference

To run the model on your device, we will walk through the instructions in the `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

以下のセクションは <em>Hello World</em> サンプルの <a><code>hello_world_test.cc</code></a>を見ていきます。 この単体テストでは、マイクロコントローラ向け TensorFlow Liteを使って推論を実行する方法を実演します。モデルを読み込み、推論を数回実行します。

### 1. Include the library headers

To use the TensorFlow Lite for Microcontrollers library, we must include the following header files:

```C++
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)モデルを実行するためにインタープリタが使用する演算を提供します。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_error_reporter.h)はデバッグ情報を出力します。
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h)にはモデルをロードして実行するためのコードが含まれています。
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h)には、TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/)デルファイル形式のスキーマが含まれています。
- [`version.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/version.h)は TensorFlow Lite スキーマのバージョン情報を提供します。

### 2. Include the model header

The TensorFlow Lite for Microcontrollers interpreter expects the model to be provided as a C++ array. The model is defined in `model.h` and `model.cc` files. The header is included with the following line:

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. Include the unit test framework header

In order to create a unit test, we include the TensorFlow Lite for Microcontrollers unit test framework by including the following line:

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

The test is defined using the following macros:

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

We now discuss the code included in the macro above.

### 4. Set up logging

To set up logging, a `tflite::ErrorReporter` pointer is created using a pointer to a `tflite::MicroErrorReporter` instance:

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

This variable will be passed into the interpreter, which allows it to write logs. Since microcontrollers often have a variety of mechanisms for logging, the implementation of `tflite::MicroErrorReporter` is designed to be customized for your particular device.

### 5. Load a model

In the following code, the model is instantiated using data from a `char` array, `g_model`, which is declared in `model.h`. We then check the model to ensure its schema version is compatible with the version we are using:

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. Instantiate operations resolver

[`AllOpsResolver`](github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h)インスタンスが宣言されています。これは、モデルで使用されている演算にアクセスするためにインタープリタが使います。

```C++
tflite::AllOpsResolver resolver;
```

The `AllOpsResolver` loads all of the operations available in TensorFlow Lite for Microcontrollers, which uses a lot of memory. Since a given model will only use a subset of these operations, it's recommended that real world applications load only the operations that are needed.

これは別のクラス、`MicroMutableOpResolver`を使用して実施されます。 *Micro speech* [`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc)の例で使い方を見ることができます。

### 7. Allocate memory

We need to preallocate a certain amount of memory for input, output, and intermediate arrays. This is provided as a `uint8_t` array of size `tensor_arena_size`:

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

The size required will depend on the model you are using, and may need to be determined by experimentation.

### 8. Instantiate interpreter

We create a `tflite::MicroInterpreter` instance, passing in the variables created earlier:

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. Allocate tensors

We tell the interpreter to allocate memory from the `tensor_arena` for the model's tensors:

```C++
interpreter.AllocateTensors();
```

### 10. Validate input shape

The `MicroInterpreter` instance can provide us with a pointer to the model's input tensor by calling `.input(0)`, where `0` represents the first (and only) input tensor:

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

We then inspect this tensor to confirm that its shape and type are what we are expecting:

```C++
// Make sure the input has the properties we expect
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
```

enum値`kTfLiteFloat32`は、TensorFlow Lite のデータ型のうちの一つへの参照であり、 [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h)で定義されています。

### 11. Provide an input value

To provide an input to the model, we set the contents of the input tensor, as follows:

```C++
input->data.f[0] = 0.;
```

In this case, we input a floating point value representing `0`.

### 12. Run the model

To run the model, we can call `Invoke()` on our `tflite::MicroInterpreter` instance:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

戻り値`TfLiteStatus`を確認でき、実行が成功したかどうか決定できます。`TfLiteStatus`の取りうる値は、[`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h)で定義されており、 `kTfLiteOk`と`kTfLiteError` です。

The following code asserts that the value is `kTfLiteOk`, meaning inference was successfully run.

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. Obtain the output

The model's output tensor can be obtained by calling `output(0)` on the `tflite::MicroInterpreter`, where `0` represents the first (and only) output tensor.

In the example, the model's output is a single floating point value contained within a 2D tensor:

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

We can read the value directly from the output tensor and assert that it is what we expect:

```C++
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. Run inference again

The remainder of the code runs inference several more times. In each instance, we assign a value to the input tensor, invoke the interpreter, and read the result from the output tensor:

```C++
input->data.f[0] = 1.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

input->data.f[0] = 3.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

input->data.f[0] = 5.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
```

### 15. Read the application code

この単体テストを一度ひととおり読み終えたら、[`main_functions.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/main_functions.cc)にあるサンプルアプリケーションのコードを理解できるはずです。 同じような処理を行いますが、実行された推論の数に基づいて入力値を生成し、それからデバイス固有の関数を呼び、モデルの出力をユーザーに表示します。

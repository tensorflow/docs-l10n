# 마이크로컨트롤러 시작하기

이 문서에서는 마이크로컨트롤러를 사용하여 모델을 훈련시키고 추론을 실행하는 방법을 설명합니다.

## Hello World 예제

The [Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) example is designed to demonstrate the absolute basics of using TensorFlow Lite for Microcontrollers. We train and run a model that replicates a sine function, i.e, it takes a single number as its input, and outputs the number's [sine](https://en.wikipedia.org/wiki/Sine) value. When deployed to the microcontroller, its predictions are used to either blink LEDs or control an animation.

The end-to-end workflow involves the following steps:

1. [모델](#train_a_model) 훈련 (Python에서): 기기에서 사용할 수 있도록 모델을 훈련, 변환 및 최적화하는 jupyter 노트북.
2. [Run inference](#run_inference) (in C++ 11): An end-to-end unit test that runs inference on the model using the [C++ library](library.md).

## 지원되는 기기 준비하기

The example application we'll be using has been tested on the following devices:

- [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers)(Arduino IDE 사용)
- [SparkFun Edge](https://www.sparkfun.com/products/15170)(소스에서 직접 빌드)
- [STM32F746 Discovery 키트](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)(Mbed 사용)
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400)(Arduino IDE 사용)
- [마이크로컨트롤러용 Adafruit TensorFlow Lite 키트](https://www.adafruit.com/product/4317)(Arduino IDE 사용)
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all) (using Arduino IDE)
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview)(ESP IDF 사용)
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview)(ESP IDF 사용)

[마이크로컨트롤러용 TensorFlow Lite](index.md)에서 지원되는 플랫폼에 대해 자세히 알아보세요.

## 모델 훈련

Note: You can skip this section and use the trained model included in the example code.

Use Google Colaboratory to [train your own model](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb). For more details, refer to the `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello World Training README.md</a>

## Run inference

To run the model on your device, we will walk through the instructions in the `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

The following sections walk through the example's [`hello_world_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc), unit test which demonstrates how to run inference using TensorFlow Lite for Microcontrollers. It loads the model and runs inference several times.

### 1. 라이브러리 헤더 포함

마이크로컨트롤러용 TensorFlow Lite 라이브러리를 사용하려면 다음 헤더 파일을 포함해야 합니다.

```C++
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h) provides the operations used by the interpreter to run the model.
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_error_reporter.h) outputs debug information.
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h) contains code to load and run models.
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h) contains the schema for the TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/) model file format.
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h) provides versioning information for the TensorFlow Lite schema.

### 2. 모델 헤더 포함

마이크로컨트롤러용 TensorFlow Lite 인터프리터는 모델이 C++ 배열로 제공될 것으로 예상합니다. 모델은 `model.h` 및 `model.cc` 파일에 정의되어 있습니다. 헤더는 다음 줄에 포함됩니다.

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. Include the unit test framework header

단위 테스트를 만들기 위해, 다음 줄을 넣어 마이크로컨트롤러용 TensorFlow Lite 단위 테스트 프레임워크를 포함합니다.

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

테스트는 다음 매크로를 사용하여 정의됩니다.

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

이제 위의 매크로에 포함된 코드에 대해 설명합니다.

### 4. 로깅 설정

로깅을 설정하기 위해 `tflite::MicroErrorReporter` 인스턴스에 대한 포인터를 사용하여 `tflite::ErrorReporter` 포인터를 생성합니다.

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

This variable will be passed into the interpreter, which allows it to write logs. Since microcontrollers often have a variety of mechanisms for logging, the implementation of `tflite::MicroErrorReporter` is designed to be customized for your particular device.

### 5. Load a model

다음 코드에서, 모델은 `model.h`에 선언된 `char` 배열인 `g_model`의 데이터를 사용하여 인스턴스화됩니다. 그런 다음 모델에서 스키마 버전이 사용 중인 버전과 호환되는지 확인합니다.

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

An [`AllOpsResolver`](github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h) instance is declared. This will be used by the interpreter to access the operations that are used by the model:

```C++
tflite::AllOpsResolver resolver;
```

`AllOpsResolver`는 마이크로컨트롤러용 TensorFlow Lite에서 사용할 수 있는 모든 연산을 로드하며, 여기에 많은 메모리가 사용됩니다. 특정 모델은 이러한 연산의 일부만 사용하므로 실제 애플리케이션에서는 필요한 연산만 로드하는 것이 좋습니다.

This is done using a different class, `MicroMutableOpResolver`. You can see how to use it in the *Micro speech* example's [`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc).

### 7. 메모리 할당

입력, 출력 및 중간 배열에 대해 일정량의 메모리를 미리 할당해야 합니다. 이 메모리는 `tensor_arena_size` 크기의 `uint8_t` 배열로 제공됩니다.

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

필요한 크기는 사용 중인 모델에 따라 다르며 실험을 통해 결정해야 할 수도 있습니다.

### 8. 인터프리터 인스턴스화

`tflite::MicroInterpreter` 인스턴스를 만들고 앞서 만든 변수를 전달합니다.

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. 텐서 할당

We tell the interpreter to allocate memory from the `tensor_arena` for the model's tensors:

```C++
interpreter.AllocateTensors();
```

### 10. Validate input shape

`MicroInterpreter` 인스턴스는 `.input(0)`을 호출하여 모델의 입력 텐서에 대한 포인터를 제공할 수 있습니다. 여기서 `0`은 첫 번째 (및 유일한) 입력 텐서를 나타냅니다.

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

The enum value `kTfLiteFloat32` is a reference to one of the TensorFlow Lite data types, and is defined in [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h).

### 11. 입력 값 제공

모델에 입력을 제공하기 위해 다음과 같이 입력 텐서의 내용을 설정합니다.

```C++
input->data.f[0] = 0.;
```

이 경우, `0`을 나타내는 부동 소수점 값을 입력합니다.

### 12. 모델 실행

모델을 실행하기 위해 `tflite::MicroInterpreter` 인스턴스에서 `Invoke()`를 호출할 수 있습니다.

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

We can check the return value, a `TfLiteStatus`, to determine if the run was successful. The possible values of `TfLiteStatus`, defined in [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h), are `kTfLiteOk` and `kTfLiteError`.

다음 코드에서 값이 `kTfLiteOk`인 것을 알수 있으며, 이는 추론이 성공적으로 실행되었음을 의미합니다.

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. Obtain the output

The model's output tensor can be obtained by calling `output(0)` on the `tflite::MicroInterpreter`, where `0` represents the first (and only) output tensor.

예제에서 모델의 출력은 2D 텐서에 포함된 단일 부동 소수점 값입니다.

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

### 14. 추론 다시 실행

나머지 코드는 추론을 여러 번 더 실행합니다. 각 인스턴스에서 입력 텐서에 값을 할당하고 인터프리터를 호출하고 출력 텐서에서 결과를 읽습니다.

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

### 15. 애플리케이션 코드 읽기

Once you have walked through this unit test, you should be able to understand the example's application code, located in [`main_functions.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/main_functions.cc). It follows a similar process, but generates an input value based on how many inferences have been run, and calls a device-specific function that displays the model's output to the user.

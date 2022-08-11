# 微控制器入门

本文介绍了如何使用微控制器训练模型并运行推断。

## Hello World 示例

[Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) 示例旨在演示 TensorFlow Lite for Microcontrollers 的最基础用法。我们会训练并运行一个复制正弦函数的模型，该模型以单个数字作为输入，并输出该数字的[正弦](https://en.wikipedia.org/wiki/Sine)值。部署到微控制器后，该模型的预测将用于使 LED 闪烁或控制动画。

端到端工作流包括以下步骤：

1. [训练模型](#train_a_model)（用 Python 编写）：Jupyter 笔记本，用于训练、转换和优化模型供设备端使用。
2. [运行推断](#run_inference)（用 C++ 11 编写）：端到端单元测试，使用 [C++ 库](library.md)在模型上运行推断。

## 获得支持的设备

我们将使用的示例应用已在以下设备上进行了测试：

- [Arduino Nano 33 BLE Sense](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense-with-headers) (using Arduino IDE)
- [SparkFun Edge](https://www.sparkfun.com/products/15170)（直接从源代码构建）
- [STM32F746 Discovery 套件](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)（使用 Mbed）
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400)（使用 Arduino IDE）
- [Adafruit TensorFlow Lite for Microcontrollers 套件](https://www.adafruit.com/product/4317)（使用 Arduino IDE）
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all)（使用 Arduino IDE）
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview)（使用 ESP IDF）
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview)（使用 ESP IDF）

请在 [TensorFlow Lite for Microcontrollers](index.md) 中了解有关所支持的平台的详细信息。

## 训练模型

注：您可以跳过本部分，使用示例代码中包含的训练好的模型。

请使用 Google Colab 来[训练您自己的模型](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb)。有关更多详细信息，请参考 `README.md`：

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/train/README.md">Hello World Training README.md</a>

## 运行推断

为了在您的设备上运行模型，我们将对 `README.md` 中的说明进行逐步介绍 ：

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

以下各部分逐步介绍了示例的 [`hello_world_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc)，这是一个演示如何使用 TensorFlow Lite for Microcontrollers 运行推断的单元测试。它会加载模型并多次运行推断。

### 1. 包括库头文件

要使用 TensorFlow Lite for Microcontrollers 库，我们必须包含以下头文件：

```C++
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h) 提供解释器用来运行模型的运算。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_error_reporter.h) 输出调试信息。
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h) 包含用于加载和运行模型的代码。
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h) 包含 TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/) 模型文件架构的模式。
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h) 提供 Tensorflow Lite 架构的版本控制信息。

### 2. 包含模型头文件

TensorFlow Lite for Microcontrollers 解释器希望以 C++ 数组的形式提供模型。模型在 `model.h` 和 `model.cc` 文件中进行定义。请使用下面这行代码来包括头文件：

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. 包含单元测试框架头文件

为了创建单元测试，我们通过包含下面这行代码来包括 TensorFlow Lite for Microcontrollers 单元测试框架：

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

该测试使用下面的宏来定义：

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

现在我们来讨论一下上面宏中包含的代码。

### 4. 设置日志记录

要设置日志记录，请使用指向 `tflite::MicroErrorReporter` 实例的指针来创建 `tflite::ErrorReporter` 指针：

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

此变量将被传递到解释器中，从而允许其写入日志。由于微控制器通常有多种日志记录机制，`tflite::MicroErrorReporter` 的实现旨在针对您的特定设备进行自定义。

### 5. 加载模型

下面的代码使用了 `model.h` 中声明的 `char` 数组和 `g_model` 中的数据实例化模型。然后，我们检查模型，以确保它的架构版本与我们正在使用的版本兼容：

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. 实例化运算解析器

声明了一个 [`AllOpsResolver`](github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/all_ops_resolver.h) 实例。解释器将使用它来访问模型所使用的运算：

```C++
tflite::AllOpsResolver resolver;
```

`AllOpsResolver` 会加载 TensorFlow Lite for Microcontrollers 中可用的所有运算，而这些运算会占用大量内存。由于给定的模型仅会用到这些运算中的一部分，因此建议在实际应用中仅加载所需的运算。

这是使用另一个类 `MicroMutableOpResolver` 来实现的。您可以在 *Micro speech* 示例的 [`micro_speech_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc) 中了解如何使用它。

### 分配内存

我们需要为输入、输出和中间数组预分配一定的内存。这由大小为 `tensor_arena_size` 的 `uint8_t` 数组提供：

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

所需的大小将取决于您使用的模型，可能需要通过实验来确定。

### 8. 实例化解释器

我们创建一个 `tflite::MicroInterpreter` 实例，并传入之前创建的变量：

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. 分配张量

我们告诉解释器从 `tensor_arena` 为模型的张量分配内存：

```C++
interpreter.AllocateTensors();
```

### 10. 验证输入形状

`MicroInterpreter` 实例可以通过调用 `.input(0)` 为我们提供指向模型输入张量的指针，其中 `0` 代表第一个（也是唯一的）输入张量：

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

然后，我们检查该张量以确认其形状和类型是否符合预期：

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

枚举值 `kTfLiteFloat32` 是对 TensorFlow Lite 其中一种数据类型的引用，并在 [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h) 中定义。

### 11. 提供输入值

为了给模型提供输入，我们设置输入张量的内容，如下所示：

```C++
input->data.f[0] = 0.;
```

在本例中，我们输入表示 `0` 的浮点值。

### 12. 运行模型

要运行模型，我们可以在 `tflite::MicroInterpreter` 实例上调用 `Invoke()`：

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

我们可以检查返回值 `TfLiteStatus`，以确定运行是否成功。在 <a><code>common.h</code></a> 中定义的 `TfLiteStatus` 的可能值为 `kTfLiteOk` 和 `kTfLiteError`。

以下代码断言该值为 `kTfLiteOk`，意味着推断已成功运行。

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. 获得输出

可以通过在 `tflite::MicroInterpreter` 上调用 `output(0)` 来获得模型的输出张量，其中 `0` 表示第一个（也是唯一的）输出张量。

在此例中，模型的输出是包含在 2D 张量中的单个浮点值：

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

我们可以直接从输出张量中读取该值，并断言这是我们期望的值：

```C++
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. 再次运行推断

代码的剩余部分又运行了几次推断。在每个实例中，我们都为输入张量分配一个值，调用解释器，并从输出张量中读取结果。

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

### 15. 阅读应用代码

完成此单元测试后，您应该能够理解位于 [`main_functions.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/main_functions.cc) 的示例应用代码。它遵循类似的过程，但会根据已运行推断的次数生成输入值，并调用特定于设备的函数，将模型的输出显示给用户。

# 理解 C++ 库

TensorFlow Lite for Microcontrollers C++ 库是 [TensorFlow 仓库](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro)的一部分。它具有可读性强、易于修改、测试良好、易于集成，并与常规的 TensorFlow Lite 兼容等特点。

以下文档概述了 C ++ 库的基本结构，并提供了有关创建自己项目的信息。

## 文件结构

[`micro`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro) 根目录的结构相对简单。但是，由于它位于内容丰富的 TensorFlow 仓库内部，我们创建了脚本和预生成的项目文件，在各种嵌入式开发环境中单独提供相关源文件。

### 关键文件

使用 TensorFlow Lite for Microcontrollers 解释器最重要的文件位于项目的根目录，并附带测试：

- [`all_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.h) 或 [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) 可以用来提供解释器运行模型时所使用的运算。由于 `all_ops_resolver.h` 会拉取每一个可用的运算，因此它会占用大量内存。在生产应用中，您应该仅使用 `micro_mutable_op_resolver.h` 拉取您的模型所需的运算。
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_error_reporter.h) 输出调试信息。
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h) 包含用于处理和运行模型的代码。

请参阅[微处理器入门](get_started_low_level.md)获取典型用法的演练。

构建系统提供了某些文件的特定于平台的实现。它们位于具有平台名称的目录中，例如 [`sparkfun_edge`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/sparkfun_edge)。

还有其他几个目录，包括：

- [`kernel`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels)，其中包含运算实现和相关代码。
- [`tools`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools)，其中包含构建工具及其输出。
- [`examples`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples)，其中包含示例代码。

## 开始新项目

我们建议使用 *Hello World* 示例作为新项目的模板。您可以按照本部分的说明获得一个适用于您所选择的平台的版本。

### 使用 Arduino 库

如果您使用的是 Arduino，则 *Hello World* 示例包含在 `Arduino_TensorFlowLite` 中。您可以从 Arduino IDE 和 [Arduino Create](https://create.arduino.cc/) 中下载 Arduino 库。

添加库后，请转到 `File -> Examples`。应该会在列表底部看到一个名为 `TensorFlowLite:hello_world` 的示例。选择它并点击 `hello_world` 来加载这个示例。然后，您可以保存该示例的副本，并将其用作自己项目的基础。

### 为其他平台生成项目

TensorFlow Lite for Microcontrollers 能够使用 `Makefile` 生成包含所有必要源文件的独立项目。目前支持的环境有 Keil、Make 和 Mbed。

要使用 Make 生成这些项目，请克隆 [TensorFlow 仓库{/ a0}，然后运行以下命令：](http://github.com/tensorflow/tensorflow)

```bash
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
```

这需要几分钟的时间，因为它要下载一些大型工具链来建立依赖关系。完成后，您应该会看到在类似 `tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/` 这样的路径（具体路径取决于您的主机操作系统）下创建了一些文件夹。这些文件夹包含了生成的项目和源文件。

运行该命令后，您将能够在 `tensorflow/ite/micro/tools/make/gen/linux_x86_64/prj/hello_world` 中找到 *Hello World* 项目。例如，`hello_world/keil` 将包含 Keil 项目。

## 写入新设备

要构建库并运行其所有的单元测试，请使用以下命令：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test
```

要运行一个单独的测试，请使用以下命令，将 `<test_name>` 替换为测试名称：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile test_<test_name>
```

您可以在项目的 Makefile 中找到测试名称。例如，`examples/hello_world/Makefile.inc` 指定了 *Hello World* 示例的测试名称。

## 构建二进制文件

要为一个给定的项目（例如一个示例应用）构建可运行的二进制文件，请使用以下命令，并将 `<project_name>` 替换为您要构建的项目：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile <project_name>_bin
```

例如，以下命令将为 *Hello World* 应用构建一个二进制文件：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
```

默认情况下，项目将针对主机操作系统进行编译。如需指定不同的目标架构，请使用 `TARGET=`。下面的示例展示了如何为 SparkFun Edge 构建 *Hello World* 示例：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge hello_world_bin
```

指定目标后，任何可用的特定于目标的源文件将被用来代替原始代码。例如，子目录 `examples/hello_world/sparkfun_edge` 中包含了文件 `constants.cc` 和 `output_handler.cc` 的 SparkFun Edge 实现，指定目标 `sparkfun_edge` 后，将使用这些文件。

您可以在项目的 Makefile 中找到项目名称。例如，`examples/hello_world/Makefile.inc` 指定了 *Hello World* 示例的二进制名称。

## 优化内核

`tensorflow/lite/micro/kernels` 根目录下的参考内核是用纯 C/C++ 实现的，并不包含特定于平台的硬件优化。

子目录中提供了内核的优化版本。例如，`kernels/cmsis-nn` 包含了几个使用 ARM 的 CMSIS-NN 库的优化内核。

要使用优化内核生成项目，请使用以下命令，将 `<subdirectory_name>` 替换为包含优化的子目录的名称：

```bash
make -f tensorflow/lite/micro/tools/make/Makefile TAGS=<subdirectory_name> generate_projects
```

您可以通过创建新的子文件夹来添加自己的优化。我们鼓励对新的优化实现进行拉取请求。

## 生成 Arduino 库

通过 Arduino IDE 的库管理器可以获得 Arduino 库的 Nightly 版本。

如果需要生成库的新版本，您可以从 TensorFlow 仓库中运行以下脚本：

```bash
./tensorflow/lite/micro/tools/ci_build/test_arduino.sh
```

生成的库可在 `tensorflow/lite/micro/tools/make/gen/arduino_x86_64/prj/tensorflow_lite.zip` 中找到。

## 移植到新设备

有关将 TensorFlow Lite for Microcontrollers 移植到新平台和设备的指南，可以在 [`micro/docs/new_platform_support.md`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md) 中找到。

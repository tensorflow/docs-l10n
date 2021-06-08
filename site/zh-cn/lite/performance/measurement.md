# 性能衡量

## 基准化分析工具

TensorFlow Lite 基准化分析工具目前可衡量和计算以下重要性能指标的统计信息：

- 初始化时间
- 预热状态的推断时间
- 稳定状态的推断时间
- 初始化期间的内存用量
- 整体内存用量

基准化分析工具的可用形式包括 Android 和 iOS 版基准化分析应用以及原生命令行二进制文件，它们共享相同的核心性能衡量逻辑。请注意，由于运行时环境的差异，可用选项和输出格式略有不同。

### Android 基准化分析应用

在 Android 上使用基准化分析工具有两种选项。一种是[原生基准化分析二进制文件](#native-benchmark-binary)，另一种是 Android 基准化分析应用，后者可以更好地衡量模型在应用中的性能。无论哪种方式，基准化分析工具测得的数字都会与在实际应用中运行模型推断略有不同。

此 Android 基准化分析应用不含界面。需要使用 `adb` 命令安装并运行该应用，并使用 `adb logcat` 命令检索结果。

#### 下载或构建应用

请使用以下链接下载每日预构建版 Android 基准化分析应用：

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk)

对于支持通过 [Flex 委托](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)执行 [TF 运算](https://www.tensorflow.org/lite/guide/ops_select)的 Android 基准化分析应用，请使用以下链接：

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex.apk)

您还可以按照以下[说明](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)以从源代码构建应用。

注：如果您想在 x86 CPU 或 Hexagon 委托上运行 Android 基准化分析 apk，或者您的模型包含[精选 TF 算子](../guide/ops_select)或[自定义算子](../guide/ops_custom)，则需要从源代码构建应用。

#### 准备基准化分析

在运行基准化分析应用之前，请按照以下方法安装应用并将模型文件推送至设备：

```shell
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```

#### 运行基准化分析

```shell
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/your_model.tflite \
              --num_threads=4"'
```

`graph` 为必要参数。

- `graph`：`string` <br> TFLite 模型文件的路径。

您可以为运行基准化分析指定更多可选参数。

- `num_threads`: `int`（默认值为 1）<br> 用于运行 TFLite 解释器的线程数。
- `use_gpu`: `bool`（默认值为 false）<br> 使用 [GPU 委托](gpu)。
- `use_nnapi`: `bool`（默认值为 false）<br> 使用 [NNAPI 委托](nnapi)。
- `use_xnnpack`: `bool`（默认值为 `false`）<br> 使用 [XNNPACK 委托](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack)。
- `use_hexagon`: `bool`（默认值为 `false`）<br> 使用 [Hexagon 委托](hexagon_delegate)。

根据所用的具体设备，其中某些选项可能不可用或无效。请参阅[参数](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters)以了解更多基准化分析应用可运行的性能参数。

使用 `logcat` 命令查看结果：

```shell
adb logcat | grep "Average inference"
```

将以如下形式报告基准化分析结果：

```
... tflite  : Average inference timings in us: Warmup: 91471, Init: 4108, Inference: 80660.1
```

### 原生基准化分析二进制文件

基准化分析工具也以原生二进制文件 `benchmark_model` 形式提供。您可以在 Linux、Mac、嵌入式设备和 Android 设备上通过 Shell 命令行执行此工具。

#### 下载或构建二进制文件

请使用以下链接下载每日预构建版原生命令行二进制文件：

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model)

对于支持通过 [Flex 委托](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex)执行 [TF 运算](https://www.tensorflow.org/lite/guide/ops_select)的每日预构建版二进制文件，请使用以下链接：

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_plus_flex)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)

您也可以在您的计算机上从[源代码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)构建原生基准化分析二进制文件。

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

要使用 Android NDK 工具链进行构建，您需要先按照本[指南](../guide/build_android#set_up_build_environment_without_docker)来设置构建环境，或者使用本[指南](../guide/build_android#set_up_build_environment_using_docker)中所述的 Docker 镜像。

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

注：直接在 Android 设备上推送并执行二进制文件进行基准化分析是一种有效方式，但与在实际 Android 应用中执行相比，可能会导致性能存在细微（但可观察到的）差异。尤其是，Android 的调度程序会根据线程和进程优先级来调整行为，而前台活动/应用和通过 `adb shell ...` 执行的常规后台二进制文件具有不同的优先级。对 TensorFlow Lite 启用多线程 CPU 执行时，这种行为调整最明显。因此，Android 基准化分析应用是性能衡量的首选工具。

#### 运行基准化分析

要在您的计算机上运行基准化分析，请从 Shell 执行二进制文件。

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

您可以对原生命令行二进制文件使用与上述相同的[参数](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters)集。

#### 分析模型运算

基准化分析模型二进制文件还支持分析模型运算以及获取每个算子的执行时间。为此，请在调用期间将标志 `--enable_op_profiling=true` 传递至 `benchmark_model`。[此处](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators)解释了详细信息。

### 使用原生基准化分析二进制文件在一次运行中分析多个性能选项

我们还提供了方便易用的 C++ 二进制文件，用于在一次运行中[基准化分析多个性能选项](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#benchmark-multiple-performance-options-in-a-single-run)。此二进制文件基于前文所述的每次只能基准化分析一个性能选项的基准化分析工具构建而成。它们采用相同的构建/安装/运行过程，但此二进制文件的构建目标名称为 `benchmark_model_performance_options`，并具有一些额外的参数。此二进制文件的一项重要参数为：

`perf_options_list`: `string`（默认值为 'all'）<br>以逗号分隔的用于基准化分析的 TFLite 性能选项列表。

您可以获取此工具的每日预构建版二进制文件，如下所示：

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_performance_options)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_performance_options)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_performance_options)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_performance_options)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)

### iOS 基准化分析应用

要在 iOS 设备上运行基准化分析，您需要从[源代码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)构建应用。将 TensorFlow Lite 模型文件放置到源代码树的 [benchmark_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/benchmark_data) 目录中，并修改 `benchmark_params.json` 文件。这些文件将打包到应用中，而应用将从目录中读取数据。请参阅 [iOS 基准化分析应用](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)以获取详细说明。

## 知名模型的性能基准化分析

本部分列出了在某些 Android 和 iOS 设备上运行知名模型时的 TensorFlow Lite 性能基准化分析。

### Android 性能基准化分析

这些性能基准化分析数据是使用[原生基准化分析二进制文件](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)生成的。

对于 Android 基准化分析，将 CPU 亲和性设置为使用设备的大核以减少偏差（请参阅[详细信息](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android)）。

假定模型已下载并解压到 `/data/local/tmp/tflite_models` 目录。使用[这些说明](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android)构建基准化分析二进制文件，假定其位于 `/data/local/tmp` 目录中。

要运行基准化分析，请执行以下操作：

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

要使用 NNAPI 委托运行，请设置 `--use_nnapi=true`。要使用 GPU 委托运行，请设置 `--use_gpu=true`。

以下性能值是在 Android 10 上测得。

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>设备</th>
      <th>CPU, 4 threads</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>     </td>
    <td>Pixel 3</td>
    <td>23.9 毫秒</td>
    <td>6.45 毫秒</td>
    <td>13.8 ms</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>14.0 毫秒</td>
    <td>9.0 毫秒</td>
    <td>14.8 毫秒</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>     </td>
    <td>Pixel 3</td>
    <td>13.4 ms</td>
    <td>---</td>
    <td>6.0 毫秒</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>5.0 毫秒</td>
    <td>---</td>
    <td>3.2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>     </td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>---</td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>34.5 毫秒</td>
    <td>---</td>
    <td>99.0 毫秒</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>     </td>
    <td>Pixel 3</td>
    <td>35.8 毫秒</td>
    <td>9.5 毫秒</td>
    <td>18.5 ms</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>23.9 毫秒</td>
    <td>11.1 ms</td>
    <td>19.0 毫秒</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>     </td>
    <td>Pixel 3</td>
    <td>422 ms</td>
    <td>99.8 毫秒</td>
    <td>201 ms</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>272.6 毫秒</td>
    <td>87.2 毫秒</td>
    <td>171.1 毫秒</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>     </td>
    <td>Pixel 3</td>
    <td>486 ms</td>
    <td>93 毫秒</td>
    <td>292 毫秒</td>
  </tr>
   <tr>
     <td>Pixel 4 </td>
    <td>324.1 毫秒</td>
    <td>97.6 毫秒</td>
    <td>186.9 毫秒</td>
  </tr>
 </table>

### iOS 性能基准化分析

这些性能基准化分析数据是使用 [iOS 基准化分析应用](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios)生成的。

为了运行 iOS 基准化分析，我们修改了基准化分析应用以包含适当的模型，并修改了 `benchmark_params.json` 以将 `num_threads` 设置为 2。为了使用 GPU 委托，我们还对 `benchmark_params.json` 添加了 `"use_gpu" : "1"` 和 `"gpu_wait_type" : "aggressive"` 选项。

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>设备</th>
      <th>CPU，2 线程</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>     </td>
    <td>iPhone XS</td>
    <td>14.8 毫秒</td>
    <td>3.4 毫秒</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>     </td>
    <td>iPhone XS</td>
    <td>11 毫秒</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>     </td>
    <td>iPhone XS</td>
    <td>30.4 毫秒</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz" class="_active_edit_href">SqueezeNet</a>     </td>
    <td>iPhone XS</td>
    <td>21.1 毫秒</td>
    <td>15.5 毫秒</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz" class="_active_edit_href">Inception_ResNet_V2</a>     </td>
    <td>iPhone XS</td>
    <td>261.1 毫秒</td>
    <td>45.7 毫秒</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz" class="_active_edit_href">Inception_V4</a>     </td>
    <td>iPhone XS</td>
    <td>309 毫秒</td>
    <td>54.4 毫秒</td>
  </tr>
 </table>

## 在 Android 中跟踪 TensorFlow Lite 内部事件

注：此功能为实验性功能，仅在使用每日构建版 Tensorflow Lite 库构建 Android 应用时可用。v2.3 及之前的稳定版库不支持此功能。

Android 应用的 TensorFlow Lite 解释器中的内部事件可通过 [Android 跟踪工具](https://developer.android.com/topic/performance/tracing)进行捕获。它与 Android [Trace](https://developer.android.com/reference/android/os/Trace) API 的事件相同，因此从 Java/Kotlin 代码捕获的事件可与 TensorFlow Lite 内部事件一起查看。

事件的一些示例包括：

- 算子调用
- 由委托执行的计算图修改
- 张量分配

在捕获跟踪的各种选项中，本指南将介绍 Android Studio CPU 性能剖析器和系统跟踪应用。请参阅 [Perfetto 命令行工具](https://developer.android.com/studio/command-line/perfetto)或 [Systrace 命令行工具](https://developer.android.com/topic/performance/tracing/command-line)以了解其他选项。

### 在 Java 代码中添加跟踪事件

以下为[图像分类](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)示例应用的代码段。TensorFlow Lite 解释器在 `recognizeImage/runInference` 部分中运行。此步骤为可选步骤，但有助于通知在何处调用了推断。

```java
  Trace.beginSection("recognizeImage");
  ...
  // Runs the inference call.
  Trace.beginSection("runInference");
  tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
  Trace.endSection();
  ...
  Trace.endSection();

```

### 启用 TensorFlow Lite 跟踪

要启用 TensorFlow Lite 跟踪，请在启动 Android 应用之前将 Android 系统属性 `debug.tflite.tracing` 设置为 1.

```shell
adb shell setprop debug.tflite.trace 1
```

如果在初始化 TensorFlow Lite 解释器时已设置了此属性，则将跟踪解释器中的关键事件（例如，算子调用）。

捕获所有跟踪后，可将属性值设置为 0 以停用跟踪。

```shell
adb shell setprop debug.tflite.trace 0
```

### Android Studio CPU 性能剖析器

请遵循以下步骤来使用 [Android Studio CPU 性能剖析器](https://developer.android.com/studio/profile/cpu-profiler)捕获跟踪：

1. 从顶部菜单中选择 **Run &gt; Profile 'app'**。

2. 在 Profiler 窗口出现后，点击 CPU 时间轴上的任意位置。

3. 在 CPU 性能剖析模式中选择“Trace System Calls”。

    ![选择“Trace System Calls”](images/as_select_profiling_mode.png)

4. 按“Record”按钮。

5. 按“Stop”按钮。

6. 调查跟踪结果。

    ![Android Studio 跟踪](images/as_traces.png)

在本例中，您可以查看线程中事件的层次结构以及每个算子时间的统计信息，还可以查看整个应用在各个线程之间的数据流。

### 系统跟踪应用

按照[系统跟踪应用](https://developer.android.com/topic/performance/tracing/on-device)中详细介绍的步骤，在不使用 Android Studio 的情况下捕获跟踪。

在本例中，我们捕获了相同的 TFLite 事件并根据 Android 设备版本将其保存为 Perfetto 或 Systrace 格式。捕获的跟踪文件可在 [Perfetto 界面](https://ui.perfetto.dev/#!/)中打开。

![Perfetto 跟踪](images/perfetto_traces.png)

### 使用跟踪数据

跟踪数据可用于确定性能瓶颈。

以下是性能剖析器可为您提供的数据分析示例，以及可以提高性能的潜在解决方案：

- 如果可用 CPU 核心数少于推断线程数，则 CPU 调度开销可能会导致性能低于标准水平。您可以在应用中重新调度其他 CPU 占用率较高的任务以避免与模型推断重叠，或者调整解释器线程数。
- 如果算子未被完全委托，则模型计算图的某些部分将在 CPU 而非预期的硬件加速器上执行。您可以将不支持的算子替换为类似的受支持算子。

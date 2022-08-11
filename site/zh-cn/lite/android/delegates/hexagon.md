# TensorFlow Lite Hexagon 委托

本文档介绍如何使用 Java 和/或 C API 在您的应用中利用 TensorFlow Lite Hexagon 委托。该委托利用 Qualcomm Hexagon 库在 DSP 上执行量化内核。请注意，该委托旨在*补充* NNAPI 功能，特别适用于 NNAPI DSP 加速不可用的设备（例如，在旧设备上，或者在没有 DSP NNAPI 驱动程序的设备上）。

注：此委托目前处于实验 (Beta) 阶段。

**支持的设备：**

目前支持以下 Hexagon 架构，包括但不限于：

- Hexagon 680
    - SoC 示例：Snapdragon 821、820、660
- Hexagon 682
    - SoC 示例：Snapdragon 835
- Hexagon 685
    - SoC 示例：Snapdragon 845、Snapdragon 710、QCS605、QCS603
- Hexagon 690
    - SoC 示例：Snapdragon 855、QCS610、QCS410、RB5

**支持的模型：**

Hexagon 委托支持符合我们的 [8 位对称量化规范](https://www.tensorflow.org/lite/performance/quantization_spec)的所有模型，包括使用[训练后整数量化](https://www.tensorflow.org/lite/performance/post_training_integer_quant)生成的模型。使用旧[量化感知训练](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)路径训练的 UInt8 模型也受支持，例如，我们的“托管模型”页面上的[这些量化版本](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models)。

## Hexagon 委托 Java API

```java
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

### 示例用法

#### 第 1 步. 编辑 app/build.gradle 以使用 Nightly 版本 Hexagon 委托 AAR

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### 第 2 步. 将 Hexagon 库添加到 Android 应用

- 下载并运行 hexagon_nn_skel.run。它应该会提供 3 个不同的共享库：“libhexagon_nn_skel.so”、“libhexagon_nn_skel_v65.so”、“libhexagon_nn_skel_v66.so”
    - v1.10.3
    - v1.14
    - v1.17
    - v1.20
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注：您需要接受许可协议。

注：从 2021 年 2 月 23 日起，您应当使用 v1.20.0.1。

注：必须将 hexagon_nn 库与兼容版本的接口库一起使用。接口库是 AAR 的一部分，由 Bazel 通过 [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl) 获取。Bazel 配置中的版本就是您应该使用的版本。

- 在应用中包含所有 3 个共享库和其他共享库。请参阅[如何将共享库添加到应用](#how-to-add-shared-library-to-your-app)。委托会根据设备自动选择性能最佳的共享库。

注：如果您要同时为 32 位和 64 位 ARM 设备构建应用，则需要将 Hexagon 共享库同时添加到 32 位和 64 位 lib 文件夹中。

#### 第 3 步. 创建委托并初始化 TensorFlow Lite 解释器

```java
import org.tensorflow.lite.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## Hexagon 委托 C API

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

### 示例用法

#### 第 1 步. 编辑 app/build.gradle 以使用 Nightly 版本 Hexagon 委托 AAR

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### 第 2 步. 将 Hexagon 库添加到 Android 应用

- 下载并运行 hexagon_nn_skel.run。它应该会提供 3 个不同的共享库：“libhexagon_nn_skel.so”、“libhexagon_nn_skel_v65.so”、“libhexagon_nn_skel_v66.so”
    - v1.10.3
    - v1.14
    - v1.17
    - v1.20
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注：您需要接受许可协议。

注：从 2021 年 2 月 23 日起，您应当使用 v1.20.0.1。

注：必须将 hexagon_nn 库与兼容版本的接口库一起使用。接口库是 AAR 的一部分，由 Bazel 通过 [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl) 获取。Bazel 配置中的版本就是您应该使用的版本。

- 在应用中包含所有 3 个共享库和其他共享库。请参阅[如何将共享库添加到应用](#how-to-add-shared-library-to-your-app)。委托会根据设备自动选择性能最佳的共享库。

注：如果您要同时为 32 位和 64 位 ARM 设备构建应用，则需要将 Hexagon 共享库同时添加到 32 位和 64 位 lib 文件夹中。

#### 第 3 步. 包含 C 头

- 头文件“hexagon_delegate.h”既可以从 [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h) 下载，也可以从 Hexagon 委托 AAR 提取。

#### 第 4 步. 创建委托并初始化 TensorFlow Lite 解释器

- 在您的代码中，确保已加载原生 Hexagon 库。在您的操作组件或 Java 入口点中调用 `System.loadLibrary("tensorflowlite_hexagon_jni");` <br>即可加载此库。

- 创建委托，示例：

```c
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Assuming shared libraries are under "/data/local/tmp/"
// If files are packaged with native lib in android App then it
// will typically be equivalent to the path provided by
// "getContext().getApplicationInfo().nativeLibraryDir"
const char[] library_directory_path = "/data/local/tmp/";
TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
::tflite::TfLiteHexagonDelegateOptions params = {0};
// 'delegate_ptr' Need to outlive the interpreter. For example,
// If use case will need to resize input or anything that can trigger
// re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
    ::tflite::TfLiteHexagonDelegateDelete(delegate);
  });
interpreter->ModifyGraphWithDelegate(delegate.get());
// After usage of delegate.
TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

## 将共享库添加到应用

- 创建文件夹“app/src/main/jniLibs”，并为每个目标架构创建一个目录。例如：
    - ARM 64 位：`app/src/main/jniLibs/arm64-v8a`
    - ARM 32 位：`app/src/main/jniLibs/armeabi-v7a`
- 将 .so 文件放在与架构相符的目录中。

注：如果您使用 App Bundle 发布应用，可能需要在 gradle.properties 文件中设置 android.bundle.enableUncompressedNativeLibs=false。

## 反馈

如有问题，请创建 [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) 议题，并提供重现问题所需的所有必要详情，包括使用的手机型号和主板（`adb shell getprop ro.product.device` 和 `adb shell getprop ro.board.platform`）。

## 常见问题解答

- 委托支持哪些运算？
    - 请参见[支持的运算和约束](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md)的最新列表
- 启用委托时，如何确定模型是否使用 DSP？
    - 启用委托时会打印两条消息：其中一条指示是否创建了委托，另一条指示有多少节点使用委托运行。<br> `Created TensorFlow Lite delegate for Hexagon.` <br> `Hexagon delegate: X nodes delegated out of Y nodes.`
- 是否需要支持模型中的所有运算才能运行委托？
    - 不需要，模型会根据支持的运算分配到子计算图中。任何不受支持的运算都将在 CPU 上运行。
- 如何从源代码构建 Hexagon 委托 AAR？
    - 使用 `bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/hexagon/java:tensorflow-lite-hexagon`。
- 为何我的 Android 设备具有受支持的 SoC，但 Hexagon 委托无法初始化？
    - 验证您的设备是否确实具有受支持的 SoC。请运行 `adb shell cat /proc/cpuinfo | grep Hardware`，并查看返回的结果是否类似于 “Hardware : Qualcomm Technologies, Inc MSMXXXX”。
    - 某些手机制造商可能会为相同的手机型号使用不同的 SoC。因此，对于某些手机型号，Hexagon 委托可能只能在部分设备上正常运行，不一定可以在所有设备上正常运行。
    - 某些手机制造商会特意限制从非系统 Android 应用使用 Hexagon DSP，从而导致 Hexagon 委托无法正常运行。
- 我的手机已锁定 DSP 访问。我已启用手机的 root 权限，但仍然无法运行委托，应该怎么办？
    - 确保通过运行 `adb shell setenforce 0` 来停用 SELinux 强制访问

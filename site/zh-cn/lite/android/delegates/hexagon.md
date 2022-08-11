# TensorFlow Lite Hexagon 委托

This document explains how to use the TensorFlow Lite Hexagon Delegate in your application using the Java and/or C API. The delegate leverages the Qualcomm Hexagon library to execute quantized kernels on the DSP. Note that the delegate is intended to *complement* NNAPI functionality, particularly for devices where NNAPI DSP acceleration is unavailable (e.g., on older devices, or devices that don’t yet have a DSP NNAPI driver).

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

The Hexagon delegate supports all models that conform to our [8-bit symmetric quantization spec](https://www.tensorflow.org/lite/performance/quantization_spec), including those generated using [post-training integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant). UInt8 models trained with the legacy [quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize) path are also supported, for e.g., [these quantized versions](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models) on our Hosted Models page.

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
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注：您需要接受许可协议。

注：从 2021 年 2 月 23 日起，您应当使用 v1.20.0.1。

Note: You must use the hexagon_nn libraries with the compatible version of interface library. Interface library is part of the AAR and fetched by bazel through the [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl) The version in the bazel config is the version you should use.

- Include all 3 in your app with other shared libraries. See [How to add shared library to your app](#how-to-add-shared-library-to-your-app). The delegate will automatically pick the one with best performance depending on the device.

Note: If your app will be built for both 32 and 64-bit ARM devices, then you will need to add the Hexagon shared libs to both 32 and 64-bit lib folders.

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
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

注：您需要接受许可协议。

注：从 2021 年 2 月 23 日起，您应当使用 v1.20.0.1。

Note: You must use the hexagon_nn libraries with the compatible version of interface library. Interface library is part of the AAR and fetched by bazel through the [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl). The version in the bazel config is the version you should use.

- 在应用中包含所有 3 个共享库和其他共享库。请参阅[如何将共享库添加到应用](#how-to-add-shared-library-to-your-app)。委托会根据设备自动选择性能最佳的共享库。

Note: If your app will be built for both 32 and 64-bit ARM devices, then you will need to add the Hexagon shared libs to both 32 and 64-bit lib folders.

#### 第 3 步. 包含 C 头

- The header file "hexagon_delegate.h" can be downloaded from [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h) or extracted from the Hexagon delegate AAR.

#### 第 4 步. 创建委托并初始化 TensorFlow Lite 解释器

- In your code, ensure the native Hexagon library is loaded. This can be done by calling `System.loadLibrary("tensorflowlite_hexagon_jni");`
     in your Activity or Java entry-point.

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

Note: If you're using App Bundle for publishing your Application, you might want to set android.bundle.enableUncompressedNativeLibs=false in the gradle.properties file.

## 反馈

For issues, please create a [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) issue with all the necessary repro details, including the phone model and board used (`adb shell getprop ro.product.device` and `adb shell getprop ro.board.platform`).

## 常见问题解答

- 委托支持哪些运算？
    - See the current list of [supported ops and constraints](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md)
- 启用委托时，如何确定模型是否使用 DSP？
    - Two log messages will be printed when you enable the delegate - one to indicate if the delegate was created and another to indicate how many nodes are running using the delegate.
         `Created TensorFlow Lite delegate for Hexagon.`
         `Hexagon delegate: X nodes delegated out of Y nodes.`
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

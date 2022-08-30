# Google Play 服务中的 TensorFlow Lite（Beta 版）

Beta 版：此页面中描述的功能是 Beta 版。此处介绍的功能和 API 可能会在未来版本中发生变化。

TensorFlow Lite is available in the Google Play services API as a public beta on all Android devices running the current version of Play services. This API lets you run machine learning (ML) models without statically bundling TensorFlow Lite libraries into your app, allowing you to:

- 缩减应用大小
- 从最新的 TensorFlow Lite 稳定版中获得更高的性能

TensorFlow Lite in Google Play services is available through the [TensorFlow Lite Task API](../api_docs/java/org/tensorflow/lite/task/core/package-summary) and [TensorFlow Lite Interpreter API](../api_docs/java/org/tensorflow/lite/InterpreterApi). The Task Library provides optimized out-of-box model interfaces for common machine learning tasks using visual, audio, and text data. The TensorFlow Lite Interpreter API, provided by the TensorFlow runtime and support libraries, provides a more general-purpose interface for building and running ML models.

本页面简要概述了如何在 Android 应用中使用新的 Google Play 服务 API 中的 TensorFlow Lite。

有关 Google Play 服务的更多信息，请参阅 [Google Play 服务](https://developers.google.com/android/guides/overview)网站。

<aside class="note"> <b>Terms:</b> By accessing or using TensorFlow Lite in Google Play services, you agree to the <a href="#tos">Terms of Service</a>. Please read and understand all applicable terms and policies before accessing the APIs. </aside>

## 将 TensorFlow Lite 添加到您的应用

You can use the TensorFlow Lite in Google Play services API by making a few changes to your app module dependencies, initializing the new API, and using a specific class as your interpreter object.

The following instructions provide more details on how to implement the Interpreter and Task Library APIs in Google Play services. While it is possible for an app to use both the Interpreter APIs and Task Library APIs, most apps should only use one set of APIs.

Note: If you are already using TensorFlow Lite in your app, you should review the [Migrating from stand-alone TensorFlow Lite](#migrating) section.

### Example app with Task Library

You can review and test an example implementation of TensorFlow Lite in Google Play services in the [example app](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services). This example app uses the Task Library within Play services to create an object detection app.

### Using the Task Library APIs

The TensorFlow Lite Task API wraps the Interpreter API and provides a high-level programming interface for common machine learning tasks that use visual, audio, and text data. You should use the Task API if your application requires one of the [supported tasks](../api_docs/java/org/tensorflow/lite/inference_with_metadata/task_library/overview#supported_tasks).

#### 1. 添加项目依赖项

Your project dependency depends on your machine learning use case. The Task APIs contain the following libraries:

- Vision library: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- Audio library: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- Text library: `org.tensorflow:tensorflow-lite-task-text-play-services`

Add one of the dependencies to your app project code to access the Play Services API for TensorFlow Lite. For example, to implement a vision task:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2-beta03'
...
}
```

#### 2. 添加 TensorFlow Lite 的初始化

Initialize the TensorFlow Lite component of the Google Play services API *before* using the TensorFlow Lite APIs. The following example initializes the vision library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">init {
  TfLiteVision.initialize(context)
    }
  }
</pre>
    </section>
  </devsite-selector>
</div>

Important: Make sure the `TfLite.initialize` task completes before executing code that accesses TensorFlow Lite APIs.

Tip: The TensorFlow Lite modules are installed at the same time your application is installed or updated from the Play Store. You can check the availability of the modules by using `ModuleInstallClient` from the Google Play services APIs. For more information on checking module availability, see [Ensuring API availability with ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Run inferences

After initializing the TensorFlow Lite component, call the `detect()` method to generate inferences. The exact code within the `detect()` method varies depending on the library and use case. The following is for a simple object detection use case with the `TfLiteVision` library:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">fun detect(...) {
  if (!TfLiteVision.isInitialized()) {
    Log.e(TAG, "detect: TfLiteVision is not initialized yet")
    return
  }

  if (objectDetector == null) {
    setupObjectDetector()
  }

  ...

}
</pre>
    </section>
  </devsite-selector>
</div>

Depending on the data format, you may also need to preprocess and convert your data within the `detect()` method before generating inferences. For example, image data for an object detector requires the following:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Using the Interpreter APIs

The Interpreter APIs offer more control and flexibility than the Task Library APIs. You should use the Interpreter APIs if your machine learning task is not supported by the Task library, or if you require a more general-purpose interface for building and running ML models.

#### 1. Add project dependencies

Add the following dependencies to your app project code to access the Play Services API for TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.0-beta03'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.0-beta03'
...
}
```

#### 2. Add initialization of TensorFlow Lite

在使用 TensorFlow Lite API *之前*，先初始化 Google Play 服务 API 的 TensorFlow Lite 组件：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">val initializeTask: Task&lt;Void&gt; by lazy { TfLite.initialize(this) }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">Task&lt;Void&gt; initializeTask = TfLite.initialize(context);
</pre>
    </section>
  </devsite-selector>
</div>

注：请确保在执行访问 TensorFlow Lite API 的代码之前完成 `TfLite.initialize` 任务。使用 `addOnSuccessListener()` 方法，如下一部分所示。

#### 3. Create an Interpreter and set runtime option {:#step_3_interpreter}

使用 `InterpreterApi.create()` 创建解释器，并通过调用 `InterpreterApi.Options.setRuntime()` 将其配置为使用 Google Play 服务运行时，如以下示例代码所示：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private lateinit var interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener {
  val interpreterOption =
    InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  interpreter = InterpreterApi.create(
    modelBuffer,
    interpreterOption
  )}
  .addOnFailureListener { e -&gt;
    Log.e("Interpreter", "Cannot initialize interpreter", e)
  }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private InterpreterApi interpreter;
...
initializeTask.addOnSuccessListener(a -&gt; {
    interpreter = InterpreterApi.create(modelBuffer,
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY));
  })
  .addOnFailureListener(e -&gt; {
    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
          e.getMessage()));
  });
</pre>
    </section>
  </devsite-selector>
</div>

您应该使用上面的实现，因为它能够避免阻塞 Android 界面线程。如果您需要更紧密地管理线程执行，可以添加一个 `Tasks.await()` 调用来创建解释器：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">import androidx.lifecycle.lifecycleScope
...
lifecycleScope.launchWhenStarted { // uses coroutine
  initializeTask.await()
}
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">@BackgroundThread
InterpreterApi initializeInterpreter() {
    Tasks.await(initializeTask);
    return InterpreterApi.create(...);
}
</pre>
    </section>
  </devsite-selector>
</div>

警告：请勿在前台界面线程上调用 `.await()`，因为这会中断界面元素的显示并产生不好的用户体验。

#### 4. 运行推断

使用您创建的 `interpreter` 对象，调用 `run()` 方法来生成推断。

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer)
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">interpreter.run(inputBuffer, outputBuffer);
</pre>
    </section>
  </devsite-selector>
</div>

## Hardware acceleration {:#hardware-acceleration}

TensorFlow Lite allows you to accelerate the performance of your model using specialized hardware processors, such as graphics processing units (GPUs). You can take advantage of these specialized processors using hardware drivers called [*delegates*](https://www.tensorflow.org/lite/performance/delegates). You can use the following hardware acceleration delegates with TensorFlow Lite in Google Play Services:

- *[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) (recommended)* - This delegate is provided through Google Play services and is dynamically loaded, just like the Play services versions of the Task API and Interpreter API.

- [*NNAPI delegate*](https://www.tensorflow.org/lite/android/delegates/nnapi) - This delegate is available as an included library dependency in your Android development project, and is bundled into your app.

For more information about hardware acceleration with TensorFlow Lite, see the [TensorFlow Lite Delegates](https://www.tensorflow.org/lite/performance/delegates) page.

### GPU with Task Library APIs

To use the GPU delegate with the Task APIs:

1. Update the project dependencies to use the GPU delegate from Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0-beta03'
    ```

2. Initialize the GPU delegate with `setEnableGpuDelegateSupport`. For example, you can initialize the GPU delegate for `TfLiteVision` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Enable the GPU delegate option with [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          val baseOptions = BaseOptions.builder().useGpu().build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
            </pre>
    </section>
    </devsite-selector>
    </div>

4. Configure the options using `.setBaseOptions`. For example, you can set up GPU in `ObjectDetector` with the following:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        ObjectDetectorOptions options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build();
            </pre>
    </section>
    </devsite-selector>
    </div>

### GPU with Interpreter APIs

To use the GPU delegate with the Interpreter APIs:

1. Update the project dependencies to use the GPU delegate from Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0-beta03'
    ```

2. Enable the GPU delegate option in the TFlite initialization:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          TfLite.initialize(this,
                TfLiteInitializationOptions.builder()
                 .setEnableGpuDelegateSupport(true)
                 .build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          TfLite.initialize(this,
                TfLiteInitializationOptions.builder()
                 .setEnableGpuDelegateSupport(true)
                 .build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Set GPU delegate in interpreter options to use `DelegateFactory` by calling `addDelegateFactory()` within `InterpreterApi.Options()`:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">          val interpreterOption = InterpreterApi.Options()
               .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
               .addDelegateFactory(GpuDelegateFactory())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">          Options interpreterOption = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                .addDelegateFactory(new GpuDelegateFactory());
            </pre>
    </section>
    </devsite-selector>
    </div>

## Migrating from stand-alone TensorFlow Lite {:#migrating}

If you are planning to migrate your app from stand-alone TensorFlow Lite to the Play services API, review the following additional guidance for updating your app project code:

1. 请查看此页面的[限制](#limitations)部分，以确保您的用例受支持。
2. 在更新代码之前，请对模型进行性能和准确率检查，特别是如果您使用的是 2.1 之前版本的 TensorFlow Lite，这样您就有一个可以用来与新实现进行比较的基准。
3. 如果您已迁移所有代码以使用适用于 TensorFlow Lite 的 Play 服务 API，则应从您的 build.gradle 文件中移除现有的 TensorFlow Lite *runtime library* 依赖项（带有 <code>org.tensorflow:**tensorflow-lite**:*</code> 的条目），以便缩减应用大小。
4. Identify all occurrences of `new Interpreter` object creation in your code, and modify it so that it uses the InterpreterApi.create() call. This new API is asynchronous, which means in most cases it's not a drop-in replacement, and you must register a listener for when the call completes. Refer to the code snippet in [Step 3](#step_3_interpreter) code.
5. 将 `import org.tensorflow.lite.InterpreterApi;` 和 `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` 添加到任何使用 `org.tensorflow.lite.Interpreter` 或 `org.tensorflow.lite.InterpreterApi` 类的源文件中。
6. 如果对 `InterpreterApi.create()` 的任何结果调用只有一个参数，请将 `new InterpreterApi.Options()` 追加到参数列表。
7. 将 `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` 追加到对 `InterpreterApi.create()` 的任何调用的最后一个参数。
8. 将 `org.tensorflow.lite.Interpreter` 类的其他匹配项替换为 `org.tensorflow.lite.InterpreterApi`。

If you want to use stand-alone TensorFlow Lite and the Play services API side-by-side, you must use TensorFlow Lite 2.9 (or later). TensorFlow Lite 2.8 and earlier versions are not compatible with the Play services API version.

## Known issues

在 Google Play 服务中实现 TensorFlow Lite 后，请确保测试您的应用并尝试使用您应用的机器学习模型功能。如果您遇到无法解决的错误或问题，请使用下面的[支持和反馈](#support)部分中列出的渠道进行报告。

### LoadingException：没有可接受的模块

The `LoadingException` error occurs when an app fails to initialize either the TFlite or GPU module. This error can occur if you are not part of the [tflite-play-services-beta-access](https://groups.google.com/g/tflite-play-services-beta-access/about) group, or when a device is not support GPU hardware acceleration.

#### TFlite module

在 Beta 版发布期间通过开发环境测试您的应用时，当您的应用尝试初始化 TensorFlow Lite 类 (`TfLite.intialize(context)`) 时，可能会出现异常：

```
com.google.android.gms.dynamite.DynamiteModule$LoadingException:
  No acceptable module com.google.android.gms.tflite_dynamite found.
  Local version is 0 and remote version is 0.
```

This error means that the TensorFlow Lite in Google Play services API is not yet available on your test device. You can resolve this exception by joining this Google group [tflite-play-services-beta-access](https://groups.google.com/g/tflite-play-services-beta-access/about) with *the user account you are using to test on your device.* Once you have been added to the beta access group, this exception should be resolved. If you are already part of the beta access group, try restarting the app.

请在您加入此群组后至少留出一个工作日，以便授予访问权限并清除错误。如果您继续遇到此错误，请使用下面的[支持和反馈](#support)部分中概述的渠道进行报告。

注：只有在开发环境中测试此 API 时才会出现该错误。使用此 API 并通过 Google Play 商店在设备上安装或更新的应用会自动接收所需的库。

#### GPU module

Not all devices support hardware acceleration for TFLite. In order to mitigate errors and potential crashes the download of the Play Services GPU module is allowlisted only for supported devices.

When initializing TFLite with the GPU delegate, also initialize TFLite with CPU as a fallback. This allows the app to use CPU with devices that do not support GPU.

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">// Instantiate the initialization task with GPU delegate
private var useGpu = true
private val initializeTask: Task&lt;Void&gt; by lazy {
   TfLite.initialize(this,
       TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
       // Add a TFLite initialization task with CPU as a failsafe
       .continueWithTask { task: Task&lt;Void&gt; -&gt;
         if (task.exception != null) {
           useGpu = false
           return@continueWithTask TfLite.initialize(this)
         } else {
           return@continueWithTask Tasks.forResult&lt;Void&gt;(null)
         }
       }
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">// Initialize TFLite asynchronously
initializeTask = TfLite.initialize(
   this, TfLiteInitializationOptions.builder()
      .setEnableGpuDelegateSupport(true)
      .build()
).continueWithTask(
      task -&gt; {
         if (task.getException() != null) {
            useGpu = false;
            return TfLite.initialize(this);
         } else {
            return Tasks.forResult(null);
         }
      }
);
    </pre>
</section>
</devsite-selector>
</div>

After initializing TFLite with the GPU delegate and a CPU fallback, you can use the GPU delegate only when it is supported.

**With the Task Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">lateinit val options
if (useGpu) {
   val baseOptions = BaseOptions.builder().useGpu().build()
   options =
      ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptions)
          .setMaxResults(1)
          .build()
} else {
   options =
      ObjectDetectorOptions.builder()
        .setMaxResults(1)
        .build()
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">ObjectDetectorOptions options;
   if (useGpu) {
      BaseOptions baseOptions = BaseOptions.builder().useGpu().build()
      options =
        ObjectDetectorOptions.builder()
          .setBaseOptions(baseOptions)
          .setMaxResults(1)
          .build()
   } else {
      options =
         ObjectDetectorOptions.builder()
           .setMaxResults(1)
           .build()
   }
    </pre>
</section>
</devsite-selector>
</div>

**With the Interpreter Api**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">val interpreterOptions = if (useGpu) {
  InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())
} else {
   InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
}
InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOption)
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">InterpreterApi.Options options;
  if (useGpu) {
    options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
        .addDelegateFactory(new GpuDelegateFactory());
  } else {
    options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  }
    </pre>
</section>
</devsite-selector>
</div>

## 限制

TensorFlow Lite in Google Play services has the following limitations:

- Support for hardware acceleration delegates is limited to the delegates listed in the [Hardware acceleration](#hardware-acceleration) section. No other acceleration delegates are supported.
- 不支持通过[原生 API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) 访问 TensorFlow Lite。只有 TensorFlow Lite Java API 可以通过 Google Play 服务访问。
- 不支持实验性或过时的 TensorFlow Lite API，包括自定义运算。

## 支持和反馈 {:#support}

You can provide feedback and get support through the TensorFlow Issue Tracker. Please report issues and support requests using the [Issue template](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) for TensorFlow Lite in Google Play services.

## Terms of service {:#tos}

Use of TensorFlow Lite in Google Play services is subject to the [Google APIs Terms of Service](https://developers.google.com/terms/).

Note: TensorFlow Lite in Google Play services is in beta and, as such, its functionality as well as associated APIs may change without advance notice.

### Privacy and data collection

当您使用 Google Play 服务 API 中的 TensorFlow Lite 时，图像、视频、文本等输入数据的处理完全在设备端进行，而 Google Play 服务中的 TensorFlow Lite 不会将这些数据发送到 Google 服务器。因此，您可以使用我们的 API 处理不应离开设备的数据。

Google Play 服务 API 中的 TensorFlow Lite 可能会不时联系 Google 服务器，以便接收错误修复、更新的模型和硬件加速器兼容性信息等内容。Google Play 服务 API 中的 TensorFlow Lite 还将有关应用中 API 的性能和利用率的指标发送给 Google。Google 使用这些指标数据来衡量性能，调试、维护和改进 API，并检测误用或滥用，有关详细信息，请参阅我们的[隐私权政策](https://policies.google.com/privacy)。

**您有责任根据适用法律的要求，将 Google 对 Google Play 服务指标数据中的 TensorFlow Lite 的处理情况告知您应用的用户。**

我们收集的数据包括：

- 设备信息（例如制造商、型号、操作系统版本和内部版本号）和可用的机器学习硬件加速器（GPU 和 DSP）。用于诊断和使用情况分析。
- 用于诊断和使用情况分析的设备标识符。
- 应用信息（软件包名称、应用版本)。用于诊断和使用情况分析。
- API 配置（例如正在使用哪些委托）。用于诊断和使用情况分析。
- 事件类型（例如解释器创建、推断）。用于诊断和使用情况分析。
- 错误代码。用于诊断。
- 性能指标。用于诊断。

## Next steps

有关使用 TensorFlow Lite 在移动应用中实现机器学习的更多信息，请参阅 [TensorFlow Lite 开发者指南](https://www.tensorflow.org/lite/guide)。您可以在 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) 上找到用于图像分类、目标检测和其他应用的其他 TensorFlow Lite 模型。

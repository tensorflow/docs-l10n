# 使用 Task Library 进行 GPU 加速委托

使用图形处理单元 (GPU) 运行机器学习 (ML) 模型可以显著提高支持机器学习的应用的性能和用户体验。在 Android 设备上，您可以使用[*委托*](../../performance/delegates)和以下 API 之一启用模型的 GPU 加速执行：

- Interpreter API - [指南](./gpu)
- Task Library API - 本指南
- 原生 (C/C++) API - 本[指南](./gpu_native)

本页介绍了如何在 Android 应用中使用 Task Library 为 TensorFlow Lite 模型启用 GPU 加速。有关将 GPU 委托用于 TensorFlow Lite 的更多信息，包括最佳做法和高级技术，请参阅 [GPU 委托](../../performance/gpu)页面。

## 将 GPU 与 Google Play 服务中的 TensorFlow Lite 一起使用

TensorFlow Lite [Task Library](../../inference_with_metadata/task_library/overview) 提供了一组用于构建机器学习应用的任务特定 API。本部分介绍如何通过 Google Play 服务中的 TensorFlow Lite 将 GPU 加速器委托与这些 API 结合使用。

[Google Play 服务中的 TensorFlow Lite](../play_services) 是在 Android 上使用 TensorFlow Lite 的推荐途径。如果您的应用面向未运行 Google Play 的设备，请参阅[通过 Task Library 将 GPU 与独立版 TensorFlow Lite 一起使用](#standalone)部分。

### 添加项目依赖项

要使用 Google Play 服务通过 TensorFlow Lite Task Library 启用对 GPU 委托的访问，请将 `com.google.android.gms:play-services-tflite-gpu` 添加到应用的 `build.gradle` 文件的依赖项中：

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### 启用 GPU 加速

然后，使用 [`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu) 类异步验证设备是否支持 GPU 委托，并使用 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 类为您的 Task API 模型类启用 GPU 委托选项。例如，您可以在 `ObjectDetector` 中设置 GPU，如以下代码示例所示：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

        lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
          val baseOptionsBuilder = BaseOptions.builder()
          if (task.result) {
            baseOptionsBuilder.useGpu()
          }
        ObjectDetectorOptions.builder()
                  .setBaseOptions(baseOptionsBuilder.build())
                  .setMaxResults(1)
                  .build()
        }
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">      Task&lt;Boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

      Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
        BaseOptions baseOptionsBuilder = BaseOptions.builder();
        if (task.getResult()) {
          baseOptionsBuilder.useGpu();
        }
        return ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setMaxResults(1)
                .build()
      });
      </pre>
    </section>
  </devsite-selector>
</div>

## 将 GPU 与独立版 TensorFlow Lite 一起使用{:#standalone}

如果您的应用面向未运行 Google Play 的设备，则可以将 GPU 委托捆绑到您的应用并将其与独立版 TensorFlow Lite 一起使用。

### 添加项目依赖项

要使用独立版 TensorFlow Lite 通过 TensorFlow Lite Task Library 启用对 GPU 委托的访问，请将 `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` 添加到应用的 `build.gradle` 文件的依赖项中：

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### 启用 GPU 加速

然后，使用 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder) 类为您的 Task API 模型类启用 GPU 委托选项。例如，您可以在 `ObjectDetector` 中设置 GPU，如以下代码示例所示：

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre>
    </section>
  </devsite-selector>
</div>

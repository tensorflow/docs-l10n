# 适用于 Android 的加速服务（测试版）

测试版：适用于 Android 的加速服务当前为测试版。请查看本页的[注意事项](#caveats)和[条款与隐私](#terms_privacy)部分。

使用 GPU、NPU 或 DSP 等专用处理器进行硬件加速可以显著提高推断性能（某些情况下可将推断速度提高 10 倍）以及支持机器学习的 Android 应用的用户体验。不过，考虑到您的用户可能拥有各种硬件和驱动程序，为每个用户的设备选择最佳硬件加速配置可能极具挑战。此外，如果在设备上启用了错误的配置，则可能会由于高延迟而导致用户体验变得糟糕，或者在极少数情况下，硬件不兼容会导致运行时错误或准确率问题。

适用于 Android 的加速服务是一种特定 API，可帮助您为给定的用户设备和您的 `.tflite` 模型选择最佳硬件加速配置，同时最大程度地降低运行时错误或准确率问题的风险。

通过利用您的 TensorFlow Lite 模型运行内部推断基准化分析，加速服务可评估用户设备上的不同加速配置。这些测试运行通常会在几秒钟内完成，具体取决于您的模型。您可以在推断时间之前在每个用户设备上运行一次基准化分析，缓存结果并在推断期间使用它。这些基准化分析在进程外运行；这有助于将您的应用的崩溃风险降至最低。

提供您的模型、数据样本和预期结果（“黄金”输入和输出）后，加速服务便可运行内部 TFLite 推断基准化分析，以便为您提供硬件建议。

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/acceleration/acceleration_service.png?raw=true)

加速服务属于 Android 自定义机器学习堆栈的一部分，可与 [Google Play 服务中的 TensorFlow Lite](https://www.tensorflow.org/lite/android/play_services) 配合使用。

## 将依赖项添加到您的项目中

将以下依赖项添加到您的应用的 build.gradle 文件中：

```
implementation  "com.google.android.gms:play-services-tflite-
acceleration-service:16.0.0-beta01"
```

加速服务 API 可与 [Google Play 服务中的 TensorFlow Lite](https://www.tensorflow.org/lite/android/play_services) 配合使用。如果您尚未使用通过 Play 服务提供的 TensorFlow Lite 运行时，则需要更新您的[依赖项](https://www.tensorflow.org/lite/android/play_services#1_add_project_dependencies_2)。

## 如何使用加速服务 API

要使用加速服务，首先请创建要针对您的模型进行评估的加速配置（例如，支持 OpenGL 的 GPU）。随后，使用您的模型、一些样本数据和预期模型输出来创建一个验证配置。最后，在传递加速配置和验证配置时调用 `validateConfig()`。

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/lite/images/acceleration/acceleration_service_steps.png?raw=true)

### 创建加速配置

加速配置是在执行过程中转换为委托的硬件配置的表示。随后，加速服务将在内部使用这些配置来执行测试推断。

目前，借助加速服务，您可以使用 [GpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig) 和 CPU 推断（使用 [CpuAccelerationConfig](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig)）来评估 GPU 配置（在执行过程中转换为 GPU 委托）。我们正在努力支持更多委托以便在将来使用其他硬件。

#### GPU 加速配置

创建 GPU 加速配置，代码如下：

```
AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder()
  .setEnableQuantizedInference(false)
  .build();
```

必须通过 [`setEnableQuantizedInference()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/GpuAccelerationConfig.Builder#public-gpuaccelerationconfig.builder-setenablequantizedinference-boolean-value) 指定您的模型是否使用量化。

#### CPU 加速配置

创建 CPU 加速配置，代码如下：

```
AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder()
  .setNumThreads(2)
  .build();
```

使用 [`setNumThreads()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CpuAccelerationConfig.Builder#setNumThreads(int)) 方法定义要用于评估 CPU 推断的线程数。

### 创建验证配置

借助验证配置，您可以定义加速服务如何评估推断。您将使用它们来传递：

- 输入样本，
- 预期输出，
- 准确率验证逻辑。

确保提供您期望模型获得良好性能的输入样本（也称为“黄金”样本）。

使用 [`CustomValidationConfig.Builder`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder) 创建 [`ValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidationConfig)，代码如下：

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenOutputs(outputBuffer)
   .setAccuracyValidator(new MyCustomAccuracyValidator())
   .build();
```

使用 [`setBatchSize()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setBatchSize(int)) 指定黄金样本的数量。使用 [`setGoldenInputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldeninputs-object...-value) 传递黄金样本的输入。为使用 [`setGoldenOutputs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setgoldenoutputs-bytebuffer...-value) 传递的输入提供预期输出。

可以使用 [`setInferenceTimeoutMillis()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#public-customvalidationconfig.builder-setinferencetimeoutmillis-long-value) 定义最长推断时间（默认为 5000 毫秒）。如果推断花费的时间长于您定义的时间，则该配置会被拒绝。

或者，您也可以创建自定义 [`AccuracyValidator`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.AccuracyValidator)，代码如下：

```
class MyCustomAccuracyValidator implements AccuracyValidator {
   boolean validate(
      BenchmarkResult benchmarkResult,
      ByteBuffer[] goldenOutput) {
        for (int i = 0; i < benchmarkResult.actualOutput().size(); i++) {
            if (!goldenOutputs[i]
               .equals(benchmarkResult.actualOutput().get(i).getValue())) {
               return false;
            }
         }
         return true;

   }
}
```

确保定义适用于您的用例的验证逻辑。

请注意，如果验证数据已嵌入到您的模型中，则可以使用 [`EmbeddedValidationConfig`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/EmbeddedValidationConfig)。

##### 生成验证输出

黄金输出是可选的，只要您提供黄金输入，加速服务就可以在内部生成黄金输出。此外，您还可以通过调用 [`setGoldenConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/CustomValidationConfig.Builder#setGoldenConfig(com.google.android.gms.tflite.acceleration.AccelerationConfig)) 来定义用于生成这些黄金输出的加速配置：

```
ValidationConfig validationConfig = new CustomValidationConfig.Builder()
   .setBatchSize(5)
   .setGoldenInputs(inputs)
   .setGoldenConfig(customCpuAccelerationConfig)
   [...]
   .build();
```

### 验证加速配置

在创建完加速配置和验证配置后，您就可以针对您的模型评估这些配置。

确保包含 Play 服务运行时的 TensorFlow Lite 已正确初始化，并且 GPU 委托可用于设备，方法是运行以下代码：

```
TfLiteGpu.isGpuDelegateAvailable(context)
   .onSuccessTask(gpuAvailable -> TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(gpuAvailable)
        .build()
      )
   );
```

通过调用 [`AccelerationService.create()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#create(android.content.Context))来实例化 [`AccelerationService`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService)。

随后，可以通过调用 [`validateConfig()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfig(com.google.android.gms.tflite.acceleration.Model,%20com.google.android.gms.tflite.acceleration.AccelerationConfig,%20com.google.android.gms.tflite.acceleration.ValidationConfig)) 来验证模型的加速配置：

```
InterpreterApi interpreter;
InterpreterOptions interpreterOptions = InterpreterApi.Options();
AccelerationService.create(context)
   .validateConfig(model, accelerationConfig, validationConfig)
   .addOnSuccessListener(validatedConfig -> {
      if (validatedConfig.isValid() && validatedConfig.benchmarkResult().hasPassedAccuracyTest()) {
         interpreterOptions.setAccelerationConfig(validatedConfig);
         interpreter = InterpreterApi.create(model, interpreterOptions);
});
```

您也可以通过调用 [`validateConfigs()`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/AccelerationService#validateConfigs(com.google.android.gms.tflite.acceleration.Model,%20java.lang.Iterable%3Ccom.google.android.gms.tflite.acceleration.AccelerationConfig%3E,%20com.google.android.gms.tflite.acceleration.ValidationConfig)) 并将 `Iterable<AccelerationConfig>` 对象作为参数传递来验证多个配置。

`validateConfig()` 将从启用异步任务的 Google Play 服务 [Task API](https://developers.google.com/android/guides/tasks) 返回 `Task<`[`ValidatedAccelerationConfigResult`](https://developers.google.com/android/reference/com/google/android/gms/tflite/acceleration/ValidatedAccelerationConfigResult)`>`。<br>要从验证调用中获得结果，请添加一个 [`addOnSuccessListener()`](https://developers.google.com/android/reference/com/google/android/gms/tasks/OnSuccessListener) 回调。

#### 在您的解释器中使用经过验证的配置

在检查回调中返回的 `ValidatedAccelerationConfigResult` 是否有效后，您可以将经过验证的配置设置为您的解释器调用 `interpreterOptions.setAccelerationConfig()` 的加速配置。

#### 配置缓存

模型的最佳加速配置不太可能在设备上改变。因此，一旦获得令人满意的加速配置，您应当将其存储在设备上，随后让您的应用检索该配置并在后续会话期间使用它来创建您的 `InterpreterOptions`，而不是运行另一个验证。利用 {code/3}ValidatedAccelerationConfigResult 中的 `serialize()` 和 `deserialize()` 方法，可以更轻松地完成存储和检索过程。

### 示例应用

要查看 Accerlation Service 的现场集成，请查看[示例应用](https://github.com/tensorflow/examples/tree/master/lite/examples/acceleration_service/android_play_services)。

## 限制

加速服务目前存在以下限制：

- 目前仅支持 CPU 和 GPU 加速配置，
- 它仅支持 Google Play 服务中的 TensorFlow Lite，如果您使用捆绑版本的 TensorFlow Lite，则无法使用它，
- 它不支持 TensorFlow Lite [Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview)，因为您无法使用 `ValidatedAccelerationConfigResult` 对象直接初始化 [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder)。
- 加速服务 SDK 仅支持 22 及更高级别的 API。

## 注意事项 {:#caveats}

请仔细阅读以下注意事项，特别是当您计划在生产中使用此 SDK 时：

- 在退出测试版并为加速服务 API 发布稳定版之前，我们会发布一个新的 SDK，它可能与当前的测试版 SDK 有一些不同。要继续使用加速服务，您需要迁移到这个新的 SDK 并及时向您的应用推送更新。如果不这样做，则可能会导致损坏，因为在一段时间后，测试版 SDK 可能不再与 Google Play 服务兼容。

- 无法保证加速服务 API 中的特定功能或整个 API 变得普遍可用。它可能会无限期地保留在测试版阶段、被关闭或者与其他功能组合到专为特定开发者受众设计的软件包中。加速服务 API 中的某些功能或整个 API 本身可能会最终变得普遍可用，但目前没有固定的时间表。

## 条款与隐私 {:#terms_privacy}

#### 服务条款

加速服务 API 的使用受 [Google API 服务条款](https://developers.google.com/terms/)约束。<br>此外，加速服务 API 当前为测试版，因此，使用它即表示您认可上述“注意事项”部分中概述的潜在问题，并认可加速服务可能并不总能按规定执行。

#### 隐私

当您使用加速服务 API 时，输入数据（如图像、视频、文本）的处理完全在设备上进行，并且**加速服务不会将这些数据发送到 Google 服务器**。因此，您可以使用我们的 API 来处理不应离开设备的输入数据。<br>加速服务 API 可能会不时连接 Google 服务器，以便接收 bug 修复、更新的模型和硬件加速器兼容性信息等内容。此外，加速服务 API 还会将有关应用中 API 的性能和利用率的指标发送给 Google。Google 会使用这些指标数据来衡量性能、调试、维护和改进 API，以及检测误用或滥用情况，我们的[隐私政策](https://policies.google.com/privacy)对此有进一步说明。<br>**您有责任根据适用法律的要求，将 Google 对加速服务指标数据的处理告知您的应用的用户**。<br>我们收集的数据包括以下各项：

- 设备信息（例如制造商、型号、操作系统版本和内部版本号）和可用的机器学习硬件加速器（GPU 和 DSP）。用于诊断和使用情况分析。
- 应用信息（软件包名称/捆绑包 ID、应用版本)。用于诊断和使用情况分析。
- API 配置（例如图像格式和分辨率）。用于诊断和使用情况分析。
- 事件类型（例如初始化、下载模型、更新、运行、检测）。用于诊断和使用情况分析。
- 错误代码。用于诊断。
- 性能指标。用于诊断。
- 不能以唯一的方式标识用户或实体设备的预安装标识符。用于远程配置和使用情况分析的操作。
- 网络请求发送方 IP 地址。用于远程配置诊断。收集到的 IP 地址会暂时保留。

## 支持和反馈

您可以通过 TensorFlow 问题跟踪器提供反馈并获得支持。请使用适用于 Google Play 服务中的 TensorFlow Lite 的[议题模板](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md)报告议题和支持请求。

# Android 快速入门

本页面向您展示如何使用 TensorFlow Lite 构建一个 Android 应用来分析实时摄像头画面并识别目标。这种机器学习用例称为*目标检测*。此示例应用通过 [Google Play 服务](./play_services)使用 TensorFlow Lite [Task library for vision](./play_services)，以实现目标检测机器学习模型的执行，这是使用 TensorFlow Lite 构建 ML 应用的推荐方式。

<aside class="note"><b>条款</b>：访问或使用 Google Play 服务 API 中的 TensorFlow Lite，即表示您同意<a href="./play_services#tos">服务条款</a>。在访问 API 之前，请阅读并理解所有适用的条款和政策。</aside>

![目标检测动画演示](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## 设置并运行示例

在本练习的第一部分中，从 GitHub 下载[示例代码](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services)并使用 [Android Studio](https://developer.android.com/studio/) 运行。本文档的以下部分将探索代码示例的相关部分，以便您可以将它们应用于您自己的 Android 应用。您需要安装以下版本的下列工具：

- Android Studio 4.2 或更高版本
- Android SDK 21 或更高版本

注：本示例会用到摄像头，因此您应该在实体 Android 设备上运行。

### 获取示例代码

创建示例代码的本地副本，以便您可以构建并运行它。

要克隆和设置示例代码，请执行以下操作：

1. 克隆 Git 仓库：
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 将您的 Git 实例配置为使用稀疏签出，这样您就只有目标检测示例应用的文件：
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android_play_services
        </pre>

### 导入并运行项目

使用 Android Studio 从下载的示例代码创建一个项目，构建并运行该项目。

要导入和构建示例代码项目，请执行以下操作：

1. 启动 [Android Studio](https://developer.android.com/studio)。
2. 在 Android Studio **Welcom** 页面中，选择 **Import Project**，或选择 **File &gt; New &gt; Import Project**。
3. 导航到包含 build.gradle 文件的示例代码目录 (`...examples/lite/examples/object_detection/android_play_services/build.gradle`) 并选择该目录。

选择此目录后，Android Studio 会创建并构建一个新项目。构建完成后，Android Studio 会在 **Build Output** 状态面板中显示 `BUILD SUCCESSFUL` 消息。

要运行项目，请执行以下操作：

1. 在 Android Studio 中，选择 **Run &gt; Run…** 和 **MainActivity** 来运行项目。
2. 选择一台带有摄像头并且已连接的 Android 设备测试该应用。

## 示例应用的运作方式

示例应用使用 TensorFlow Lite 格式的预训练目标检测模型，例如 [mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite)，在来自 Android 设备摄像头的视频流中查找目标。此功能的代码主要在以下文件中：

- [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt) - 初始化运行时环境，启用硬件加速，并运行目标检测 ML 模型。
- [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt) - 构建摄像头图像数据流，为模型准备数据，并显示目标检测结果。

注：此示例应用使用 TensorFlow Lite [Task Library](../inference_with_metadata/task_library/overview#supported_tasks)，它提供了易于使用的针对特定任务的 API，用于执行常见的机器学习操作。对于具有更具体需求和自定义 ML 功能的应用，请考虑使用 [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)。

接下来的部分将向您展示这些代码文件的关键组件，以便您可以修改 Android 应用来添加此功能。

## 构建应用 {:#build_app}

以下部分解释了构建您自己的 Android 应用并运行示例应用中显示的模型的关键步骤。这些说明使用前面显示的示例应用作为参考点。

注：要按照这些说明构建您自己的应用，请使用 Android Studio 创建一个[基本的 Android 项目](https://developer.android.com/studio/projects/create-project)。

### 添加项目依赖项 {:#add_dependencies}

在您的基本 Android 应用中，添加用于运行 TensorFlow Lite 机器学习模型和访问 ML 数据效用函数的项目依赖项。这些效用函数将图像等数据转换为模型可以处理的张量数据格式。

示例应用使用来自 [Google Play 服务](./play_services)的 TensorFlow Lite [Task library for vision](../inference_with_metadata/task_library/overview#supported_tasks) 来实现目标检测机器学习模型的执行。以下说明解释了如何将所需的库依赖项添加到您自己的 Android 应用项目中。

要添加模块依赖项，请执行以下操作：

1. 在使用 TensorFlow Lite 的 Android 应用模块中，更新模块的 `build.gradle` 文件以包含以下依赖项。在示例代码中，此文件位于以下位置：`...examples/lite/examples/object_detection/android_play_services/app/build.gradle`
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
    ...
    }
    ```
2. 在 Android Studio 中，通过选择以下选项来同步项目依赖项：**File &gt; Sync Project with Gradle Files**。

### 初始化 Google Play 服务

当您使用 [Google Play 服务](./play_services)运行 TensorFlow Lite 模型时，您必须先初始化该服务，然后才能使用它。如果您想对该服务使用硬件加速支持（例如 GPU 加速），您还需要在此初始化过程中启用该支持。

要使用 Google Play 服务初始化 TensorFlow Lite，请执行以下操作：

1. 创建一个 `TfLiteInitializationOptions` 对象并修改它以启用 GPU 支持：

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

2. 使用 `TfLiteVision.initialize()` 方法启用 Play 服务运行时的使用，并设置一个侦听器来验证它是否成功加载：

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### 初始化机器学习模型解释器

通过加载模型文件并设置模型参数来初始化 TensorFlow Lite 机器学习模型解释器。TensorFlow Lite 模型包括一个包含模型代码的 `.tflite` 文件。您应该将模型存储在开发项目的 `src/main/assets` 目录中，例如：

```
.../src/main/assets/mobilenetv1.tflite`
```

提示：如果您未指定文件路径，Task Library 解释器代码会自动在 `src/main/assets` 目录中查找模型。

要初始化模型，请执行以下操作：

1. 将一个 `.tflite` 模型文件添加到开发项目的 `src/main/assets` 目录中，例如 [ssd_mobilenet_v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2)。
2. 设置 `modelName` 变量以指定 ML 模型的文件名：
    ```
    val modelName = "mobilenetv1.tflite"
    ```
3. 设置模型的选项，例如预测阈值和结果集大小：
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
4. 如果设备不支持加速，则使用选项启用 GPU 加速并允许代码优雅地失败：
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
5. 使用此对象中的设置来构造一个包含模型的 TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) 对象：
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

有关对 TensorFlow Lite 使用硬件加速委托的更多信息，请参阅 [TensorFlow Lite 委托](../performance/delegates)。

### 为模型准备数据

您通过将现有数据（例如图像）转换为[张量](../api_docs/java/org/tensorflow/lite/Tensor)数据格式来准备数据以供模型解释，以便您的模型对其进行处理。张量中的数据必须具有与用于训练模型的数据格式相匹配的特定维度或形状。根据您使用的模型，您可能需要转换数据以符合模型的预期。示例应用使用 [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) 对象从相机子系统中提取图像帧。

要准备数据以供模型处理，请执行以下操作：

1. 构建一个 `ImageAnalysis` 对象以提取所需格式的图像：
    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```
2. 将分析器连接到摄像头子系统并创建一个位图缓冲区以包含从摄像头接收到的数据：
    ```
            .also {
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
3. 提取模型所需的特定图像数据，并传递图像旋转信息：
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }
    ```
4. 完成所有最终数据转换并将图像数据添加到 `TensorImage` 对象，如示例应用的 `ObjectDetectorHelper.detect()` 方法所示：
    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

### 运行预测

使用正确格式的图像数据创建 [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) 对象后，您可以在数据上运行该模型以生成预测或*推断*。在示例应用中，此代码包含在 `ObjectDetectorHelper.detect()` 方法中。

要运行模型并从图像数据生成预测，请执行以下操作：

- 通过将图像数据传递给预测函数来运行预测：
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### 处理模型输出

在您针对目标检测模型运行图像数据后，它会生成一个预测结果列表，您的应用代码必须通过执行其他业务逻辑、向用户显示结果或采取其他操作来处理这些预测结果。示例应用中的目标检测模型会为检测到的目标生成一个预测列表和边界框。在示例应用中，预测结果会被传递到侦听器对象以供进一步处理并显示给用户。

要处理模型预测结果，请执行以下操作：

1. 使用侦听器模式将结果传递给您的应用代码或界面对象。示例应用使用此模式将检测结果从 `ObjectDetectorHelper` 对象传递到 `CameraFragment` 对象：
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
2. 根据结果采取操作，例如向用户显示预测。示例应用会在 `CameraPreview` 对象上绘制叠加层来显示结果：
    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

## 后续步骤

- 详细了解 [Task Library API](../inference_with_metadata/task_library/overview#supported_tasks)
- 详细了解 [Interpreter API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi)。
- 在[示例](../examples)中探索 TensorFlow Lite 的用法。
- 在[模型](../models)部分中详细了解如何在 TensorFlow Lite 中使用和构建机器学习模型。
- 有关在您的移动应用中实现机器学习的更多信息，请参阅 [TensorFlow Lite 开发者指南](../guide)。

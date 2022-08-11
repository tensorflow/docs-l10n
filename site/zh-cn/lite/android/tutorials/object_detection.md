# Android 对象检测

本教程展示如何使用 TensorFlow Lite 构建 Android 应用，以连续检测设备摄像头捕获的帧中的对象。此应用专为实体 Android 设备设计。如果您要更新现有项目，可以使用代码示例作为参考，并跳至[修改项目](#add_dependencies)的说明。

## 对象检测概述

*对象检测*是识别图像中多类对象的存在和位置的机器学习任务。对象检测模型是在包含一组已知对象的数据集上训练出来的。

经过训练的模型接收图像帧作为输入，并尝试根据它被训练识别的已知类别集合对图像中的项目进行分类。对于每个图像帧，对象检测模型都会输出它检测到的对象列表、每个对象的边框位置以及指示对象被正确分类的置信度的分数。

## 模型和数据集

本教程使用的模型是用 [COCO 数据集](http://cocodataset.org/)训练的。 COCO 是一个大规模对象检测数据集，其中包含 33 万个图像、150 万个对象实例和 80 个对象类别。

您可以选择使用以下预训练模型之一：

- [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) *[推荐]* - 一个轻量级对象检测模型，包含 BiFPN 特征提取器、共享框预测器和焦点损失。COCO 2017 验证数据集的 mAP（平均精度均值）为 25.69%。

- [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1) - 一个中型 EfficientDet 对象检测模型。COCO 2017 验证数据集的 mAP 为 30.55%。

- [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1) - 一个大型 EfficientDet 对象检测模型。COCO 2017 验证数据集的 mAP 为 33.97%。

- [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2) - 一个极其轻量级的模型，经过优化，可与 TensorFlow Lite 配合使用进行对象检测。COCO 2017 验证数据集的 mAP 为 21%。

对于本教程，*EfficientDet-Lite0* 模型在大小和准确性之间取得了良好的平衡。

下载、提取和放置模型到资源文件夹是由 `download.gradle` 文件自动管理的，该文件在构建时运行。您无需手动将 TFLite 模型下载到项目中。

## 设置并运行示例

要设置对象检测应用，请从 [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) 下载示例并使用 [Android Studio](https://developer.android.com/studio/) 运行该示例。本教程的以下部分将探索代码示例的相关部分，以便您将它们应用于您自己的 Android 应用。

### 系统要求

- **[Android Studio](https://developer.android.com/studio/index.html)** 2021.1.1 (Bumblebee) 或更高版本。
- Android SDK 31 或更高版本
- 最低操作系统版本为 SDK 24 (Android 7.0 - Nougat) 并且已启用开发者模式的 Android 设备。

注：本示例使用摄像头，因此请在实体 Android 设备上运行。

### 获取示例代码

创建示例代码的本地副本。您将在 Android Studio 中使用此代码创建项目并运行示例应用。

要克隆和设置示例代码，请执行以下操作：

1. 克隆 git 仓库
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. （可选）将您的 git 实例配置为使用稀疏签出，这样您就只有对象检测示例应用的文件：
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android
        </pre>

### 导入并运行项目

从下载的示例代码创建一个项目，构建并运行该项目。

要导入和构建示例代码项目，请执行以下操作：

1. 启动 [Android Studio](https://developer.android.com/studio)。
2. 在 Android Studio 中，选择 **File &gt; New &gt; Import Project**。
3. 导航到包含 build.gradle 文件的示例代码目录 (`.../examples/lite/examples/object_detection/android/build.gradle`) 并选择该目录。
4. 如果 Android Studio 请求 Gradle Sync，请选择 OK。
5. 确保您的 Android 设备已连接到计算机并且已启用开发者模式。单击绿色 `Run` 箭头。

如果您选择了正确的目录，Android Studio 会创建一个新项目并进行构建。此过程可能需要几分钟，具体取决于您的计算机速度，以及您是否将 Android Studio 用于其他项目。构建完成后，Android Studio 会在 <strong>Build Output</strong> 状态面板中显示 <code>BUILD SUCCESSFUL</code> 消息。

注：示例代码使用 Android Studio 4.2.2 构建，但也适用于早期版本的 Studio。如果您使用的是早期版本的 Android Studio，可以尝试调整 Android 插件的版本号，以便完成构建，而无需升级 Studio。

**可选**：要通过更新 Android 插件版本来修正构建错误，请执行以下操作：

1. 打开项目目录中的 build.gradle 文件。

2. 按如下方式更改 Android 工具版本：

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

3. 选择 **File &gt; Sync Project with Gradle Files** 来同步项目。

要运行项目，请执行以下操作：

1. 在 Android Studio 中，选择 **Run &gt; Run…** 来运行项目。
2. 选择一台已连接的带摄像头的 Android 设备来测试应用。

接下来的部分将以此示例应用作为参考点，展示要将此功能添加到您自己的应用中，您需要对现有项目进行的修改。

## 添加项目依赖项 {:#add_dependencies}

在您自己的应用中，您必须添加特定的项目依赖项才能运行 TensorFlow Lite 机器学习模型，并访问能够将图像等数据转换为您所使用的模型可以处理的张量数据格式的效用函数。

示例应用使用 TensorFlow Lite [Task library for vision](../../inference_with_metadata/task_library/overview#supported_tasks) 来实现对象检测机器学习模型的执行。以下说明解释了如何将所需的库依赖项添加到您自己的 Android 应用项目中。

以下说明解释了如何将所需的项目和模块依赖项添加到您自己的 Android 应用项目中。

要添加模块依赖项，请执行以下操作：

1. 在使用 TensorFlow Lite 的模块中，更新模块的 `build.gradle` 文件以包含以下依赖项。在示例代码中，此文件位于以下位置：`...examples/lite/examples/object_detection/android/app/build.gradle`（[代码引用](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle)）

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    项目必须包含 Vision Task Library (`tensorflow-lite-task-vision`)。图形处理单元 (GPU) 库 (`tensorflow-lite-gpu-delegate-plugin`) 提供了在 GPU 上运行应用的基础结构，委托 (`tensorflow-lite-gpu`) 提供了兼容性列表。

2. 在 Android Studio 中，选择 **File &gt; Sync Project with Gradle Files** 来同步项目依赖项。

## 初始化机器学习模型

在您的 Android 应用中，必须先使用参数初始化 TensorFlow Lite 机器学习模型，然后才能使用该模型运行预测。这些初始化参数在对象检测模型中是一致的，并且可以包括预测的最小准确度阈值等设置。

TensorFlow Lite 模型包括一个含有模型代码的 `.tflite` 文件，并且经常包括一个含有模型预测的类别名称的标签文件。在进行对象检测时，类别是人、狗、猫或汽车等对象。

此示例将下载 `download_models.gradle` 中指定的几个模型，`ObjectDetectorHelper` 类为模型提供选择器：

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

关键点：模型应存储在开发项目的 `src/main/assets` 目录中。当指定模型文件名后，TensorFlow Lite Task Library 会自动检查该目录。

要在您的应用中初始化模型，请执行以下操作：

1. 将一个 `.tflite` 模型文件添加到您的开发项目的 `src/main/assets` 目录中，例如：[EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1)。

2. 为模型的文件名设置一个静态变量。在示例应用中，您将 `modelName` 变量设置为 `MODEL_EFFICIENTDETV0` 以使用 EfficientDet-Lite0 检测模型。

3. 设置模型的选项，例如预测阈值、结果集大小以及可选的硬件加速委托：

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

4. 使用此对象的设置来构造一个包含模型的 TensorFlow Lite [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) 对象：

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

`setupObjectDetector` 设置以下模型参数：

- 检测阈值
- 最大检测结果数
- 要使用的处理线程数 (`BaseOptions.builder().setNumThreads(numThreads)`)
- 实际模型 (`modelName`)
- ObjectDetector 对象 (`objectDetector`)

### 配置硬件加速器

在您的应用中初始化 TensorFlow Lite 模型时，可以使用硬件加速功能来加快模型的预测计算。

TensorFlow Lite *委托*是使用移动设备上的专用处理硬件（如图形处理单元 (GPU)、张量处理单元 (TPU) 和数字信号处理器 (DSP)）加速机器学习模型执行的软件模块。建议使用委托来运行 TensorFlow Lite 模型，但非必需。

对象检测器通过正在使用它的线程上的当前设置进行初始化。对于在主线程上创建和在后台线程上使用的检测器，可以使用 CPU 和 [NNAPI](../../android/delegates/nnapi) 委托，但对于初始化了检测器的线程，必须使用 GPU 委托。

委托在 `ObjectDetectionHelper.setupObjectDetector()` 函数内设置：

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

有关对 TensorFlow Lite 使用硬件加速委托的更多信息，请参阅 [TensorFlow Lite 委托](../../performance/delegates)。

## 为模型准备数据

在您的 Android 应用中，您的代码通过将现有数据（如图像帧）转换为可以被模型处理的张量数据格式，向模型提供数据进行解释。传递给模型的张量中的数据必须具有与用于训练模型的数据格式相匹配的特定尺寸或形状。

此代码示例中使用的 [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) 模型接受的张量代表尺寸为 320 x 320 且每个像素具有三个通道（红色、蓝色和绿色）的图像。张量中的每个值都是 0 到 255 之间的单个字节。因此，要对新图像运行预测，您的应用必须将图像数据转换为该尺寸和形状的张量数据对象。TensorFlow Lite Task Library Vision API 可为您处理数据转换。

该应用使用 [`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) 对象从摄像头拉取图像。此对象对来自摄像头的位图调用 `detectObject` 函数。数据由 `ImageProcessor` 自动调整大小并旋转，以符合模型的图像数据要求。然后，图像转换为一个 [`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage) 对象。

要从摄像头子系统准备供机器学习模型处理的数据，请执行以下操作：

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
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

3. 提取模型所需的特定图像数据，并传递图像旋转信息：

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
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

注：从 Android 摄像头子系统提取图像信息时，请确保获取 RGB 格式的图像。此格式是 TensorFlow Lite <a>ImageProcessor</a> 类必需的，该类用于准备供模型分析的图像。如果 RGB 格式的图像包含 Alpha 通道，则该透明度数据将被忽略。

## 运行预测

在您的 Android 应用中，当使用正确格式的图像数据创建 TensorImage 对象后，就可以针对该数据运行模型来生成预测或*推断*。

当示例应用连接到摄像头后，`fragments/CameraFragment.kt` 类中的 `bindCameraUseCases` 函数内的 `imageAnalyzer` 对象会自动将数据传递给模型进行预测。

该应用使用 `cameraProvider.bindToLifecycle()` 方法来处理摄像头选择器、预览窗口和机器学习模型处理。`ObjectDetectorHelper.kt` 类负责将图像数据传递到模型中。要运行模型并从图像数据生成预测，请执行以下操作：

- 通过将图像数据传递给预测函数来运行预测：

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

TensorFlow Lite Interpreter 对象接收该数据，在模型上运行该数据，并生成预测列表。要通过模型连续处理数据，请使用 `runForMultipleInputsOutputs()` 方法，这样系统就不会为每次预测运行创建并移除 Interpreter 对象。

## 处理模型输出

在您的 Android 应用中，在目标检测模型上运行图像数据后，它会生成一个预测列表，您的应用代码必须通过执行额外的业务逻辑来处理这些预测，从而向用户显示结果或采取其他操作。

任何给定的 TensorFlow Lite 模型的输出都根据其产生的预测数量（一个或多个）以及每个预测的描述性信息而有所不同。在使用对象检测模型的情况下，预测通常包括用于指示在图像中检测到目标的位置的边框数据。在示例代码中，结果传递给 `CameraFragment.kt` 中的 `onResults` 函数，该函数在对象检测过程中充当 DetectorListener。

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

对于本示例中使用的模型，每个预测都包括对象的边框位置、对象的标签以及一个介于 0 和 1 之间的预测分数（作为表示预测置信度的浮点数），其中 1 是最高置信度。一般来说，分数低于 50% (0.5) 的预测被认为是不确定的。但是，如何处理低值预测结果取决于您和您应用的需求。

要处理模型预测结果，请执行以下操作：

1. 使用侦听器模式将结果传递给您的应用代码或用户界面对象。示例应用使用此模式将检测结果从 `ObjectDetectorHelper` 对象传递到 `CameraFragment` 对象：

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

2. 根据结果采取操作，例如向用户显示预测。该示例在 CameraPreview 对象上绘制覆盖来显示结果：

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

一旦模型返回预测结果，您的应用就可以根据预测执行操作，将结果呈现给用户或执行其他逻辑。对于示例代码，应用在识别的对象周围绘制边框，并在屏幕上显示类名。

## 后续步骤

- 在[示例](../../examples)中探索 TensorFlow Lite 的各种用法。
- 在[模型](../../models)部分中详细了解如何在 TensorFlow Lite 中使用机器学习模型。
- 在 [TensorFlow Lite 开发者指南](../../guide)中详细了解如何在您的移动应用中实现机器学习。

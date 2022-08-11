# Android 快速入门

本教程将向您展示如何使用 TensorFlow Lite 构建 Android 应用，以使用最少的代码来分析实时摄像头画面并使用机器学习模型识别目标。如果您正在更新现有项目，则可以使用代码示例作为参考，并跳到[修改项目](#add_dependencies)的说明。

## 基于机器学习的目标检测

![目标检测动画演示](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/qs-obj-detect.gif){: .attempt-right width="250px"} 本教程中的机器学习模型执行目标检测。目标检测模型接受特定格式的图像数据，对其进行分析，并尝试将图像中的条目分类为其被训练识别的一组已知类别中的一个。模型识别已知目标（称为目标*预测*或*推断*）的速度通常以毫秒为单位测量。在实践中，推断速度会根据托管模型的硬件、处理数据的大小和机器学习模型的大小而有所不同。

## 设置并运行示例

在本教程的第一部分中，从 GitHub 下载示例并使用 [Android Studio](https://developer.android.com/studio/) 运行。本教程的以下部分将探索代码示例的相关部分，以便您可以将它们应用于您自己的 Android 应用。您需要安装以下版本的下列工具：

- Android Studio 4.2.2 或更高版本
- Android SDK 31 或更高版本

Note: This example uses the camera, so you should run it on a physical Android device.

### 获取示例代码

创建一个示例代码的本地副本。您将使用此代码在 Android Studio 中创建一个项目并运行示例应用。

要克隆和设置示例代码，请执行以下操作：

1. 克隆 Git 仓库：
    <pre class="devsite-click-to-copy">    git clone https://github.com/android/camera-samples.git
        </pre>
2. 将您的 Git 实例配置为使用稀疏签出，这样您就只有目标检测示例应用的文件：
    ```
    cd camera-samples
    git sparse-checkout init --cone
    git sparse-checkout set CameraXAdvanced
    ```

### 导入并运行项目

从下载的示例代码创建一个项目，构建并运行该项目。

要导入和构建示例代码项目，请执行以下操作：

1. 启动 [Android Studio](https://developer.android.com/studio)。
2. 在 Android Studio **Welcom** 页面中，选择 **Import Project**，或选择 **File &gt; New &gt; Import Project**。
3. 导航到包含 build.gradle 文件的示例代码目录 (`.../android/camera-samples/CameraXAdvanced/build.gradle`) 并选择该目录。

如果您选择了正确的目录，Android Studio 会创建并构建一个新项目。这个过程可能需要几分钟，具体取决于您的计算机速度，以及您是否曾将 Android Studio 用于其他项目。构建完成后，Android Studio 会在 **Build Output** 状态面板中显示 `BUILD SUCCESSFUL` 消息。

Note: The example code is built with Android Studio 4.2.2, but works with earlier versions of Studio. If you are using an earlier version of Android Studio you can try to adjust the version number of the Android plugin so that the build completes, instead of upgrading Studio.

**可选**：要通过更新 Android 插件版本来修正构建错误，请执行以下操作：

1. 打开项目目录中的 build.gradle 文件。
2. 按如下方式更改 Android 工具版本：
    ```
    // from:
    classpath 'com.android.tools.build:gradle:4.2.2'
    // to:
    classpath 'com.android.tools.build:gradle:4.1.2'
    ```
3. 通过选择以下选项来同步项目：**File &gt; Sync Project with Gradle Files**。

要运行项目，请执行以下操作：

1. 在 Android Studio 中，通过选择以下选项来运行项目：**Run &gt; Run…** 和 **CameraActivity**。
2. 选择一台带有摄像头并且已连接的 Android 设备测试该应用。

以下各部分将使用此示例应用作为参考点，向您展示需要对现有项目进行的修改，以便将此功能添加到您自己的应用中。

## 添加项目依赖项 {:#add_dependencies}

在您自己的应用中，您必须添加特定的项目依赖项才能运行 TensorFlow Lite 机器学习模型，并访问能够将图像等数据转换为您所使用的模型可以处理的张量数据格式的效用函数。

The example app uses several TensorFlow Lite libraries to enable the execution of the object detection machine learning model:

- *TensorFlow Lite 主库* - 提供所需的数据输入类、机器学习模型的执行，以及模型处理的输出结果。
- *TensorFlow Lite Support 库* - 该库提供了一个辅助类，用于将摄像头中的图像转换为可由机器学习模型处理的 [`TensorImage`](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) 数据对象。
- *TensorFlow Lite GPU 库* - 该库支持使用设备上的 GPU 处理器（如果可用）来加速模型执行。

以下说明解释了如何将所需的项目和模块依赖项添加到您自己的 Android 应用项目中。

要添加模块依赖项，请执行以下操作：

1. 在使用 TensorFlow Lite 的模块中，更新模块的 `build.gradle` 文件以包含以下依赖项。在示例代码中，此文件位于以下位置：`.../android/camera-samples/CameraXAdvanced/tflite/build.gradle`（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/build.gradle#L69-L71)）
    ```
    ...
    dependencies {
    ...
        // Tensorflow lite dependencies
        implementation 'org.tensorflow:tensorflow-lite:2.8.0'
        implementation 'org.tensorflow:tensorflow-lite-gpu:2.8.0'
        implementation 'org.tensorflow:tensorflow-lite-support:2.8.0'
    ...
    }
    ```
2. 在 Android Studio 中，通过选择以下选项来同步项目依赖项：**File &gt; Sync Project with Gradle Files**。

## 初始化机器学习模型解释器

在您的 Android 应用中，您必须使用参数初始化 TensorFlow Lite 机器学习模型解释器，然后才能使用该模型运行预测。这些初始化参数取决于您使用的模型，并且可以包括预测的最小准确率阈值和已标识对象类的标签等设置。

TensorFlow Lite 模型包括包含模型代码的 `.tflite` 文件，并且经常包括包含模型预测的类的名称的标签文件。在目标检测中，类是诸如人、狗、猫或汽车等目标。模型通常存储在主模块的 `src/main/assets` 目录中，如代码示例所示：

- CameraXAdvanced/tflite/src/main/assets/coco_ssd_mobilenet_v1_1.0_quant.tflite
- CameraXAdvanced/tflite/src/main/assets/coco_ssd_mobilenet_v1_1.0_labels.txt

为方便起见和提升代码可读性，该示例声明了一个为模型定义设置的伴随对象。

要在应用中初始化模型，请执行以下操作：

1. 创建一个伴随对象以定义模型的设置：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L342-L347)）
    ```
    companion object {
       private val TAG = CameraActivity::class.java.simpleName

       private const val ACCURACY_THRESHOLD = 0.5f
       private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
       private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }
    ```
2. 使用此对象中的设置来构造包含模型的 TensorFlow Lite [Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter) 对象：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L90-L94)）
    ```
    private val tflite by lazy {
       Interpreter(
           FileUtil.loadMappedFile(this, MODEL_PATH),
           Interpreter.Options().addDelegate(nnApiDelegate))
    }
    ```

### 配置硬件加速器

在应用中初始化 TensorFlow Lite 模型时，可以使用硬件加速功能来加快模型的预测计算。上面的代码示例使用 NNAPI 委托处理模型执行的硬件加速：

```
Interpreter.Options().addDelegate(nnApiDelegate)
```

TensorFlow Lite *delegates* are software modules that accelerate the execution of machine learning models using specialized processing hardware on a mobile device, such as GPUs, TPUs, or DSPs. Using delegates for running TensorFlow Lite models is recommended, but not required.

有关在 TensorFlow Lite 中使用委托的更多信息，请参阅 [TensorFlow Lite 委托](../performance/delegates)。

## 向模型提供数据

在您的 Android 应用中，您的代码通过将图像等现有数据转换为您的模型可以处理的[张量](../api_docs/java/org/tensorflow/lite/Tensor)数据格式，向模型提供数据以进行解释。张量中的数据必须具有与用于训练模型的数据格式相匹配的特定维度或形状。

To determine the required tensor shape for a model:

- 使用已初始化的 [Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter) 对象来确定模型使用的张量的形状，如以下代码片段所示：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L102-L106)）
    ```
    private val tfInputSize by lazy {
       val inputIndex = 0
       val inputShape = tflite.getInputTensor(inputIndex).shape()
       Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }
    ```

示例代码中使用的目标检测模型需要 300 x 300 像素大小的正方形图像。

在您可以提供来自摄像头的图像之前，您的应用必须获取图像，使其符合预期的大小，调整其旋转，并对图像数据进行归一化。当使用 TensorFlow Lite 模型处理图像时，您可以使用 TensorFlow Lite Support Library [ImageProcessor](../api_docs/java/org/tensorflow/lite/support/image/ImageProcessor) 类来处理此数据预处理，如下所示。

要转换模型的图像数据，请执行以下操作：

1. 使用 Support Library [ImageProcessor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/ImageProcessor) 创建用于将图像数据转换为模型可用于运行预测的格式的对象：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L75-L84)）
    ```
    private val tfImageProcessor by lazy {
       val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
       ImageProcessor.Builder()
           .add(ResizeWithCropOrPadOp(cropSize, cropSize))
           .add(ResizeOp(
               tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
           .add(Rot90Op(-imageRotationDegrees / 90))
           .add(NormalizeOp(0f, 1f))
           .build()
    }
    ```
2. 从 Android 摄像头系统复制图像数据，并准备使用您的 [ImageProcessor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/ImageProcessor) 对象进行分析：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L198-L202)）
    ```
    // Copy out RGB bits to the shared buffer
    image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)  }

    // Process the image in Tensorflow
    val tfImage =  tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
    ```

注：从 Android 摄像头子系统提取图像信息时，请确保获取的是 RGB 格式的图像。此格式是 TensorFlow Lite [ImageProcessor](../api_docs/java/org/tensorflow/lite/support/image/ImageProcessor) 类所必需的，用于准备供模型分析的图像。如果 RGB 格式的图像包含 Alpha 通道，则该透明度数据将被忽略。

## 运行预测

在您的 Android 应用中，使用正确格式的图像数据创建 [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) 对象后，您可以在模型上运行该数据以生成预测或*推断*。本教程的示例代码使用了 [ObjectDetectionHelper](https://github.com/android/camera-samples/blob/main/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt) 类，该类将此代码封装在 `predict()` 方法中。

要对一组图像数据运行预测，请执行以下操作：

1. 通过将图像数据传递给预测函数来运行预测：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L204-L205)）
    ```
    // Perform the object detection for the current frame
    val predictions = detector.predict(tfImage)
    ```
2. 使用图像数据对您的 `tflite` 对象实例调用 Run 方法以生成预测：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L60-L63)）
    ```
    fun predict(image: TensorImage): List<ObjectPrediction> {
       tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)
       return predictions
    }
    ```

TensorFlow Lite Interpreter 对象接收该数据，在模型上运行该数据，并生成预测列表。要通过模型连续处理数据，请使用 `runForMultipleInputsOutputs()` 方法，这样系统就不会为每次预测运行创建并移除 Interpreter 对象。

## 处理模型输出

In your Android app, after you run image data against the object detection model, it produces a list of predictions that your app code must handle by executing additional business logic, displaying results to the user, or taking other actions.

任何给定的 TensorFlow Lite 模型的输出都根据其产生的预测数量（一个或多个）以及每个预测的描述性信息而有所不同。对于目标检测模型来说，预测通常包括用于指示在图像中检测到目标的位置的边界框数据。在示例代码中，返回的数据被格式化为 [ObjectPrediction](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L42-L58) 对象的列表，如下所示：（[代码引用](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/ObjectDetectionHelper.kt#L42-L58)）

```
val predictions get() = (0 until OBJECT_COUNT).map {
   ObjectPrediction(

       // The locations are an array of [0, 1] floats for [top, left, bottom, right]
       location = locations[0][it].let {
           RectF(it[1], it[0], it[3], it[2])
       },

       // SSD Mobilenet V1 Model assumes class 0 is background class
       // in label file and class labels start from 1 to number_of_classes + 1,
       // while outputClasses correspond to class index from 0 to number_of_classes
       label = labels[1 + labelIndices[0][it].toInt()],

       // Score is a single value of [0, 1]
       score = scores[0][it]
   )
}
```

![目标检测屏幕截图](../../images/lite/android/qs-obj-detect.jpeg){: .attempt-right width="250px"} 对于在该示例中使用的模型，每个预测包括目标的边界框位置、目标的标签，以及介于 0 和 1 之间的预测分数作为表示预测的置信度的浮点，其中 1 是最高置信度评级。一般来说，得分低于 50% (0.5) 的预测会被认为不确定。但是，如何处理低值预测结果取决于您和您应用的需要。

Once the model has returned a prediction result, your application can act on that prediction by presenting the result to your user or executing additional logic. In the case of the example code, the application draws a bounding box around the identified object and displays the class name on the screen. Review the [`CameraActivity.reportPrediction()`](https://github.com/android/camera-samples/blob/b0f4ec3a81ec30e622bb1ccd55f30e54ddac223f/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite/CameraActivity.kt#L236-L262) function in the example code for details.

## 后续步骤

- 在[示例](../examples)中探索 TensorFlow Lite 的各种用法。
- 在[模型](../models)部分中详细了解如何在 TensorFlow Lite 中使用机器学习模型。
- 有关在您的移动应用中实现机器学习的更多信息，请参阅 [TensorFlow Lite 开发者指南](../guide)。

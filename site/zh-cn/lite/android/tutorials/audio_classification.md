# 在 Android 中进行声音和单词识别

本教程展示如何在 Android 应用中使用 TensorFlow Lite 和预构建的机器学习模型来识别声音和口语。像本教程中所示的音频分类模型可用于检测活动、识别动作或识别语音命令。

![音频识别动画演示](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/audio_classification.gif){: .attempt-right} 本教程展示如何下载示例代码，将项目加载到 [Android Studio](https://developer.android.com/studio/)，并解释代码示例的关键部分，以便您可以开始将此功能添加到您自己的应用中。示例应用代码使用 TensorFlow [Task Library for Audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)，它处理大部分音频数据录制和预处理。有关如何预处理音频以供机器学习模型使用的更多信息，请参阅[音频数据准备和增强](https://www.tensorflow.org/io/tutorials/audio)。

## 基于机器学习的音频分类

本教程中的机器学习模型可识别在 Android 设备上使用麦克风录制的音频样本中的声音或单词。本教程中的示例应用允许您在 [YAMNet/分类器](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)（一个识别声音的模型）和一个识别特定口语的模型（使用 TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) 工具进行[训练{/a1）之间切换。这两个模型对音频剪辑进行预测，其中每个剪辑包含 15600 个单独样本，长度约为 1 秒。](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)

## 设置并运行示例

在本教程的第一部分，您从 GitHub 下载示例并使用 Android Studio 运行。本教程的以下部分将探索示例的相关部分，以便您将它们应用于您自己的 Android 应用。

### 系统要求

- [Android Studio](https://developer.android.com/studio/index.html) 2021.1.1 (Bumblebee) 或更高版本。
- Android SDK 31 或更高版本
- 最低操作系统版本为 SDK 24 (Android 7.0 - Nougat) 并且已启用开发者模式的 Android 设备。

### 获取示例代码

创建示例代码的本地副本。您将在 Android Studio 中使用此代码创建项目并运行示例应用。

要克隆和设置示例代码，请执行以下操作：

1. 克隆 Git 仓库
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. （可选）将您的 git 实例配置为使用稀疏签出，这样您就只有示例应用的文件：
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### 导入并运行项目

从下载的示例代码创建一个项目，构建并运行该项目。

要导入和构建示例代码项目，请执行以下操作：

1. 启动 [Android Studio](https://developer.android.com/studio)。
2. 在 Android Studio 中，选择 **File &gt; New &gt; Import Project**。
3. 导航到包含 `build.gradle` 文件的示例代码目录 (`.../examples/lite/examples/audio_classification/android/build.gradle`) 并选择该目录。

如果您选择了正确的目录，Android Studio 会创建一个新项目并进行构建。此过程可能需要几分钟，具体取决于您的计算机速度，以及您是否将 Android Studio 用于其他项目。构建完成后，Android Studio 会在 <strong>Build Output</strong> 状态面板中显示 <code>BUILD SUCCESSFUL</code> 消息。

要运行该项目，请执行以下操作：

1. 在 Android Studio 中，选择 **Run &gt; Run 'app'** 来运行项目。
2. 选择一台已连接的带麦克风的 Android 设备来测试应用。

注：如果使用模拟器运行应用，请确保从主机[启用音频输入](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio)。

接下来的部分将以此示例应用作为参考点，展示要将此功能添加到您自己的应用中，您需要对现有项目进行的修改。

## 添加项目依赖项

在您自己的应用中，您必须添加特定的项目依赖项才能运行 TensorFlow Lite 机器学习模型，并访问能够将音频等标准数据格式转换为您所使用的模型可以处理的张量数据格式的效用函数。

示例应用使用以下 TensorFlow Lite 库：

- [TensorFlow Lite Task library Audio API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/package-summary) - 提供所需的音频数据输入类、机器学习模型的执行以及模型处理的输出结果。

以下说明介绍如何将所需的项目依赖项添加到您自己的 Android 应用项目中。

要添加模块依赖项，请执行以下操作：

1. 在使用 TensorFlow Lite 的模块中，更新模块的 `build.gradle` 文件以包含以下依赖项。在示例代码中，此文件位于以下位置：`.../examples/lite/examples/audio_classification/android/build.gradle`
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
2. 在 Android Studio 中，选择 **File &gt; Sync Project with Gradle Files** 来同步项目依赖项。

## 初始化机器学习模型

在您的 Android 应用中，必须先使用参数初始化 TensorFlow Lite 机器学习模型，然后才能使用该模型运行预测。这些初始化参数取决于模型，并且可以包括设置，例如预测的默认最小准确度阈值以及模型可以识别的单词或声音的标签。

TensorFlow Lite 模型包含一个 `*.tflite` 文件，其中含有模型。该模型文件包含预测逻辑，并且通常包含有关如何解释预测结果的[元数据](../../models/convert/metadata)，例如预测类名称。模型文件应存储在开发项目的 `src/main/assets` 目录中，如代码示例中所示：

- `<project>/src/main/assets/yamnet.tflite`

为方便起见和提升代码可读性，该示例声明了一个为模型定义设置的伴随对象。

要在您的应用中初始化模型，请执行以下操作：

1. 创建一个伴随对象以定义模型的设置：
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
2. 通过构建一个 `AudioClassifier.AudioClassifierOptions` 对象为模型创建设置：
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
3. 使用此设置对象构造一个包含模型的 TensorFlow Lite [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) 对象：
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### 启用硬件加速

在您的应用中初始化 TensorFlow Lite 模型时，您应该考虑使用硬件加速功能来加速模型的预测计算。TensorFlow Lite [委托](https://www.tensorflow.org/lite/performance/delegates)是使用移动设备上的专用处理硬件（如图形处理单元 (GPU) 或张量处理单元 (TPU)）加速机器学习模型执行的软件模块。代码示例使用 NNAPI 委托来处理模型执行的硬件加速：

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

建议使用委托来运行 TensorFlow Lite 模型，但不是必需的。有关在 TensorFlow Lite 中使用委托的更多信息，请参阅 [TensorFlow Lite 委托](https://www.tensorflow.org/lite/performance/delegates)。

## 为模型准备数据

在您的 Android 应用中，您的代码通过将现有数据（如音频剪辑）转换为可以被模型处理的[张量](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)数据格式，向模型提供数据进行解释。传递给模型的张量中的数据必须具有与用于训练模型的数据格式相匹配的特定尺寸或形状。

此代码示例中使用的 [YAMNet/分类器模型](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)和自定义[语音命令](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition)模型接受的张量数据对象代表以 16kHz 录制 0.975 秒的单通道或单声道音频剪辑（15600 个样本）。对新的音频数据运行预测时，您的应用必须将音频数据转换为该大小和形状的张量数据对象。TensorFlow Lite Task Library [Audio API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) 为您处理数据转换。

在示例代码 `AudioClassificationHelper` 类中，应用使用 Android [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) 对象从设备麦克风录制实时音频。代码使用 [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) 来构建和配置该对象，以便以适合模型的采样率录制音频。该代码还使用 AudioClassifier 构建 [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) 对象来存储转换后的音频数据。然后，TensorAudio 对象将传递给模型进行分析。

要向机器学习模型提供音频数据，请执行以下操作：

- 使用 `AudioClassifier` 对象创建一个 `TensorAudio` 对象和一个 `AudioRecord` 对象：
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

注：您的应用必须请求使用 Android 设备麦克风录制音频的权限。有关示例，请查看项目中的 `fragments/PermissionsFragment` 类获取示例。有关请求权限的更多信息，请参阅 [Android 中的权限](https://developer.android.com/guide/topics/permissions/overview)。

## 运行预测

在您的 Android 应用中，将 [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) 对象和 [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) 对象连接到 AudioClassifier 对象后，就可以针对该数据运行模型来生成预测或{ em2}推断。本教程的示例代码对来自特定速率的实时录制音频输入流的剪辑运行预测。

模型执行会消耗大量资源，因此在单独的后台线程上运行机器学习模型预测非常重要。示例应用使用 `[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)` 对象将模型处理与应用的其他功能隔离开来。

识别具有清晰开头和结尾的声音（例如单词）的音频分类模型可以分析重叠的音频剪辑，从而对传入的音频流生成更准确的预测。此方法有助于模型避免错过对剪辑结尾处被截断的单词的预测。在示例应用中，每次运行预测时，代码都会从音频录制缓冲区中抓取最新的 0.975 秒剪辑并对其进行分析。您可以将模型分析线程执行池的 `interval` 值设置为短于正在分析的剪辑的长度，从而使模型分析重叠的音频剪辑。例如，如果模型分析 1 秒的剪辑并将间隔设置为 500 毫秒，则模型每次将分析前一个剪辑的后半部分和 500 毫秒的新音频数据，从而产生 50% 的剪辑分析重叠。

要开始对音频数据运行预测，请执行以下操作：

1. 使用 `AudioClassificationHelper.startAudioClassification()` 方法开始为模型录音：
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
2. 通过在 `ScheduledThreadPoolExecutor` 对象中设置固定速率 `interval` 来设置模型从音频剪辑生成推断的频率：
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
3. 上面代码中的 `classifyRunnable` 对象执行 `AudioClassificationHelper.classifyAudio()` 方法，该方法从录制器加载最新的可用音频数据并执行预测：
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

注意：不要在应用的主执行线程上运行机器学习模型预测。这样做可能会导致您的应用用户界面变慢或无响应。

### 停止预测处理

确保您的应用代码在应用的音频处理 Fragment 或 Activity 失去焦点时停止进行音频分类。持续运行机器学习模型会对 Android 设备的电池寿命产生重大影响。使用与音频分类关联的 Android Activity 或 Fragment 的 `onPause()` 方法来停止音频录制和预测处理。

要停止录音和分类，请执行以下操作：

- 使用 `AudioClassificationHelper.stopAudioClassification()` 方法停止录音和模型执行，如以下 `AudioFragment` 类所示：
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## 处理模型输出

在您的 Android 应用中，处理一个音频剪辑后，模型会生成一个预测列表，您的应用代码必须通过执行额外的业务逻辑、向用户显示结果或采取其他操作来处理这些预测。任何给定的 TensorFlow Lite 模型的输出都会根据其产生的预测数量（一个或多个）以及每个预测的描述信息而有所不同。对于示例应用程序中的模型，预测是识别的声音或单词的列表。代码示例中使用的 AudioClassifier 选项对象允许您使用 `setMaxResults()` 方法设置最大预测数，如[初始化机器学习模型](#Initialize_the_ML_model)部分中所示。

要从模型获取预测结果，请执行以下操作：

1. 获取 [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) 对象的 `classify()` 方法的结果并将它们传递给侦听器对象（代码引用）：
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
2. 使用侦听器的 onResult() 函数通过执行业务逻辑或向用户显示结果来处理输出：
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

本示例中使用的模型会生成一个预测列表，其中包含分类的声音或单词的标签，以及一个介于 0 和 1 之间的预测分数（作为表示预测置信度的浮点数），其中 1 是最高置信度。一般来说，分数低于 50% (0.5) 的预测被认为是不确定的。但是，如何处理低值预测结果取决于您和您应用的需求。

一旦模型返回一组预测结果，您的应用就可以根据预测执行操作，将结果呈现给用户或执行其他逻辑。对于示例代码，应用会在应用用户界面中列出已识别的声音或单词。

## 后续步骤

您可以在 [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt) 以及[预训练模型指南](https://www.tensorflow.org/lite/models/trained)页面上找到其他用于音频处理的 TensorFlow Lite 模型。有关在移动应用中使用 TensorFlow Lite 实现机器学习的更多信息，请参阅 [TensorFlow Lite 开发者指南](https://www.tensorflow.org/lite/guide)。

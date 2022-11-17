# 使用 Android 实现问答

![Android 中的问题示例应用](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

本教程向您展示如何使用 TensorFlow Lite 构建 Android 应用，以提供以自然语言文本组织的问题的答案。[示例应用](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)使用[自然语言 (NL) 的 Task 库](../../inference_with_metadata/task_library/overview#supported_tasks)中的 *BERT 问答器* ([`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer)) API 来启用问答机器学习模型。此应用是为实体 Android 设备设计的，但也可以在设备模拟器上运行。

如果您要更新现有项目，则可以将示例应用用作参考或模板。有关如何向现有应用添加问答的说明，请参阅[更新和修改应用](#modify_applications)。

## 问答概览

*问答*是回答以自然语言提出的问题的机器学习任务。经过训练的问答模型接收文本段落和问题作为输入，并尝试根据其对段落中信息的解释来回答问题。

问答模型在问答数据集上进行训练，该数据集由阅读理解数据集以及基于不同文本段的问答对组成。

有关本教程中模型生成方式的更多信息，请参阅[使用 TensorFlow Lite Model Maker 进行 BERT 问答](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)教程。

## 模型和数据集

示例应用使用 Mobile BERT Q&amp;A ([`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1)) 模型，它是 [BERT](https://arxiv.org/abs/1810.04805)（来自 Transformer 的双向编码器表示）的更轻、更快的版本。有关 `mobilebert` 的更多信息，请参阅 [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) 研究论文。

`mobilebert` 模型使用 Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) 数据集进行训练，该数据集是一个由 Wikipedia 中的文章和一组针对每篇文章的问答对组成的阅读理解数据集。

## 设置并运行示例应用

要设置问答应用，请从 [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) 下载示例应用并使用 [Android Studio](https://developer.android.com/studio/) 运行。

### 系统要求

- **[Android Studio](https://developer.android.com/studio/index.html)** 2021.1.1 (Bumblebee) 或更高版本。
- Android SDK 31 或更高版本
- 最低操作系统版本为 SDK 21 (Android 7.0 - Nougat) 并且已启用[开发者模式](https://developer.android.com/studio/debug/dev-options)的 Android 设备，或者 Android 模拟器。

### 获取示例代码

创建一个示例代码的本地副本。您将使用此代码在 Android Studio 中创建一个项目并运行示例应用。

要克隆和设置示例代码，请执行以下操作：

1. 克隆 git 仓库
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. （可选）将您的 git 实例配置为使用稀疏签出，这样您就只有问答示例应用的文件：
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/bert_qa/android
        </pre>

### 导入并运行项目

从下载的示例代码创建一个项目，构建并运行该项目。

要导入和构建示例代码项目，请执行以下操作：

1. 启动 [Android Studio](https://developer.android.com/studio)。
2. 在 Android Studio 中，选择 **File &gt; New &gt; Import Project**。
3. 导航到包含 build.gradle 文件的示例代码目录 (`.../examples/lite/examples/bert_qa/android/build.gradle`) 并选择该目录。
4. 如果 Android Studio 请求 Gradle Sync，请选择 OK。
5. 确保您的 Android 设备已连接到计算机并且已启用开发者模式。点击绿色 `Run` 箭头。

如果您选择了正确的目录，Android Studio 会创建并构建一个新项目。这个过程可能需要几分钟，具体取决于您的计算机速度，以及您是否曾将 Android Studio 用于其他项目。构建完成后，Android Studio 会在 <strong>Build Output</strong> 状态面板中显示 <code>BUILD SUCCESSFUL</code> 消息。

要运行项目，请执行以下操作：

1. 在 Android Studio 中，选择 **Run &gt; Run…** 来运行项目。
2. 选择一台已连接的 Android 设备（或模拟器）来测试应用。

### 使用应用

在 Android Studio 中运行项目后，应用会自动在连接的设备或设备模拟器上打开。

要使用问答器示例应用，请执行以下操作：

1. 从主题列表中选择一个主题。
2. 选择一个建议的问题或在文本框中输入您自己的问题。
3. 切换橙色箭头以运行模型。

应用尝试从段落文本中确定问题的答案。如果模型在文章中检测到答案，应用会为用户突出显示相关的文本范围。

您现在有一个正常运行的问答应用。使用以下部分可以更好地了解示例应用的运作方式，以及如何在生产应用中实现问答功能：

- [应用的运作方式](#how_it_works) - 示例应用的结构和关键文件的演练。

- [修改您的应用](#modify_applications) - 将问答添加到现有应用的说明。

## 示例应用的运作方式 {:#how_it_works}

该应用程序使用[自然语言 (NL) 的 Task 库](../../inference_with_metadata/task_library/overview#supported_tasks)软件包中的 `BertQuestionAnswerer` API。MobileBERT 模型使用 TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) 进行训练。该应用默认在 CPU 上运行，也可以选择使用 GPU 或 NNAPI 委托进行硬件加速。

以下文件和目录包含此应用的关键代码：

- [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt) - 初始化问答器并处理模型和委托选择。
- [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt) - 处理和格式化结果。
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt) - 提供应用的组织逻辑。

## 修改您的应用 {:#modify_applications}

以下部分介绍了修改您自己的 Android 应用以运行示例应用中显示的模型的关键步骤。这些说明使用示例应用作为参考点。您自己的应用所需的具体更改可能与示例应用不同。

### 打开或创建一个 Android 项目

您需要在 Android Studio 中创建一个 Android 开发项目来遵循这些说明的其余部分。按照以下说明打开现有项目或创建新项目。

打开现有 Android 开发项目：

- 在 Android Studio 中，选择 *File &gt; Open* 并选择一个现有项目。

创建一个基本的 Android 开发项目：

- 按照 Android Studio 中的说明[创建一个基本项目](https://developer.android.com/studio/projects/create-project)。

有关使用 Android Studio 的更多信息，请参阅 [Android Studio 文档](https://developer.android.com/studio/intro)。

### 添加项目依赖项

在您自己的应用中，添加特定的项目依赖项来运行 TensorFlow Lite 机器学习模型并访问效用函数。这些函数可将字符串等数据转换为模型可以处理的张量数据格式。以下说明阐述了如何将所需的项目和模块依赖项添加到您自己的 Android 应用项目中。

要添加模块依赖项，请执行以下操作：

1. 在使用 TensorFlow Lite 的模块中，更新模块的 `build.gradle` 文件以包含以下依赖项。

    在示例应用中，依赖项位于 [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle) 中：

    ```
    dependencies {
      ...
      // Import tensorflow library
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'

      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    项目必须包含 Text 任务库 (`tensorflow-lite-task-text`)。

    如果您想修改此应用以在图形处理单元 (GPU) 上运行，GPU 库 (`tensorflow-lite-gpu-delegate-plugin`) 提供了在 GPU 上运行应用的基础架构，而委托 (`tensorflow-lite-gpu`) 提供了兼容性列表。

2. 在 Android Studio 中，通过选择以下选项来同步项目依赖项：**File &gt; Sync Project with Gradle Files**。

### 初始化机器学习模型 {:#initialize_models}

在您的 Android 应用中，您必须先使用参数初始化 TensorFlow Lite 机器学习模型，然后才能使用模型运行预测。

TensorFlow Lite 模型存储为 `*.tflite` 文件。该模型文件包含预测逻辑，并且通常包含有关如何解释预测结果的[元数据](../../models/convert/metadata)。通常，模型文件应存储在开发项目的 `src/main/assets` 目录中，如代码示例中所示：

- `<project>/src/main/assets/mobilebert_qa.tflite`

注：示例应用使用 [`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle) 文件在构建时下载 [mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) 模型和[段落文本](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json)。生产应用不需要这种方法。

为方便起见和提升代码可读性，该示例声明了一个为模型定义设置的伴随对象。

要在您的应用中初始化模型，请执行以下操作：

1. 创建一个伴随对象来定义模型的设置。在示例应用中，此对象位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106) 中：

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

2. 通过构建 `BertQaHelper` 对象为模型创建设置，并使用 `bertQuestionAnswerer` 构造 TensorFlow Lite 对象。

    在示例应用中，它位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76) 的 `setupBertQuestionAnswerer()` 函数中：

    ```
    class BertQaHelper(
        ...
    ) {
        ...
        init {
            setupBertQuestionAnswerer()
        }

        fun clearBertQuestionAnswerer() {
            bertQuestionAnswerer = null
        }

        private fun setupBertQuestionAnswerer() {
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
            ...
            val options = BertQuestionAnswererOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .build()

            try {
                bertQuestionAnswerer =
                    BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
            } catch (e: IllegalStateException) {
                answererListener
                    ?.onError("Bert Question Answerer failed to initialize. See error logs for details")
                Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            }
        }
        ...
        }
    ```

### 启用硬件加速（可选）{:#hardware_acceleration}

在您的应用中初始化 TensorFlow Lite 模型时，您应该考虑使用硬件加速功能来加速模型的预测计算。TensorFlow Lite [委托](https://www.tensorflow.org/lite/performance/delegates)是使用移动设备上的专用处理硬件（如图形处理单元 (GPU) 或张量处理单元 (TPU)）加速机器学习模型执行的软件模块。

要在您的应用中启用硬件加速，请执行以下操作：

1. 创建一个变量来定义应用将使用的委托。在示例应用中，此变量位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31) 中：

    ```
    var currentDelegate: Int = 0
    ```

2. 创建一个委托选择器。在示例应用中，委托选择器位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62) 的 `setupBertQuestionAnswerer` 函数中：

    ```
    when (currentDelegate) {
        DELEGATE_CPU -> {
            // Default
        }
        DELEGATE_GPU -> {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                baseOptionsBuilder.useGpu()
            } else {
                answererListener?.onError("GPU is not supported on this device")
            }
        }
        DELEGATE_NNAPI -> {
            baseOptionsBuilder.useNnapi()
        }
    }
    ```

建议使用委托来运行 TensorFlow Lite 模型，但不是必需的。有关在 TensorFlow Lite 中使用委托的更多信息，请参阅 [TensorFlow Lite 委托](https://www.tensorflow.org/lite/performance/delegates)。

### 为模型准备数据

在您的 Android 应用中，您的代码通过将现有数据（如原始文本）转换为可以被模型处理的[张量](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor)数据格式，向模型提供数据进行解释。传递给模型的张量必须具有与用于训练模型的数据格式相匹配的特定尺寸或形状。此问答应用接受[字符串](https://developer.android.com/reference/java/lang/String.html)作为文本段落和问题的输入。该模型不识别特殊字符和非英文单词。

要向模型提供段落文本数据，请执行以下操作：

1. 使用 `LoadDataSetClient` 对象将段落文本数据加载到应用。在示例应用中，它位于 [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45) 中

    ```
    fun loadJson(): DataSet? {
        var dataSet: DataSet? = null
        try {
            val inputStream: InputStream = context.assets.open(JSON_DIR)
            val bufferReader = inputStream.bufferedReader()
            val stringJson: String = bufferReader.use { it.readText() }
            val datasetType = object : TypeToken<DataSet>() {}.type
            dataSet = Gson().fromJson(stringJson, datasetType)
        } catch (e: IOException) {
            Log.e(TAG, e.message.toString())
        }
        return dataSet
    }
    ```

2. 使用 `DatasetFragment` 对象列出每个文本段落的标题并启动 **TFL Question and Answer** 屏幕。在示例应用中，它位于 [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt) 中：

    ```
    class DatasetFragment : Fragment() {
        ...
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)
            val client = LoadDataSetClient(requireActivity())
            client.loadJson()?.let {
                titles = it.getTitles()
            }
            ...
        }
       ...
    }
    ```

3. 使用 `DatasetAdapter` 对象中的 `onCreateViewHolder` 函数来显示每个文本段落的标题。在示例应用中，它位于 [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt) 中：

    ```
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }
    ```

要将用户问题提供给模型，请执行以下操作：

1. 使用 `QaAdapter` 对象向模型提供问题。在示例应用中，它位于 [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt) 中：

    ```
    class QaAdapter(private val question: List<String>, private val select: (Int) -> Unit) :
      RecyclerView.Adapter<QaAdapter.ViewHolder>() {

      inner class ViewHolder(private val binding: ItemQuestionBinding) :
          RecyclerView.ViewHolder(binding.root) {
          init {
              binding.tvQuestionSuggestion.setOnClickListener {
                  select.invoke(adapterPosition)
              }
          }

          fun bind(question: String) {
              binding.tvQuestionSuggestion.text = question
          }
      }
      ...
    }
    ```

### 运行预测

在您的 Android 应用中，一旦初始化 [BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer) 对象，就可以开始以自然语言文本的形式向模型输入问题。模型会尝试在文本段落中识别答案。

要运行预测，请执行以下操作：

1. 创建一个 `answer` 函数，它运行模型并测量识别答案所花费的时间（`inferenceTime`）。在示例应用中，`answer` 函数位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98) 中：

    ```
    fun answer(contextOfQuestion: String, question: String) {
        if (bertQuestionAnswerer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val answers = bertQuestionAnswerer?.answer(contextOfQuestion, question)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        answererListener?.onResults(answers, inferenceTime)
    }
    ```

2. 将 `answer` 的结果传递给侦听器对象。

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### 处理模型输出

输入问题后，模型会在段落中提供最多五个可能的答案。

要从模型获取结果，请执行以下操作：

1. 为侦听器对象创建一个 `onResult` 函数来处理输出。在示例应用中，侦听器对象位于 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98) 中

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

2. 根据结果突出显示段落的部分。在示例应用中，它位于 [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208) 中：

    ```
    override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
        results?.first()?.let {
            highlightAnswer(it.text)
        }

        fragmentQaBinding.tvInferenceTime.text = String.format(
            requireActivity().getString(R.string.bottom_view_inference_time),
            inferenceTime
        )
    }
    ```

一旦模型返回一组结果，您的应用就可以根据预测执行操作，将结果呈现给用户或执行其他逻辑。

## 后续步骤

- 要从头开始训练和实现模型，请参阅[使用 TensorFlow Lite Model Maker 进行问答](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)教程。
- 探索更多[适用于 TensorFlow 的文本处理工具](https://www.tensorflow.org/text)。
- 在 [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1) 上下载其他 BERT 模型。
- 在[示例](../../examples)中探索 TensorFlow Lite 的各种用法。

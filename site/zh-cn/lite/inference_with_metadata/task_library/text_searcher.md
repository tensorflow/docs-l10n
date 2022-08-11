# 集成文本搜索器

文本搜索允许在语料库中搜索语义相似的文本。它的工作原理是将搜索查询嵌入到表示查询语义的高维向量中，然后在预定义的自定义索引中使用 [ScaNN](https://github.com/google-research/google-research/tree/master/scann)（可扩缩最近邻）进行相似度搜索。

与文本分类（例如，[BERT 自然语言分类器](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier)）不同，扩展可识别的项的数量不需要重新训练整个模型。只需重新构建索引即可添加新的项。这还可以处理更大（超过 10 万项）的语料库。

使用 Task Library `TextSearcher` API 将您的自定义文本搜索器部署到您的移动应用中。

## TextSearcher API 的主要功能

- 将单个字符串作为输入，在索引中执行嵌入向量提取和最近邻搜索。

- 输入文本处理，包括对输入文本的计算图内或计算图外的 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 或 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 标记。

## 前提条件

在使用 `TextSearcher` API 之前，需要基于要搜索的自定义语料库构建索引。这可以使用 [Model Maker Searcher API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) 按照并改编[教程](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher)来实现。

为此，您需要进行以下准备：

- TFLite 文本嵌入器模型，如通用语句编码器。例如，
    - 在此 [Colab](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/colab/on_device_text_to_image_search_tflite.ipynb) 中重新训练，并针对设备端推断进行了优化的[模型](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/searcher/text_to_image_blogpost/text_embedder.tflite)。在 Pixel 6 上查询一个文本字符串只需要 6ms。
    - [量化](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)模型，比上述模型小，但每个嵌入向量查询需要 38ms。
- 您的文本语料库。

完成这一步后，您应该有了一个独立的 TFLite 搜索器模型（例如，`mobilenet_v3_searcher.tflite`），它是原始的文本嵌入器模型，并将索引附加到 [TFLite 模型元数据](https://www.tensorflow.org/lite/models/convert/metadata)中。

## 用 Java 运行推断

### 步骤 1：导入 Gradle 依赖项和其他设置

将 `.tflite` 搜索器模型文件复制到将要运行模型的 Android 模块的资源目录下。指定不压缩该文件，并将 TensorFlow Lite 库添加到模块的 `build.gradle` 文件中。

```java
android {
    // Other settings

    // Specify tflite index file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
}
```

### 第 2 步：使用模型

```java
// Initialization
TextSearcherOptions options =
    TextSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
TextSearcher textSearcher =
    textSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = textSearcher.search(text);
```

请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/searcher/TextSearcher.java)，了解有关配置 `TextSearcher` 的更多选项。

## 用 C++ 运行推断

```c++
// Initialization
TextSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<TextSearcher> text_searcher = TextSearcher::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
const SearchResult result = text_searcher->Search(input_text).value();
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_searcher.h)，了解有关配置 `TextSearcher` 的更多选项。

## 在 Python 中运行推断

### 第 1 步：安装 TensorFlow Lite Support Pypi 软件包

您可以使用以下命令安装 TensorFlow Lite Support Pypi 软件包：

```sh
pip install tflite-support
```

### 第 2 步：使用模型

```python
from tflite_support.task import text

# Initialization
text_searcher = text.TextSearcher.create_from_file(model_path)

# Run inference
result = text_searcher.search(text)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_searcher.py)，了解有关配置 `TextSearcher` 的更多选项。

## 结果示例

```
Results:
 Rank#0:
  metadata: The sun was shining on that day.
  distance: 0.04618
 Rank#1:
  metadata: It was a sunny day.
  distance: 0.10856
 Rank#2:
  metadata: The weather was excellent.
  distance: 0.15223
 Rank#3:
  metadata: The cat is chasing after the mouse.
  distance: 0.34271
 Rank#4:
  metadata: He was very happy with his newly bought car.
  distance: 0.37703
```

用您自己的模型和测试数据试用简单的 [TextSearcher CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textsearcher)。

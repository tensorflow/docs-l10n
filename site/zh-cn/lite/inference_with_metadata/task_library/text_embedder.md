# 集成文本嵌入器

文本嵌入器允许将文本嵌入到代表文本语义的高维特征向量中，然后将其与其他文本的特征向量进行比较，以评估它们的语义相似度。

与[文本搜索](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher)不同，文本嵌入器允许动态计算文本之间的相似度，而不是通过从文本语料库构建的预定义索引进行搜索。

使用 Task Library `TextEmbedder` API 将您的自定义文本嵌入器部署到您的移动应用中。

## TextEmbedder API 的主要功能

- 输入文本处理，包括对输入文本的计算图内或计算图外的 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 或 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 标记。

- 内置效用函数，用于计算特征向量之间的[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)。

## 支持的文本嵌入器模型

以下模型保证可与 `TextEmbedder` API 兼容。

- [来自 TensorFlow Hub 的通用句子编码器 TFLite 模型](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 C++ 运行推断

```c++
// Initialization.
TextEmbedderOptions options:
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<TextEmbedder> text_embedder = TextEmbedder::CreateFromOptions(options).value();

// Run inference with your two inputs, `input_text1` and `input_text2`.
const EmbeddingResult result_1 = text_embedder->Embed(input_text1);
const EmbeddingResult result_2 = text_embedder->Embed(input_text2);

// Compute cosine similarity.
double similarity = TextEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector()
    result_2.embeddings[0].feature_vector());
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_embedder.h)，了解有关配置 `TextEmbedder` 的更多选项。

## 用 Python 运行推断

### 第 1 步：安装 TensorFlow Lite Support Pypi 软件包

您可以使用以下命令安装 TensorFlow Lite Support Pypi 软件包：

```sh
pip install tflite-support
```

### 第 2 步：使用模型

```python
from tflite_support.task import text

# Initialization.
text_embedder = text.TextEmbedder.create_from_file(model_path)

# Run inference on two texts.
result_1 = text_embedder.embed(text_1)
result_2 = text_embedder.embed(text_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = text_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_embedder.py)，了解有关配置 `TextEmbedder` 的更多选项。

## 结果示例

归一化特征向量之间的余弦相似度返回 -1 到 1 之间的分数。分数越高越好，即余弦相似度为 1 表示两个向量完全相同。

```
Cosine similarity: 0.954312
```

用您自己的模型和测试数据试用简单的 [TextEmbedder CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder)。

## 模型兼容性要求

`TextEmbedder` API 需要具有强制性 [TFLite 模型元数据](https://www.tensorflow.org/lite/models/convert/metadata)的 TFLite 模型。

支持三种主要类型的模型：

- 基于 BERT 的模型（请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h)，了解详细信息）：

    - 正好 3 个输入张量 (kTfLiteString)

        - IDs 张量，元数据名称为 "ids"。
        - 掩码张量，元数据名称为 "mask"。
        - 分割 IDs 张量，元数据名称为 "segment_ids"。

    - 正好一个输出张量 (kTfLiteUInt8/kTfLiteFloat32)

        - 其中 `N` 分量对应于该输出层的返回特征向量的 `N` 维
        - 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`。

    - 用于 WordPiece/SentencePiece 标记器的 input_process_units

- 基于通用语句编码器的模型（请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h)，了解详细信息）：

    - 恰好 3 个输入张量 (kTfLiteString)

        - 查询文本张量，元数据名称为 "inp_text"。
        - 响应上下文张量，元数据名称为 "res_context"。
        - 响应文本张量，元数据名称为 "res_text"。

    - 正好 2 个输出张量 (kTfLiteUInt8/kTfLiteFloat32)

        - 查询编码张量，元数据名称为 "query_encoding"。
        - 响应编码张量，元数据名称为 "response_encoding"。
        - 均有 `N` 个分量对应于该输出层的返回特征向量的 `N` 个维度。
        - 均有 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`。

- 具有以下条件的任何文本嵌入器模型：

    - 一个文本张量 (kTfLiteString)

    - 至少一个输出嵌入向量张量 (kTfLiteUInt8/kTfLiteFloat32)

        - 其中 `N` 个分量对应于该输出层的返回特征向量的 `N` 个维度。
        - 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`。

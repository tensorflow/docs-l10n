# 集成图像嵌入器

图像嵌入器允许将图像嵌入到代表图像语义的高维特征向量中，然后将其与其他图像的特征向量进行比较，以评估它们的语义相似度。

与[图像搜索](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher)不同，图像嵌入器允许动态计算图像之间的相似度，而不是通过从图像语料库构建的预定义索引进行搜索。

使用 Task Library `ImageEmbedder` API 将您的自定义图像嵌入器部署到您的移动应用中。

## ImageEmbedder API 的主要功能

- 输入图像处理，包括旋转、调整大小和色彩空间转换。

- 输入图像的感兴趣区域。

- 内置效用函数，用于计算特征向量之间的[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)。

## 支持的图像嵌入器模型

以下模型保证可与 `ImageEmbedder` API 兼容。

- 来自 [TensorFlow Hub 上的 Google 图像模块集合](https://tfhub.dev/google/collections/image/1)的特征向量模型。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 C++ 运行推断

```c++
// Initialization
ImageEmbedderOptions options:
options.mutable_model_file_with_metadata()->set_file_name(model_path);
options.set_l2_normalize(true);
std::unique_ptr<ImageEmbedder> image_embedder = ImageEmbedder::CreateFromOptions(options).value();

// Create input frame_buffer1 and frame_buffer_2 from your inputs `image_data1`, `image_data2`, `image_dimension1` and `image_dimension2`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer1 = CreateFromRgbRawBuffer(
      image_data1, image_dimension1);
std::unique_ptr<FrameBuffer> frame_buffer1 = CreateFromRgbRawBuffer(
      image_data2, image_dimension2);

// Run inference on two images.
const EmbeddingResult result_1 = image_embedder->Embed(*frame_buffer_1);
const EmbeddingResult result_2 = image_embedder->Embed(*frame_buffer_2);

// Compute cosine similarity.
double similarity = ImageEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector(),
    result_2.embeddings[0].feature_vector());
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_embedder.h)，了解有关配置 `ImageEmbedder` 的更多选项。

## 用 Python 运行推断

### 第 1 步：安装 TensorFlow Lite Support Pypi 软件包

您可以使用以下命令安装 TensorFlow Lite Support Pypi 软件包：

```sh
pip install tflite-support
```

### 第 2 步：使用模型

```python
from tflite_support.task import vision

# Initialization.
image_embedder = vision.ImageEmbedder.create_from_file(model_path)

# Run inference on two images.
image_1 = vision.TensorImage.create_from_file('/path/to/image1.jpg')
result_1 = image_embedder.embed(image_1)
image_2 = vision.TensorImage.create_from_file('/path/to/image2.jpg')
result_2 = image_embedder.embed(image_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = image_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_embedder.py)，了解有关配置 `ImageEmbedder` 的更多选项。

## 结果示例

归一化特征向量之间的余弦相似度返回 -1 到 1 之间的分数。分数越高越好，即余弦相似度为 1 表示两个向量完全相同。

```
Cosine similarity: 0.954312
```

用您自己的模型和测试数据试用简单的 [ImageEmbedder CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder)。

## 模型兼容性要求

`ImageEmbedder` API 要求 TFLite 模型具有可选但强烈推荐的 [TFLite 模型元数据](https://www.tensorflow.org/lite/models/convert/metadata)。

兼容的图像嵌入器模型应满足以下要求：

- 输入图像张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 图像输入大小为 `[batch x height x width x channels]`。
    - 不支持批量推断（`batch` 必须为 1）。
    - 仅支持 RGB 输入（`channels` 必须为 3）。
    - 如果类型为 kTfLiteFloat32，则必须将 NormalizationOptions 附加到元数据以进行输入归一化。

- 至少一个输出张量 (kTfLiteUInt8/kTfLiteFloat32)

    - 其中 `N` 个分量对应于该输出层的返回特征向量的 `N` 个维度。
    - 2 或 4 个维度，即 `[1 x N]` 或 `[1 x 1 x 1 x N]`。

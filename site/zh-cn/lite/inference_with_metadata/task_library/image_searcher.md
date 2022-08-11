# 集成图像搜索器

图像搜索允许在图像数据库中搜索相似的图像。它的工作原理是将搜索查询嵌入到表示查询语义的高维向量中，然后在预定义的自定义索引中使用 [ScaNN](https://github.com/google-research/google-research/tree/master/scann)（可扩缩最近邻）进行相似度搜索。

与[图像分类](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)不同，扩展可识别项的数量不需要重新训练整个模型。只需重新构建索引即可添加新项。这还可以处理更大的（超过 10 万项）图像数据库。

使用 Task Library `ImageSearcher` API 将您的自定义图像搜索器部署到您的移动应用中。

## ImageSearcher API 的主要功能

- 将单个图像作为输入，在索引中执行嵌入向量提取和最近邻搜索。

- 输入图像处理，包括旋转、调整大小和色彩空间转换。

- 输入图像的感兴趣区域。

## 前提条件

在使用 `ImageSearcher` API 之前，需要基于要搜索的自定义图片语料库构建索引。这可以使用 [Model Maker Searcher API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) 按照并改编[教程](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher)来实现。

为此，您需要进行以下准备：

- TFLite 图像嵌入器模型，如 [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1)。在 [TensorFlow Hub 上的 Google 图像模块集合](https://tfhub.dev/google/collections/image/1)中查看更多预训练的嵌入器模型（也称为特征向量模型）。
- 您的图片库。

完成这一步后，您应该有了一个独立的 TFLite 搜索器模型（例如，`mobilenet_v3_searcher.tflite`），它是原始的图像嵌入器模型，并将索引附加到 [TFLite 模型元数据](https://www.tensorflow.org/lite/models/convert/metadata)中。

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
ImageSearcherOptions options =
    ImageSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
ImageSearcher imageSearcher =
    ImageSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = imageSearcher.search(image);
```

请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/searcher/ImageSearcher.java)，了解有关配置 `ImageSearcher` 的更多选项。

## 用 C++ 运行推断

```c++
// Initialization
ImageSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<ImageSearcher> image_searcher = ImageSearcher::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const SearchResult result = image_searcher->Search(*frame_buffer).value();
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_searcher.h)，了解有关配置 `ImageSearcher` 的更多选项。

## 用 Python 运行推断

### 第 1 步：安装 TensorFlow Lite Support Pypi 软件包

您可以使用以下命令安装 TensorFlow Lite Support Pypi 软件包：

```sh
pip install tflite-support
```

### 第 2 步：使用模型

```python
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_searcher.py)，了解有关配置 `ImageSearcher` 的更多选项。

## 结果示例

```
Results:
 Rank#0:
  metadata: burger
  distance: 0.13452
 Rank#1:
  metadata: car
  distance: 1.81935
 Rank#2:
  metadata: bird
  distance: 1.96617
 Rank#3:
  metadata: dog
  distance: 2.05610
 Rank#4:
  metadata: cat
  distance: 2.06347
```

用您自己的模型和测试数据试用简单的 [ImageSearcher CLI 演示工具](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher)。

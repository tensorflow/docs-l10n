# 文本分类

Use a TensorFlow Lite model to category a paragraph into predefined groups.

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## 开始


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the guide of [TensorFLow Lite Task Library](../../inference_with_metadata/task_library/nl_classifier) to integrate text classification models within just a few lines of code. You can also integrate the model using the [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android 示例</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android example</a>

如果您使用的不是 Android 平台，或者您已经熟悉 <a>TensorFlow Lite API</a>，则可以下载我们的起始文本分类模型。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Download starter model</a>

## 工作方式

文本分类根据内容将段落分类到预定义组中。

此预训练模型可预测段落的情感是正面的还是负面的。它在 Mass 等人提供的 [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) 上进行训练，该数据集包含标记为正面或负面的 IMDB 电影评论。

下面是使用该模型对段落进行分类的步骤：

1. Tokenize the paragraph and convert it to a list of word ids using a predefined vocabulary.
2. 将该列表馈送到 TensorFlow Lite 模型。
3. 从模型输出获取该段落为正面或负面评价的概率。

### 说明

- 仅支持英语。
- 此模型在电影评论数据集上进行训练，因此，对其他领域的文本进行分类时，您可能发现准确率有所降低。

## 性能基准

性能基准数值使用[此处所述](https://www.tensorflow.org/lite/performance/benchmarks)工具生成。

<table>
  <thead>
    <tr>
      <th>模型名称</th>
      <th>模型大小</th>
      <th>设备</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite">文本分类</a></td>
    <td rowspan="3">       0.6 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>0.025ms**</td>
  </tr>
</table>

* 使用 4 个线程。

** 为了获得最佳性能结果，在 iPhone 上使用 2 个线程。

## 示例输出

Text | Negative (0) | Positive (1)
--- | --- | ---
This is the best movie I’ve seen in recent | 25.3% | 74.7%
: years. Strongly recommend it!              :              :              : |  |
What a waste of my time. | 72.5% | 27.5%

## 使用训练数据集

Follow this [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) to apply the same technique used here to train a text classification model using your own datasets. With the right dataset, you can create a model for use cases such as document categorization or toxic comments detection.

## 详细了解文本分类

- 使用 4 个线程。

# 文本分类

使用预训练的模型将段落分类到预定义组中。

## 开始

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example applications that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android example</a>

如果您使用的不是 Android 平台，或者您已经熟悉 <a>TensorFlow Lite API</a>，则可以下载我们的起始文本分类模型。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Download starter model</a>

## 工作方式

Text classification categorizes a paragraph into predefined groups based on its content.

This pretrained model predicts if a paragraph's sentiment is positive or negative. It was trained on [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) from Mass et al, which consists of IMDB movie reviews labeled as either positive or negative.

下面是使用该模型对段落进行分类的步骤：

1. 对段落进行分词，并使用预定义词汇表将其转换为一个单词 ID 列表。
2. 将该列表馈送到 TensorFlow Lite 模型。
3. Get the probability of the paragraph being positive or negative from the model outputs.

### 说明

- 仅支持英语。
- 此模型在电影评论数据集上进行训练，因此，对其他领域的文本进行分类时，您可能发现准确率有所降低。

## 性能基准

Performance benchmark numbers are generated with the tool [described here](https://www.tensorflow.org/lite/performance/benchmarks).

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
    <td rowspan="3">       0.6 Mb     </td>
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

使用您自己的数据集，按照本[教程](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker/demo/text_classification.ipynb)运用本文使用的相同技术训练文本分类模型。利用正确的数据集，您可以为文档分类或负面评论检测等用例创建模型。

## 详细了解文本分类

- [单词嵌入向量和训练此模型的教程](https://www.tensorflow.org/tutorials/text/word_embeddings)

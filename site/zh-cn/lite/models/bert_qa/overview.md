# BERT Question and Answer

Use a TensorFlow Lite model to answer questions based on the content of a given passage.

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## 开始

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

如果您是 TensorFlow Lite 新用户，并且使用的是 Android 或 iOS 平台，我们建议您研究以下可以帮助您入门的示例应用。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Android example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">iOS
example</a>

如果您使用的不是 Android/iOS 平台，或者您已经熟悉 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)，则可以下载我们的起始问答模型。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Download starter model and vocab</a>

For more information about metadata and associated fields (e.g. `vocab.txt`) see <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">Read the metadata from models</a>.

## 工作方式

该模型可用于构建以自然语言回答用户问题的系统。它使用在 SQuAD 1.1 数据集上经过微调的预训练 BERT 模型进行创建。

[BERT](https://github.com/google-research/bert)（或称基于 Transformer 的双向编码器表示）是一种预训练语言表示方法，它可以在广泛的自然语言处理任务上获得一流的结果。

此应用使用 BERT 的压缩版本 MobileBERT。该版本的运行速度快 4 倍，而模型大小只有 BERT 模型的四分之一。

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)（或称 Stanford Question Answering Dataset）是一个由 Wikipedia 中的文章和一组针对每篇文章的问答对组成的阅读理解数据集。

该模型将一篇文章和一个问题作为输入，然后返回文章中最有可能回答问题的一段内容。它需要半复合预处理，包括在 BERT [论文](https://arxiv.org/abs/1810.04805)中介绍和在示例应用中实现的分词和后处理步骤。

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
    <td rowspan="3">
      <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Mobile Bert</a>
    </td>
    <td rowspan="3">       100.5 Mb     </td>
    <td>Pixel 3 (Android 10)</td>
    <td>123ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>74ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>257ms**</td>
  </tr>
</table>

* 使用 4 个线程。

** 为了获得最佳性能结果，在 iPhone 上使用 2 个线程。

## 示例输出

### 文章（输入）

> Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. It is considered one of the Big Four technology companies, alongside Amazon, Apple, and Facebook.
>
> Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.

### 问题（输入）

> Who is the CEO of Google?

### 回答（输出）

> Sundar Pichai

## 详细了解 BERT

- 学术论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Open-source implementation of BERT](https://github.com/google-research/bert)

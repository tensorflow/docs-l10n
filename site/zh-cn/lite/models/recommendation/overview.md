# 推荐

<table class="tfo-notebook-buttons" align="left">   <td>     <a target="_blank" href="https://www.tensorflow.org/lite/examples/recommendation/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">View on TensorFlow.org</a>   </td>   {% dynamic if request.tld != 'cn' %}<td>     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>   </td>{% dynamic endif %}   <td>     <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">View source on GitHub</a>   </td>
</table>

个性化推荐广泛应用于移动设备上的各种用例，例如媒体内容检索、产品选购建议以及新应用推荐等。如果您尊重用户隐私，同时希望在自己的应用中提供个性化推荐，我们建议您研究以下示例和工具包。

注：若要自定义模型，请尝试 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)。

## 开始

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

我们提供了一个 TensorFlow Lite 示例应用，为您演示如何在 Android 上向用户推荐相关内容。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android 示例</a>

如果您使用的不是 Android 平台，或者您已经熟悉 TensorFlow Lite API，则可以下载我们的入门推荐模型。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">下载入门模型</a>

我们还在 Github 中提供了训练脚本，以便以可配置的方式训练您自己的模型。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">训练代码</a>

## 了解模型架构

我们利用双编码器模型架构，通过上下文编码器对序列式用户历史记录编码，并通过标签编码器对预测推荐候选项编码。上下文编码与标签编码之间的相似度用于表示预测的候选项与用户需求相符的几率。

此代码库中提供了三种不同的序列式用户历史记录编码技术：

- 词袋编码器 (BOW)：不考虑上下文顺序，对用户活动的嵌入向量求平均值。
- 卷积神经网络编码器 (CNN)：应用多层卷积神经网络来生成上下文编码。
- 循环神经网络编码器 (RNN)：应用循环神经网络来编码上下文序列。

要对每个用户活动建模，我们可以使用活动项的 ID（基于 ID），或项的多个特征（基于特征），或二者的组合。基于特征的模型利用多个特征来共同编码用户的行为。使用此代码库，您可以用可配置的方式创建基于 ID 或基于特征的模型。

训练后，将输出一个 TensorFlow Lite 模型，该模型可以直接在推荐候选项中提供 top-K 预测。

## 使用训练数据

除了经过训练的模型，我们在 GitHub 中还提供了一个开放源代码[工具包](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)，以便帮助您使用自己的数据训练模型。您可以按本教程学习如何使用此工具包，并在自己的移动应用中部署经过训练的模型。

请按本[教程](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)运用此处使用的相同技术，利用自己的数据集训练推荐模型。

## 示例

作为示例，我们使用基于 ID 和基于特征的方法来训练推荐模型。基于 ID 的模型只将电影 ID 作为输入，而基于特征的模型将电影 ID 和电影类型 ID 都作为输入。请查看以下输入和输出示例。

输入

- 上下文电影 ID：

    - The Lion King (ID: 362)
    - Toy Story (ID: 1)
    - （等等）

- 上下文电影类型 ID：

    - 动画 (ID: 15)
    - 少儿 (ID: 9)
    - 音乐 (ID: 13)
    - 动画 (ID: 15)
    - 少儿 (ID: 9)
    - 喜剧 (ID: 2)
    - （等等）

输出：

- 推荐的电影 ID：
    - Toy Story 2 (ID: 3114)
    - （等等）

注：预训练模型基 [MovieLens](https://grouplens.org/datasets/movielens/1m/) 数据集构建，用于研究目的。

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
  <tbody>
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">推荐（电影 ID 作为输入）</a>
</td>
        <td rowspan="3">       0.52 Mb</td>
        <td>Pixel 3</td>
        <td>0.09ms*</td>
      </tr>
       <tr>
         <td>Pixel 4</td>
        <td>0.05ms*</td>
      </tr>
    
    <tr>
      </tr>
<tr>
        <td rowspan="3">           <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20210317/recommendation_cnn_i10i32o100.tflite">推荐（电影 ID 和电影类型作为输入）</a>
</td>
        <td rowspan="3">           1.3 Mb</td>
        <td>Pixel 3</td>
        <td>0.13ms*</td>
      </tr>
       <tr>
         <td>Pixel 4 </td>
        <td>0.06ms*</td>
      </tr>
    
  </tbody>
</table>

* 使用 4 个线程。

## Use your training data

In addition to the trained model, we provide an open-sourced [toolkit in GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) to train models with your own data. You can follow this tutorial to learn how to use the toolkit and deploy trained models in your own mobile applications.

Please follow this [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) to apply the same technique used here to train a recommendation model using your own datasets.

## 利用数据自定义模型的提示

此演示应用中集成的预训练模型使用 [MovieLens](https://grouplens.org/datasets/movielens/1m/) 数据集进行训练，您可能希望根据自己的数据修改模型配置，例如词汇大小、嵌入向量维度和输入上下文长度。下面是一些提示：

- 输入上下文长度：最佳输入上下文长度因数据集而异。我们建议根据标签事件与长期兴趣和短期上下文之间的相关性来选择输入上下文长度。

- 编码器类型选择：我们建议根据输入上下文长度选择编码器类型。对于较短的输入上下文（如小于 10），词袋编码器效果良好；对于较长的输入上下文，CNN 和 RNN 编码器的归纳能力更强。

- 使用底层特征来表示项或用户活动可以提高模型性能，更好地适应新鲜项，可能会缩小嵌入向量空间，从而减少内存消耗，并对设备更加友好。

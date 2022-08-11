# 推荐

<table class="tfo-notebook-buttons" align="left">   <td>     <a target="_blank" href="https://www.tensorflow.org/lite/examples/recommendation/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">View on TensorFlow.org</a>   </td>   {% dynamic if request.tld != 'cn' %}<td>     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>   </td>{% dynamic endif %}   <td>     <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">View source on GitHub</a>   </td> </table>

个性化推荐广泛应用于移动设备上的各种用例，例如媒体内容检索、产品选购建议以及新应用推荐等。如果您尊重用户隐私，同时希望在自己的应用中提供个性化推荐，我们建议您研究以下示例和工具包。

Note: To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker).

## 开始

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

我们提供了一个 TensorFlow Lite 示例应用，为您演示如何在 Android 上向用户推荐相关内容。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android example</a>

如果您使用的不是 Android 平台，或者您已经熟悉 TensorFlow Lite API，则可以下载我们的入门推荐模型。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">Download starter model</a>

We also provide training script in Github to train your own model in a configurable way.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">Training code</a>

## 了解模型架构

我们利用双编码器模型架构，通过上下文编码器对序列式用户历史记录编码，并通过标签编码器对预测推荐候选项编码。上下文编码与标签编码之间的相似度用于表示预测的候选项与用户需求相符的几率。

此代码库中提供了三种不同的序列式用户历史记录编码技术：

- 词袋编码器 (BOW)：不考虑上下文顺序，对用户活动的嵌入向量求平均值。
- 卷积神经网络编码器 (CNN)：应用多层卷积神经网络来生成上下文编码。
- 循环神经网络编码器 (RNN)：应用循环神经网络来编码上下文序列。

To model each user activity, we could use the ID of the activity item (ID-based) , or multiple features of the item (feature-based), or a combination of both. The feature-based model utilizing multiple features to collectively encode users’ behavior. With this code base, you could create either ID-based or feature-based models in a configurable way.

After training, a TensorFlow Lite model will be exported which can directly provide top-K predictions among the recommendation candidates.

## 使用训练数据

除了经过训练的模型，我们在 GitHub 中还提供了一个开放源代码[工具包](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)，以便帮助您使用自己的数据训练模型。您可以按本教程学习如何使用此工具包，并在自己的移动应用中部署经过训练的模型。

请按本[教程](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)运用此处使用的相同技术，利用自己的数据集训练推荐模型。

## 示例

As examples, we trained recommendation models with both ID-based and feature-based approaches. The ID-based model takes only the movie IDs as input, and the feature-based model takes both movie IDs and movie genre IDs as inputs. Please find the following inputs and outputs examples.

Inputs

- Context movie IDs:

    - The Lion King (ID: 362)
    - Toy Story (ID: 1)
    - （等等）

- Context movie genre IDs:

    - Animation (ID: 15)
    - Children's (ID: 9)
    - Musical (ID: 13)
    - Animation (ID: 15)
    - Children's (ID: 9)
    - Comedy (ID: 2)
    - （等等）

Outputs:

- Recommended movie IDs:
    - Toy Story 2 (ID: 3114)
    - (and more)

Note: The pretrained model is built based on [MovieLens](https://grouplens.org/datasets/movielens/1m/) dataset for research purpose.

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
        <td rowspan="3">
          <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">recommendation (movie ID as input)</a>
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
        <td rowspan="3">
          <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20210317/recommendation_cnn_i10i32o100.tflite">recommendation (movie ID and movie genre as inputs)</a>
        </td>
        <td rowspan="3">           1.3 Mb         </td>
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

- Using underlying features to represent items or user activities could improve model performance, better accommodate fresh items, possibly down scale embedding spaces hence reduce memory consumption and more on-device friendly.

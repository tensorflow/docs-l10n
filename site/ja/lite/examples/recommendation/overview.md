# レコメンド

<table class="tfo-notebook-buttons" align="left">   <td>     <a target="_blank" href="https://www.tensorflow.org/lite/examples/recommendation/overview"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">View on TensorFlow.org</a>   </td>   {% dynamic if request.tld != 'cn' %}<td>     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>   </td>{% dynamic endif %}   <td>     <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">View source on GitHub</a>   </td> </table>

個別のレコメンドは、メディアコンテンツの検索、ショッピングの商品提案、アプリの次のレコメンドなど、モバイルデバイス上のさまざまなユースケースで広く使用されています。ユーザーのプライバシーを尊重しながらアプリで個別のレコメンドを提供することに興味がある場合には、以下の例とツールキットの検討をお勧めします。

Note: To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker).

## はじめに


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Android 上でユーザーに関連するアイテムを推奨する方法を実演する TensorFlow Lite のサンプルアプリを提供しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android example</a>

Android 以外のプラットフォームを使用する場合、または、すでに TensorFlow Lite API に精通している場合は、レコメンドスターターモデルをダウンロードしてください。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">Download starter model</a>

We also provide training script in Github to train your own model in a configurable way.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">トレーニング用コード</a>

## モデルアーキテクチャの理解

Sequential にユーザー履歴をエンコードするコンテキストエンコーダと、予測されたレコメンドをエンコードするラベルエンコーダから成る、デュアルエンコーダ モデルアーキテクチャを活用しています。コンテキストエンコーディングとラベルエンコーディング間の類似性は、予測された候補がユーザーのニーズを満たす可能性を表現するために使用されます。

このコードベースでは、Sequential なユーザー履歴のエンコーディング手法を 3 種類提供しています。

- Bag of Words エンコーダ (BOW): コンテキストの順序を考慮しないでユーザーアクティビティの埋め込みを平均化します。
- 畳み込みニューラルネットワークエンコーダ (CNN): 畳み込みニューラルネットワークの複数のレイヤーを適用してコンテキストエンコーディングを生成します。
- 再帰ニューラルネットワークエンコーダ (RNN): 再帰ニューラルネットワークを適用してコンテキストシーケンスをエンコードします。

To model each user activity, we could use the ID of the activity item (ID-based) , or multiple features of the item (feature-based), or a combination of both. The feature-based model utilizing multiple features to collectively encode users’ behavior. With this code base, you could create either ID-based or feature-based models in a configurable way.

After training, a TensorFlow Lite model will be exported which can directly provide top-K predictions among the recommendation candidates.

## 独自のトレーニングデータを使用する

トレーニング済みモデルに加え、オープンソースの[ツールキットを GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) で提供しているので、独自のデータを使用してモデルをトレーニングすることができます。本チュートリアルでは、ツールキットの使い方、そしてトレーニング済みモデルを独自のモバイルアプリにデプロイする方法を学ぶことができます。

独自のデータセットを用いてレコメンドモデルをトレーニングする場合は、この[チュートリアル](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)に従い、ここで使用したのと同じ手法を適用してください。

## 使用例

As examples, we trained recommendation models with both ID-based and feature-based approaches. The ID-based model takes only the movie IDs as input, and the feature-based model takes both movie IDs and movie genre IDs as inputs. Please find the following inputs and outputs examples.

Inputs

- Context movie IDs:

    - The Lion King (ID: 362)
    - Toy Story (ID: 1)
    - （その他）

- Context movie genre IDs:

    - Animation (ID: 15)
    - Children's (ID: 9)
    - Musical (ID: 13)
    - Animation (ID: 15)
    - Children's (ID: 9)
    - Comedy (ID: 2)
    - （その他）

Outputs:

- Recommended movie IDs:
    - Toy Story 2 (ID: 3114)
    - (and more)

Note: The pretrained model is built based on [MovieLens](https://grouplens.org/datasets/movielens/1m/) dataset for research purpose.

## パフォーマンスベンチマーク

パフォーマンスベンチマークの数値は、[ここで説明する](https://www.tensorflow.org/lite/performance/benchmarks)ツールで生成されます。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>モデルサイズ</th>
      <th>デバイス</th>
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

* 4 つのスレッドを使用。

## Use your training data

In addition to the trained model, we provide an open-sourced [toolkit in GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml) to train models with your own data. You can follow this tutorial to learn how to use the toolkit and deploy trained models in your own mobile applications.

Please follow this [tutorial](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb) to apply the same technique used here to train a recommendation model using your own datasets.

## 独自のデータによるモデルカスタマイズのヒント

このデモアプリケーションに統合されている事前トレーニング済みモデルは [MovieLens](https://grouplens.org/datasets/movielens/1m/) データセットでトレーニングしていますが、独自のデータに基づいて、語彙サイズ、埋め込み次元、入力コンテキストの長さなどのモデル構成を変更することができます。以下にヒントを示します。

- 入力コンテキストの長さ: 入力コンテキストの最適な長さはデータセットによって異なります。ラベルのイベントの長期的関心と短期的コンテキストとの相関関係に基づいて、入力コンテキストの長さを選択することをお勧めします。

- エンコーダタイプの選択: 入力コンテキストの長さに基づいて適切なエンコーダのタイプを選択することをお勧めします。Bag of Words エンコーダは入力コンテキストの長さが短い場合（例えば 10 未満）でよく機能しますが、CNN や RNN エンコーダは入力コンテキストの長さが長い場合により優れた要約力を発揮します。

- Using underlying features to represent items or user activities could improve model performance, better accommodate fresh items, possibly down scale embedding spaces hence reduce memory consumption and more on-device friendly.

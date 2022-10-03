# テキスト分類

TensorFlow Lite モデルを使用して、段落を定義済みのグループに分類します。

注意: (1) 既存のモデルを統合するには、[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) を試してください。(2) モデルをカスタマイズするには、[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) を試してください。

## はじめに


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite を初めて使用する場合、Android を使用する場合は、TensorFLow Lite タスクライブラリのガイドを参考にして、数行のコードを使って[テキスト分類モデルの統合を行う](../../inference_with_metadata/task_library/nl_classifier)ことをお勧めします。また、[TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java) を使用してもモデルの統合が可能です。

以下の Android の例では、両方のメソッドをそれぞれ [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_task_api) および [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter) として実装しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android の例</a>

Android 以外のプラットフォームを使用する場合、または、すでに TensorFlow Lite API に精通している場合は、テキスト分類スターターモデルをダウンロードしてください。

スターターモデルをダウンロードする

## 使い方

テキスト分類は、その内容に基づいて段落を事前定義済みのグループに分類します。

This pretrained model predicts if a paragraph's sentiment is positive or negative. It was trained on [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) from Mass et al, which consists of IMDB movie reviews labeled as either positive or negative.

ここでは、モデルを使用して段落を分類する手順を紹介します。

1. 段落をトークン化し、事前定義済みの語彙を使用して単語 ID のリストに変換します。
2. リストを TensorFlow Lite モデルに与えます。
3. モデルの出力から、段落がポジティブかネガティブかの確率を取得します。

### 注意事項

- 英語のみに対応しています。
- このモデルは映画レビューのデータセットでトレーニングされているため、他のドメインのテキスト分類をする場合には精度が低下する可能性があります。

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
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Text Classification</a> </td>
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

* 4 threads used.

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## 出力例

テキスト | ネガティブ (0) | ポジティブ (1)
--- | --- | ---
ここ数年で見た中で最高の映画。 | 25.3% | 74.7%
強くお勧め！ |  |
時間の無駄。 | 72.5% | 27.5%

## 独自のトレーニングデータセットを使用する

独自のデータセットを使用してテキスト分類モデルをトレーニングする場合は、この[チュートリアル](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification)に従い、ここで使用するのと同じ手法を適用してください。適切なデータセットを使用して、ドキュメント分類や有害コメント検出などのユースケースに用いるモデルを作成することができます。

## テキスト分類についてもっと読む

- [このモデルをトレーニングするための単語埋め込みとチュートリアル](https://www.tensorflow.org/tutorials/text/word_embeddings)

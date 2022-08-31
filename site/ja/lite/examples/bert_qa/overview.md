# BERT Question and Answer

Use a TensorFlow Lite model to answer questions based on the content of a given passage.

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## はじめに


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Android example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">iOS
example</a>

Android または iOS 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、質問と回答スターターモデルをダウンロードしてください。

スターターモデルと語彙をダウンロードする

For more information about metadata and associated fields (e.g. `vocab.txt`) see <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">Read the metadata from models</a>.

## 使い方

このモデルを使用すると、ユーザーの質問に自然言語で回答できるシステムを構築できます。これは、SQuAD 1.1 データセットでファインチューニングされた事前トレーニング済み BERT モデルを使用して作成されました。

[BERT](https://github.com/google-research/bert) (Bidirectional Encoder Representations from Transformers) は、言語表現を事前トレーニングする方法で、さまざまな自然言語処理タスクで最先端の結果を取得できます。

このアプリは、BERT の圧縮バージョンである MobileBERT を使用します。これは 4 倍の速度で実行し、モデルサイズは 1/4 になります。

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset) は、ウィキペディアの記事と、各記事の一連の質問と回答のペアで構成される読解データセットです。

モデルは、パッセージと質問を入力として取り、質問の回答しての可能性が最も高いパッセージのセグメントを返します。これには、トークン化と後処理ステップを含むやや複雑な前処理が必要です。これらは BERT に関する[論文](https://arxiv.org/abs/1810.04805)で説明され、サンプルアプリで実装されています。

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

* 4 threads used.

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## 出力例

### パッセージ (入力)

> Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. It is considered one of the Big Four technology companies, alongside Amazon, Apple, and Facebook.
>
> Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.

### 質問 (入力)

> Google の CEO は誰ですか？

### 回答 (出力)

> スンダー・ピチャイ

## BERT の詳細を読む

- Academic paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Open-source implementation of BERT](https://github.com/google-research/bert)

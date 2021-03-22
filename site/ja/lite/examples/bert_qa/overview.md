# 質問と回答

事前トレーニング済みモデルを使用して、与えられたパッセージの内容に基づいて質問に答えます。

## はじめに

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Android の例</a>

Android または iOS 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、質問と回答スターターモデルをダウンロードしてください。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">スターターモデルと語彙をダウンロードする</a>

メタデータと関連フィールド (`vocab.txt`など) の詳細については、「<a href="https://www.tensorflow.org/lite/convert/metadata#read_the_metadata_from_models">モデルからメタデータを読み取る</a>」をご覧ください。

## 使い方

このモデルを使用すると、ユーザーの質問に自然言語で回答できるシステムを構築できます。これは、SQuAD 1.1 データセットでファインチューニングされた事前トレーニング済み BERT モデルを使用して作成されました。

[BERT](https://github.com/google-research/bert) (Bidirectional Encoder Representations from Transformers) は、言語表現を事前トレーニングする方法で、さまざまな自然言語処理タスクで最先端の結果を取得できます。

このアプリは、BERT の圧縮バージョンである MobileBERT を使用します。これは 4 倍の速度で実行し、モデルサイズは 1/4 になります。

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset) は、ウィキペディアの記事と、各記事の一連の質問と回答のペアで構成される読解データセットです。

モデルは、パッセージと質問を入力として取り、質問の回答しての可能性が最も高いパッセージのセグメントを返します。これには、トークン化と後処理ステップを含むやや複雑な前処理が必要です。これらは BERT に関する[論文](https://arxiv.org/abs/1810.04805)で説明され、サンプルアプリで実装されています。

## パフォーマンスベンチマーク

パフォーマンスベンチマークの数値は、[ここで説明する](https://www.tensorflow.org/lite/performance/benchmarks)ツールで生成されています。

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
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_qa_vocab.zip">Mobile Bert</a></td>
    <td rowspan="3">       100.5 Mb</td>
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

* 4 つのスレッドを使用。

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## 出力例

### パッセージ (入力)

> Google LLC は、オンライン広告技術、検索エンジン、クラウドコンピューティング、ソフトウェア、ハードウェアなど、インターネット関連のサービスと製品を専門とするアメリカの多国籍テクノロジー企業です。アマゾン、アップル、フェイスブックと並んで、4 大テクノロジー企業の 1 つと見なされています。
> 当時カリフォルニア州のスタンフォード大学の博士課程に在籍していたラリー・ペイジとサーゲイ・ブリンにより設立されました。現在でも  2 人 合わせて約 14% の株を保有し、スーパー投票株を通じて株主投票権の 56% を制御しています。Google は 1998 年 9 月 4 日にカリフォルニア州でカリフォルニアの非公開会社として設立されました。その後、Google は 2002 年 10 月 22 日にデラウェア州で再度法人として設立されました。2004 年 8 月 19 日に株式公開 (IPO) が行われ、Google は Googleplex と呼ばれるカリフォルニア州マウンテンビューの本社に移転しました。2015 年 8 月、Google は Alphabet Inc と呼ばれるコングロマリットとしてさまざまな事業を再編成する計画を発表しました。Google は Alphabet 社の主要な子会社として、引き続きインターネット関係の事業に包括的に取り組みます。 ラリー・ペイジの後任としてスンダー・ピチャイが新 CEO に就任し、ラリー・ペイジは Alphabet の CEO に着任しました。

### 質問 (入力)

> Google の CEO は誰ですか？

### 回答 (出力)

> スンダー・ピチャイ

## BERT の詳細を読む

- 学術論文: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (自然言語処理のための Transformer を用いたディープ双方向型事前トレーニング)](https://arxiv.org/abs/1810.04805)
- [BERT のオープンソース実装](https://github.com/google-research/bert)

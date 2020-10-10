# テキストクックブック

このページでは、TensorFlow Hub を使ってテキスト分野の問題を解決する既知のガイドやツールの一覧を提供しています。典型的な ML の問題をゼロからではなく、トレーニング済みの ML コンポーネントを使って解決しようとしているユーザーの出発点としてご利用ください。

## 分類

When we want to predict a class for a given example, for example **sentiment**, **toxicity**, **article category**, or any other characteristic.

![Text Classification Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-classification.png)

以下のチュートリアルでは、同一のタスクを異なるツールを使って異なる観点から解決しています。

### Keras

[Keras によるテキスト分類](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) - Keras と TensorFlow の Dataset を使用して IMDB のセンチメント分類器を構築する例。

### Estimator

[テキスト分類](https://github.com/tensorflow/hub/blob/master/docs/tutorials/text_classification_with_tf_hub.ipynb) - Estimator を使用して IMDB のセンチメント分類器を構築する例。改善に関するさまざまなヒントやモジュールの比較セクションが含まれます。

### BERT

[Predicting Movie Review Sentiment with BERT on TF Hub](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) - shows how to use a BERT module for classification. Includes use of `bert` library for tokenization and preprocessing.

### Kaggle

[IMDB classification on Kaggle](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) - shows how to easily interact with a Kaggle competition from a Colab, including downloading the data and submitting the results.

 | Estimator | Keras | TF2 | TF Dataset | BERT | Kaggle API
--- | --- | --- | --- | --- | --- | ---
[テキスト分類](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  |
[Keras によるテキスト分類](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |
[TF Hub で BERT を使用して映画レビューを予測する](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |
[Kaggle での IMDB 分類](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

### FastText 埋め込みによる Bangla タスク

現在のところ、TensorFlow Hub はモジュールをすべての言語で提供していません。次のチュートリアルでは、TensorFlow を使用して高速実験とモジュール式 ML 開発を行う方法が示されています。

[Bangla 記事分類器](https://github.com/tensorflow/hub/blob/master/examples/colab/bangla_article_classifier.ipynb) - 再利用可能な TensorFlow Hub テキスト埋め込みを作成し、それを使用して [BARD Bangla Article データセット](https://github.com/tanvirfahim15/BARD-Bangla-Article-Classifier)用に Keras 分類器をトレーニングする方法を実演しています。

## 意味的類似性

When we want to find out which sentences correlate with each other in zero-shot setup (no training examples).

![Semantic Similarity Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

### 基本

[意味的類似性](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) - 文章エンコーダモジュールを使用して文章の類似性を計算する方法を示します。

### クロスリンガル

[Cross-lingual semantic similarity](https://github.com/tensorflow/hub/blob/master/examples/colab/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb) - shows how to use one of the cross-lingual sentence encoders to compute sentence similarity across languages.

### セマンティック検索

[セマンティック検索](https://github.com/tensorflow/hub/blob/master/examples/colab/retrieval_with_tf_hub_universal_encoder_qa.ipynb) - QA 文章エンコーダを使用して、意味的類似性に基づく検索を行えるように、ドキュメントコレクションのインデックスを作成する方法を示します。

### SentencePiece 入力

[ユニバーサルエンコーダー Lite による意味的類似性](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb) - テキストの代わりに入力の [SentencePiece](https://github.com/google/sentencepiece) id を受け付ける文章エンコーダモジュールの使用方法を示します。

## モジュールの作成

[tfhub.dev](https://tfhub.dev) のモジュールのみを使用する代わりに、独自のモジュールを作成する方法があります。優れた ML コードベースモジュール性と以降での共有に有用なツールです。

### 既存のトレーニング済みの埋め込みをラッピングする

[テキスト埋め込みモジュールエクスポータ](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) - 既存のトレーニング済みの埋め込みをモジュールにラッピングするツールです。テキストの事前処理演算をモジュールに含める方法を示します。こうすることで、トークン埋め込みから文章埋め込みを作成することが可能となります。

[Text embedding module exporter v2](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings_v2/export_v2.py) - same as above, but compatible with TensorFlow 2 and eager execution.

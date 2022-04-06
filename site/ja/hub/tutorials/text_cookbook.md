# テキストクックブック

このページでは、TensorFlow Hub を使ってテキスト分野の問題を解決する既知のガイドやツールの一覧を提供しています。典型的な ML の問題をゼロからではなく、トレーニング済みの ML コンポーネントを使って解決しようとしているユーザーの出発点としてご利用ください。

## 分類

特定の例のクラスを予測する場合、特に**センチメント**、**毒性**、**記事カテゴリ**、またはその他の特性を予測する場合。

![Text Classification Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-classification.png)

以下のチュートリアルでは、同一のタスクを異なるツールを使ってさまざまな観点から解決しています。

### Keras

[Keras によるテキスト分類](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) - Keras と TensorFlow の Dataset を使用して IMDB のセンチメント分類器を構築する例。

### Estimator

[テキスト分類](https://github.com/tensorflow/hub/blob/master/docs/tutorials/text_classification_with_tf_hub.ipynb) - Estimator を使用して IMDB のセンチメント分類器を構築する例。改善に関するさまざまなヒントやモジュールの比較セクションが含まれます。

### BERT

[TF Hub の BERT による映画レビューのセンチメントの予測](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) - BERT モジュールを使用して分類を行う方法が紹介されています。トークン化と前処理を行うための `bert` ライブラリの使用方法が含まれます。

### Kaggle

[Kaggle での IMDB 分類](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) - データのダウンロードと結果の送信など、Colab から Kaggle コンペと簡単に連携する方法が示されています。

 | Estimator | Keras | TF2 | TF Datasets | BERT | Kaggle API
--- | --- | --- | --- | --- | --- | ---
[テキスト分類](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  |
[Keras によるテキスト分類](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |
[TF Hub の BERT による映画レビューのセンチメントの予測](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |
[Kaggle での IMDB 分類](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

### FastText 埋め込みによる Bangla タスク

| Estimator | Keras | TF2 | TF Datasets | BERT | Kaggle API --- | --- | --- | --- | --- | --- | --- [テキスト分類](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  | [Keras によるテキスト分類](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  | [TF Hub の BERT による映画レビューのセンチメントの予測](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | [Kaggle での IMDB 分類](https://github.com/tensorflow/hub/blob/master/examples/colab/text_classification_with_tf_hub_on_kaggle.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |  |  |  |  | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

[Bangla 記事分類器](https://github.com/tensorflow/hub/blob/master/examples/colab/bangla_article_classifier.ipynb) - 再利用可能な TensorFlow Hub テキスト埋め込みを作成し、それを使用して [BARD Bangla Article データセット](https://github.com/tanvirfahim15/BARD-Bangla-Article-Classifier)用に Keras 分類器をトレーニングする方法を実演しています。

## 意味的類似性

ゼロショットセットアップ（トレーニングサンプルなしのセットアップ）で、どの文章が相関しているかを見つけ出す場合。

![Semantic Similarity Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

### 基本

[意味的類似性](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) - 文章エンコーダモジュールを使用して文章の類似性を計算する方法を示します。

### クロスリンガル

[クロスリンガル意味的類似性](https://github.com/tensorflow/hub/blob/master/examples/colab/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb) - クロスリンガル文章エンコーダの 1 つを使用して言語間の文章の類似性を計算する方法を示します。

### セマンティック検索

[セマンティック検索](https://github.com/tensorflow/hub/blob/master/examples/colab/retrieval_with_tf_hub_universal_encoder_qa.ipynb) - QA 文章エンコーダを使用して、意味的類似性に基づく検索を行えるように、ドキュメントコレクションのインデックスを作成する方法を示します。

### SentencePiece 入力

[ユニバーサルエンコーダー Lite による意味的類似性](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb) - テキストの代わりに入力の [SentencePiece](https://github.com/google/sentencepiece) id を受け付ける文章エンコーダモジュールの使用方法を示します。

## モジュールの作成

[tfhub.dev](https://tfhub.dev) のモジュールのみを使用する代わりに、独自のモジュールを作成する方法があります。優れた ML コードベースモジュール性と以降での共有に有用なツールです。

### 既存のトレーニング済みの埋め込みをラッピングする

[テキスト埋め込みモジュールエクスポータ](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) - 既存のトレーニング済みの埋め込みをモジュールにラッピングするツールです。テキストの事前処理演算をモジュールに含める方法を示します。こうすることで、トークン埋め込みから文章埋め込みを作成することが可能となります。

[テキスト埋め込みモジュールエクスポータ v2](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings_v2/export_v2.py) - 上記と同じですが、TensorFlow 2 と Eager execution との互換性があります。

# テキスト埋め込みの統合

テキストエンベッダでは、テキストの語義を表す高次元の特徴量ベクトルにテキストを埋め込み、他のテキストの特徴量ベクトルと比較して、語義の類似度を評価することができます。

[テキスト検索](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher)とは対照的に、テキスト埋め込みでは、コーパスから構築された定義済みのインデックスを使用して検索するのではなく、テキスト間の類似度をその場で計算することができます。

Task Library `TextEmbedder` API を使用して、カスタムテキスト埋め込みをモバイルアプリにデプロイします。

## TextEmbedder API の主な機能

- 入力するテキストに対してグラフ内またはグラフ外の [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) または [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 字句解析を実行します。

- 特徴量ベクトル間の[コサイン類似度](https://en.wikipedia.org/wiki/Cosine_similarity)を計算するためのビルトインのユーティリティ関数。

## サポートされているテキスト埋め込みモデル

次のモデルは、`TextSegmenter` API との互換性が保証されています。

- [TensorFlow Hub の Universal Sentence Encoder TFLite モデル](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## C++ で推論を実行する

```c++
// Initialization.
TextEmbedderOptions options:
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<TextEmbedder> text_embedder = TextEmbedder::CreateFromOptions(options).value();

// Run inference with your two inputs, `input_text1` and `input_text2`.
const EmbeddingResult result_1 = text_embedder->Embed(input_text1);
const EmbeddingResult result_2 = text_embedder->Embed(input_text2);

// Compute cosine similarity.
double similarity = TextEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector()
    result_2.embeddings[0].feature_vector());
```

<code>TextEmbedder</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## Python で推論を実行する

### ステップ 1: TensorFlow Lite Support Pypi パッケージをインストールする

次のコマンドを使用して、TensorFlow Lite Support Pypi パッケージをインストールします。

```sh
pip install tflite-support
```

### ステップ 2: モデルを使用する

```python
from tflite_support.task import text

# Initialization.
text_embedder = text.TextEmbedder.create_from_file(model_path)

# Run inference on two texts.
result_1 = text_embedder.embed(text_1)
result_2 = text_embedder.embed(text_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = text_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

<code>TextEmbedder</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## 結果の例

正規化された特徴量ベクトル間のコサイン類似度は、-1 ～ 1 のスコアを返します。スコアが高いほど、類似度が高くなります。コサイン類似度 1 は、2 つのベクトルが同一であることを意味します。

```
Cosine similarity: 0.954312
```

独自のモデルとテストデータを使用して、シンプルな [TextEmbedder 用 CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder)をお試しください。

## モデルの互換性要件

`TextEmbedder` API では、TFLite モデルが必要です。また、[TFLite モデルメタデータ](https://www.tensorflow.org/lite/models/convert/metadata)は必須です。

次の 3 つの主な種類のモデルがサポートされます。

- BERT ベースのモデル (詳細については、[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h)を参照)

    - 3 つの入力テンソル (kTfLiteString)

        - ID テンソル。メタデータ名は「ids」
        - マスクテンソル。メタデータ名は「mask」
        - セグメント ID テンソル。メタデータ名は「segment_ids」

    - 1 つの出力テンソル (kTfLiteUInt8/kTfLiteFloat32)

        - この出力レイヤーに対して返された特徴量ベクトルの `N` 次元に対応する`N` コンポーネント。
        - 2 次元または 4 次元。例: `[1 x N]` または `[1 x 1 x 1 x N]`

    - Wordpiece/Sentencepiece Tokenizer の input_process_units

- Universal Sentence Encoder ベースのモデル (詳細については、[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h)を参照)

    - 3 つの入力テンソル (kTfLiteString)

        - クエリテキストテンソル。メタデータ名は「inp_text」
        - 応答コンテキストテンソル。メタデータ名は「res_context」
        - 応答テキストテンソル。メタデータ名は「res_text」

    - 2 つの出力テンソル (kTfLiteUInt8/kTfLiteFloat32)

        - クエリエンコーディングテンソル。メタデータ名は「query_encoding」
        - 応答エンコーディングテンソル。メタデータ名は「response_encoding」
        - この出力レイヤーに対して返された特徴量ベクトルの `N` 次元に対応する`N` コンポーネント。
        - いずれも 2 次元または 4 次元。例: `[1 x N]` または `[1 x 1 x 1 x N]`

- すべてのテキスト埋め込みモデルには、次のテンソルがあります。

    - 1 つの入力テンソル (kTfLiteString)

    - 1 つ以上の出力埋め込みテンソル (kTfLiteUInt8/kTfLiteFloat32)

        - この出力レイヤーに対して返された特徴量ベクトルの `N` 次元に対応する`N` コンポーネント。
        - 2 次元または 4 次元。例: `[1 x N]` または `[1 x 1 x 1 x N]`

# テキスト検索の統合

テキスト検索では、コーパスの語義的に類似したテキストを検索できます。クエリの語義を表す高次元のベクトルに検索クエリを埋め込み、[ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Scalable Nearest Neighbors) を使用した定義済みのカスタムインデックスで類似検索を実行します。

テキスト分類 (例: [Bert 自然言語分類器](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier)) とは対照的に、認識可能な項目の数が増えても、モデル全体を再トレーニングする必要がありません。インデックスを再構築するだけで、新しい項目を追加できます。このため、大規模なコーパス (項目 10 万件以上) でも動作します。

Task Library `TextSearcher` API を使用して、カスタムテキスト検索をモバイルアプリにデプロイします。

## TextSearcher API の主な機能

- 単一の文字列を入力として取り、インデックスで埋め込み抽出と最近傍探索を実行します。

- 入力するテキストに対してグラフ内またはグラフ外の [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) または [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 字句解析を実行します。

## 前提条件

`TextSearcher` API を使用する前に、検索するテキストのカスタムコーパスに基づいて、インデックスを構築する必要があります。構築するには、[チュートリアル](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher)に従い、調整しながら、[Model Maker Searcher API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) を使用します。

次の項目が必要です。

- TFLite テキスト埋め込みモデル (例: Universal Sentence Encoder)。たとえば、次のようなモデルがあります。
    - この [Colab](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/colab/on_device_text_to_image_search_tflite.ipynb) で維持されている[モデル](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/searcher/text_to_image_blogpost/text_embedder.tflite)。オンデバイス推論用に最適化されています。Pixel 6 でのテキスト文字列のクエリにかかる時間は、わずが 6 ミリ秒です。
    - [量子化](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)されたモデル。上記より小さいモデルですが、各埋め込みに 38 ミリ秒かかります。
- テキストのコーパス

このステップの後、スタンドアロン TFLite 検索モデル (例: `mobilenet_v3_searcher.tflite`) が必要です。このモデルは、インデックスが [TFLite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata) に関連付けられた、元のテキスト埋め込みモデルです。

## Java で推論を実行する

### ステップ 1: Gradle の依存関係とその他の設定をインポートする

`.tflite` 検索モデルファイルを、モデルが実行される Android モジュールのアセットディレクトリにコピーします。ファイルを圧縮しないように指定し、TensorFlow Lite ライブラリをモジュールの`build.gradle`ファイルに追加します。

```java
android {
    // Other settings

    // Specify tflite index file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
}
```

### ステップ 2: モデルを使用する

```java
// Initialization
TextSearcherOptions options =
    TextSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
TextSearcher textSearcher =
    textSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = textSearcher.search(text);
```

`TextSearcher` を構成するその他のオプションについては、[ソースコードと javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/searcher/TextSearcher.java) を参照してください。

## C++ で推論を実行する

```c++
// Initialization
TextSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<TextSearcher> text_searcher = TextSearcher::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
const SearchResult result = text_searcher->Search(input_text).value();
```

<code>TextSearcher</code> を構成するその他のオプションについては、<a>ソースコード</a>を参照してください。

## Python で推論を実行する

### ステップ 1: TensorFlow Lite Support Pypi パッケージをインストールする

次のコマンドを使用して、TensorFlow Lite Support Pypi パッケージをインストールします。

```sh
pip install tflite-support
```

### ステップ 2: モデルを使用する

```python
from tflite_support.task import text

# Initialization
text_searcher = text.TextSearcher.create_from_file(model_path)

# Run inference
result = text_searcher.search(text)
```

<code>TextSearcher</code> を構成するその他のオプションについては、<a>ソースコード</a>を参照してください。

## 結果の例

```
Results:
 Rank#0:
  metadata: The sun was shining on that day.
  distance: 0.04618
 Rank#1:
  metadata: It was a sunny day.
  distance: 0.10856
 Rank#2:
  metadata: The weather was excellent.
  distance: 0.15223
 Rank#3:
  metadata: The cat is chasing after the mouse.
  distance: 0.34271
 Rank#4:
  metadata: He was very happy with his newly bought car.
  distance: 0.37703
```

独自のモデルとテストデータを使用して、シンプルな [TextSearcher 用 CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textsearcher)をお試しください。

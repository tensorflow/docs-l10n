# 画像検索の統合

画像検索では、画像データベースの類似した画像を検索できます。クエリの語義を表す高次元のベクトルに検索クエリを埋め込み、[ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Scalable Nearest Neighbors) を使用した定義済みのカスタムインデックスで類似検索を実行します。

[画像分類](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)とは対照的に、認識可能な項目の数が増えても、モデル全体を再トレーニングする必要がありません。インデックスを再構築するだけで、新しい項目を追加できます。このため、大規模な画像データベース (項目 10 万件以上) でも動作します。

Task Library `ImageSearcher` API を使用して、カスタム画像検索をモバイルアプリにデプロイします。

## ImageSearcher API の主な機能

- 単一の画像を入力として取り、インデックスで埋め込み抽出と最近傍探索を実行します。

- 回転、サイズ変更、色空間変換などの入力画像処理。

- 入力画像の関心領域。

## 前提条件

`ImageSearcher` API を使用する前に、検索する画像のカスタムコーパスに基づいて、インデックスを構築する必要があります。構築するには、[チュートリアル](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher)に従い、調整しながら、[Model Maker Searcher API](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher) を使用します。

次の項目が必要です。

- [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1) などの TFLite 画像埋め込みモデル。[TensorFlow Hub の Google Image Modules コレクション](https://tfhub.dev/google/collections/image/1)のトレーニング済み埋め込みモデル (特徴量ベクトルモデル) を参照してください。
- 画像のコーパス

このステップの後、スタンドアロン TFLite 検索モデル (例: `mobilenet_v3_searcher.tflite`) が必要です。このモデルは、インデックスが [TFLite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata) に関連付けられた、元の画像埋め込みモデルです。

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
ImageSearcherOptions options =
    ImageSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
ImageSearcher imageSearcher =
    ImageSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = imageSearcher.search(image);
```

[ソースコードと javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/searcher/ImageSearcher.java) を参照し、`ImageSearcher` を構成するその他のオプションについてご覧ください。

## C++ で推論を実行する

```c++
// Initialization
ImageSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<ImageSearcher> image_searcher = ImageSearcher::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const SearchResult result = image_searcher->Search(*frame_buffer).value();
```

`ImageSearcher` を構成するその他のオプションについては、[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_searcher.h)を参照してください。

## Python で推論を実行する

### ステップ 1: TensorFlow Lite Support Pypi パッケージをインストールする

次のコマンドを使用して、TensorFlow Lite Support Pypi パッケージをインストールします。

```sh
pip install tflite-support
```

### ステップ 2: モデルを使用する

```python
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

<code>ImageSearcher</code> を構成するその他のオプションについては、<a>ソースコード</a>を参照してください。

## 結果の例

```
Results:
 Rank#0:
  metadata: burger
  distance: 0.13452
 Rank#1:
  metadata: car
  distance: 1.81935
 Rank#2:
  metadata: bird
  distance: 1.96617
 Rank#3:
  metadata: dog
  distance: 2.05610
 Rank#4:
  metadata: cat
  distance: 2.06347
```

独自のモデルとテストデータを使用して、シンプルな[ImageSearcher 用 CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher)をお試しください。

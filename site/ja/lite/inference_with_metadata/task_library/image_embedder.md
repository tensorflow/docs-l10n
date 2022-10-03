# 画像埋め込みの統合

画像埋め込みでは、画像の語義を表す高次元の特徴量ベクトルに画像を埋め込み、他の画像の特徴量ベクトルと比較して、語義の類似度を評価することができます。

[画像検索](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher)とは対照的に、画像エンベッダでは、画像のコーパスから構築された定義済みのインデックスを使用して検索するのではなく、画像間の類似性をその場で計算することができます。

Task Library `ImageEmbedder` API を使用して、カスタム画像埋め込みをモバイルアプリにデプロイします。

## ImageEmbedder API の主な機能

- 回転、サイズ変更、色空間変換などの入力画像処理。

- 入力画像の関心領域。

- 特徴量ベクトル間の[コサイン類似度](https://en.wikipedia.org/wiki/Cosine_similarity)を計算するためのビルトインのユーティリティ関数。

## サポートされている画像埋め込みモデル

次のモデルは、`ImageEmbedder` API との互換性が保証されています。

- [TensorFlow Hub の Google Image Modules コレクション](https://tfhub.dev/google/collections/image/1)の特徴量ベクトルモデル。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## C++ で推論を実行する

```c++
// Initialization
ImageEmbedderOptions options:
options.mutable_model_file_with_metadata()->set_file_name(model_path);
options.set_l2_normalize(true);
std::unique_ptr<ImageEmbedder> image_embedder = ImageEmbedder::CreateFromOptions(options).value();

// Create input frame_buffer1 and frame_buffer_2 from your inputs `image_data1`, `image_data2`, `image_dimension1` and `image_dimension2`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer1 = CreateFromRgbRawBuffer(
      image_data1, image_dimension1);
std::unique_ptr<FrameBuffer> frame_buffer1 = CreateFromRgbRawBuffer(
      image_data2, image_dimension2);

// Run inference on two images.
const EmbeddingResult result_1 = image_embedder->Embed(*frame_buffer_1);
const EmbeddingResult result_2 = image_embedder->Embed(*frame_buffer_2);

// Compute cosine similarity.
double similarity = ImageEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector(),
    result_2.embeddings[0].feature_vector());
```

<code>ImageEmbedder</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## Python で推論を実行する

### ステップ 1: TensorFlow Lite Support Pypi パッケージをインストールする

次のコマンドを使用して、TensorFlow Lite Support Pypi パッケージをインストールします。

```sh
pip install tflite-support
```

### ステップ 2: モデルを使用する

```python
from tflite_support.task import vision

# Initialization.
image_embedder = vision.ImageEmbedder.create_from_file(model_path)

# Run inference on two images.
image_1 = vision.TensorImage.create_from_file('/path/to/image1.jpg')
result_1 = image_embedder.embed(image_1)
image_2 = vision.TensorImage.create_from_file('/path/to/image2.jpg')
result_2 = image_embedder.embed(image_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = image_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

<code>ImageEmbedder</code>を構成するその他のオプションについては、<a>ソースコード</a>をご覧ください。

## 結果の例

正規化された特徴量ベクトル間のコサイン類似度は、-1 ～ 1 のスコアを返します。スコアが高いほど、類似度が高くなります。コサイン類似度 1 は、2 つのベクトルが同一であることを意味します。

```
Cosine similarity: 0.954312
```

独自のモデルとテストデータを使用して、シンプルな [ImageEmbedder 用 CLI デモツール](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder)をお試しください。

## モデルの互換性要件

`ImageEmbedder` API では、TFLite モデル が必要です。また、任意ですが、[TFLite モデルメタデータ](https://www.tensorflow.org/lite/models/convert/metadata)を使用することを強くお勧めします。

互換性のある画像埋め込みモデルは、次の要件を満たす必要があります。

- 入力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - サイズ`[batch x height x width x channels]`の画像入力。
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - RGB 入力のみがサポートされています (`channels`は 3 である必要があります)。
    - タイプが kTfLiteFloat32 の場合、入力の正規化のためにメタデータに NormalizationOptions をアタッチする必要があります。

- 1 つ以上の出力画像テンソル (kTfLiteUInt8/kTfLiteFloat32)

    - この出力レイヤーに対して返された特徴量ベクトルの `N` 次元に対応する`N` コンポーネント。
    - 2 次元または 4 次元。例: `[1 x N]` または `[1 x 1 x 1 x N]`

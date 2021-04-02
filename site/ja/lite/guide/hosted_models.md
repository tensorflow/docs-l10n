# ホステッドモデル

以下は、TensorFlow Lite で動作するように最適化された事前トレーニング済みモデルのリストの一部です。

モデルの選択を開始するには、エンドツーエンドの例が記載された<a href="../models">モデル</a>ページにアクセスするか、TensorFlow Hub からの TensorFlow Lite モデルを選択してください。

注: 特定のアプリケーションに最適なモデルは、要件によって異なります。たとえば、アプリケーションによっては、より高い精度が有用な場合がありますが、小さなモデルサイズを必要とする場合もあります。そのため、さまざまなモデルを使用してアプリケーションをテストし、サイズ、パフォーマンス、および精度の最適なバランスを見つける必要があります。

## 画像分類

画像分類の詳細については、<a href="../models/image_classification/overview.md">画像分類</a>を参照してください。わずか数行のコードで[画像分類モデルを統合する](../inference_with_metadata/task_library/image_classifier)には、TensorFlow Lite Task Library をご覧ください。

### 量子化モデル

<a href="../performance/post_training_quantization">量子化</a>画像分類モデルは、精度を低くして、最小のモデルサイズと最速のパフォーマンスを提供します。パフォーマンス値は、Android 10 が搭載された Pixel 3 で測定されています。

TensorFlow Hub には多くの[量子化モデル](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification&q=quantized)が提供されているため、TensorFlow Hub からさらに詳しいモデル情報を取得できます。

モデル名 | 論文とモデル | モデルサイズ | トップ1の精度 | トップ5の精度 | CPU、4 スレッド | NNAPI
--- | :-: | --: | --: | --: | --: | --:
Mobilenet_V1_0.25_128_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) | 0.5 Mb | 39.5% | 64.4% | 0.8 ms | 2 ms
Mobilenet_V1_0.25_160_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) | 0.5 Mb | 42.8% | 68.1% | 1.3 ms | 2.4 ms
Mobilenet_V1_0.25_192_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) | 0.5 Mb | 45.7% | 70.8% | 1.8 ms | 2.6 ms
Mobilenet_V1_0.25_224_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) | 0.5 Mb | 48.2% | 72.8% | 2.3 ms | 2.9 ms
Mobilenet_V1_0.50_128_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz) | 1.4 Mb | 54.9% | 78.1% | 1.7 ms | 2.6 ms
Mobilenet_V1_0.50_160_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz) | 1.4 Mb | 57.2% | 80.5% | 2.6 ms | 2.9 ms
Mobilenet_V1_0.50_192_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) | 1.4 Mb | 59.9% | 82.1% | 3.6 ms | 3.3 ms
Mobilenet_V1_0.50_224_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz) | 1.4 Mb | 61.2% | 83.2% | 4.7 ms | 3.6 ms
Mobilenet_V1_0.75_128_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) | 2.6 Mb | 55.9% | 79.1% | 3.1 ms | 3.2 ms
Mobilenet_V1_0.75_160_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) | 2.6 Mb | 62.4% | 83.7% | 4.7 ms | 3.8 ms
Mobilenet_V1_0.75_192_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) | 2.6 Mb | 66.1% | 86.2% | 6.4 ms | 4.2 ms
Mobilenet_V1_0.75_224_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) | 2.6 Mb | 66.9% | 86.9% | 8.5 ms | 4.8 ms
Mobilenet_V1_1.0_128_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz) | 4.3 Mb | 63.3% | 84.1% | 4.8 ms | 3.8 ms
Mobilenet_V1_1.0_160_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz) | 4.3 Mb | 66.9% | 86.7% | 7.3 ms | 4.6 ms
Mobilenet_V1_1.0_192_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz) | 4.3 Mb | 69.1% | 88.1% | 9.9 ms | 5.2 ms
Mobilenet_V1_1.0_224_quant | [論文](https://arxiv.org/pdf/1712.05877.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | 4.3 Mb | 70.0% | 89.0% | 13 ms | 6.0 ms
Mobilenet_V2_1.0_224_quant | [論文](https://arxiv.org/abs/1806.08342)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | 3.4 Mb | 70.8% | 89.9% | 12 ms | 6.9 ms
Inception_V1_quant | [論文](https://arxiv.org/abs/1409.4842)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz) | 6.4 Mb | 70.1% | 89.8% | 39 ms | 36 ms
Inception_V2_quant | [論文](https://arxiv.org/abs/1512.00567)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz) | 11 Mb | 73.5% | 91.4% | 59 ms | 18 ms
Inception_V3_quant | [論文](https://arxiv.org/abs/1806.08342)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz) | 23 Mb | 77.5% | 93.7% | 148 ms | 74 ms
Inception_V4_quant | [論文](https://arxiv.org/abs/1602.07261)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) | 41 Mb | 79.5% | 93.9% | 268 ms | 155 ms

注: モデルファイルには、TF Lite FlatBuffer と Tensorflow フリーズグラフの両方が含まれます。

注: パフォーマンスの数値は、Pixel-3 (Android 10) でベンチマークされ、精度の数値は、[TFLite 画像分類評価ツール](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)を使用して計算されています。

### 浮動小数点モデル

浮動小数点モデルは、モデルのサイズとパフォーマンスを犠牲にして、最高の精度を提供します。<a href="../performance/gpu">GPU アクセラレーション</a>では、浮動小数点モデルを使用する必要があります。パフォーマンス値は、Android 10 を搭載した Pixel 3 で測定されています。

TensorFlow Hub には多くの[画像分類モデル](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification)が提供されているため、TensorFlow Hub からさらに詳しいモデル情報を取得できます。

モデル名 | 論文とモデル | モデルサイズ | トップ 1 精度 | トップ 5 精度 | CPU、4 スレッド | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
DenseNet | [論文](https://arxiv.org/abs/1608.06993)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz) | 43.6 Mb | 64.2% | 85.6% | 195 ms | 60 ms | 1656 ms
SqueezeNet | [論文](https://arxiv.org/abs/1602.07360)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) | 5.0 Mb | 49.0% | 72.9% | 36 ms | 9.5 ms | 18.5 ms
NASNet mobile | [論文](https://arxiv.org/abs/1707.07012)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz) | 21.4 Mb | 73.9% | 91.5% | 56 ms | --- | 102 ms
NASNet large | [論文](https://arxiv.org/abs/1707.07012)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz) | 355.3 Mb | 82.6% | 96.1% | 1170 ms | --- | 648 ms
ResNet_V2_101 | [論文](https://arxiv.org/abs/1603.05027)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz) | 178.3 Mb | 76.8% | 93.6% | 526 ms | 92 ms | 1572 ms
Inception_V3 | [論文](http://arxiv.org/abs/1512.00567)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | 95.3 Mb | 77.9% | 93.8% | 249 ms | 56 ms | 148 ms
Inception_V4 | [論文](http://arxiv.org/abs/1602.07261)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz) | 170.7 Mb | 80.1% | 95.1% | 486 ms | 93 ms | 291 ms
Inception_ResNet_V2 | [論文](https://arxiv.org/abs/1602.07261)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | 121.0 Mb | 77.5% | 94.0% | 422 ms | 100 ms | 201 ms
Mobilenet_V1_0.25_128 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz) | 1.9 Mb | 41.4% | 66.2% | 1.2 ms | 1.6 ms | 3 ms
Mobilenet_V1_0.25_160 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_160.tgz) | 1.9 Mb | 45.4% | 70.2% | 1.7 ms | 1.7 ms | 3.2 ms
Mobilenet_V1_0.25_192 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_192.tgz) | 1.9 Mb | 47.1% | 72.0% | 2.4 ms | 1.8 ms | 3.0 ms
Mobilenet_V1_0.25_224 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz) | 1.9 Mb | 49.7% | 74.1% | 3.3 ms | 1.8 ms | 3.6 ms
Mobilenet_V1_0.50_128 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz) | 5.3 Mb | 56.2% | 79.3% | 3.0 ms | 1.7 ms | 3.2 ms
Mobilenet_V1_0.50_160 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz) | 5.3 Mb | 59.0% | 81.8% | 4.4 ms | 2.0 ms | 4.0 ms
Mobilenet_V1_0.50_192 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_192.tgz) | 5.3 Mb | 61.7% | 83.5% | 6.0 ms | 2.5 ms | 4.8 ms
Mobilenet_V1_0.50_224 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz) | 5.3 Mb | 63.2% | 84.9% | 7.9 ms | 2.8 ms | 6.1 ms
Mobilenet_V1_0.75_128 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_128.tgz) | 10.3 Mb | 62.0% | 83.8% | 5.5 ms | 2.6 ms | 5.1 ms
Mobilenet_V1_0.75_160 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_160.tgz) | 10.3 Mb | 65.2% | 85.9% | 8.2 ms | 3.1 ms | 6.3 ms
Mobilenet_V1_0.75_192 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_192.tgz) | 10.3 Mb | 67.1% | 87.2% | 11.0 ms | 4.5 ms | 7.2 ms
Mobilenet_V1_0.75_224 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz) | 10.3 Mb | 68.3% | 88.1% | 14.6 ms | 4.9 ms | 9.9 ms
Mobilenet_V1_1.0_128 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz) | 16.9 Mb | 65.2% | 85.7% | 9.0 ms | 4.4 ms | 6.3 ms
Mobilenet_V1_1.0_160 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_160.tgz) | 16.9 Mb | 68.0% | 87.7% | 13.4 ms | 5.0 ms | 8.4 ms
Mobilenet_V1_1.0_192 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_192.tgz) | 16.9 Mb | 69.9% | 89.1% | 18.1 ms | 6.3 ms | 10.6 ms
Mobilenet_V1_1.0_224 | [論文](https://arxiv.org/pdf/1704.04861.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | 16.9 Mb | 71.0% | 89.9％ | 24.0 ms | 6.5 ms | 13.8 ms
Mobilenet_V2_1.0_224 | [論文](https://arxiv.org/pdf/1801.04381.pdf)、[tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | 14.0 Mb | 71.8% | 90.6% | 17.5 ms | 6.2 ms | 11.23 ms

### AutoML モバイルモデル

次の画像分類モデルは、<a href="https://cloud.google.com/automl/">Cloud AutoML</a> を使用して作成されました。パフォーマンス値は、Android 10 が搭載された Pixel 3 で測定されています。

これらのモデルは [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=MnasNet) にあり、TensorFlow Hub からより多くのモデル情報を取得できます。

モデル名 | 論文とモデル | モデルサイズ | トップ1 精度 | トップ 5 精度 | CPU、4 スレッド | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
MnasNet_0.50_224 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz) | 8.5 Mb | 68.03% | 87.79% | 9.5 ms | 5.9 ms | 16.6 ms
MnasNet_0.75_224 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.75_224_09_07_2018.tgz) | 12 Mb | 71.72% | 90.17% | 13.7 ms | 7.1 ms | 16.7 ms
MnasNet_1.0_96 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_96_09_07_2018.tgz) | 17 Mb | 62.33% | 83.98% | 5.6 ms | 5.4 ms | 12.1 ms
MnasNet_1.0_128 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_128_09_07_2018.tgz) | 17 Mb | 67.32% | 87.70% | 7.5 ms | 5.8 ms | 12.9 ms
MnasNet_1.0_160 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_160_09_07_2018.tgz) | 17 Mb | 70.63% | 89.58% | 11.1 ms | 6.7 ms | 14.2 ms
MnasNet_1.0_192 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_192_09_07_2018.tgz) | 17 Mb | 72.56% | 90.76% | 14.5 ms | 7.7 ms | 16.6 ms
MnasNet_1.0_224 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_224_09_07_2018.tgz) | 17 Mb | 74.08% | 91.75% | 19.4 ms | 8.7 ms | 19 ms
MnasNet_1.3_224 | [論文](https://arxiv.org/abs/1807.11626)、[tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz) | 24 Mb | 75.24% | 92.55% | 27.9 ms | 10.6 ms | 22.0 ms

注: パフォーマンスの数値は、Pixel-3 (Android 10) でベンチマークされ、精度の数値は、[TFLite 画像分類評価ツール](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)を使用して計算されています。

## 物体検出

物体検出の詳細については、<a href="../models/object_detection/overview.md">物体検出</a>を参照してください。わずか数行のコードで[物体検出モデルを統合する](../inference_with_metadata/task_library/object_detector)には、TensorFlow Lite Task Library をご覧ください。

TensorFlow Hub から[物体検出モデル](https://tfhub.dev/s?deployment-format=lite&module-type=image-object-detection)をご覧ください。

## ポーズ推定

ポーズ推定についての詳細は、<a href="../models/pose_estimation/overview.md">ポーズ推定</a>をご覧ください。

TensorFlow Hub から[ポーズ推定モデル](https://tfhub.dev/s?deployment-format=lite&module-type=image-pose-detection)をご覧ください。

## 画像セグメンテーション

画像セグメンテーションの詳細については、<a href="../models/segmentation/overview.md">セグメンテーション</a>を参照してください。わずか数行のコードで[画像セグメンテーションモデルを統合する](../inference_with_metadata/task_library/image_segmenter)には、TensorFlow Lite Task Library をご覧ください。

TensorFlow Hub から[画像セグメンテーションモデル](https://tfhub.dev/s?deployment-format=lite&module-type=image-segmentation)をご覧ください。

## 質問応答

MobileBERT を使用した質問応答の詳細については、<a href="../models/bert_qa/overview.md">質問応答</a>を参照してください。わずか数行のコードで[質問応答モデルを統合する](../inference_with_metadata/task_library/bert_question_answerer)には、TensorFlow Lite Task Library をご覧ください。

TensorFlow Hub から[Mobile BERT モデル](https://tfhub.dev/tensorflow/mobilebert/1)をご覧ください。

## スマートリプライ

スマートリプライについての詳細は、<a href="../models/smart_reply/overview.md">スマートリプライ</a><br>をご覧ください。

TensorFlow Hub から[スマートリプライモデル](https://tfhub.dev/tensorflow/smartreply/1)をご覧ください。

# 호스팅된 모델

다음은 TensorFlow Lite와 함께 동작하도록 최적화된 사전 훈련된 모델의 일부를 수록한 목록입니다.

모델 선택을 시작하려면 엔드 투 엔드 예제가 있는 <a href="../models">모델</a> 페이지를 방문하거나 [TensorFlow Hub에서 TensorFlow Lite 모델을](https://tfhub.dev/s?deployment-format=lite) 선택하세요.

참고: 특정 애플리케이션에 가장 적합한 모델은 요구 사항에 따라 다릅니다. 예를 들어, 일부 애플리케이션에는 높은 정확성이 유익할 수 있지만 다른 애플리케이션에는 작은 모델 크기가 필요할 수 있습니다. 다양한 모델로 애플리케이션을 테스트하여 크기, 성능 및 정확성 간의 최적 균형을 찾아야 합니다.

## 이미지 분류

이미지 분류에 대한 자세한 내용은 <a href="../models/image_classification/overview.md">이미지 분류</a>를 참조하세요. TensorFlow Lite 작업 라이브러리에서 단 몇 줄의 코드만으로 [이미지 분류 모델을 통합하는 방법](../inference_with_metadata/task_library/image_classifier)에 대한 지침을 살펴보세요.

### 양자화된 모델

<a href="../performance/post_training_quantization">양자화된</a> 이미지 분류 모델은 정확성을 희생하면서 가장 작은 모델 크기와 가장 빠른 성능을 제공합니다. 성능 값은 Android 10의 Pixel 3에서 측정했습니다.

TensorFlow Hub에서 많은 [양자화된 모델](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification&q=quantized)을 찾아 더 많은 모델 정보를 얻을 수 있습니다.

모델 이름 | 논문과 모델 | 모델 크기 | Top-1 정확성 | Top-5 정확성 | CPU, 4 스레드 | NNAPI
--- | :-: | --: | --: | --: | --: | --:
Mobilenet_V1_0.25_128_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz) | 0.5Mb | 39.5% | 64.4% | 0.8ms | 2ms
Mobilenet_V1_0.25_160_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) | 0.5Mb | 42.8% | 68.1% | 1.3ms | 2.4ms
Mobilenet_V1_0.25_192_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz) | 0.5Mb | 45.7% | 70.8% | 1.8ms | 2.6ms
Mobilenet_V1_0.25_224_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz) | 0.5Mb | 48.2% | 72.8% | 2.3ms | 2.9ms
Mobilenet_V1_0.50_128_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz) | 1.4Mb | 54.9% | 78.1% | 1.7ms | 2.6ms
Mobilenet_V1_0.50_160_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz) | 1.4Mb | 57.2% | 80.5% | 2.6ms | 2.9ms
Mobilenet_V1_0.50_192_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) | 1.4Mb | 59.9% | 82.1% | 3.6ms | 3.3ms
Mobilenet_V1_0.50_224_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz) | 1.4Mb | 61.2% | 83.2% | 4.7ms | 3.6ms
Mobilenet_V1_0.75_128_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz) | 2.6Mb | 55.9% | 79.1% | 3.1ms | 3.2ms
Mobilenet_V1_0.75_160_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz) | 2.6Mb | 62.4% | 83.7% | 4.7ms | 3.8ms
Mobilenet_V1_0.75_192_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz) | 2.6Mb | 66.1% | 86.2% | 6.4ms | 4.2ms
Mobilenet_V1_0.75_224_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz) | 2.6Mb | 66.9% | 86.9% | 8.5ms | 4.8ms
Mobilenet_V1_1.0_128_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz) | 4.3Mb | 63.3% | 84.1% | 4.8ms | 3.8ms
Mobilenet_V1_1.0_160_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz) | 4.3Mb | 66.9% | 86.7% | 7.3ms | 4.6ms
Mobilenet_V1_1.0_192_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz) | 4.3Mb | 69.1% | 88.1% | 9.9ms | 5.2ms
Mobilenet_V1_1.0_224_quant | [논문](https://arxiv.org/pdf/1712.05877.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | 4.3Mb | 70.0% | 89.0% | 13ms | 6.0ms
Mobilenet_V2_1.0_224_quant | [논문](https://arxiv.org/abs/1806.08342), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | 3.4Mb | 70.8% | 89.9% | 12ms | 6.9ms
Inception_V1_quant | [논문](https://arxiv.org/abs/1409.4842), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz) | 6.4Mb | 70.1% | 89.8% | 39ms | 36ms
Inception_V2_quant | [논문](https://arxiv.org/abs/1512.00567), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz) | 11Mb | 73.5% | 91.4% | 59ms | 18ms
Inception_V3_quant | [논문](https://arxiv.org/abs/1806.08342), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz) | 23Mb | 77.5% | 93.7% | 148ms | 74ms
Inception_V4_quant | [논문](https://arxiv.org/abs/1602.07261), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) | 41Mb | 79.5% | 93.9% | 268ms | 155ms

참고: 모델 파일에는 TF Lite FlatBuffer 및 Tensorflow 고정 그래프가 모두 포함됩니다.

참고: 성능 수치는 Pixel-3(Android 10)에서 벤치마킹했습니다. 정확성 수치는 [TFLite 이미지 분류 평가 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)를 사용하여 계산했습니다.

### 부동 소수점 모델

부동 소수점 모델은 모델 크기와 성능을 희생하면서 최고의 정확성을 제공합니다. <a href="../performance/gpu">GPU 가속</a>을 사용하려면 부동 소수점 모델을 사용해야 합니다. 성능 값은 Android 10의 Pixel 3에서 측정했습니다.

TensorFlow Hub에서 많은 [이미지 분류 모델](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification)을 찾아 더 많은 모델 정보를 얻을 수 있습니다.

모델 이름 | 논문과 모델 | 모델 크기 | Top-1 정확성 | Top-5 정확성 | CPU, 4 스레드 | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
DenseNet | [논문](https://arxiv.org/abs/1608.06993), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz) | 43.6Mb | 64.2% | 85.6% | 195ms | 60ms | 1656ms
SqueezeNet | [논문](https://arxiv.org/abs/1602.07360), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz) | 5.0Mb | 49.0% | 72.9% | 36ms | 9.5ms | 18.5ms
NASNet 모바일 | [논문](https://arxiv.org/abs/1707.07012), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz) | 21.4Mb | 73.9% | 91.5% | 56ms | --- | 102ms
NASNet 대형 | [논문](https://arxiv.org/abs/1707.07012), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz) | 355.3Mb | 82.6% | 96.1% | 1170ms | --- | 648ms
ResNet_V2_101 | [논문](https://arxiv.org/abs/1603.05027), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz) | 178.3Mb | 76.8% | 93.6% | 526ms | 92ms | 1572ms
Inception_V3 | [논문](http://arxiv.org/abs/1512.00567), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | 95.3Mb | 77.9% | 93.8% | 249ms | 56ms | 148ms
Inception_V4 | [논문](http://arxiv.org/abs/1602.07261), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz) | 170.7Mb | 80.1% | 95.1% | 486ms | 93ms | 291ms
Inception_ResNet_V2 | [논문](https://arxiv.org/abs/1602.07261), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) | 121.0Mb | 77.5% | 94.0% | 422ms | 100ms | 201ms
Mobilenet_V1_0.25_128 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz) | 1.9MB | 41.4% | 66.2% | 1.2ms | 1.6ms | 3ms
Mobilenet_V1_0.25_160 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_160.tgz) | 1.9MB | 45.4% | 70.2% | 1.7ms | 1.7ms | 3.2ms
Mobilenet_V1_0.25_192 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_192.tgz) | 1.9MB | 47.1% | 72.0% | 2.4ms | 1.8ms | 3.0ms
Mobilenet_V1_0.25_224 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz) | 1.9MB | 49.7% | 74.1% | 3.3ms | 1.8ms | 3.6ms
Mobilenet_V1_0.50_128 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_128.tgz) | 5.3Mb | 56.2% | 79.3% | 3.0ms | 1.7ms | 3.2ms
Mobilenet_V1_0.50_160 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz) | 5.3Mb | 59.0% | 81.8% | 4.4ms | 2.0ms | 4.0ms
Mobilenet_V1_0.50_192 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_192.tgz) | 5.3Mb | 61.7% | 83.5% | 6.0ms | 2.5ms | 4.8ms
Mobilenet_V1_0.50_224 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz) | 5.3Mb | 63.2% | 84.9% | 7.9ms | 2.8ms | 6.1ms
Mobilenet_V1_0.75_128 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_128.tgz) | 10.3Mb | 62.0% | 83.8% | 5.5ms | 2.6ms | 5.1ms
Mobilenet_V1_0.75_160 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_160.tgz) | 10.3Mb | 65.2% | 85.9% | 8.2ms | 3.1ms | 6.3ms
Mobilenet_V1_0.75_192 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_192.tgz) | 10.3Mb | 67.1% | 87.2% | 11.0ms | 4.5ms | 7.2ms
Mobilenet_V1_0.75_224 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz) | 10.3Mb | 68.3% | 88.1% | 14.6ms | 4.9ms | 9.9ms
Mobilenet_V1_1.0_128 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_128.tgz) | 16.9Mb | 65.2% | 85.7% | 9.0ms | 4.4ms | 6.3ms
Mobilenet_V1_1.0_160 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_160.tgz) | 16.9Mb | 68.0% | 87.7% | 13.4ms | 5.0ms | 8.4ms
Mobilenet_V1_1.0_192 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_192.tgz) | 16.9Mb | 69.9% | 89.1% | 18.1ms | 6.3ms | 10.6ms
Mobilenet_V1_1.0_224 | [논문](https://arxiv.org/pdf/1704.04861.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | 16.9Mb | 71.0% | 89.9% | 24.0ms | 6.5ms | 13.8ms
Mobilenet_V2_1.0_224 | [논문](https://arxiv.org/pdf/1801.04381.pdf), [tflite&amp;pb](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | 14.0Mb | 71.8% | 90.6% | 17.5ms | 6.2ms | 11.23ms

### AutoML 모바일 모델

다음 이미지 분류 모델은 <a href="https://cloud.google.com/automl/">Cloud AutoML</a>을 사용하여 생성되었습니다. 성능 값은 Android 10의 Pixel 3에서 측정했습니다.

[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=MnasNet) 에서 이러한 모델을 찾아 더 많은 모델 정보를 얻을 수 있습니다.

모델 이름 | 논문과 모델 | 모델 크기 | Top-1 정확성 | Top-5 정확성 | CPU, 4 스레드 | GPU | NNAPI
--- | :-: | --: | --: | --: | --: | --: | --:
MnasNet_0.50_224 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.5_224_09_07_2018.tgz) | 8.5MB | 68.03% | 87.79% | 9.5ms | 5.9ms | 16.6ms
MnasNet_0.75_224 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_0.75_224_09_07_2018.tgz) | 12Mb | 71.72% | 90.17% | 13.7ms | 7.1ms | 16.7ms
MnasNet_1.0_96 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_96_09_07_2018.tgz) | 17Mb | 62.33% | 83.98% | 5.6ms | 5.4ms | 12.1ms
MnasNet_1.0_128 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_128_09_07_2018.tgz) | 17Mb | 67.32% | 87.70% | 7.5ms | 5.8ms | 12.9ms
MnasNet_1.0_160 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_160_09_07_2018.tgz) | 17Mb | 70.63% | 89.58% | 11.1ms | 6.7ms | 14.2ms
MnasNet_1.0_192 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_192_09_07_2018.tgz) | 17Mb | 72.56% | 90.76% | 14.5ms | 7.7ms | 16.6ms
MnasNet_1.0_224 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.0_224_09_07_2018.tgz) | 17Mb | 74.08% | 91.75% | 19.4ms | 8.7ms | 19ms
MnasNet_1.3_224 | [논문](https://arxiv.org/abs/1807.11626), [tflite&amp;pb](https://storage.cloud.google.com/download.tensorflow.org/models/tflite/mnasnet_1.3_224_09_07_2018.tgz) | 24Mb | 75.24% | 92.55% | 27.9ms | 10.6ms | 22.0ms

참고: 성능 수치는 Pixel-3(Android 10)에서 벤치마킹했습니다. 정확성 수치는 [TFLite 이미지 분류 평가 도구](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)를 사용하여 계산했습니다.

## 객체 감지

물체 감지에 대한 자세한 내용은 <a href="../models/object_detection/overview.md">물체 감지</a>를 참조하세요. TensorFlow Lite 작업 라이브러리에서 단 몇 줄의 코드만으로 [물체 감지 모델을 통합하는 방법](../inference_with_metadata/task_library/object_detector)에 대한 지침은 살펴보세요.

TensorFlow Hub에서 [객체 감지 모델](https://tfhub.dev/s?deployment-format=lite&module-type=image-object-detection)을 찾아보세요.

## 포즈 예측

포즈 예측에 대한 자세한 내용은 <a href="../models/pose_estimation/overview.md">포즈 예측</a>을 참조하세요.

TensorFlow Hub에서 [포즈 예측 모델](https://tfhub.dev/s?deployment-format=lite&module-type=image-pose-detection)을 찾아보세요.

## 이미지 분할

이미지 분할에 대한 자세한 내용은 <a href="../models/segmentation/overview.md">분할</a>을 참조하세요. TensorFlow Lite 작업 라이브러리에서 단 몇 줄의 코드만으로 [이미지 분할 모델을 통합하는 방법](../inference_with_metadata/task_library/image_segmenter)에 대한 지침을 살펴보세요.

TensorFlow Hub에서 [이미지 분할 모델](https://tfhub.dev/s?deployment-format=lite&module-type=image-segmentation)을 찾아보세요.

## 질문과 답변

MobileBERT를 이용한 질문과 답변에 대한 자세한 내용은 <a href="../models/bert_qa/overview.md">질문과 답변</a>을 참조하세요. TensorFlow Lite 작업 라이브러리에서 몇 줄의 코드만으로 [질문 및 답변 모델을 통합하는 방법](../inference_with_metadata/task_library/bert_question_answerer)에 대한 지침은 살펴보세요.

TensorFlow Hub에서 [모바일 BERT 모델](https://tfhub.dev/tensorflow/mobilebert/1)을 찾아보세요.

## 스마트 답장

스마트 답장에 대한 자세한 내용은 <a href="../models/smart_reply/overview.md">스마트 답장</a>을 참조하세요.

TensorFlow Hub에서 [스마트 답장 모델](https://tfhub.dev/tensorflow/smartreply/1)을 찾아보세요.

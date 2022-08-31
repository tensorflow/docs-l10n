# 이미지 임베더 통합

이미지 임베더를 사용하면 이미지의 의미론적 의미를 나타내는 고차원 요소 벡터에 이미지를 임베딩할 수 있으며, 이 벡터를 다른 이미지의 요소 벡터와 비교하여 의미론적 유사성을 평가할 수 있습니다.

[이미지 검색](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher)과 달리 이미지 임베더를 사용하면 이미지 코퍼스에서 구축된 미리 정의된 인덱스를 통해 검색하는 대신 이미지 간의 유사성을 즉석에서 계산할 수 있습니다.

작업 라이브러리 `ImageEmbedder` API를 사용하여 사용자 지정 이미지 임베더를 모바일 앱에 배포합니다.

## ImageEmbedder API의 주요 기능

- 회전, 크기 조정 및 색 공간 변환을 포함한 입력 이미지 처리

- 입력 이미지의 관심 영역

- 요소 벡터 간의 [코사인 유사성](https://en.wikipedia.org/wiki/Cosine_similarity)를 계산하는 내장 유틸리티 함수

## 지원되는 이미지 임베더 모델

다음 모델은 `ImageEmbedder` API와 호환이 보장됩니다.

- [TensorFlow Hub에 있는 Google 이미지 모듈 컬렉션](https://tfhub.dev/google/collections/image/1)의 요소 벡터 모델

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## C++에서 추론 실행하기

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

<code>ImageEmbedder</code> 구성에 대한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## Python에서 추론 실행하기

### 1단계: TensorFlow Lite 지원 Pypi 패키지 설치하기

다음 명령을 사용하여 TensorFlow Lite Support Pypi 패키지를 설치할 수 있습니다.

```sh
pip install tflite-support
```

### 2단계: 모델 사용하기

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

<code>ImageEmbedder</code> 구성에 대한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## 예제 결과

정규화된 요소 벡터 간의 코사인 유사성은 -1과 1 사이의 점수를 반환합니다. 높을수록 좋습니다. 즉, 코사인 유사성이 1이면 두 벡터가 동일하다는 의미입니다.

```
Cosine similarity: 0.954312
```

고유한 모델 및 테스트 데이터로 간단한 [ImageEmbedder용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder)를 사용해 보세요.

## 모델 호환성 요구 사항

`ImageEmbedder` API는 선택 사항이지만 강력하게 권장되는 [TFLite 모델 메타데이터](https://www.tensorflow.org/lite/models/convert/metadata)가 있는 TFLite 모델을 예상합니다.

호환되는 이미지 임베더 모델은 다음 요구 사항을 충족해야 합니다.

- 입력 이미지 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - 이미지 입력 크기가 `[batch x height x width x channels]`입니다.
    - 배치 추론은 지원되지 않습니다(`batch`는 1이어야 함).
    - RGB 입력만 지원됩니다(`channels`은 3이어야 함).
    - 유형이 kTfLiteFloat32인 경우, 입력 정규화를 위해 NormalizationOptions를 메타데이터에 첨부해야 합니다.

- 하나 이상의 출력 텐서(kTfLiteUInt8/kTfLiteFloat32)

    - 이 출력 레이어에 대해 반환된 요소 벡터의 `N` 차원에 해당하는 `N` 구성 요소가 있습니다.
    - 2 또는 4차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`

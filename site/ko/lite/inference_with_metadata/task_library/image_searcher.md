# 이미지 검색기 통합

이미지 검색을 사용하면 이미지 데이터베이스에서 유사한 이미지를 검색할 수 있습니다. 구체적으로, 검색 쿼리를 쿼리의 의미론적 의미를 나타내는 고차원 벡터에 포함시킨 다음 [ScanNN](https://github.com/google-research/google-research/tree/master/scann)(Scalable Nearest Neighbors)을 사용하여 미리 정의된 사용자 지정 인덱스에서 유사성 검색을 수행하는 식으로 작동합니다.

[이미지 분류](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier)와 달리 인식할 수 있는 항목의 수를 늘리는 데 전체 모델을 다시 학습할 필요가 없습니다. 인덱스를 다시 빌드하기만 하여 새 항목을 추가할 수 있습니다. 또한 더 큰(100,000개 이상의 항목) 이미지 데이터베이스로 작업할 수 있습니다.

작업 라이브러리 `ImageSearcher` API를 사용하여 사용자 지정 이미지 검색기를 모바일 앱에 배포합니다.

## ImageSearcher API의 주요 기능

- 단일 이미지를 입력으로 사용하고 인덱스에서 임베딩 추출 및 NN(nearest-neighbor) 검색을 수행

- 회전, 크기 조정 및 색 공간 변환을 포함한 입력 이미지 처리

- 입력 이미지의 관심 영역

## 전제 조건

`ImageSearcher` API를 사용하기 전에 검색할 이미지의 사용자 지정 코퍼스를 기반으로 인덱스를 구축해야 합니다. 이를 위해 [Model Maker Searcher API](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher)를 사용할 수 있고, 이 [튜토리얼](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher)을 따라한 다음 필요한 부분을 조정하면 됩니다.

이를 위해서는 다음이 필요합니다.

- [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1)와 같은 TFLite 이미지 임베더 모델. [TensorFlow Hub의 Google 이미지 모듈 컬렉션](https://tfhub.dev/google/collections/image/1)에서 더 많은 사전 훈련된 임베더 모델(요소 벡터 모델이라고도 함)을 참조하세요.
- 자신의 이미지 코퍼스

이 단계가 끝나면 독립형 TFLite 검색기 모델(예: `mobilenet_v3_searcher.tflite`)을 갖게 됩니다. 이는 [TFLite 모델 메타데이터](https://www.tensorflow.org/lite/models/convert/metadata)에 인덱스가 첨부된 원본 이미지 임베더 모델입니다.

## Java에서 추론 실행하기

### 1단계: Gradle 종속성 및 기타 설정 가져오기

`.tflite` 검색기 모델 파일을 모델이 실행될 Android 모듈의 assets 디렉터리에 복사합니다. 파일을 압축하지 않도록 지정하고 TensorFlow Lite 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

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

### 2단계: 모델 사용하기

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

<code>ImageSearcher</code>를 구성하는 추가 옵션은 <a>소스 코드와 javadoc</a>을 참조하세요.

## C++에서 추론 실행하기

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

<code>ImageSearcher</code> 구성을 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## Python에서 추론 실행하기

### 1단계: TensorFlow Lite 지원 Pypi 패키지 설치하기

다음 명령을 사용하여 TensorFlow Lite Support Pypi 패키지를 설치할 수 있습니다.

```sh
pip install tflite-support
```

### 2단계: 모델 사용하기

```python
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

<code>ImageSearcher</code> 구성을 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## 예제 결과

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

고유한 모델 및 테스트 데이터로 간단한 [ImageSearcher용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher)를 사용해 보세요.

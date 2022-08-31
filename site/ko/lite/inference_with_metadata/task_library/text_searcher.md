# 텍스트 검색기 통합

텍스트 검색을 사용하면 코퍼스에서 의미적으로 유사한 텍스트를 검색할 수 있습니다. 구체적으로, 검색 쿼리를 쿼리의 의미론적 의미를 나타내는 고차원 벡터에 포함시킨 다음 [ScanNN](https://github.com/google-research/google-research/tree/master/scann)(Scalable Nearest Neighbors)을 사용하여 미리 정의된 사용자 지정 인덱스에서 유사성 검색합니다.

텍스트 분류(예: [Bert 자연어 분류기](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier))와 달리 인식할 수 있는 항목 수를 확장하기 위해 전체 모델을 다시 학습할 필요가 없습니다. 인덱스를 다시 빌드하기만 하여 새 항목을 추가할 수 있습니다. 또한, 더 큰(100,000개 이상의 항목) 코퍼스로 작업할 수 있습니다.

작업 라이브러리 `TextSearcher` API를 사용하여 사용자 지정 텍스트 검색기를 모바일 앱에 배포합니다.

## TextSearcher API의 주요 기능

- 단일 이미지를 입력으로 사용하고 인덱스에서 임베딩 추출 및 NN(nearest-neighbor) 검색을 수행

- 입력 텍스트에 대한 그래프 내 또는 그래프 외 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 또는 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 토큰화를 포함한 입력 텍스트 처리

## 전제 조건

`TextSearcher` API를 사용하기 전에 검색할 텍스트의 사용자 지정 코퍼스를 기반으로 인덱스를 구축해야 합니다. 이를 위해 [Model Maker Searcher API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher)를 사용할 수 있고, 이 [튜토리얼](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher)을 따라한 다음 필요한 부분을 조정하면 됩니다.

이를 위해서는 다음이 필요합니다.

- Universal Sentence Encoder와 같은 TFLite 텍스트 임베더 모델. 예를 들어,
    - 기기 내 추론에 최적화된 이 [Colab](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/searcher/text_to_image_blogpost/text_embedder.tflite)에서 다시 훈련된 [모델](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/colab/on_device_text_to_image_search_tflite.ipynb). Pixel 6에서 텍스트 문자열을 쿼리하는 데 6ms밖에 걸리지 않습니다.
    - 위의 것보다 작지만 각 임베딩에 38ms가 걸리는 [양자화된](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1) 모델
- 자신의 텍스트 코퍼스

이 단계가 끝나면 독립형 TFLite 검색기 모델(예: `mobilenet_v3_searcher.tflite`)을 갖게 됩니다. 이는 [TFLite 모델 메타데이터](https://www.tensorflow.org/lite/models/convert/metadata)에 인덱스가 첨부된 원본 텍스트 임베더 모델입니다.

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

<code>TextSearcher</code>를 구성하기 위한 추가 옵션은 <a>소스 코드와 javadoc</a>을 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
TextSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<TextSearcher> text_searcher = TextSearcher::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
const SearchResult result = text_searcher->Search(input_text).value();
```

<code>TextSearcher</code>를 구성하기 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## Python에서 추론 실행하기

### 1단계: TensorFlow Lite 지원 Pypi 패키지 설치하기

다음 명령을 사용하여 TensorFlow Lite Support Pypi 패키지를 설치할 수 있습니다.

```sh
pip install tflite-support
```

### 2단계: 모델 사용하기

```python
from tflite_support.task import text

# Initialization
text_searcher = text.TextSearcher.create_from_file(model_path)

# Run inference
result = text_searcher.search(text)
```

<code>TextSearcher</code>를 구성하기 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## 예제 결과

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

자신의 고유한 모델 및 테스트 데이터로 간단한 [TextSearcher용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textsearcher)를 사용해 보세요.

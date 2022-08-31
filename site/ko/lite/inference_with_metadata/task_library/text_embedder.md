# 텍스트 임베더 통합

텍스트 임베더를 사용하면 의미론적 의미를 나타내는 고차원 요소 벡터에 텍스트를 임베딩할 수 있으며, 이를 다른 텍스트의 요소 벡터와 비교하여 의미론적 유사성을 평가할 수 있습니다.

[텍스트 검색](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher)과 달리 텍스트 임베더를 사용하면 코퍼스에서 구축된 미리 정의된 인덱스를 통해 검색하는 대신 즉석에서 텍스트 간의 유사성을 계산할 수 있습니다.

작업 라이브러리 `TextEmbedder` API를 사용하여 사용자 지정 텍스트 임베더를 모바일 앱에 배포합니다.

## TextEmbedder API의 주요 기능

- 입력 텍스트에 대한 그래프 내 또는 그래프 외 [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) 또는 [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) 토큰화를 포함한 입력 텍스트 처리

- 요소 벡터 간의 [코사인 유사성](https://en.wikipedia.org/wiki/Cosine_similarity)을 계산하는 내장 유틸리티 함수

## 지원되는 텍스트 임베더 모델

다음 모델은 `TextEmbedder` API와 호환이 보장됩니다.

- [TensorFlow Hub의 Universal Sentence Encoder TFLite 모델](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## C++에서 추론 실행하기

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

<code>TextEmbedder</code>를 구성하기 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## Python에서 추론 실행하기

### 1단계: TensorFlow Lite 지원 Pypi 패키지 설치하기

다음 명령을 사용하여 TensorFlow Lite Support Pypi 패키지를 설치할 수 있습니다.

```sh
pip install tflite-support
```

### 2단계: 모델 사용하기

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

<code>TextEmbedder</code>를 구성하기 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## 예제 결과

정규화된 요소 벡터 간의 코사인 유사성은 -1과 1 사이의 점수를 반환합니다. 높을수록 좋습니다. 즉, 코사인 유사성이 1이면 두 벡터가 동일하다는 의미입니다.

```
Cosine similarity: 0.954312
```

고유한 모델 및 테스트 데이터로 간단한 [TextEmbedder용 CLI 데모 도구](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder)를 사용해 보세요.

## 모델 호환성 요구 사항

`TextEmbedder` API는 필수 [TFLite 모델 메타데이터](https://www.tensorflow.org/lite/models/convert/metadata)가 있는 TFLite 모델을 예상합니다.

세 가지 주요 유형의 모델이 지원됩니다.

- BERT 기반 모델(자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h) 참조):

    - 정확히 3개의 입력 텐서(kTfLiteString)

        - 메타데이터 이름이 "ids"인 ID 텐서
        - 메타데이터 이름이 "mask"인 마스크 텐서
        - 메타데이터 이름이 "segment_ids"인 세그먼트 ID 텐서

    - 정확히 하나의 출력 텐서(kTfLiteUInt8/kTfLiteFloat32)

        - 이 출력 레이어에 대해 반환된 요소 벡터의 `N` 차원에 해당하는 `N` 구성 요소가 있음
        - 2 또는 4차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`

    - Wordpiece/Sentencepiece Tokenizer용 input_process_units

- Universal Sentence Encoder 기반 모델(자세한 내용은 [소스 코드](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h) 참조):

    - 정확히 3개의 입력 텐서(kTfLiteString)

        - 메타데이터 이름이 "inp_text"인 쿼리 텍스트 텐서
        - 메타데이터 이름이 "res_context"인 응답 컨텍스트 텐서
        - 메타데이터 이름이 "res_text"인 응답 텍스트 텐서

    - 정확히 2개의 출력 텐서(kTfLiteUInt8/kTfLiteFloat32)

        - 메타데이터 이름이 "query_encoding"인 쿼리 인코딩 텐서
        - 메타데이터 이름이 "response_encoding"인 응답 인코딩 텐서
        - 둘 다 이 출력 레이어에 대해 반환된 요소 벡터의 `N` 차원에 해당하는 `N` 구성 요소가 있음
        - 둘 다 2차원 또는 4차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`

- 다음을 포함하는 모든 텍스트 임베더 모델:

    - 입력 텍스트 텐서(kTfLiteString)

    - 하나 이상의 출력 임베딩 텐서(kTfLiteUInt8/kTfLiteFloat32)

        - 이 출력 레이어에 대해 반환된 요소 벡터의 `N` 차원에 해당하는 `N` 구성 요소가 있음
        - 2 또는 4차원, 즉 `[1 x N]` 또는 `[1 x 1 x 1 x N]`

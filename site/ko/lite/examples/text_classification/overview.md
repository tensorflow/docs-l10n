# 텍스트 분류

Use a TensorFlow Lite model to category a paragraph into predefined groups.

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## 시작하기


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the guide of [TensorFLow Lite Task Library](../../inference_with_metadata/task_library/nl_classifier) to integrate text classification models within just a few lines of code. You can also integrate the model using the [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java).

The Android example below demonstrates the implementation for both methods as [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_task_api) and [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android/lib_interpreter), respectively.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android example</a>

Android이외의 플랫폼을 사용 중이거나 TensorFlow Lite API에 이미 익숙한 경우 스타터 텍스트 분류 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Download starter model</a>

## 동작 원리

텍스트 분류는 콘텐츠에 따라 미리 정의된 그룹으로 단락을 분류합니다.

이 사전 훈련된 모델은 단락의 감정이 긍정적인지 부정적인지 예측합니다. Mass et al의 [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/)에 대해 훈련되었으며, 이는 긍정적 또는 부정적이라고 표시된 IMDB 영화 리뷰로 구성됩니다.

모델로 단락을 분류하는 단계는 다음과 같습니다.

1. Tokenize the paragraph and convert it to a list of word ids using a predefined vocabulary.
2. TensorFlow Lite 모델에 목록을 제공합니다.
3. Get the probability of the paragraph being positive or negative from the model outputs.

### 참고

- 영어만 지원됩니다.
- 이 모델은 영화 리뷰 데이터세트에서 훈련되었으므로 다른 도메인의 텍스트를 분류할 때 정확성이 떨어질 수 있습니다.

## 성능 벤치마크

성능 벤치 마크 수치는 [여기에 설명된](https://www.tensorflow.org/lite/performance/benchmarks) 도구를 사용하여 생성됩니다.

<table>
  <thead>
    <tr>
      <th>모델명</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite">텍스트 분류</a></td>
    <td rowspan="3">       0.6 Mb     </td>
    <td>Pixel 3(Android 10)</td>
    <td>0.05ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>0.05ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
    <td>0.025ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## 예제 출력

본문 | 부정적 (0) | 긍정적 (1)
--- | --- | ---
이 영화는 내가 최근에 본 것 중 최고입니다 | 25.3 % | 74.7 %
: 년. 강력히 추천합니다! : : : |  |
내 시간 낭비입니다. | 72.5 % | 27.5 %

## 훈련 데이터세트 사용하기

Follow this [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) to apply the same technique used here to train a text classification model using your own datasets. With the right dataset, you can create a model for use cases such as document categorization or toxic comments detection.

## 텍스트 분류에 대해 자세히 알아보기

- [4개의 스레드가 사용되었습니다](https://www.tensorflow.org/tutorials/text/word_embeddings).

# Text classification

사전 훈련된 모델을 사용하여 단락을 사전 정의된 그룹으로 분류합니다.

## 시작하기


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example applications that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android">Android example</a>

Android이외의 플랫폼을 사용 중이거나 TensorFlow Lite API에 이미 익숙한 경우 스타터 텍스트 분류 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite">Download starter model</a>

## 동작 원리

Text classification categorizes a paragraph into predefined groups based on its content.

This pretrained model predicts if a paragraph's sentiment is positive or negative. It was trained on [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) from Mass et al, which consists of IMDB movie reviews labeled as either positive or negative.

Here are the steps to classify a paragraph with the model:

1. 단락을 토큰화하고 사전 정의된 어휘를 사용하여 단어 ID 목록으로 변환하세요.
2. Feed the list to the TensorFlow Lite model.
3. Get the probability of the paragraph being positive or negative from the model outputs.

### 참고

- 영어만 지원됩니다.
- 이 모델은 영화 리뷰 데이터세트에서 훈련되었으므로 다른 도메인의 텍스트를 분류할 때 정확성이 떨어질 수 있습니다.

## 성능 벤치마크

Performance benchmark numbers are generated with the tool [described here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite">Text Classification</a></td>
    <td rowspan="3">       0.6 Mb     </td>
    <td>Pixel 3(Android 10)</td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>0.05ms*</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
    <td>0.025ms** </td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## Example output

Text | Negative (0) | Positive (1)
--- | --- | ---
이 영화는 내가 최근에 본 것 중 최고입니다 | 25.3% | 74.7%
: years. Strongly recommend it!              :              :              : |  |
What a waste of my time. | 72.5% | 27.5%

## 훈련 데이터세트 사용하기

[튜토리얼](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker/demo/text_classification.ipynb)에 따라 여기에 사용된 것과 같은 기술을 적용하여 자체 데이터세트로 텍스트 분류 모델을 훈련하세요. 올바른 데이터세트를 사용하면 설명서 분류 또는 악성 댓글 감지와 같은 사용 사례에 대한 모델을 만들 수 있습니다.

## Read more about text classification

- [모델 훈련을 위한 단어 임베딩 및 튜토리얼](https://www.tensorflow.org/tutorials/text/word_embeddings)

# BERT Question and Answer

Use a TensorFlow Lite model to answer questions based on the content of a given passage.

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## 시작하기


<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite를 처음 사용하고 Android 또는 iOS로 작업하는 경우, 다음 예제 애플리케이션을 탐색하면 시작하는 데 도움이 됩니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Android example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">iOS
example</a>

Android/iOS 이외의 플랫폼을 사용 중이거나 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 이미 익숙한 경우 스타터 질문 및 답변 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Download starter model and vocab</a>

For more information about metadata and associated fields (e.g. `vocab.txt`) see <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">Read the metadata from models</a>.

## 동작 원리

The model can be used to build a system that can answer users’ questions in natural language. It was created using a pre-trained BERT model fine-tuned on SQuAD 1.1 dataset.

[BERT](https://github.com/google-research/bert), or Bidirectional Encoder Representations from Transformers, is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing tasks.

이 앱은 BERT의 압축 버전인 MobileBERT를 사용하는데 실행 속도는 4배가 더 빠르고 모델 크기는 4배 더 작습니다.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 또는 Stanford Question Answering Dataset는 Wikipedia의 문서와 각 문서에 대한 질문-답변 세트로 구성된 독해 데이터세트입니다.

모델은 구절과 질문을 입력으로 사용한 다음 질문에 가장 잘 답할 수 있는 구절의 일부를 반환합니다. BERT [논문](https://arxiv.org/abs/1810.04805)에 설명되어 있고 샘플 앱에 구현된 토큰화 및 사후 처리 단계를 포함한 약간 복잡한 전처리가 필요합니다.

## 성능 벤치마크

성능 벤치 마크 수치는 [여기에 설명된](https://www.tensorflow.org/lite/performance/benchmarks) 도구를 사용하여 생성됩니다.

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
    <td rowspan="3">
      <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Mobile Bert</a>
    </td>
    <td rowspan="3">       100.5 Mb     </td>
    <td>Pixel 3(Android 10)</td>
    <td>123ms*</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>74ms*</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
    <td>257ms** </td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## Example output

### 구절(입력)

> Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. It is considered one of the Big Four technology companies, alongside Amazon, Apple, and Facebook.
>
> Google은 1998년 9월 Larry Page와 Sergey Brin이 캘리포니아 스탠포드 대학의 박사 학위 시절에 설립했습니다.  그들은 함께 주식의 약 14%를 소유하고 슈퍼 투표 주식을 통해 주주의 투표권의 56%를 통제합니다. 1998년 9월 4일 캘리포니아에서 Google을 캘리포니아 비상장 기업으로 통합했습니다. 그 후 Google은 2002 년 10 월 22 일 델라웨어에서 재편되었습니다. 2004년 8 월 19 일에 기업 공개 (IPO)가 이루어졌고 Google은 Googleplex라는 별명을 가진 캘리포니아 주 마운틴 뷰에있는 본사로 이전했습니다. 2015 년 8 월 Google은 Alphabet Inc.라는 대기업으로 다양한 이해 관계를 재편 할 계획을 발표했습니다. Google은 Alphabet의 선도적 인 자회사이며 앞으로도 Alphabet의 인터넷 이익을위한 우산 회사가 될 것입니다. Sundar Pichai는 Google의 CEO로 임명되어 Alphabet의 CEO가 된 Larry Page를 대신했습니다.

### 질문(입력)

> Who is the CEO of Google?

### 답변(출력)

> Sundar Pichai

## Read more about BERT

- 학술 논문: [BERT: 언어 이해를 위한 Deep Bidirectional Transformers의 사전 교육](https://arxiv.org/abs/1810.04805)
- [Open-source implementation of BERT](https://github.com/google-research/bert)

# BERT 질문 및 답변

TensorFlow Lite 모델을 사용하여 주어진 구절의 내용을 기반으로 질문에 답합니다.

참고: (1) 기존 모델을 통합하려면 [TensorFlow Lite 작업 라이브러리](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer)를 사용해 보세요. (2) 모델을 사용자 지정하려면 [TensorFlow Lite 모델 제작기](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)를 사용해 보세요.

## 시작하기

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

TensorFlow Lite를 처음 사용하고 Android 또는 iOS로 작업하는 경우, 다음 예제 애플리케이션을 탐색하면 시작하는 데 도움이 됩니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Android 예제</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">iOS 예제</a>

Android/iOS 이외의 플랫폼을 사용 중이거나 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)에 이미 익숙한 경우 스타터 질문 및 답변 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">스타터 모델 및 어휘 다운로드</a>

메타 데이터 및 관련 필드(예: `vocab.txt`)에 대한 자세한 내용은 <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">모델에서 메타 데이터 읽기</a>를 참조하세요.

## 동작 원리

모델은 자연어로 사용자의 질문에 답할 수 있는 시스템을 구축하는 데 사용할 수 있습니다. 모델은 SQuAD 1.1 데이터세트에서 미세 조정되고 사전 훈련된 BERT 모델을 사용하여 생성되었습니다.

[BERT](https://github.com/google-research/bert) 또는 Bidirectional Encoder Representations from Transformers는 다양한 배열의 자연어 처리 작업에 대한 최신 결과를 얻는 언어 표현을 사전 훈련하는 메서드입니다.

이 앱은 BERT의 압축 버전인 MobileBERT를 사용하는데 실행 속도는 4배가 더 빠르고 모델 크기는 4배 더 작습니다.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 또는 Stanford Question Answering Dataset는 Wikipedia의 문서와 각 문서에 대한 질문-답변 세트로 구성된 독해 데이터세트입니다.

모델은 구절과 질문을 입력으로 사용한 다음 질문에 가장 잘 답할 수 있는 구절의 일부를 반환합니다. BERT [논문](https://arxiv.org/abs/1810.04805)에 설명되어 있고 샘플 앱에 구현된 토큰화 및 사후 처리 단계를 포함한 약간 복잡한 전처리가 필요합니다.

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
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Mobile Bert</a>
</td>
    <td rowspan="3">       100.5Mb</td>
    <td>Pixel 3(Android 10)</td>
    <td>123ms *</td>
  </tr>
   <tr>
     <td>Pixel 4(Android 10)</td>
    <td>74ms *</td>
  </tr>
   <tr>
     <td>iPhone XS(iOS 12.4.1)</td>
    <td>257ms **</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

** 최상의 결과를 위해 iPhone에서 2개의 스레드가 사용되었습니다.

## 예제 출력

### 구절(입력)

> Google LLC는 온라인 광고 기술, 검색 엔진, 클라우드 컴퓨팅, 소프트웨어 및 하드웨어를 포함하는 인터넷 관련 서비스 및 제품을 전문으로 하는 미국의 다국적 기술 회사입니다. Amazon, Apple 및 Facebook과 함께 Big Four 기술 회사 중 하나로 간주합니다.
>
> Google은 1998년 9월 Larry Page와 Sergey Brin이 캘리포니아 스탠포드 대학의 박사 학위 시절에 설립했습니다.  그들은 함께 주식의 약 14%를 소유하고 슈퍼 투표 주식을 통해 주주의 투표권의 56%를 통제합니다. 1998년 9월 4일 캘리포니아에서 Google을 캘리포니아 비상장 기업으로 통합했습니다. 그 후 Google은 2002 년 10 월 22 일 델라웨어에서 재편되었습니다. 2004년 8 월 19 일에 기업 공개 (IPO)가 이루어졌고 Google은 Googleplex라는 별명을 가진 캘리포니아 주 마운틴 뷰에있는 본사로 이전했습니다. 2015 년 8 월 Google은 Alphabet Inc.라는 대기업으로 다양한 이해 관계를 재편 할 계획을 발표했습니다. Google은 Alphabet의 선도적 인 자회사이며 앞으로도 Alphabet의 인터넷 이익을위한 우산 회사가 될 것입니다. Sundar Pichai는 Google의 CEO로 임명되어 Alphabet의 CEO가 된 Larry Page를 대신했습니다.

### 질문(입력)

> Google의 CEO는 누구입니까?

### 답변(출력)

> Sundar Pichai

## BERT에 대해 자세히 알아보기

- 학술 논문: [BERT: 언어 이해를 위한 Deep Bidirectional Transformers의 사전 교육](https://arxiv.org/abs/1810.04805)
- [BERT의 오픈 소스 구현](https://github.com/google-research/bert)

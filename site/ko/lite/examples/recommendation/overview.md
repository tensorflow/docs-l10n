# 추천

개인화된 추천은 미디어 콘텐츠 검색, 쇼핑 제품 제안 및 다음 앱 추천과 같은 모바일 기기의 다양한 사용 사례에 널리 사용됩니다. 사용자 개인 정보를 보호하면서 애플리케이션에 개인화된 추천을 제공하는 데 관심이 있다면 다음 예제와 도구 키트를 살펴보세요.

## 시작하기

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Android 사용자에게 관련 항목을 추천하는 방법을 보여주는 TensorFlow Lite 샘플 애플리케이션을 제공합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/android">Android 예제</a>

Android 이외의 플랫폼을 사용 중이거나 TensorFlow Lite API에 이미 익숙한 경우, 스타터 추천 모델을 다운로드할 수 있습니다.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/recommendation.tar.gz">스타터 모델 다운로드하기</a>

Github에서 고유한 모델을 훈련하기 위한 훈련 스크립트도 제공합니다.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml">훈련 코드</a>

## 모델 아키텍처 이해하기

컨텍스트 인코더를 사용하여 순차 사용자 기록을 인코딩하고 레이블 인코더를 사용하여 예측된 추천 후보를 인코딩하는 이중 인코더 모델 아키텍처를 이용합니다. 컨텍스트와 레이블 인코딩 간의 유사성은 예측된 후보가 사용자의 요구를 충족할 가능성을 나타내는 데 사용됩니다.

이 코드베이스에는 다음의 세 가지 순차적 사용자 기록 인코딩 기술이 제공됩니다.

- BOW(bag-of-words encoder): 컨텍스트 순서를 고려하지 않고 사용자 활동의 임베딩을 평균화합니다.
- CNN(컨볼루셔널 신경망 인코더): 여러 레이어의 컨볼루셔널 신경망을 적용하여 컨텍스트 인코딩을 생성합니다.
- RNN(순환 신경망): 순환 신경망을 적용하여 컨텍스트 시퀀스를 인코딩합니다.

*참고: 모델은 연구 목적으로 [MovieLens](https://grouplens.org/datasets/movielens/1m/) 데이터세트에 기초하여 훈련됩니다.

## 예

입력 ID:

- 매트릭스(ID: 260)
- 라이언 일병 구하기(ID: 2028)
- (기타 등등)

출력 ID:

- 스타워즈: 에피소드 VI - 제다이의 귀환(ID: 1210)
- (기타 등등)

## 성능 벤치마크

성능 벤치마크 수치는 [여기에 설명된](https://www.tensorflow.org/lite/performance/benchmarks) 도구를 사용하여 생성됩니다.

<table>
  <thead>
    <tr>
      <th>모델 이름</th>
      <th>모델 크기</th>
      <th>기기</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/recommendation/20200720/model.tar.gz">추천</a></td>
    <td rowspan="3">0.52Mb</td>
    <td>Pixel 3</td>
    <td>0.09ms*</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>0.05ms*</td>
  </tr>
</table>

* 4개의 스레드가 사용되었습니다.

## 고유한 훈련 데이터 사용하기

훈련된 모델 외에도 고유한 데이터로 모델을 훈련할 수 있는 오픈 소스 [GitHub 도구 키트](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml)가 제공됩니다. 이 튜토리얼의 설명에 따라 도구 키트를 사용하고 자신의 모바일 애플리케이션에서 훈련된 모델을 배포하는 방법을 배울 수 있습니다.

이 [튜토리얼](https://github.com/tensorflow/examples/tree/master/lite/examples/recommendation/ml/ondevice_recommendation.ipynb)에 따라 여기서 사용된 것과 같은 기술을 적용하여 고유한 데이터세트로 추천 모델을 훈련해 보세요.

## 고유 데이터를 사용한 모델 사용자 정의 팁

이 데모 애플리케이션에 통합된 미리 훈련된 모델은 [MovieLens](https://grouplens.org/datasets/movielens/1m/) 데이터세트로 훈련했으며 어휘 크기, 내장 차원 및 입력 컨텍스트 길이와 같은 사용자 고유의 데이터를 기반으로 모델 구성을 수정해야 할 수 있습니다. 다음은 몇 가지 팁입니다.

- 입력 컨텍스트 길이: 최상의 입력 컨텍스트 길이는 데이터세트에 따라 다릅니다. 레이블 이벤트가 장기 관심사 및 단기 컨텍스트와 얼마나 관련성이 있는지에 따라 입력 컨텍스트 길이를 선택하는 것이 좋습니다.

- 인코더 유형 선택: 입력 컨텍스트 길이에 따라 인코더 유형을 선택하는 것이 좋습니다. Bag-of-words 인코더는 짧은 입력 컨텍스트 길이(예: <10)에 효과적이며 CNN 및 RNN 인코더는 긴 입력 컨텍스트 길이에서 더 많은 요약 기능을 제공합니다.

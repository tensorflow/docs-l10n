**업데이트: 2020년 8월 7일**

## 양자화

- 동적 범위 커널에 대한 훈련 후 양자화 -- [출시됨](https://blog.tensorflow.org/2018/09/introducing-model-optimization-toolkit.html)
- (8b) 고정 소수점 커널에 대한 훈련 후 양자화 -- [출시됨](https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html)
- (8b) 고정 소수점 커널에 대한 양자화 인식 훈련 및 <8b에 대한 실험 -- [출시됨](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)
- [WIP] (8b) 고정 소수점 RNN에 대한 훈련 후 양자화
- (8b) 고정 소수점 RNN에 대한 양자화 인식 훈련
- [WIP] 훈련 후 동적 범위 양자화를 위한 품질 및 성능 개선

## 잘라내기/희소성

- 훈련 중 규모 기반 가중치 잘라내기 -- [출시됨](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)
- TensorFlow Lite의 희소 모델 실행 지원 -- [WIP](https://github.com/tensorflow/model-optimization/issues/173)

## 가중치 클러스터링

- 훈련 중 가중치 클러스터링 -- [출시됨](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

## 연계 압축 기술

- [WIP] 다양한 압축 기술을 결합하기 위한 추가 지원, 오늘날 사용자는 하나의 훈련 중 기술만 훈련 후 양자화와 결합할 수 있습니다. 결합 제안이 곧 출시됩니다.

## 압축

- [WIP] 텐서 압축 API

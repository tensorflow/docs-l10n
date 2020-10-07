# TensorFlow 모델 최적화 시작하기

## 1. 작업에 가장 적합한 모델 선택하기

작업에 따라 모델 복잡성과 크기 간에 균형을 맞춰야 합니다. 작업에 높은 정확성이 필요한 경우, 크고 복잡한 모델이 필요할 수 있습니다. 정밀도가 낮은 작업의 경우, 디스크 공간과 메모리를 적게 사용할 뿐만 아니라 일반적으로 더 빠르고 에너지 효율적이기 때문에 더 작은 모델을 사용하는 것이 좋습니다.

## 2. 사전 최적화된 모델

기존 [TensorFlow Lite 사전 최적화 모델](https://www.tensorflow.org/lite/models)이 애플리케이션에 필요한 효율성을 제공하는지 확인하세요.

## 3. 훈련 후 도구

애플리케이션에 사전 훈련된 모델을 사용할 수 없는 경우, [TensorFlow Lite 변환](./quantization/post_training) 중에 [TensorFlow Lite 훈련 후 양자화 도구](https://www.tensorflow.org/lite/convert)를 사용해보세요. 이미 훈련된 TensorFlow 모델을 최적화할 수 있습니다.

자세한 내용은 [훈련 후 양자화 튜토리얼](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/post_training_quant.ipynb)을 참조하세요.

## 다음 단계: 훈련 시간 도구

위의 간단한 솔루션이 사용자의 요구를 충족하지 못하는 경우, 훈련 시간 최적화 기술을 포함해야 할 수 있습니다. 훈련 시간 도구를 사용하여 [더 최적화하고](optimize_further.md) 더 깊이 알아보세요.

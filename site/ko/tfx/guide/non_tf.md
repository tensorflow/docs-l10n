# TFX에서 다른 프레임워크 사용하기

플랫폼으로서의 TFX는 프레임워크 중립적이며 JAX, scikit-learn과 같은 다른 ML 프레임워크와 함께 사용할 수 있습니다.

이에 모델 개발자의 경우 다른 ML 프레임워크에서 구현된 모델 코드를 다시 작성할 필요가 없는 대신 TFX에서 대부분의 훈련 코드를 그대로 다시 사용할 수 있으며 다른 기능 TFX 및 TensorFlow 생태계의 일부 혜택의 이점을 누릴 수 있습니다.

TFX 파이프라인 SDK와 TFX 대부분의 모듈(예: 파이프라인 오케스트레이터)은 TensorFlow에 대한 직접적인 종속성은 없지만 데이터 형식과 같이 TensorFlow를 지향하는 몇 가지 측면이 있습니다. 특정 모델링 프레임워크의 요구 사항에 대한 고려와 함께 TFX 파이프라인으로 다른 Python 기반 ML 프레임워크에서 모델을 훈련할 수 있습니다. 여기에는 Scikit-learn, XGBoost 및 PyTorch 등이 포함됩니다. 다른 프레임워크와 함께 표준 TFX 구성 요소를 사용할 때 고려해야 할 몇 가지 사항은 다음과 같습니다.

- **ExampleGen**은 TFRecord 파일에서 [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)을 출력합니다. 이는 학습 데이터의 일반적인 표현이며 다운스트림 구성요소는 [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md)를 사용하여 메모리에서 Arrow/RecordBatch로 읽습니다. 이 데이터는 향후에 `tf.dataset`, `Tensors` 혹은 기타 형식으로 변환할 수 있습니다. tf.train.Example/TFRecord 이외의 페이로드/파일 형식도 고려하고 있지만 TFXIO 사용자의 경우 블랙박스여야 합니다.
- **Transform**은 훈련에 사용하는 프레임워크와 관계없이 변환된 훈련 예시 생성에 사용할 수 있지만 모델 형식이 `saved_model`이 아닌 경우 사용자가 모델에 변환된 그래프를 삽입할 수 없게 됩니다. 이 경우 모델 예측은 원시 기능 대신 변환된 기능을 가져와야 하며, 사용자는 제공할 때 모델 예측을 호출하기 전에 전처리 단계로 변환을 실행할 수 있습니다.
- **Trainer**는 [GenericTraining](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer)을 지원하며 사용자는 모든 ML 프레임워크를 사용하여 모델을 훈련할 수 있습니다.
- **Evaluator**는 기본적으로 `saved_model`만 지원하지만 사용자는 모델 평가를 위한 예측을 생성하는 UDF를 제공할 수 있습니다.

Python 기반이 아닌 프레임워크에서 모델을 훈련하려면 Docker 컨테이너에서 Kubernetes와 같은 컨테이너화된 환경에서 실행되는 파이프라인의 일부로 사용자 정의 훈련 구성 요소를 고립시켜야 합니다.

## JAX

[JAX](https://github.com/google/jax)는 고성능 머신러닝 연구를 위해 결합한 Autograd와 XLA입니다. [Flax](https://github.com/google/flax)는 유연성을 위해 설계된 JAX용 신경망 라이브러리 및 생태계입니다.

[jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf)를 사용하여 학습된 JAX/Flax 모델을 일반 학습 및 모델 평가와 함께 TFX에서 원활하게 사용할 수 있는 `saved_model` 형식으로 변환할 수 있습니다. 자세한 내용은 이 [예시](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py)를 확인하세요.

## scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/)은 Python 프로그래밍 언어용 머신러닝 라이브러리입니다. TFX-Addons에서 사용자 정의한 훈련 및 평가가 포함된 e2e [예시](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins)를 확인할 수 있습니다.

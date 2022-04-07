<!--* freshness: { owner: 'maringeo' reviewed: '2022-01-12'} *-->

# TensorFlow 2에서 TF 허브의 SavedModel

[TensorFlow 2의 SavedModel 형식](https://www.tensorflow.org/guide/saved_model)은 TensorFlow 허브에서 사전 훈련된 모델 및 모델 조각을 공유하는 권장 방법입니다. 이 형식은 이전 [TF1 허브 형식](tf1_hub_module.md)을 대체하고 새로운 API 세트와 함께 제공됩니다.

이 페이지에서는 하위 수준 `hub.load()` API 및 해당 `hub.KerasLayer` 래퍼를 사용하여 TensorFlow 2 프로그램에서 TF2 SavedModel을 재사용하는 방법을 설명합니다. 일반적으로, `hub.KerasLayer`는 다른 `tf.keras.layers`와 결합하여 Keras 모델 또는 TF2 Estimator의 `model_fn`을 빌드합니다. 이러한 API는 또한 제한 내에서 TF1 허브 형식으로 레거시 모델을 로드할 수도 있습니다. [호환성 가이드](model_compatibility.md)를 참조하세요.

TensorFlow 1 사용자는 TF 1.15로 업데이트한 다음 동일한 API를 사용할 수 있습니다. 이전 버전의 TF1은 작동하지 않습니다.

## TF 허브의 SavedModel 사용하기

### Keras에서 SavedModel 사용하기

[Keras](https://www.tensorflow.org/guide/keras/)는 Keras Layer 객체를 구성하여 딥 러닝 모델을 빌드하기 위한 TensorFlow의 상위 수준 API입니다. `tensorflow_hub` 라이브러리는 SavedModel의 URL(또는 파일 시스템 경로)로 초기화된 다음 사전 훈련된 가중치를 포함해 SavedModel의 계산을 제공하는 클래스 `hub.KerasLayer`를 제공합니다.

다음은 사전 훈련된 텍스트 임베딩을 사용하는 예입니다.

```python
import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
```

여기서 일반적인 Keras 방식으로 텍스트 분류자를 빌드할 수 있습니다.

```python
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

[텍스트 분류 colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)은 이러한 분류자를 훈련하고 평가하는 방법을 보여주는 완벽한 예입니다.

`hub.KerasLayer`의 모델 가중치는 기본적으로 훈련 불가능한 것으로 설정됩니다. 이를 변경하는 방법은 아래 미세 조정 섹션을 참조하세요. Keras에서 일반적으로 그렇듯이 동일한 레이어 객체의 모든 애플리케이션 간에 가중치가 공유됩니다.

### Estimator에서 SavedModel 사용하기

분산 훈련을 위한 TensorFlow의 [Estimator](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator)를 사용하는 개발자는 여러 `tf.keras.layers` 중에서 `hub.KerasLayer`와 관련해 `model_fn`을 작성하여 TF 허브의 SavedModel을 사용할 수 있습니다.

### 배경: SavedModel 다운로드 및 캐싱

TensorFlow 허브(또는 [호스팅](hosting.md) 프로토콜을 구현하는 다른 HTTPS 서버)의 SavedModel을 사용하면 로컬 파일 시스템에 이러한 모델(아직 없는 경우)이 다운로드되어 압축이 풀립니다. 환경 변수 `TFHUB_CACHE_DIR` 설정을 통해 다운로드하고 압축 해제하지 않은 SavedModel을 캐싱하기 위한 기본 임시 위치를 재정의할 수 있습니다. 자세한 내용은 [캐싱](caching.md)을 참조하세요.

### 하위 수준 TensorFlow에서 SavedModel 사용하기

` hub.load(handle)` 함수는 SavedModel을 다운로드하고 압축을 해제한 다음(`handle`이 이미 파일 시스템 경로인 경우는 제외) TensorFlow의 내장 함수 `tf.saved_model.load()`를 이용해 로드 결과를 반환합니다. 따라서 `hub.load()`는 유효한 모든 SavedModel을 처리할 수 있습니다(이 점에서 TF1의 이전 `hub.Module`과 다름).

#### 고급 주제: 로드 후 SavedModel에서 기대할 사항

SavedModel의 내용에 따라 `obj = hub.load(...)`의 결과는 다양한 방식으로 호출될 수 있습니다(TensorFlow의 [SavedModel 가이드](https://www.tensorflow.org/guide/saved_model)에 매우 자세히 설명되어 있음).

- SavedModel(있는 경우)의 서비스 서명은 구체적인 함수 사전으로 표현되며 `tensors_out = obj.signatures["serving_default"](**tensors_in)`와 같이 호출할 수 있습니다. 이때 텐서 사전은 각 입력 및 출력에 의해 입력되고 서명의 형상과 dtype 제약 조건이 적용됩니다.

- 저장된 객체(있는 경우)의 [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) 데코레이션 메서드는 저장하기 전에 tf.function이 [추적된](https://www.tensorflow.org/tutorials/customization/performance#tracing) 텐서 및 텐서가 아닌 인수의 모든 조합에 의해 호출될 수 있는 tf.function 객체로 복원됩니다. 특히 적절한 트레이스를 가진 `obj.__call__` 메서드가 있으면 `obj` 자체를 Python 함수처럼 호출할 수 있습니다. `output_tensor = obj(input_tensor, training=False)`를 간단한 예로 들 수 있습니다.

그 결과로 SavedModel가 구현할 수 있는 인터페이스에 엄청난 자유가 생깁니다. `obj`의 [Reusable SavedModel 인터페이스](reusable_saved_models.md)는 `hub.KerasLayer`와 같은 어댑터를 포함해 클라이언트 코드에서 SavedModel의 사용 방법을 알 수 있도록 규칙을 설정합니다.

특히 더 큰 모델에서 재사용할 수 없는 전체 모델과 같은 일부 SavedModel은 이 규칙을 따르지 않을 수 있으며 서비스 서명만 제공합니다.

SavedModel의 훈련 가능한 변수는 훈련 가능한 것으로 다시 로드되며 `tf.GradientTape`는 기본적으로 이들 변수를 감시합니다. 아래 미세 조정 섹션에서 몇 가지 주의 사항을 참조하여 우선 몇 가지 사항을 피하는 것이 좋습니다. 미세 조정을 원하는 경우에도 `obj.trainable_variables`가 원래 훈련 가능한 변수의 하위 세트만 다시 훈련하도록 권장하는지 확인해야 합니다.

## TF 허브용 SavedModel 만들기

### 개요

SavedModel은 훈련된 모델 또는 모델 조각에 대한 TensorFlow의 표준 직렬화 형식입니다. 이 모델은 계산을 수행하기 위해 모델의 훈련된 가중치를 정확한 TensorFlow 연산과 함께 저장하며, 이 모델이 생성된 출처 코드와 독립적으로 사용할 수 있습니다. 특히, TensorFlow 연산이 공통된 기본 언어이기 때문에 Keras와 같은 다양한 상위 수준 모델 구축 API에서 재사용할 수 있습니다.

### Keras에서 저장하기

TensorFlow 2부터 `tf.keras.Model.save()` 및 `tf.keras.models.save_model()`은 기본적으로 SavedModel 형식(HDF5 아님)을 사용합니다. 이렇게 얻어진 SavedModel은 `hub.load()`, `hub.KerasLayer` 및 앞으로 제공될 다른 유사한 고수준 API 어댑터와 함께 사용할 수 있습니다.

전체 Keras 모델을 공유하려면 간단히 `include_optimizer=False`를 저장합니다.

Keras 모델의 조각을 공유하려면 해당 부분 자체를 모델로 만든 다음 저장합니다. 처음부터 그와 같이 코드를 배치할 수 있습니다.

```python
piece_to_share = tf.keras.Model(...)
full_model = tf.keras.Sequential([piece_to_share, ...])
full_model.fit(...)
piece_to_share.save(...)
```

또는 팩트 이후에 공유할 조각을 잘라냅니다(전체 모델의 레이어와 일치하는 경우).

```python
full_model = tf.keras.Model(...)
sharing_input = full_model.get_layer(...).get_output_at(0)
sharing_output = full_model.get_layer(...).get_output_at(0)
piece_to_share = tf.keras.Model(sharing_input, sharing_output)
piece_to_share.save(..., include_optimizer=False)
```

[TensorFlow Models](https://github.com/tensorflow/models) on GitHub uses the former approach for BERT (see [nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py), note the split between `core_model` for export and the `pretrainer` for restoring the checkpoint) and the latter approach for ResNet (see [legacy/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/tfhub_export.py)).

### 저수준 TensorFlow에서 저장하기

이를 위해서는 TensorFlow의 [SavedModel 가이드](https://www.tensorflow.org/guide/saved_model) 내용을 잘 알고 있어야 합니다.

서비스 서명 그 이상을 제공하려면 [재사용 가능한 SavedModel 인터페이스](reusable_saved_models.md)를 구현해야 합니다. 개념적으로 이 내용은 다음과 같습니다.

```python
class MyMulModel(tf.train.Checkpoint):
  def __init__(self, v_init):
    super().__init__()
    self.v = tf.Variable(v_init)
    self.variables = [self.v]
    self.trainable_variables = [self.v]
    self.regularization_losses = [
        tf.function(input_signature=[])(lambda: 0.001 * self.v**2),
    ]

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, inputs):
    return tf.multiply(inputs, self.v)

tf.saved_model.save(MyMulModel(2.0), "/tmp/my_mul")

layer = hub.KerasLayer("/tmp/my_mul")
print(layer([10., 20.]))  # [20., 40.]
layer.trainable = True
print(layer.trainable_weights)  # [2.]
print(layer.losses)  # 0.004
```

## 미세 조정

가져온 SavedModel의 이미 훈련된 변수를 주변 모델의 변수와 함께 훈련하는 작업을 SavedModel의 *미세 조정*이라고 합니다. 이 조정의 결과로 품질이 향상될 수 있지만 종종 훈련이 더 까다로워집니다(시간이 더 많이 걸리고, 옵티마이저와 하이퍼 매개변수에 대한 종속성이 커지며, 과대적합 위험이 높아지고, 특히 CNN의 경우 데이터세트 확대가 필요할 수 있음). SavedModel 소비자는 바람직한 훈련 체계를 수립한 후에만, 그리고 SavedModel 게시자가 권장하는 경우에만 미세 조정을 고려하는 것이 좋습니다.

미세 조정으로 훈련된 "연속" 모델의 매개변수가 변경됩니다. 텍스트 입력을 토큰화하고 토큰을 임베딩 행렬의 해당 항목에 매핑하는 것과 같은 하드 코딩된 변환은 변경하지 않습니다.

### SavedModel 소비자의 경우

`hub.KerasLayer`를 다음과 같이 만들면

```python
layer = hub.KerasLayer(..., trainable=True)
```

레이어에서 로드한 SavedModel을 미세 조정할 수 있습니다. SavedModel에 선언된 훈련 가능한 가중치 및 가중치 regularizer를 Keras 모델에 추가하고 훈련 모드에서 SavedModel의 계산을 실행합니다(드롭아웃 등을 고려).

[이미지 분류 colab](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)에 선택적 미세 조정이 포함된 엔드 투 엔드 예제가 있습니다.

#### 미세 조정 결과 다시 내보내기

고급 사용자는 SavedModel에 미세 조정 결과를 다시 저장하여 원래 로드된 모델 대신 이 모델이 사용되도록 할 수 있습니다. 이 목적으로 다음과 같은 코드를 이용합니다.

```python
loaded_obj = hub.load("https://tfhub.dev/...")
hub_layer = hub.KerasLayer(loaded_obj, trainable=True, ...)

model = keras.Sequential([..., hub_layer, ...])
model.compile(...)
model.fit(...)

export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
tf.saved_model.save(loaded_obj, export_module_dir)
```

### SavedModel 작성자의 경우

TensorFlow 허브에서 공유하기 위해 SavedModel을 만들 때 소비자가 모델을 미세 조정해야 하는지, 해야 한다면 어떻게 해야 하는지 미리 생각하고 문서에 지침을 제공하세요.

Keras 모델에서 저장하면 미세 조정의 모든 메커니즘이 동작합니다(가중치 정규화 손실 방지, 훈련 가능한 변수 선언, `training=True` 및 `training=False` 모두에 대해 `__call__` 추적 등).

그래디언트 흐름과 잘 어울리는 모델 인터페이스(예: 소프트맥스 확률 또는 top-k 예측 대신 출력 로짓)를 선택합니다.

모델이 드롭아웃, 배치 정규화 또는 하이퍼 매개변수를 포함하는 유사한 학습 기술을 사용하는 경우 예상되는 많은 대상 문제 및 배치 크기에서 의미가 있는 값으로 설정합니다(이 글을 쓰는 시점에서 Keras에서 저장하면 소비자가 쉽게 조정할 수 없습니다).

개별 레이어의 가중치 regularizer는 (정규화 강도 계수와 함께) 저장되지만 옵티마이저 내에서 가중치 정규화(예: `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`)는 손실됩니다. 따라서 SavedModel의 소비자에게 적절한 고지를 해주기 바랍니다.

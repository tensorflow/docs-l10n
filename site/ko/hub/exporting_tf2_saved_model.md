<!--* freshness: { owner: 'maringeo' reviewed: '2021-10-10' review_interval: '6 months' } *-->

# SavedModel 내보내기

This page describes the details of exporting (saving) a model from a TensorFlow program to the [SavedModel format of TensorFlow 2](https://www.tensorflow.org/guide/saved_model). This format is the recommended way to share pre-trained models and model pieces on TensorFlow Hub. It replaces the older [TF1 Hub format](tf1_hub_module.md) and comes with a new set of APIs. You can find more information on exporting the TF1 Hub format models in [TF1 Hub format export](exporting_hub_format.md). You can find details on how to compress the SavedModel for sharing it on TensorFlow Hub [here](writing_documentation.md#model-specific_asset_content).

일부 모델 구축 도구 키트는 이미 이를 위한 도구를 제공합니다(예: 아래에서 [TensorFlow Model Garden](#tensorflow-model-garden) 참조).

## 개요

SavedModel은 훈련된 모델 또는 모델 조각에 대한 TensorFlow의 표준 직렬화 형식입니다. 이 모델은 계산을 수행하기 위해 모델의 훈련된 가중치를 정확한 TensorFlow 연산과 함께 저장하며, 이 모델이 생성된 출처 코드와 독립적으로 사용할 수 있습니다. 특히, TensorFlow 연산이 공통된 기본 언어이기 때문에 Keras와 같은 다양한 상위 수준 모델 구축 API에서 재사용할 수 있습니다.

## Keras에서 저장하기

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

[TensorFlow Models](https://github.com/tensorflow/models) on GitHub uses the former approach for BERT (see [nlp/tools/export_tfhub_lib.py](https://github.com/tensorflow/models/blob/master/official/nlp/tools/export_tfhub_lib.py), note the split between `core_model` for export and the `pretrainer` for restoring the checkpoint) and the the latter approach for ResNet (see [legacy/image_classification/tfhub_export.py](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/tfhub_export.py)).

## 저수준 TensorFlow에서 저장하기

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

## SavedModel 제작자를 위한 조언

TensorFlow 허브에서 공유하기 위해 SavedModel을 만들 때 소비자가 모델을 미세 조정해야 하는지, 해야 한다면 어떻게 해야 하는지 미리 생각하고 문서에 지침을 제공하세요.

Keras 모델에서 저장하면 미세 조정의 모든 메커니즘이 동작합니다(가중치 정규화 손실 방지, 훈련 가능한 변수 선언, `training=True` 및 `training=False` 모두에 대해 `__call__` 추적 등).

그래디언트 흐름과 잘 어울리는 모델 인터페이스(예: 소프트맥스 확률 또는 top-k 예측 대신 출력 로짓)를 선택합니다.

모델이 드롭아웃, 배치 정규화 또는 하이퍼 매개변수를 포함하는 유사한 학습 기술을 사용하는 경우 예상되는 많은 대상 문제 및 배치 크기에서 의미가 있는 값으로 설정합니다(이 글을 쓰는 시점에서 Keras에서 저장하면 소비자가 쉽게 조정할 수 없습니다).

개별 레이어의 가중치 regularizer는 (정규화 강도 계수와 함께) 저장되지만 옵티마이저 내에서 가중치 정규화(예: `tf.keras.optimizers.Ftrl.l1_regularization_strength=...)`)는 손실됩니다. 따라서 SavedModel의 소비자에게 적절한 고지를 해주기 바랍니다.

<a name="tensorflow-model-garden"></a>

## TensorFlow Model Garden

[TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official) 리포지토리에는 [tfhub.dev](https://tfhub.dev/)에 업로드할 재사용 가능한 TF2 저장된 모델을 만드는 많은 예제가 포함되어 있습니다.

## 커뮤니티 요청

TensorFlow 허브 팀은 tfhub.dev에서 사용할 수 있는 자산의 극히 일부만 생성합니다. 주로 Google 및 Deepmind의 연구원, 기업 및 학술 연구 기관, ML 애호가에게 모델 제작을 의존하고 있습니다. 결과적으로 특정 자산에 대한 커뮤니티 요청을 이행할 수 있다고 보장할 수 없으며 새로운 자산을 언제 제공할 수 있을 지에 대한 확답도 해줄 수 없습니다.

아래의 [커뮤니티 모델 요청 이정표](https://github.com/tensorflow/hub/milestone/1)에는 특정 자산에 대한 커뮤니티의 요청이 포함되어 있습니다. 여러분 또는 알고 있는 다른 사람이 자산을 제작하고 tfhub.dev에서 공유하는 데 관심이 있다면 제출을 환영합니다!

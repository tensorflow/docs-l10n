<!--* freshness: { owner: 'maringeo' reviewed: '2022-01-12' } *-->

# TensorFlow Hub를 사용하여 TF1에서 TF2로 마이그레이션하기

이 페이지에서는 TensorFlow 코드를 TensorFlow 1에서 TensorFlow 2로 마이그레이션하는 동안 TensorFlow Hub를 계속 사용하는 방법을 설명합니다. TensorFlow의 일반 [마이그레이션 가이드](https://www.tensorflow.org/guide/migrate)를 보완합니다.

TF2의 경우, TF Hub는 `tf.contrib.v1.layers`처럼 `tf.compat.v1.Graph`를 빌드하기 위해 레거시 `hub.Module` API에서 전환했습니다. 대신 `tf.keras.Model`(일반적으로 TF2의 새로운 [즉시 실행 환경](https://www.tensorflow.org/guide/eager_)) 및 하위 수준 TensorFlow 코드에 대한 기본 `hub.load()` 메서드를 빌드하기 위해 다른 Keras 레이어와 함께 사용할 수 있는 `hub.KerasLayer`가 있습니다.

`hub.Module` API는 TF1 및 TF2의 TF1 호환성 모드에서 사용할 수 있도록 `tensorflow_hub` 라이브러리에서 계속 사용할 수 있습니다. [TF1 Hub 형식](tf1_hub_module.md)의 모델만 로드할 수 있습니다.

새로운 `hub.load()` 및 `hub.KerasLayer` API는 TensorFlow 1.15(즉시 및 그래프 모드) 및 TensorFlow 2에서 동작합니다. 이 새로운 API는 새로운 [TF2 SavedModel](tf2_saved_model.md) 자산을 로드할 수 있으며, TF1 Hub 형식의 레거시 모델을 [모델 호환성 가이드](model_compatibility.md)에 명시된 제한 사항에 따라 로드할 수 있습니다.

일반적으로, 가능한 한 새로운 API를 사용하는 것이 좋습니다.

## 새 API의 요약

`hub.load()`는 TensorFlow Hub (또는 호환되는 서비스)에서 SavedModel을 로드하는 하위 수준의 새로운 함수입니다. TF2의 `tf.saved_model.load()`를 래핑합니다. TensorFlow의 [SavedModel 가이드](https://www.tensorflow.org/guide/saved_model)는 그 결과로 무엇을 할 수 있는지를 설명합니다.

```python
m = hub.load(handle)
outputs = m(inputs)
```

`hub.KerasLayer` 클래스는 `hub.load()`를 호출하고 다른 Keras 레이어와 함께 Keras에서 사용할 수 있도록 결과를 조정합니다. (다른 방식으로 사용할 수 있는 로드된 SavedModel에 대한 편리한 래퍼가 될 수도 있습니다.)

```python
model = tf.keras.Sequential([
    hub.KerasLayer(handle),
    ...])
```

많은 튜토리얼에서 이러한 API가 실제로 동작하는 것을 보여줍니다. 특히, 다음을 참조하세요.

- [텍스트 분류 예제 노트북](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)
- [이미지 분류 예제 노트북](https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)

### Estimator 훈련에서 새 API 사용하기

매개변수 서버(또는 원격 기기에 배치된 변수가 있는 TF1 세션에서)로 훈련하기 위해 Estimator에서 TF2 SavedModel을 사용하는 경우, tf.Session의 ConfigProto에서 `experimental.share_cluster_devices_in_session`를 설정해야 합니다. 그렇지 않으면 "Assigned device '/job:ps/replica:0/task:0/device:CPU:0' does not match any device"와 같은 오류가 발생합니다.

필요한 옵션은 다음과 같이 설정할 수 있습니다.

```python
session_config = tf.compat.v1.ConfigProto()
session_config.experimental.share_cluster_devices_in_session = True
run_config = tf.estimator.RunConfig(..., session_config=session_config)
estimator = tf.estimator.Estimator(..., config=run_config)
```

TF2.2부터 이 옵션은 더 이상 실험용이 아니며, `.experimental` 조각을 삭제할 수 있습니다.

## TF1 Hub 형식으로 레거시 모델 로드하기

새 TF2 SavedModel을 아직 사용 사례에 사용할 수 없으며 레거시 모델을 TF1 Hub 형식으로 로드해야 할 수 있습니다. `tensorflow_hub` 릴리스 0.7부터는 아래와 같이 `hub.KerasLayer`와 함께 TF1 Hub 형식의 레거시 모델을 사용할 수 있습니다.

```python
m = hub.KerasLayer(handle)
tensor_out = m(tensor_in)
```

또한, `KerasLayer`는 TF1 Hub 형식 및 레거시 SavedModel에서 레거시 모델의 보다 구체적인 사용을 위해 `tags`, `signature`, `output_key` 및 `signature_outputs_as_dict`를 지정하는 기능을 제공합니다.

TF1 Hub 형식 호환성에 대한 자세한 내용은 [모델 호환성 가이드](model_compatibility.md)를 참조하세요.

## 하위 수준 API 사용하기

레거시 TF1 Hub 형식 모델은 `tf.saved_model.load`를 통해 로드할 수 있습니다. 다음 코드 대신에

```python
# DEPRECATED: TensorFlow 1
m = hub.Module(handle, tags={"foo", "bar"})
tensors_out_dict = m(dict(x1=..., x2=...), signature="sig", as_dict=True)
```

다음 코드를 사용하는 것이 좋습니다.

```python
# TensorFlow 2
m = hub.load(path, tags={"foo", "bar"})
tensors_out_dict = m.signatures["sig"](x1=..., x2=...)
```

이 예에서 `m.signatures`는 서명 이름으로 키가 지정된 TensorFlow [concrete 함수](https://www.tensorflow.org/tutorials/customization/performance#tracing)의 사전입니다. 이러한 함수를 호출하면 사용되지 않더라도 모든 출력이 계산됩니다. (이는 TF1 그래프 모드의 지연 평가와 다릅니다.)

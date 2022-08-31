# TensorFlow 모델 변환

이 페이지에서는 TensorFlow Lite 변환기를 사용하여 TensorFlow 모델을 TensorFlow Lite 모델(<code>.tflite</code> 파일 확장자로 식별되는 최적화된 <a>FlatBuffer</a> 형식)로 변환하는 방법을 설명합니다.

참고: 이 가이드에서는 [TensorFlow 2.x를 설치](https://www.tensorflow.org/install/pip#tensorflow-2-packages-are-available)하고 TensorFlow 2.x에서 모델을 학습했다고 가정합니다. 모델이 TensorFlow 1.x에서 학습된 경우 [TensorFlow 2.x로 마이그레이션](https://www.tensorflow.org/guide/migrate/tflite)을 고려하세요. 설치된 TensorFlow 버전을 확인하려면 `print(tf.__version__)`을 실행하세요.

## 변환 워크플로

아래 다이어그램은 모델 변환을 위한 고차원적 워크플로를 보여줍니다.

![TFLite converter workflow](../../images/convert/convert.png)

**그림 1.** 변환기 워크플로.

다음 옵션 중 하나를 사용하여 모델을 변환할 수 있습니다.

1. [Python API](#python_api)(***권장***): 이를 통해 변환을 개발 파이프라인에 통합하고, 최적화를 적용하고, 메타데이터를 추가하고, 변환 프로세스를 단순화하는 기타 여러 작업을 수행할 수 있습니다.
2. [명령줄](#cmdline): 기본 모델 변환만 지원합니다.

참고: 모델 변환 중에 문제가 발생하는 경우 [GitHub 이슈](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md)를 생성하세요.

## Python API <a name="python_api"></a>

*도우미 코드: TensorFlow Lite 변환기 API에 대해 자세히 알아보려면 `print(help(tf.lite.TFLiteConverter))`를 실행하세요.*

[`tf.lite.TFLiteConverter`](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)를 사용하여 TensorFlow 모델을 변환합니다. TensorFlow 모델은 SavedModel 형식을 사용하여 저장되며 상위 수준 `tf.keras.*` API(Keras 모델) 또는 하위 수준 `tf.*` API(구체적인 함수 생성)를 사용하여 생성됩니다. 결과적으로 다음 세 가지 옵션이 있습니다(다음 몇 개 섹션에서 예제 소개).

- `tf.lite.TFLiteConverter.from_saved_model()`(**권장**): [SavedModel](https://www.tensorflow.org/guide/saved_model)을 변환합니다.
- `tf.lite.TFLiteConverter.from_keras_model()`: [Keras](https://www.tensorflow.org/guide/keras/overview) 모델을 변환합니다.
- `tf.lite.TFLiteConverter.from_concrete_functions()`: [구체적인 함수](https://www.tensorflow.org/guide/intro_to_graphs)를 변환합니다.

### SavedModel 변환(권장)<a name="saved_model"></a>

다음 예는 [SavedModel](https://www.tensorflow.org/guide/saved_model)을 TensorFlow Lite 모델로 변환하는 방법을 보여줍니다.

```python
import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### Keras 모델 변환<a name="keras"></a>

다음 예는 [Keras](https://www.tensorflow.org/guide/keras/overview) 모델을 TensorFlow Lite 모델로 변환하는 방법을 보여줍니다.

```python
import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### 구체적인 함수 변환<a name="concrete_function"></a>

다음 예제는 [구체적인 함수](https://www.tensorflow.org/guide/intro_to_graphs)를 TensorFlow Lite 모델로 변환하는 방법을 보여줍니다.

```python
import tensorflow as tf

# Create a model using low-level tf.* APIs
class Squared(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
  def __call__(self, x):
    return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()

# Convert the model.

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

### 다른 기능

- [최적화](../../performance/model_optimization.md)를 적용합니다. 일반적으로 사용되는 최적화는 [훈련 후 양자화](../../performance/post_training_quantization.md)로, 정확도 손실을 최소화하면서 모델 지연 시간과 크기를 더욱 줄일 수 있습니다.

- [메타데이터](metadata.md)를 추가하여 기기에 모델을 배포할 때 플랫폼별 래퍼 코드를 더 쉽게 생성할 수 있습니다.

### 변환 오류

다음은 일반적인 변환 오류 및 해결 방법입니다.

- 오류: `Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select. TF Select ops: ..., .., ...`

    해결 방법: 모델에 해당 TFLite 구현이 없는 TF op가 있기 때문에 이 오류가 발생합니다. [TFLite 모델에서 TF op를 사용하여](../../guide/ops_select.md) 이 문제를 해결할 수 있습니다(권장). TFLite op만 있는 모델을 생성하려면 [Github 이슈 #21526](https://github.com/tensorflow/tensorflow/issues/21526) 에서 누락된 TFLite op에 대한 요청을 추가하거나(요청이 아직 언급되지 않은 경우 의견을 남겨주세요) 직접 [TFLite op를 생성](../../guide/ops_custom#create_and_register_the_operator)할 수 있습니다.

- 오류: `.. is neither a custom op nor a flex op`

    해결 방법: 이 TF op가 다음과 같은 경우:

    - TF에서 지원됨: TF op가 [허용 목록](../../guide/op_select_allowlist.md)(TFLite에서 지원하는 TF op 전체 목록)에서 누락되어 이 오류가 발생합니다. 다음과 같이 해결할 수 있습니다.

        1. [허용 목록에 누락된 op를 추가합니다](../../guide/op_select_allowlist.md#add_tensorflow_core_operators_to_the_allowed_list).
        2. [TF 모델을 TFLite 모델로 변환하고 추론을 실행합니다](../../guide/ops_select.md).

    - TF에서 지원되지 않음: TFLite가 사용자가 정의한 사용자 지정 TF op를 인식하지 못하기 때문에 이 오류가 발생합니다. 다음과 같이 해결할 수 있습니다.

        1. [TF op를 만듭니다](https://www.tensorflow.org/guide/create_op).
        2. [TF 모델을 TFLite 모델로 변환합니다](../../guide/op_select_allowlist.md#users_defined_operators).
        3. [TFLite op](../../guide/ops_custom.md#create_and_register_the_operator)를 만들고 TFLite 런타임에 연결하여 추론을 실행합니다.

## 명령줄 도구<a name="cmdline"></a>

**참고:** 가능하면 위에 나열된 [Python API](#python_api)를 대신 사용하는 것이 좋습니다.

[pip에서 TensorFlow 2.x를 설치](https://www.tensorflow.org/install/pip)했다면 `tflite_convert` 명령을 사용하세요. 사용 가능한 모든 플래그를 보려면 다음 명령을 사용합니다.

```sh
$ tflite_convert --help

`--output_file`. Type: string. Full path of the output file.
`--saved_model_dir`. Type: string. Full path to the SavedModel directory.
`--keras_model_file`. Type: string. Full path to the Keras H5 model file.
`--enable_v1_converter`. Type: bool. (default False) Enables the converter and flags used in TF 1.x instead of TF 2.x.

You are required to provide the `--output_file` flag and either the `--saved_model_dir` or `--keras_model_file` flag.
```

[TensorFlow 2.x 소스](https://www.tensorflow.org/install/source)가 다운로드되어 있고 패키지를 빌드 및 설치하지 않고 해당 소스에서 변환기를 실행하려는 경우 명령에서 '`tflite_convert`'를 '`bazel run tensorflow/lite/python:tflite_convert --`'로 바꿀 수 있습니다.

### SavedModel 변환<a name="cmdline_saved_model"></a>

```sh
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

### Keras H5 모델 변환<a name="cmdline_keras_model"></a>

```sh
tflite_convert \
  --keras_model_file=/tmp/mobilenet_keras_model.h5 \
  --output_file=/tmp/mobilenet.tflite
```

## 다음 단계

[TensorFlow Lite 인터프리터](../../guide/inference.md)를 사용하여 클라이언트 장치(예: 모바일, 임베디드)에서 추론을 실행합니다.

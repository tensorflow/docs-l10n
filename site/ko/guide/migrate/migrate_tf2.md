# TF1.x -&gt; TF2 마이그레이션 개요

TensorFlow 2는 여러 면에서 TF1.x와 근본적으로 다릅니다. 다음과 같이 TF2 바이너리 설치에 대해 수정되지 않은 TF1.x 코드( [contrib 제외)를 계속 실행할 수 있습니다.](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

그러나 이것은 *TF2 동작 및 API를 실행하지 않으며* TF2용으로 작성된 코드에서 예상대로 작동하지 않을 수 있습니다. TF2 동작이 활성화된 상태로 실행하지 않는 경우 TF2 설치 위에 TF1.x를 효과적으로 실행하고 있는 것입니다. TF2가 TF1.x와 어떻게 다른지에 대한 자세한 내용은 [TF1 대 TF2 동작 가이드를 읽어보세요.](./tf1_vs_tf2.ipynb)

이 가이드는 TF1.x 코드를 TF2로 마이그레이션하는 프로세스에 대한 개요를 제공합니다. 이를 통해 새로운 기능과 향후 개선된 기능을 활용할 수 있으며 코드를 더 간단하고 성능을 높이고 유지 관리하기 쉽게 만들 수 있습니다.

`tf.keras` 의 고급 API를 사용하고 `model.fit` 으로만 훈련하는 경우 다음 주의 사항을 제외하고 코드가 TF2와 거의 완전히 호환되어야 합니다.

- TF2에는 Keras 옵티마이저에 대한 [새로운 기본 학습률이 있습니다.](../../guide/effective_tf2.ipynb#optimizer_defaults)
- TF2가 메트릭이 기록되는 "이름"을 [변경했을 수 있습니다.](../../guide/effective_tf2.ipynb#keras_metric_names)

## TF2 마이그레이션 프로세스

[마이그레이션하기 전에 가이드](./tf1_vs_tf2.ipynb) 를 읽고 TF1.x와 TF2 간의 동작 및 API 차이점에 대해 알아보세요.

1. 자동화된 스크립트를 실행하여 일부 TF1.x API 사용을 `tf.compat.v1` 로 변환합니다.
2. `tf.contrib` 기호를 제거합니다 [(TF Addons](https://github.com/tensorflow/addons) 및 [TF-Slim 확인](https://github.com/google-research/tf-slim) ).
3. TF1.x 모델 포워드 패스를 즉시 실행이 활성화된 TF2에서 실행합니다.
4. 훈련 루프 및 모델 저장/로드를 위한 TF1.x 코드를 TF2에 해당하는 코드로 업그레이드하십시오.
5. (선택 사항) TF2 호환 `tf.compat.v1` API를 관용적인 TF2 API로 마이그레이션합니다.

다음 섹션에서는 위에서 설명한 단계를 확장합니다.

## 기호 변환 스크립트 실행

이렇게 하면 TF 2.x 바이너리에 대해 실행하도록 코드 기호를 다시 작성할 때 초기 단계가 실행되지만 코드가 TF 2.x에 관용적이게 만들거나 코드가 TF2 동작과 자동으로 호환되도록 만들지 않습니다.

코드는 여전히 `tf.compat.v1` 끝점을 사용하여 자리 표시자, 세션, 컬렉션 및 기타 TF1.x 스타일 기능에 액세스할 가능성이 높습니다.

기호 변환 스크립트 사용에 대한 모범 사례에 대해 [자세히 알아보려면 가이드](./upgrade.ipynb) 를 읽으십시오.

## `tf.contrib` 사용 제거

`tf.contrib` 모듈은 중단되었으며 여러 하위 모듈이 핵심 TF2 API에 통합되었습니다. [다른 하위 모듈은 이제 TF IO](https://github.com/tensorflow/io) 및 [TF Addons](https://www.tensorflow.org/addons/overview) 와 같은 다른 프로젝트로 분리되었습니다.

오래된 TF1.x 코드의 많은 양이 사용하는 [슬림](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) 으로 TF1.x와 함께 제공 라이브러리, `tf.contrib.layers` . Slim 코드를 TF2로 마이그레이션할 때 Slim API 사용이 [tf-slim pip 패키지](https://pypi.org/project/tf-slim/) 를 가리키도록 전환합니다. 그런 다음 [모델 매핑 가이드](https://tensorflow.org/guide/migrate/model_mapping#a_note_on_slim_and_contriblayers) 를 읽고 Slim 코드를 변환하는 방법을 알아보세요.

또는 Slim 사전 훈련된 모델을 사용하는 경우 원래 Slim 코드에서 내보낸 `tf.keras.applications` 또는 [TF Hub](https://tfhub.dev/s?tf-version=tf2&q=slim) 의 TF2 `SavedModel`

## TF2 동작이 활성화된 상태에서 TF1.x 모델 전달 패스를 실행합니다.

### 변수 및 손실 추적

[TF2는 글로벌 컬렉션을 지원하지 않습니다.](./tf1_vs_tf2.ipynb#no_more_globals)

TF2에서 Eager 실행은 `tf.Graph` 컬렉션 기반 API를 지원하지 않습니다. 이는 변수를 구성하고 추적하는 방법에 영향을 줍니다.

새로운 TF2 코드에 대해 다음을 사용 `tf.Variable` 대신 `v1.get_variable` 파이썬 수집하고 대신 변수를 추적하는 객체를 사용 `tf.compat.v1.variable_scope` . 일반적으로 다음 중 하나입니다.

- `tf.keras.layers.Layer`
- `tf.keras.Model`
- `tf.Module`

`Layer` , `Module` 또는 `Model` `.variables` 및 `.trainable_variables` 속성을 사용하여 변수 목록(예: `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)` )을 집계합니다.

`Layer` 및 `Model` 클래스는 전역 컬렉션의 필요성을 제거하는 몇 가지 다른 속성을 구현합니다. `.losses` `tf.GraphKeys.LOSSES` 컬렉션 사용을 대체할 수 있습니다.

[모델 매핑 가이드](./model_mapping.ipynb) 를 읽고 TF2 코드 모델링 shim을 사용하여 기존 `get_variable` 및 `variable_scope` 기반 코드를 `Layers` , `Models` 및 `Modules` 내부에 포함하는 방법에 대해 자세히 알아보세요. 이렇게 하면 주요 재작성 없이 즉시 실행이 활성화된 전달 전달을 실행할 수 있습니다.

### 다른 행동 변화에 적응하기

[모델 매핑 가이드](./model_mapping.ipynb) 자체가 더 자세한 다른 동작 변경을 실행하는 모델 전달을 얻기에 충분하지 않은 경우 [TF1.x 대 TF2 동작](./tf1_vs_tf2.ipynb) 에 대한 가이드를 참조하여 다른 동작 변경과 이에 적응할 수 있는 방법에 대해 알아보세요. . 또한 자세한 내용은 [하위 분류 가이드를 통해 새 레이어 및 모델 만들기를 확인하세요.](https://tensorflow.org/guide/keras/custom_layers_and_models.ipynb)

### 결과 확인

빠른 실행이 활성화되었을 때 모델이 올바르게 작동하는지 (숫자적으로) 검증하는 방법에 대한 쉬운 도구와 지침 [은 모델 검증 가이드](./validate_correctness.ipynb) 를 참조하십시오. [모델 매핑 가이드](./model_mapping.ipynb) 와 함께 사용하면 특히 유용할 수 있습니다.

## 교육, 평가 및 가져오기/내보내기 코드 업그레이드

로 구축 TF1.x 교육 루프 `v1.Session` 스타일 `tf.estimator.Estimator` 의 다른 컬렉션을 기반 접근 방식은 TF2의 새로운 행동와 호환되지 않습니다. TF2 코드와 결합하면 예기치 않은 동작이 발생할 수 있으므로 모든 TF1.x 교육 코드를 마이그레이션하는 것이 중요합니다.

이를 위해 여러 전략 중에서 선택할 수 있습니다.

최고 수준의 접근 방식은 `tf.keras` 를 사용하는 것입니다. Keras의 고수준 기능은 자체 훈련 루프를 작성할 경우 놓치기 쉬운 많은 저수준 세부 정보를 관리합니다. 예를 들어, 자동으로 정규화 손실을 수집하고 모델을 호출할 때 `training=True`

참고하여주십시오 [견적 마이그레이션 가이드](./migrating_estimator.ipynb) 마이그레이션하는 방법을 배울 수 `tf.estimator.Estimator` 사용에의 코드를 [바닐라](./migrating_estimator.ipynb#tf2_keras_training_api) 및 [사용자 정의](./migrating_estimator.ipynb#tf2_keras_training_api_with_custom_training_step) `tf.keras` 교육 루프.

사용자 지정 훈련 루프를 사용하면 개별 레이어의 가중치 추적과 같이 모델을 더 세밀하게 제어할 수 있습니다. `tf.GradientTape` 를 사용하여 모델 가중치를 검색하고 이를 사용하여 모델을 업데이트하는 방법을 배우려면 [처음부터 학습 루프 구축](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) 에 대한 가이드를 읽어보세요.

### TF1.x 옵티마이저를 Keras 옵티마이저로 변환

[Adam 옵티마이저](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer) 및 [경사 하강 옵티](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer) 마이저와 같은 `tf.compat.v1.train` 의 옵티마이저는 tf.keras.optimizers에 `tf.keras.optimizers` 합니다.

아래 표에는 이러한 레거시 옵티마이저를 Keras에 상응하는 것으로 변환하는 방법이 요약되어 있습니다. [추가 단계(예: 기본 학습률 업데이트](../../guide/effective_tf2.ipynb#optimizer_defaults) )가 필요하지 않는 한 TF1.x 버전을 TF2 버전으로 직접 교체할 수 있습니다.

최적화 프로그램을 변환 [하면 오래된 체크포인트가 호환되지 않을 수 있습니다](./migrating_checkpoints.ipynb) .

<table>
  <tr>
    <th>TF1.x</th>
    <th>TF2</th>
    <th>추가 단계</th>
  </tr>
  <tr>
    <td>`tf.v1.train.GradientDescentOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>없음</td>
  </tr>
  <tr>
    <td>`tf.v1.train.MomentumOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>'모멘텀' 인수 포함</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdamOptimizer`</td>
    <td>`tf.keras.optimizers.Adam`</td>
    <td>`beta1` 및 `beta2` 인수의 이름을 `beta_1` 및 `beta_2`로 변경</td>
  </tr>
  <tr>
    <td>`tf.v1.train.RMSPropOptimizer`</td>
    <td>`tf.keras.optimizers.RMSprop`</td>
    <td>`decay` 인수의 이름을 `rho`로 바꿉니다.</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdadeltaOptimizer`</td>
    <td>`tf.keras.optimizers.Adadelta`</td>
    <td>없음</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdagradOptimizer`</td>
    <td>`tf.keras.optimizers.Adagrad`</td>
    <td>없음</td>
  </tr>
  <tr>
    <td>`tf.v1.train.FtrlOptimizer`</td>
    <td>`tf.keras.optimizers.Ftrl`</td>
    <td>`accum_name` 및 `linear_name` 인수 제거</td>
  </tr>
  <tr>
    <td>`tf.contrib.AdamaxOptimizer`</td>
    <td>`tf.keras.optimizers.Adamax`</td>
    <td>`beta1` 및 `beta2` 인수의 이름을 `beta_1` 및 `beta_2`로 바꿉니다.</td>
  </tr>
  <tr>
    <td>`tf.contrib.나담`</td>
    <td>`tf.keras.optimizers.Nadam`</td>
    <td>`beta1` 및 `beta2` 인수의 이름을 `beta_1` 및 `beta_2`로 바꿉니다.</td>
  </tr>
</table>

참고: TF2에서 모든 엡실론(숫자 안정성 상수)은 이제 `1e-8` 대신 `1e-7` 기본 설정됩니다. 이 차이는 대부분의 사용 사례에서 무시할 수 있습니다.

### 데이터 입력 파이프라인 업그레이드

`tf.keras` 모델에 데이터를 공급하는 방법에는 여러 가지가 있습니다. Python 생성기와 Numpy 배열을 입력으로 받아들입니다.

모델에 데이터를 제공하는 권장 방법은 데이터 조작을 위한 고성능 클래스 모음이 포함된 `tf.data` `tf.data` 에 속한 `dataset` 는 효율적이고 표현력이 풍부하며 TF2와 잘 통합됩니다.

`tf.keras.Model.fit` 메소드에 직접 전달할 수 있습니다.

```python
model.fit(dataset, epochs=5)
```

직접 표준 Python을 통해 반복할 수 있습니다.

```python
for example_batch, label_batch in dataset:
    break
```

여전히 `tf.queue` 를 사용하는 경우 이제 입력 파이프라인이 아닌 데이터 구조로만 지원됩니다.

`tf.feature_columns` 를 사용하는 모든 기능 전처리 코드를 마이그레이션해야 합니다. 자세한 내용은 [마이그레이션 가이드를 읽어보세요.](./migrating_feature_columns.ipynb)

### 모델 저장 및 로드

TF2는 객체 기반 체크포인트를 사용합니다. 이름 기반 TF1.x 체크포인트에서 마이그레이션하는 방법에 대해 자세히 알아보려면 [체크포인트 마이그레이션 가이드를 읽어보세요.](./migrating_checkpoints.ipynb) 또한 핵심 TensorFlow 문서에서 [체크포인트 가이드를 읽어보세요.](https://www.tensorflow.org/guide/checkpoint)

저장된 모델에 대한 심각한 호환성 문제는 없습니다. `SavedModel` 의 SavedModel 을 TF2로 마이그레이션하는 방법에 대한 자세한 내용은 <a href="./saved_model.ipynb" data-md-type="link">`SavedModel` 가이드</a> 를 읽어보세요. 일반적으로,

- TF1.x stored_models는 TF2에서 작동합니다.
- TF2 stored_models는 모든 작업이 지원되는 경우 TF1.x에서 작동합니다.

`Graph.pb` 및 `Graph.pbtxt` 개체 작업에 대한 자세한 내용은 `SavedModel` [`GraphDef` 섹션](./saved_model.ipynb#graphdef_and_metagraphdef) 을 참조하십시오.

## (선택 사항) `tf.compat.v1` 기호에서 마이그레이션

`tf.compat.v1` 모듈에는 완전한 TF1.x API와 원래 의미가 포함되어 있습니다.

위의 단계를 수행하고 모든 TF2 동작과 완전히 호환되는 코드로 끝난 후에도 TF2와 호환되는 `compat.v1` API에 대한 언급이 많이 있을 수 있습니다. 이미 작성된 코드에서는 계속 작동하지만 새로 작성하는 코드에는 `compat.v1` API를 사용하지 않아야 합니다.

그러나 기존 사용을 레거시가 아닌 TF2 API로 마이그레이션하도록 선택할 수 있습니다. 개별 `compat.v1` 은 종종 레거시가 아닌 TF2 API로 마이그레이션하는 방법을 설명합니다. 또한 [관용적 TF2 API로의 증분 마이그레이션에](./model_mapping.ipynb#incremental_migration_to_native_tf2) 대한 모델 매핑 가이드 섹션도 이에 도움이 될 수 있습니다.

## 리소스 및 추가 읽을거리

이전에 언급했듯이 모든 TF1.x 코드를 TF2로 마이그레이션하는 것이 좋습니다. 자세히 [알아보려면 TensorFlow 가이드의 TF2로 마이그레이션 섹션](https://tensorflow.org/guide/migrate) 의 가이드를 읽어보세요.

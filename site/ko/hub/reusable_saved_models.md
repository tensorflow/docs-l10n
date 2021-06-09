<!--* freshness: { owner: 'kempy' reviewed: '2021-03-09' } *-->

# 재사용 가능한 SavedModel

## 시작하기

TensorFlow Hub는 다른 자산 중에서 TensorFlow 2용 SavedModel을 호스팅합니다. `obj = hub.load(url)`를 사용하여 Python 프로그램에 다시 로드할 수 있습니다[[자세히 알아보기](tf2_saved_model)]. 반환된 `obj`는 `tf.saved_model.load()`의 결과입니다(TensorFlow의 [SavedModel 가이드](https://www.tensorflow.org/guide/saved_model) 참조). 이 객체는 tf.functions, tf.Variables(사전 훈련된 값에서 초기화됨), 기타 리소스 및 반복적으로 더 많은 객체가 될 수 있는 임의의 속성을 가질 수 있습니다.

이 페이지는 TensorFlow Python 프로그램에서 *재사용*하기 위해 로드된 `obj`에 의해 구현되는 인터페이스를 설명합니다. 이 인터페이스를 준수하는 SavedModel을 *Reusable SavedModel*이라고 합니다.

재사용은 미세 조정 기능을 포함하여 `obj` 중심의 더 큰 모델을 빌드하는 것을 의미합니다. 미세 조정은 주변 모델의 일부로 로드된 `obj`의 가중치를 추가로 훈련하는 것을 의미합니다. 손실 함수와 옵티마이저는 주변 모델에 의해 결정됩니다. `obj`는 출력 활성화에 대한 입력 매핑("포워드 패스")만 정의하며 드롭아웃 또는 배치 정규화와 같은 기술을 포함할 수 있습니다.

**TensorFlow Hub 팀은 위의 의미에서 재사용할 예정인 모든 SavedModel에서 Reusable SavedModel 인터페이스**를 구현할 것을 권장합니다. `tensorflow_hub` 라이브러리의 많은 유틸리티, 특히 `hub.KerasLayer`는 이를 구현하기 위해 SavedModel가 필요합니다.

### SignatureDefs와의 관계

tf.functions 및 기타 TF2 기능 측면에서 이 인터페이스는 TF1 이후로 사용 가능하며 추론을 위해 TF2에서 계속 사용되는 SavedModel의 서명과는 별개입니다(예: TF Serving 또는 TF Lite에 SavedModel 배포). 추론을 위한 서명은 미세 조정을 지원할 만큼 충분히 다양하지 않고, [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)은 재사용된 모델에 대해 보다 자연스럽고 다양한 [Python API](https://www.tensorflow.org/tutorials/customization/performance)를 제공합니다.

### 모델 구축 라이브러리와의 관계

Reusable SavedModel 모델은 Keras 또는 Sonnet과 같은 특정 모델 구축 라이브러리와 관계없이 TensorFlow 2 프리미티브만 사용합니다. 이를 통해 원래 모델 구축 코드에 대한 종속성이 없는 모델 구축 라이브러리에서 재사용이 용이합니다.

Reusable SavedModel을 주어진 모델 구축 라이브러리에 로드하거나 저장하려면 약간의 조정이 필요합니다. Keras의 경우, [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer)가 로딩을 제공하며, 이 인터페이스의 상위 세트를 제공하기 위해 SavedModel에 저장하는 Keras의 내장 형식이 TF2용으로 재설계되었습니다(2019년 5월부터 [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md) 참조).

### 작업별 "Common SavedModel API"와의 관계

이 페이지의 인터페이스 정의는 모든 수와 유형의 입력 및 출력을 허용합니다. [TF 허브용 Common SavedModel API](common_saved_model_apis/index.md)는 모델을 쉽게 교환할 수 있도록 특정 작업에 대한 사용 규칙으로 이 일반 인터페이스를 구체화합니다.

## 인터페이스 정의

### 속성

Reusable SavedModel은 `obj = tf.saved_model.load(...)`에서 다음 속성을 가진 객체를 반환하는 TensorFlow 2 SavedModel입니다.

- `__call__`: 필수입니다. 아래 사양에 따라 모델의 계산("순방향 전달")을 구현하는 tf.function.

- `variables`: tf.Variable 객체의 목록으로, 훈련 가능 및 훈련 불가능을 포함하여 가능한 모든 `__call__` 호출에 사용되는 모든 변수를 나열합니다.

    이 목록은 비어 있는 경우 생략할 수 있습니다.

    참고: 편리하게도, 이 이름은 TF1 SavedModel을 로드하여 `GLOBAL_VARIABLES` 모음을 나타낼 때 `tf.saved_model.load(...)`에 의해 합성된 속성과 일치합니다.

- `trainable_variables`: 모든 요소에 대해 `v.trainable`이 true인 tf.Variable 객체의 목록입니다. 이들 변수는 `variables`의 하위 집합이어야 합니다. 객체를 미세 조정할 때 학습할 변수입니다. SavedModel 생성자는 미세 조정 중에 수정해서는 안 된다는 것을 나타내기 위해 원래 훈련 가능한 일부 변수를 여기에서 생략할 수 있습니다.

    이 목록은 비어 있는 경우, 특히 SavedModel에서 미세 조정을 지원하지 않는 경우 생략할 수 있습니다.

- `regularization_losses`: 각각 0개 입력을 받아 단일 스칼라 부동 텐서를 반환하는 tf.functions의 목록입니다. 미세 조정을 위해 SavedModel 사용자는 이들 tf.functions를 추가 정규화 조건으로 손실에 포함하는 것이 좋습니다(가장 간단한 경우, 추가 확장 없이). 일반적으로, 가중치 regularizer를 나타내는 데 사용됩니다. (입력 부족으로 인해 이들 tf.functions는 activity regularizer를 표현할 수 없습니다.)

    이 목록은 비어 있는 경우, 특히 SavedModel에서 미세 조정을 지원하지 않거나 가중치 정규화를 규정하지 않으려는 경우 생략할 수 있습니다.

### `__call__` 함수

Restored SavedModel `obj`에는 복원된 tf.function인 `obj.__call__` 속성이 있으며, 다음과 같이 `obj`를 호출할 수 있습니다.

시놉시스(의사 코드):

```python
outputs = obj(inputs, trainable=..., **kwargs)
```

#### 인수

인수는 다음과 같습니다.

- SavedModel의 입력 활성화 배치와 함께 하나의 위치 필수 인수가 있습니다. 그 유형은 다음 중 하나입니다.

    - 단일 입력을 위한 단일 Tensor
    - 명명되지 않은 입력의 순서가 지정된 시퀀스에 대한 Tensor 목록
    - 특정 입력 이름 세트로 키가 지정된 텐서 dict

    (이 인터페이스의 향후 개정판에서는 보다 일반적인 중첩이 허용될 수 있습니다.) SavedModel 생성자는 이중 하나와 텐서 형상 및 dtype을 선택합니다. 유용한 경우, 형상의 일부 차원은 정의되지 않아야 합니다(특히 배치 크기).

- Python boolean, `True` 또는 `False`를 허용하는 선택적 키워드 인수 `training`이 있을 수 있습니다. 기본값은 `False`입니다. 모델이 미세 조정을 지원하고 계산이 둘 사이(예: 드롭아웃 및 배치 정규화에서와 같이)에서 다른 경우, 이 인수로 구분이 구현됩니다. 그렇지 않으면, 이 인수는 없을 수 있습니다.

    `__call__`이 Tensor 값 `training` 인수를 허용할 필요는 없습니다. 디스패치를 위해 필요한 경우, `tf.cond()`를 사용하는 것은 호출자의 선택입니다.

- SavedModel 생성자는 특정 이름의 더 많은 선택적 `kwargs`를 허용하도록 선택할 수 있습니다.

    - Tensor 값 인수의 경우, SavedModel 생성자는 허용 가능한 dtype 및 형상을 정의합니다. `tf.function`은 tf.TensorSpec 입력으로 추적되는 인수에 대한 Python 기본값을 허용합니다. 이러한 인수를 사용하여 `__call__`에 포함된 숫자 하이퍼 매개변수(예: 드롭아웃 비율)를 사용자 정의할 수 있습니다.

    - Python 값 인수의 경우, SavedModel 생성자가 허용 가능한 값을 정의합니다. 이러한 인수는 추적된 함수에서 개별 선택을 하기 위한 플래그로 사용할 수 있습니다(그러나 추적의 조합 확산에 유의하세요).

복원된 `__call__` 함수는 허용되는 모든 인수 조합에 대한 추적을 제공해야 합니다. `True`과 `False` 간에 `training`을 뒤집는다고 해서 인수의 허용성이 변경되어서는 안 됩니다.

#### 결과

`obj` 호출의 `outputs`은 다음과 같을 수 있습니다.

- 단일 출력을 위한 단일 Tensor
- 이름이 지정되지 않은 출력의 순서가 지정된 시퀀스에 대한 Tensor의 목록
- 특정 출력 이름 세트로 키가 지정된 Tensor dict

(이 인터페이스의 향후 개정판에서는 보다 일반적인 중첩을 허용할 수 있습니다.) 반환 유형은 Python 값 kwarg에 따라 다를 수 있습니다. 추가 출력을 생성하는 플래그를 허용합니다. SavedModel 생성자는 출력 dtype 및 형상과 입력에 대한 종속성을 정의합니다.

### 명명된 callable

Reusable SavedModel은 명명된 하위 객체(예: `obj.foo`, `obj.bar` 등)에 넣어 위에서 설명한 방식으로 여러 모델 조각을 제공할 수 있습니다. 각 하위 객체는 `__call__` 메서드와 해당 모델 조각에 특정한 변수 등에 대한 속성을 지원합니다. 위의 예에서는 `obj.foo.__call__`, `obj.foo.variables` 등이 해당합니다.

이 인터페이스에서는 기본 tf.function를 `tf.foo`로 직접 추가하는 방법은 다루지 *않습니다.*

Reusable SavedModel의 사용자는 한 수준의 중첩만 처리해야 합니다(`obj.bar.baz`가 아닌 `obj.bar`). (이 인터페이스의 향후 개정판에서는 더 깊은 중첩을 허용할 수 있으며, 최상위 수준 객체는 자체적으로 호출 가능해야 한다는 요구 사항이 면제될 수 있습니다.)

## 맺음말

### 진행 중인 API와의 관계

이 문서는 `tf.saved_model.save()` 및 `tf.saved_model.load()`를 통한 직렬화를 통해 왕복을 유지하는 tf.function 및 tf.Variable과 같은 기본 요소로 구성된 Python 클래스의 인터페이스를 설명합니다. 그러나 인터페이스는 `tf.saved_model.save()`에 전달된 원래 객체에 이미 존재했습니다. 해당 인터페이스에 적응하면 단일 TensorFlow 프로그램 내에서 모델 빌드 API 간에 모델 조각을 교환할 수 있습니다.

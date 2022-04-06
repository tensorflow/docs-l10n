<!--* freshness: { owner: 'akhorlin' reviewed: '2021-10-15'  } *-->

# 이미지 작업을 위한 일반적인 SavedModel API

이 페이지에서는 이미지 관련 작업용 [TF2 SavedModel](../tf2_saved_model.md)에서 [Reusable SavedModel API](../reusable_saved_models.md)를 구현하는 방법을 설명합니다. (이는 현재 지원 중단된 [TF1 Hub 형식](../common_signatures/images.md)의 [이미지에 대한 일반적인 서명](../tf1_hub_module)을 대체합니다.)

<a name="feature-vector"></a>

## 이미지 특성 벡터

### 사용법 요약

**이미지 특성 벡터**는 전체 이미지를 나타내는 밀집 1-D 텐서로, 일반적으로 소비자 모델의 단순 피드 포워드 분류자에서 사용됩니다. (기존 CNN의 경우, 공간 범위가 풀링되거나 평평해진 후, 분류가 완료되기 전 병목 상태의 값입니다. 아래 [이미지 분류](#classification)를 참조하세요.)

이미지 특성 추출을 위한 Reusable SavedModel에는 이미지 배치를 특성 벡터 배치에 매핑하는 루트 객체에 대한 `__call__` 메서드가 있습니다. 다음과 같이 사용할 수 있습니다.

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
features = obj(images)   # A batch with shape [batch_size, num_features].
```

Keras에서는 다음에 해당합니다.

```python
features = hub.KerasLayer("path/to/model")(images)
```

입력은 [이미지 입력](#input)에 대한 일반적인 규칙을 따릅니다. 모델 설명서는 입력의 `height`와 `width`에 대한 허용 범위를 지정합니다.

출력은 dtype `float32` 및 형상 `[batch_size, num_features]`의 단일 텐서입니다. `batch_size`는 입력에서와 동일합니다. `num_features`는 입력 크기와 무관한 모듈별 상수입니다.

### API 세부 정보

[Reusable SavedModel API](../reusable_saved_models.md)는 또한 `obj.variables`(예: 즉시 로딩하지 않을 때 초기화를 위해)의 목록을 제공합니다.

미세 조정을 지원하는 모델은 `obj.trainable_variables` 목록을 제공합니다. 훈련 모드에서 실행하려면 `training=True`를 전달해야 할 수 있습니다(예: 드롭아웃). 일부 모델에서는 선택적 인수가 하이퍼 매개변수를 재정의할 수 있습니다(예: 드롭아웃 비율, 모델 설명서에 설명됨). 모델은 `obj.regularization_losses` 목록을 제공할 수도 있습니다. 자세한 내용은 [Reusable SavedModel API](../reusable_saved_models.md)를 참조하세요.

Keras에서는 `hub.KerasLayer`에서 처리합니다. `trainable=True`로 초기화하여 미세 조정을 활성화하고, `arguments=dict(some_hparam=some_value, ...)`를 사용합니다(hparam 재정의가 적용되는 드문 경우).

### 메모

출력 특성에 드롭아웃을 적용하거나 적용하지 않는 것은 모델 소비자에게 맡겨야 합니다. SavedModel 자체는 실제 출력에서 드롭아웃을 수행해서는 안 됩니다(다른 곳에서 내부적으로 드롭아웃을 사용하는 경우에도).

### 예

이미지 특성 벡터용 Reusable SavedModel은 다음에서 사용됩니다.

- Colab 튜토리얼 [이미지 분류자 다시 훈련하기](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb)
- 명령 줄 도구 [make_image_classifier](https://github.com/tensorflow/hub/tree/master/tensorflow_hub/tools/make_image_classifier)

<a name="classification"></a>

## 이미지 분류

### 사용법 요약

**이미지 분류**는 *모듈 게시자가 선택한 * 분류 체계의 등급(class)에서 이미지의 픽셀을 멤버십에 대한 선형 점수(logit)에 매핑합니다. 이를 통해 모델 소비자는 게시자 모듈에서 학습한 특정 분류에서 결론을 도출할 수 있습니다. (새로운 등급의 세트를 사용한 이미지 분류의 경우, 대신 새 분류자로 [이미지 특성 벡터](#feature-vector) 모델을 재사용하는 것이 일반적입니다.)

이미지 분류를 위한 Reusable SavedModel에는 이미지 배치를 로짓 배치에 매핑하는 루트 객체에 대한 `__call__` 메서드가 있습니다. 다음과 같이 사용할 수 있습니다.

```python
obj = hub.load("path/to/model")  # That's tf.saved_model.load() after download.
images = ...  # A batch of images with shape [batch_size, height, width, 3].
logits = obj(images)   # A batch with shape [batch_size, num_classes].
```

Keras에서는 다음에 해당합니다.

```python
logits = hub.KerasLayer("path/to/model")(images)
```

입력은 [이미지 입력](#input)에 대한 일반적인 규칙을 따릅니다. 모델 설명서는 입력의 `height`와 `width`에 대한 허용 범위를 지정합니다.

출력 `logits`는 dtype `float32` 및 형상 `[batch_size, num_classes]`의 단일 텐서입니다. `batch_size`는 입력에서와 동일합니다. `num_classes`는 분류에서 등급의 수이며 모델별 상수입니다.

`logits[i, c]` 값은 인덱스가 `c`인 등급에서 예제 `i`의 멤버십을 예측하는 점수입니다.

이러한 점수가 소프트맥스(상호 배타적인 등급의 경우), 시그모이드(직교 등급의 경우) 또는 다른 것과 함께 사용되는지 여부는 기본 분류에 따라 다르며, 모듈 설명서에서 설명되어 있습니다. 등급 인덱스의 정의를 참조하세요.

### API 세부 정보

[Reusable SavedModel API](../reusable_saved_models.md)는 또한 `obj.variables`(예: 즉시 로딩하지 않을 때 초기화를 위해)의 목록을 제공합니다.

미세 조정을 지원하는 모델은 `obj.trainable_variables` 목록을 제공합니다. 훈련 모드에서 실행하려면 `training=True`를 전달해야 할 수 있습니다(예: 드롭아웃). 일부 모델에서는 선택적 인수가 하이퍼 매개변수를 재정의할 수 있습니다(예: 드롭아웃 비율, 모델 설명서에 설명됨). 모델은 `obj.regularization_losses` 목록을 제공할 수도 있습니다. 자세한 내용은 [Reusable SavedModel API](../reusable_saved_models.md)를 참조하세요.

Keras에서는 `hub.KerasLayer`에서 처리합니다. `trainable=True`로 초기화하여 미세 조정을 활성화하고, `arguments=dict(some_hparam=some_value, ...)`를 사용합니다(hparam 재정의가 적용되는 드문 경우).

<a name="input"></a>

## 이미지 입력

이미지 입력은 모든 유형의 이미지 모델에 공통입니다.

이미지 배치를 입력으로 사용하는 모델은 이미지를 dtype `float32` 및 형상 `[batch_size, height, width, 3]`의 밀집 4-D 텐서로 받아들입니다. 텐서의 요소는 [0, 1] 범위로 정규화된 픽셀의 RGB 색상 값입니다. 이 값은 `tf.image.decode_*()`, 그리고 이어서 `tf.image.convert_image_dtype(..., tf.float32)`에서 얻어집니다.

모델은 모든 `batch_size`를 허용합니다. 모델 설명서에서 `height`와 `width`의 허용 범위를 지정합니다. 마지막 차원은 3개의 RGB 채널로 고정됩니다.

모델은 전체적으로 Tensor의 `channels_last` (또는 `NHWC`) 레이아웃을 사용하고 TensorFlow의 그래프 옵티마이저에서 `channels_first` (또는 필요한 경우 `NCHW`)를 재작성하는 것이 좋습니다.

<!--* freshness: { owner: 'arnoegw' reviewed: '2020-09-11' } *-->

# 이미지에 대한 공통 서명

이 페이지에서는 이미지 관련 작업을 위해 [TF1 Hub 형식](../tf1_hub_module.md)의 모듈에서 구현해야 하는 일반적인 서명을 설명합니다. [TF2 SavedModel 형식](../tf2_saved_model.md)에 대해서는 유사한 [SavedModel API](../common_saved_model_apis/images.md)를 참조하세요.

일부 모듈은 하나 이상의 작업에 사용할 수 있습니다(예: 이미지 분류 모듈은 특성 추출을 함께 수행하는 경우가 많음). 따라서 각 모듈은 (1) 게시자가 예상하는 모든 작업에 대해 명명된 서명을 제공하고, (2) 지정된 기본 작업에 대해 기본 서명 `output = m(images)`를 제공합니다.

<a name="feature-vector"></a>

## 이미지 특성 벡터

### 사용 요약

**이미지 특성 벡터**는 일반적으로 소비자 모델에 의한 분류를 위해 전체 이미지를 나타내는 밀집 1-D 텐서입니다. 이 텐서는 CNN의 중간 활성화와 달리 공간 분석을 제공하지 않고, [이미지 분류](#classification)와 달리 게시자 모델에서 학습한 분류를 폐기합니다.

이미지 특성 추출을 위한 모듈에는 이미지 배치를 특성 벡터 배치에 매핑하는 기본 서명이 있습니다. 이 서명은 다음과 같이 사용할 수 있습니다.

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  features = module(images)   # A batch with shape [batch_size, num_features].
```

또한 해당 명명된 서명을 정의합니다.

### 서명 사양

이미지 특성 벡터를 추출하기 위한 명명된 서명은 다음과 같이 호출됩니다.

```python
  outputs = module(dict(images=images), signature="image_feature_vector",
                   as_dict=True)
  features = outputs["default"]
```

입력은 [이미지 입력](#input)에 대한 일반적인 규칙을 따릅니다.

출력 사전에는 dtype `float32` 및 형상 `[batch_size, num_features]`의 `"default"` 출력이 포함됩니다. `batch_size`는 입력에서와 동일하지만 그래프 생성 시 알려지지 않습니다. `num_features`는 입력 크기에 무관하게 알려진 모듈별 상수입니다.

이러한 특성 벡터는 이미지 분류를 위한 일반적인 CNN의 최상위 컨볼루셔널 레이어에서 풀링된 특성과 같이 간단한 피드 전달 분류자를 사용하여 분류하는 데 사용할 수 있습니다.

출력 특성에 드롭아웃을 적용하는 작업은 모듈 소비자에게 맡겨야 합니다. 모듈 자체는 실제 출력에서 드롭아웃을 수행하지 않아야 합니다(다른 위치에서 내부적으로 드롭아웃을 사용하는 경우에도).

출력 사전은 예를 들어 모듈 내부에 숨겨진 레이어의 활성화와 같은 추가적인 출력을 제공할 수 있습니다. 키와 값은 모듈에 따라 다릅니다. 아키텍처 종속 키에 아키텍처 이름을 접두사로 지정하는 것이 좋습니다(예: 중간 레이어 `"InceptionV3/Mixed_5c"`와 최상위 컨볼루셔널 레이어 `"InceptionV2/Mixed_5c"`를 혼동하지 않기 위해).

<a name="classification"></a>

## 이미지 분류

### 사용 요약

**이미지 분류**는 이미지의 픽셀을 *모듈 게시자가 선택한* 분류법 클래스의 구성원 자격에 대한 선형 점수(로짓)에 매핑합니다. 이를 통해 소비자는 기본 특성뿐만 아니라 게시자 모듈에서 학습한 특정 분류로부터 결론을 도출할 수 있습니다([이미지 특성 벡터](#feature-vector) 참조).

이미지 특성 추출을 위한 모듈에는 이미지 배치를 로짓 배치에 매핑하는 기본 서명이 있습니다. 이 서명은 다음과 같이 사용할 수 있습니다.

```python
  module_spec = hub.load_module_spec("path/to/module")
  height, width = hub.get_expected_image_size(module_spec)
  images = ...  # A batch of images with shape [batch_size, height, width, 3].
  module = hub.Module(module_spec)
  logits = module(images)   # A batch with shape [batch_size, num_classes].
```

또한 해당 명명된 서명을 정의합니다.

### 서명 사양

이미지 특성 벡터를 추출하기 위한 명명된 서명은 다음과 같이 호출됩니다.

```python
  outputs = module(dict(images=images), signature="image_classification",
                   as_dict=True)
  logits = outputs["default"]
```

입력은 [이미지 입력](#input)에 대한 일반적인 규칙을 따릅니다.

출력 사전에는 dtype `float32` 및 형상 `[batch_size, num_classes]`의 `"default"` 출력이 포함됩니다. `batch_size`는 입력에서와 동일하지만 그래프 생성시 알려지지 않습니다. `num_classes`는 분류의 클래스 수이며 입력 크기와 무관하게 알려진 상수입니다.

`outputs["default"][i, c]`를 평가하면 인덱스 `c`가 있는 클래스에서 예제 `i`의 구성원 자격을 예측하는 점수가 산출됩니다.

이러한 점수가 소프트맥스(상호 배타적 클래스의 경우), 시그모이드(직교 클래스의 경우) 또는 기타 다른 요소와 함께 사용되는지 여부는 기본 분류에 따라 다릅니다. 모듈 설명서에 이 내용이 설명되어 있으며 클래스 인덱스의 정의를 참조하기 바랍니다.

출력 사전은 예를 들어 모듈 내부에 숨겨진 레이어의 활성화와 같은 추가적인 출력을 제공할 수 있습니다. 키와 값은 모듈에 따라 다릅니다. 아키텍처 종속 키에 아키텍처 이름을 접두사로 지정하는 것이 좋습니다(예: 중간 레이어 `"InceptionV3/Mixed_5c"`와 최상위 컨볼루셔널 레이어 `"InceptionV2/Mixed_5c"`를 혼동하지 않기 위해).

<a name="input"></a>

## 이미지 입력

이 부분은 모든 유형의 이미지 모듈과 이미지 서명에 공통적입니다.

이미지 배치를 입력으로 사용하는 서명은 이미지를 dtype `float32` 형상 `[batch_size, height, width, 3]`의 밀집 4D 텐서로 받아들입니다. 텐서의 요소는 [0, 1] 범위로 정규화된 픽셀의 RGB 색상 값입니다. 이 값은 `tf.image.decode_*()`, 그리고 이어서 `tf.image.convert_image_dtype(..., tf.float32)`에서 얻어집니다.

정확히 하나(또는 하나의 기본)의 이미지 입력이 있는 모듈은 이 입력에 `"images"`라는 이름을 사용합니다.

모듈은 모든 `batch_size`를 허용하고 이에 따라 TensorInfo.tensor_shape의 첫 번째 차원을 "unknown"으로 설정합니다. 마지막 차원은 RGB 채널 수 `3` 으로 고정됩니다. `height` 및 `width` 차원은 입력 이미지의 예상 크기로 고정됩니다(향후 작업에서 완전한 컨볼루션 모듈을 얻기 위해 이러한 제한을 없앨 수 있음).

모듈 소비자는 형상을 직접 검사하지 않아야 하지만 모듈 또는 모듈 사양에서 hub.get_expected_image_size()를 호출하여 크기 정보를 가져오고 그에 따라 입력 이미지 크기를 조정해야 합니다(일반적으로 배치 처리 중 또는 그 이전).

단순화를 위해 TF-Hub 모듈은 텐서의 `channels_last`(또는 `NHWC`) 레이아웃을 사용하고, 필요 시 `channels_first`(또는 `NCHW`)로 다시 작성하는 일은 TensorFlow 그래프 옵티마이저의 몫으로 남겨 놓습니다.  TensorFlow 버전 1.7부터 기본적으로 이런 방식으로 동작합니다.

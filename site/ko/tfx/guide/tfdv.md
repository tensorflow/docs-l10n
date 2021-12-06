# TensorFlow 데이터 검증: 데이터 확인 및 분석하기

데이터가 TFX 파이프라인에 있으면 TFX 구성 요소를 사용하여 분석하고 변환할 수 있습니다. 모델을 훈련하기 전에도 이러한 도구를 사용할 수 있습니다.

데이터를 분석하고 변환하는 데는 여러 가지 이유가 있습니다.

- 데이터에서 문제를 찾기 위함입니다. 일반적인 문제는 다음과 같습니다.
    - 값이 비어있는 특성과 같은 누락된 데이터
    - 모델이 훈련 중에 올바른 답을 엿볼 수 있도록 특성으로 취급되는 레이블
    - 예상 범위를 벗어난 값을 가진 특성
    - 데이터 이상
    - 전이 학습된 모델에 훈련 데이터와 일치하지 않는 전처리가 있음
- 더 효과적인 특성 세트를 설계하기 위함입니다. 예를 들어, 다음을 식별할 수 있습니다.
    - 특히 유용한 특성
    - 중복 특성
    - 규모 면에서 매우 광범위하여 학습 속도가 느려질 수 있는 특성
    - 고유한 예측 정보가 거의 또는 전혀 없는 특성

TFX 도구는 데이터 버그를 찾는 데 도움이 되고 특성 엔지니어링(feature engineering)에 도움이 됩니다.

## TensorFlow 데이터 검증

- [개요](#overview)
- [스키마 기반 예제 검증](#schema_based_example_validation)
- [Training-Serving Skew Detection](#skewdetect)
- [편향 감지](#drift_detection)

### 개요

TensorFlow 데이터 검증은 훈련 및 적용 데이터에서 이상을 식별하고 데이터를 검사하여 스키마를 자동으로 만들 수 있습니다. 구성 요소는 데이터에서 서로 다른 클래스의 이상을 감지하도록 구성할 수 있습니다. 이는 다음을 수행합니다.

1. 데이터 통계와 사용자의 기대치를 코드화하는 스키마를 비교하여 검사를 수행합니다.
2. 훈련 및 적용 데이터의 예제를 비교하여 훈련-적용 편향을 감지합니다.
3. 일련의 데이터를 보고 데이터 편향을 감지합니다.

이러한 각 기능을 독립적으로 문서화합니다.

- [스키마 기반 예제 검증](#schema_based_example_validation)
- [Training-Serving Skew Detection](#skewdetect)
- [편향 감지](#drift_detection)

### 스키마 기반 예제 검증

TensorFlow 데이터 검증은 데이터 통계를 스키마와 비교하여 입력 데이터의 이상을 식별합니다. 스키마는 데이터 유형 또는 범주 값과 같이 입력 데이터가 충족할 것으로 예상되는 속성을 코드화하고 사용자가 수정하거나 바꿀 수 있습니다.

Tensorflow Data Validation은 일반적으로 (i) ExampleGen에서 얻은 모든 분할, (ii) 변환에서 사용하는 모든 사전 변환된 데이터 및 (iii) 변환에 의해 생성된 모든 사후 변환 데이터를 대상으로 TFX 파이프라인의 컨텍스트 내에서 여러 번 호출됩니다. 변환 (ii-iii)의 컨텍스트에서 호출되었을 때 [`stats_options_updater_fn`](tft.md)을 정의하여 통계 옵션과 스키마 기반 제약 조건을 설정할 수 있습니다. 이것은 구조화되지 않은 데이터(예: 텍스트 특성)의 유효성을 검사할 때 특히 유용합니다. 예제는 [사용자 코드](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py)를 참조하세요.

#### 고급 스키마 특성

이 섹션에서는 특수 설정에 도움이 될 수 있는 고급 스키마 구성을 다룹니다.

##### 희소 특성

예제에서 희소 특성을 인코딩하면 일반적으로 모든 예제에서 같은 수의 값을 가질 것으로 예상되는 여러 특성이 도입됩니다. 예를 들어, 희소 특성은 다음과 같습니다.

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

인덱스 및 값에 대해 별도의 특성을 사용하여 인코딩됩니다.

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

모든 예에서 인덱스 및 값 특성의 개수가 일치해야 한다는 제한이 있습니다. 이 제한은 sparse_feature를 정의하여 스키마에서 명시적으로 만들 수 있습니다.

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

희소 특성 정의에는 스키마에 있는 특성을 참조하는 하나 이상의 인덱스와 하나의 값 특성이 필요합니다. 희소 특성을 명시적으로 정의하면 TFDV가 참조된 모든 특성의 값 개수가 일치하는지 확인할 수 있습니다.

일부 사용 사례는 특성 간에 값 개수에 대한 유사한 제한을 도입하지만 희소 특성을 반드시 인코딩할 필요는 없습니다. 희소 특성을 사용하면 제한이 완화되지만 이상적인 방법은 아닙니다.

##### 스키마 환경

기본적으로 검증에서는 파이프라인의 모든 예제가 단일 스키마를 준수한다고 가정합니다. 때에 따라 약간의 스키마 변형을 도입해야 할 필요가 있습니다. 예를 들어, 레이블로 사용되는 특성은 훈련 중에 필요하고 검증을 해야 하지만 서빙 중에 누락됩니다. 환경을 사용하여 이러한 요구 사항, 특히 `default_environment()`, `in_environment()`, `not_in_environment()`를 표현할 수 있습니다.

예를 들어, 'LABEL'이라는 특성이 훈련에 필요하지만 적용될 때 누락될 것으로 예상된다고 가정해 보겠습니다. 이는 다음과 같이 표현할 수 있습니다.

- 스키마에 [ "SERVING", "TRAINING"]의 두 가지 고유한 환경을 정의하고 'LABEL'을 'TRAINING' 환경에만 연결합니다.
- 훈련 데이터를 'TRAINING' 환경과 연결하고 적용 데이터를 'SERVING' 환경과 연결합니다.

##### 스키마 생성

입력 데이터 스키마는 TensorFlow [스키마](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto)의 인스턴스로 지정됩니다.

개발자는 처음부터 수동으로 스키마를 구성하는 대신 TensorFlow 검증의 자동 스키마 구성을 사용할 수 있습니다. 특히 TensorFlow 검증은 파이프라인에서 사용 가능한 훈련 데이터에 대해 계산된 통계를 기반으로 초기 스키마를 자동으로 구성합니다. 사용자는 이 자동 생성된 스키마를 검토하고 필요에 따라 수정하고 버전 제어 시스템에 체크인한 다음, 추가 검증을 위해 파이프라인에 명시적으로 푸시할 수 있습니다.

TFDV에는 스키마를 자동으로 생성하는 `infer_schema()`가 포함되어 있습니다. 예를 들면, 다음과 같습니다.

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

그러면 다음 규칙에 따라 자동 스키마 생성이 트리거됩니다.

- 스키마가 이미 자동 생성된 경우 그대로 사용됩니다.

- 그렇지 않으면 TensorFlow 검증은 사용 가능한 데이터 통계를 검사하고 데이터에 적합한 스키마를 계산합니다.

*참고: 자동 생성된 스키마는 최상의 결과이며 데이터의 기본 속성만 추론하려고 합니다. 사용자가 필요에 따라 검토하고 수정해야 합니다.*

### 훈련-서빙 편향 감지<a name="skewdetect"></a>

#### 개요

TensorFlow 데이터 검증은 훈련 데이터와 서빙 데이터 간의 분포 편향을 감지할 수 있습니다. 분포 편향은 훈련 데이터의 특성 값 분포가 서빙 데이터와 크게 다를 때 발생합니다. 분포 편향의 주요 원인 중 하나는 원하는 코퍼스의 초기 데이터 부족을 극복하기 위해 훈련 데이터 생성에 완전히 다른 코퍼스를 사용하는 것입니다. 또 다른 이유는 훈련할 서빙 데이터의 하위 샘플만 선택하는 잘못된 샘플링 메커니즘입니다.

##### 예제 시나리오

참고: 예를 들어, 과소 표현된 데이터 조각을 보상하기 위해 다운샘플링된 예제에 적절하게 높은 가중치를 부여하지 않고 편향된 샘플링을 사용하면 훈련 데이터와 서빙 데이터 간의 특성 값 분포가 인위적으로 왜곡됩니다.

훈련-서빙 편향 감지 구성에 대한 내용은 [TensorFlow 데이터 검증 시작하기 가이드](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)를 참조하세요.

### 편향 감지

편향 감지는 서로 다른 훈련 데이터 날짜 사이와 같이 연속 데이터 범위(즉, 범위 N과 범위 N+1 사이) 사이에서 지원됩니다. 범주형 특성에 대한 [L-무한 거리](https://en.wikipedia.org/wiki/Chebyshev_distance)와 숫자 특성에 대한 대략적 [Jensen-Shannon 발산](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)의 관점에서 편향을 표현합니다. 편향이 허용 가능한 수준보다 높을 때 경고를 받도록 임계 거리를 설정할 수 있습니다. 올바른 거리를 설정하려면 일반적으로 도메인 지식을 바탕으로 반복적인 시도를 해 보아야 합니다.

편향 감지 구성에 대한 내용은 [TensorFlow 데이터 검증 시작하기 가이드](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)를 참조하세요.

## 시각화로 데이터 확인하기

TensorFlow 데이터 검증은 특성값의 분포를 시각화하는 도구를 제공합니다. [Facets](https://pair-code.github.io/facets/)을 사용하여 Jupyter 노트북에서 이러한 분포를 검사하면 데이터의 일반적인 문제를 파악할 수 있습니다.

![Feature stats](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/feature_stats.png?raw=true)

### 의심스러운 분포 식별하기

특성값의 의심스러운 분포를 찾기 위해 Facets Overview 디스플레이를 사용하여 데이터에서 일반적인 버그를 식별할 수 있습니다.

#### 불균형 데이터

불균형 특성은 하나의 값이 우세한 특성입니다. 불균형 특성은 자연스럽게 발생할 수 있지만 특성이 항상 같은 값을 갖는 경우 데이터 버그가 있을 수 있습니다. Facets Overview에서 불균형 특성을 감지하려면 'Sort by' 드롭다운에서 'Non-uniformity'을 선택합니다.

가장 불균형한 특성이 각 특성 유형 목록의 맨 위에 나열됩니다. 예를 들어, 다음 스크린샷은 'Numeric Features' 목록의 맨 위에 모두 0인 특성과 매우 불균형한 두 번째 특성을 보여줍니다.

![Visualization of unbalanced data](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/unbalanced.png?raw=true)

#### 균일하게 분포된 데이터

균일하게 분포된 특성은 가능한 모든 값이 같은 빈도에 가깝게 나타나는 특성입니다. 불균형 데이터와 마찬가지로 이 분포는 자연스럽게 발생할 수 있지만 데이터 버그로 생성될 수도 있습니다.

Facets Overview에서 균일하게 분포된 특성을 감지하려면 'Sorty by' 드롭다운에서 'Non-uniformity'을 선택하고 'Reverse order' 확인란을 선택합니다.

![Histogram of uniform data](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/uniform.png?raw=true)

문자열 데이터는 고유한 값이 20개 이하인 경우 막대 차트를 사용하고 고유한 값이 20개 이상인 경우 누적 분포 그래프로 표시됩니다. 따라서 문자열 데이터의 경우 균일한 분포는 위와 같은 평평한 막대 그래프 또는 아래와 같은 직선으로 나타날 수 있습니다.

![Line graph: cumulative distribution of uniform data](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/uniform_cumulative.png?raw=true)

##### 균일하게 분포된 데이터를 생성할 수 있는 버그

다음은 균일하게 분포된 데이터를 생성할 수 있는 몇 가지 일반적인 버그입니다.

- 문자열을 사용하여 날짜와 같은 비 문자열 데이터 유형을 나타냅니다. 예를 들어, '2017-03-01-11-45-03'과 같은 표현이 있는 날짜/시간 특성에 대한 고유한 값이 많이 있습니다. 고유한 값이 균일하게 분포됩니다.

- 특성으로 '행 번호'와 같은 인덱스를 포함합니다. 여기에서도 고유한 값이 많이 있습니다.

#### 누락된 데이터

특성값이 완전히 누락되었는지 확인하려면 다음을 확인합니다.

1. 'Sort by' 드롭다운에서 'Amount missing/0'을 선택합니다.
2. 'Reverse order' 확인란을 선택합니다.
3. 특성에 대해 누락된 값이 있는 인스턴스의 비율을 보려면 'missing' 열을 확인하세요.

데이터 버그로 인해 불완전한 특성값이 발생할 수도 있습니다. 예를 들어, 특성의 값 목록에 항상 3개의 요소가 있을 것으로 예상했지만 때로는 하나만 있는 것을 발견할 수 있습니다. 불완전한 값 또는 특성값 목록에 예상한 개수의 요소가 없는 기타 경우를 확인하려면 다음을 수행하세요.

1. 오른쪽의 'Chart to show' 드롭다운 메뉴에서 'Value list length'를 선택합니다.

2. 각 특성 행의 오른쪽에 있는 차트를 봅니다. 차트는 특성에 대한 값 목록 길이의 범위를 보여줍니다. 예를 들어, 아래 스크린샷에서 강조 표시된 행은 값 목록 길이가 0인 특성을 보여줍니다.

![Facets Overview display with feature with zero-length feature value lists](https://github.com/tensorflow/docs-l10n/blob/master/site/ko/tfx/guide/images/zero_length.png?raw=true)

#### 특성 간 큰 규모 차이

특성의 규모가 크게 다를 경우 모델이 학습하는 데 어려움이 있을 수 있습니다. 예를 들어, 일부 특성은 0에서 1까지 다양하고 다른 특성은 0에서 1,000,000,000까지 다양하다면 규모에 큰 차이가 있습니다. 특성 전체에서 '최대' 및 '최소' 열을 비교하여 크게 변화하는 범위를 찾습니다.

이러한 광범위한 변형을 줄이려면 특성값을 정규화하는 것이 좋습니다.

#### 잘못된 레이블이 있는 레이블

TensorFlow의 Estimator는 레이블로 허용하는 데이터 유형에 제한이 있습니다. 예를 들어, 이진 분류자는 일반적으로 {0, 1} 레이블에서만 동작합니다.

Facets Overview에서 레이블 값을 검토하고 [Estimator 요구 사항](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md)을 준수하는지 확인하세요.

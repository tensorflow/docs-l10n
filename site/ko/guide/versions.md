# TensorFlow 버전 호환성

이 문서는 서로 다른 버전의 TensorFlow(코드 또는 데이터)에서 하위 호환성이 필요한 사용자, 그리고 호환성을 유지하면서 TensorFlow를 수정하려는 개발자를 위해 작성되었습니다.

## 유의적 버저닝 2.0

TensorFlow는 공개 API에 유의적 버저닝 2.0([semver](http://semver.org))을 준수합니다. TensorFlow의 각 릴리즈 버전은 `MAJOR.MINOR.PATCH` 형식입니다. 이를테면 TensorFlow 버전 1.2.3은 `MAJOR` 버전 1, `MINOR` 버전 2, `PATCH` 버전 3을 뜻합니다. 각 숫자의 변화는 다음을 뜻합니다:

- **MAJOR**: 하위 호환성이 없는 변동일 수 있습니다. 이전 주(major) 버전에서 동작했었던 코드와 데이터는 새로운 버전에서 동작하지 않을 수 있습니다. 그러나, 어떤 경우에는 기존 TensorFlow 그래프와 체크포인트를 새로운 버전에 마이그레이션할 수도 있습니다. 자세한 사항은 [그래프와 체크포인트의 호환성](#compatibility_of_graphs_and_checkpoints)을 참고하세요.

- **MINOR**: 하위 호환되는 특성, 속도 개선 등입니다. 이전 부(minor) 버전 *및* 이전 부 버전의 비실험적인 공개 API를 사용했던 코드와 데이터는 정상적으로 동작합니다. 공개 API에 무엇이 포함되고 포함되지 않는지에 대해서는 [포함되는 사항](#What_is_covered)을 참조하세요.

- **PATCH**: 하위 호환되는 버그 픽스

이를테면 릴리즈 1.0.0은 릴리즈 0.12.1에서 하위 호환성이 *없는* 변동사항이 있습니다. 그러나, 릴리즈 1.1.1은 릴리즈 1.0.0과 하위 *호환성이 있습니다*.

## 포함되는 사항

TensorFlow의 공개 API만이 부 버전 및 패치 버전에서 하위 호환성을 가집니다. 공개 API는 다음을 포함합니다.

- 모든 문서화된 [Python](https://gitlocalize.com/repo/4592/ko/site/en-snapshot/api_docs/python) `tensorflow` 모듈과 서브 모듈에 있는 함수와 클래스, 다음은 제외

    - 비공개 심볼: `_`로 시작하는 함수나 클래스 등
    - 실험적인 및 `tf.contrib` 심볼, 자세한 내용은 [아래](#not_covered) 내용 참조

    `examples/`와 `tools/` 경로에 있는 코드는 `tensorflow` Python 모듈을 통해 접근할 수 없고 따라서 호환성을 보장할 수 없습니다.

    한 심볼이 `tensorflow` Python 모듈이나 서브 모듈에서 사용가능 하지만 문서화되지는 않은 경우, 공개 API의 일부로 간주하지 **않습니다**.

- 호환성 API(Python의 `tf.compat` 모듈). 주 버전에서 사용자들이 새로운 주 버전으로 옮겨가는 것을 도와주는 유틸리티와 추가적인 엔드포인트가 공개될 수도 있습니다. 이러한 API 심볼들은 없어지고 지원되지 않지만(즉, 기능을 추가하지 않고 취약성 이외의 버그를 수정하지 않음) 호환성은 보장됩니다.

- [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)

- 다음의 프로토콜 버퍼 파일:

    - [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    - [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    - [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    - [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    - [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    - [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    - [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    - [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    - [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    - [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>

## 포함되지 *않는* 사항

TensorFlow의 일부분은 어떤 면에서도 하위 호환성이 없도록 변동될 수 있습니다. 여기에는 다음이 포함됩니다.

- **실험적인 API**: 개발을 용이하게 하기 위해, 어떤 API 심볼들은 실험적인 것으로 규정하고 하위 호환성을 보장하지 않습니다. 특히 다음은 어떠한 호환성 보장도 하지 않습니다.

    - `tf.contrib` 모듈이나 서브 모듈에 있는 모든 심볼
    - `experimental` 또는 `Experimental`이라는 이름을 포함하는 모든 심볼(모듈, 함수, 매개변수, 속성, 클래스, 상수); 또는
    - 모듈이나 클래스가 포함하는 절대 표기가 그 자체로 실험적인 모든 심볼. `experimental`로 분류되는 모든 프로토콜 버퍼의 필드나 서브 메시지 포함.

- **다른 언어:** Python과 C 이외의 다음과 같은 TensorFlow API 언어:

    - [C++](https://www.tensorflow.org/api_guides/cc/guide.md)([`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc)의 헤더 파일을 통해 공개되어 있음).
    -  [Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
    - [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
    - [JavaScript](https://js.tensorflow.org)

- **합성 연산 세부사항:** Python의 많은 공개 함수가 일부 그래프의 원시 연산에 확장됩니다, 그리고 이러한 세부사항은 `GraphDef`로 디스크에 저장되는 그래프의 한 부분입니다. 이러한 세부사항은 부(minor) 버전에서 변경될 수 있습니다. 특히, 그래프간 정확한 매칭이 되는지 확인하는 회귀 테스트는 그래프의 행동이 변경되지 않고 기존의 체크포인트가 아직 동작할지라도 서로 다른 부 버전에서는 호환되지 않을 가능성이 높습니다.

- **부동 소수점 세부사항:** 연산을 통해 계산되는 특정 부동 소수점 값은 언제든지 변경될 수 있습니다. 사용자는 계산된 특정 비트에 의존하면 안되고, 근사적인 정밀도와 수치적 안정성에 초점을 두어야 합니다. 부 버전과 패치에서 수식의 변화는 상당히 정확도를 높입니다. 머신러닝에서 특정 공식의 향상된 정확도는 전체 시스템에서의 정확도를 낮추는 경우도 있습니다.

- **랜덤 숫자:** 특정한 랜덤 숫자가 [random ops](https://www.tensorflow.org/api_guides/python/constant_op.md#Random_Tensors)를 통해 계산되고 언제든지 바뀔 수 있습니다. 사용자는 계산된 특정 비트에 의존하지 말고, 근사적으로 적절한 분포와 통계적 강도에 중점을 두어야 합니다. 그러나, 패치 버전에서는 특정한 비트를 거의 바꾸지 않도록 합니다. 당연히 이러한 모든 변동사항은 문서화합니다.

- **분산 Tensorflow에서의 버전 엇갈림:** 하나의 클러스터에서 서로 다른 두 버전의 Tensorflow를 실행하는 것은 지원되지 않습니다. 와이어 프로토콜(wire protocol)의 하위 호환성을 보장할 수 없습니다.

- **버그:** 현재의 구현이 명백하게 문제가 있는 경우, 하위 호환성을 유지하지 않는 변동사항을 만들 수 있습니다. 문서와 구현이 서로 모순되는 경우 또는 잘 알려져 있고 잘 정의된 의도를 가진 행동이 버그 때문에 적절하게 구현되지 않은 경우가 이에 해당됩니다. 이를테면, 잘 알려진 최적화 알고리즘이 옵티마이저에 구현되어야 하지만 버그 때문에 그 알고리즘과 매치되지 않는다면, 옵티마이저를 수정할 것입니다. 수정사항은 통합을 위해서 잘못 동작하는 부분에 의존하는 코드를 포함합니다. 릴리즈 노트에 그러한 변동사항이 기록될 것입니다.

- **사용되지 않는 API:** 당사는 문서화된 용도가 없는 API에 대해 하위 호환되지 않는 변경을 수행할 권리를 보유합니다(GitHub 검색을 통해 TensorFlow 사용 감사를 수행함). 이러한 변경을 하기 전에 [announce@ 메일링 리스트](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce)에 변경 의사를 발표하여 중단을 해결하는 방법(해당되는 경우)을 제공하고 2주 동안 기다리면서 커뮤니티에서 피드백을 공유할 수 있는 기회를 줍니다.

- **오류 동작:** 오류를 오류가 아닌 동작으로 대체할 수 있습니다. 예를 들어 오류가 문서화되어 있어도 오류를 발생시키는 대신 결과를 계산하도록 함수를 변경할 수 있습니다. 또한 오류 메시지의 텍스트를 변경할 권리가 있습니다. 이와 함께, 문서에 특정 오류 조건에 대한 예외 유형이 지정되어 있지 않으면 오류 유형이 변경될 수 있습니다.

<a name="compatibility_of_graphs_and_checkpoints"></a>

## 저장된 모델, 그래프 및 체크포인트의 호환성

저장된 모델은 TensorFlow 프로그램에서 사용하기 위해 직렬화된 형식이 좋습니다. 저장된 모델은 두 부분으로 이루어져 있습니다. 하나 이상의 그래프가 `GraphDefs`와 체크포인트로 인코딩됩니다. 그래프는 실행할 연산의 데이터 흐름를 기술하고 체크포인트는 그래프 변수들의 저장된 텐서값을 포함합니다.

많은 TensorFlow 사용자들이 저장된 모델을 만들고 나중에 릴리즈된 TensorFlow에서 로드하여 실행합니다. [semver](https://semver.org)에 따라 한 버전의 TensorFlow에서 작성된 모델이 같은 주 버전에 속한 나중 버전의 TensorFlow에서 로드되고 평가될 수 있습니다.

*지원하는* 저장된 모델에서는 추가적인 보장이 있습니다. TensorFlow 주 버전 `N`에서 **사라지지 않고 실험적이지도 않으며 호환되지 않는 API**를 사용하여 만든 저장된 모델은 <em data-md-type="emphasis">버전 `N`에서 지원됩니다.</em> TensorFlow 주 버전 `N`에서 지원하는 모든 저장된 모델은 TensorFlow 주 버전 `N+1`에서도 로드되고 실행될 수 있습니다. 그러나, 그 모델을 만들고 수정하기 위해 필요한 기능들을 더 이상 사용할 수 없는 경우, 이 보장은 수정하지 않은 저장된 모델에만 적용됩니다.

가능하면 하위 호환성을 유지하기 위해 노력할 것이므로 직렬화된 파일들은 오랫동안 사용 가능합니다.

### GraphDef 호환성

그래프는 `GraphDef` 프로토콜 버퍼를 통해 직렬화됩니다. 이전 버전과 호환되지 않는 그래프 변경을 용이하게 하기 위해 각 `GraphDef`에는 TensorFlow 버전과 별도로 버전 번호가 있습니다. 예를 들어  `GraphDef` 버전 17에서는 `reciprocal`를 위해 `inv` op를 없앴습니다. 의미는 다음과 같습니다.

- TensorFlow의 각 버전은 `GraphDef` 버전의 간격을 지원합니다. 이 간격은 패치 릴리스 사이에서 일정하며 부 릴리스에서만 증가합니다. `GraphDef` 버전에 대한 지원 중단은 TensorFlow의 주요 릴리스에 대해서만 발생합니다(저장된 모델에 대해 보장되는 버전 지원과만 일치함).

- 새로 생성된 그래프에는 최신 `GraphDef` 버전 번호가 할당됩니다.

- 특정 버전의 TensorFlow가 그래프의 `GraphDef` 버전을 지원하는 경우, TensorFlow의 주요 버전에 관계없이 그래프를 생성하는 데 사용된 TensorFlow 버전과 동일한 동작으로 로드 및 평가됩니다(위에 설명한 대로 부동 소수점 숫자 정보와 난수는 제외). 특히, 한 버전의 TensorFlow(예: 저장된 모델의 경우)의 체크포인트 파일과 호환되는 GraphDef는 GraphDef가 지원되는 한 후속 버전에서 해당 체크포인트와 호환됩니다.

    이는 GraphDefs(및 저장된 모델)의 직렬화된 그래프에만 적용됩니다. 체크포인트를 읽는 *Code*는 다른 버전의 TensorFlow를 실행하는 동일한 코드에서 생성된 체크포인트를 읽지 못할 수 있습니다.

- 부 릴리스에서 `GraphDef` *위쪽* 경계가 X로 증가하면 *아래쪽* 경계가 X까지 증가하는 데는 최소 6개월이 걸립니다. 예를 들면 다음과 같습니다(여기서는 가상 버전 번호를 사용함).

    - TensorFlow 1.2는 `GraphDef` 버전 4부터 7까지 지원할 수 있습니다.
    - TensorFlow 1.3은 `GraphDef` 버전 8을 추가하고 버전 4부터 8까지 지원할 수 있습니다.
    - 적어도 6개월 후, TensorFlow 2.0.0은 버전 4부터 7까지에 대한 지원을 중단하고 버전 8만 남길 수 있습니다.

    TensorFlow의 주요 버전은 일반적으로 6개월 이상 간격을 두고 출시되기 때문에 위에 설명한 지원되는 저장된 모델에 대한 보증은 GraphDefs에 대한 6개월 보증보다 훨씬 강력합니다.

마지막으로, `GraphDef` 버전에 대한 지원이 중단되면 그래프를 지원되는 최신 `GraphDef` 버전으로 자동 변환하는 도구를 제공할 것입니다.

## TensorFlow 확장시 그래프 및 체크포인트 호환성

이 섹션은 연산 추가, 연산 제거 또는 기존 연산의 기능 변경과 같이 `GraphDef` 형식에 호환되지 않는 변경을 수행하는 경우에만 관련됩니다. 대부분의 사용자에게는 이전 섹션의 내용으로 충분합니다.

<a id="backward_forward"></a>

### 역방향 및 부분 순방향 호환성

버저닝 계획의 세 가지 요건:

- 이전 버전의 TensorFlow에서 만들어진 그래프와 체크포인트의 로딩을 지원하기 위한 **하위 호환성**.
- 그래프나 체크포인트의 생산자가 소비자 이전에 새 버전의 TensorFlow로 업그레이드되는 시나리오를 위한 **상위 호환성**.
- TensorFlow가 호환하지 않는 방향으로 개선되는 것을 가능하게 함. 이를테면, 연산을 제거하거나 속성을 추가하고 제거함.

`GraphDef` 버전 메커니즘은 TensorFlow 버전과는 분리되어 있지만, `GraphDef` 형식에 하위 호환되지 않는 변동사항은 유의적 버저닝에서 제한됩니다. 즉, TensorFlow `주 (MAJOR)` 버전간(이를테면 `1.7`과 `2.0`) 기능이 제거되거나 변할 수 있습니다. 상위 호환성은 패치 릴리즈(이를테면 `1.x.1`에서 `1.x.2`) 안에서 강제됩니다.

상위 호환성과 하위 호환성을 가능하게 하고 형식의 변동을 언제 강제해야 할지 알기 위해서, 그래프와 체크포인트는 언제 생성되었는지에 대한 메타데이터를 가집니다. 아래의 섹션에 TensorFlow 구현과 `GraphDef` 버전업에 대한 가이드라인이 자세히 나와 있습니다.

### 독립적인 데이터 버전 계획

그래프와 체크포인트에는 서로 다른 데이터 버전이 있습니다. 두 데이터 형식은 서로 다른 비율로 버전업되고 또한 TensorFlow와도 서로 다른 비율로 버전업됩니다. 두 버저닝 시스템 모두 [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)에서 정의됩니다. 새 버전이 추가될 때마다 어떤 사항이 변했고 날짜는 어떻게 되는 지가 헤더에 추가됩니다.

### 데이터, 생산자 및 소비자

다음과 같은 데이터 버전 정보를 구분할 것입니다.

- **생산자**: 데이터를 생성하는 바이너리입니다. 생산자는 버전(`producer`) 및 호환되는 최소 소비자 버전(`min_consumer`)을 가지고 있습니다.
- **소비자**: 데이터를 소비하는 바이너리입니다. 소비자는 버전(`consumer`) 및 호환되는 최소 생산자 버전(`min_producer`)을 가지고 있습니다.

데이터 버전은 데이터를 만든 `생산자`와 호환이 되는 `min_consumer`, 그리고 허용되지 않은 `bad_consumers` 버전 리스트를 기록하는 [`VersionDef versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto) 필드를 가지고 있습니다.

기본적으로, 생산자가 데이터를 만들면 데이터는 생산자의 `producer`와 `min_consumer` 버전을 물려 받습니다. 특정한 소비자 버전이 버그를 포함하고 있거나 반드시 피해야 한다면 `bad_consumers`가 설정될 수 있습니다. 소비자는 다음이 모두 성립하는 경우 데이터를 받아들일 수 있습니다.

- `consumer` &gt;= 데이터의 `min_consumer`
- 데이터의 `producer` &gt;= 소비자의 `min_producer`
- 데이터의 `bad_consumers`에 없는 `consumer`

생산자와 소비자 모두 같은 TensorFlow 코드베이스로부터 나온 것이기 때문에, [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)는 문맥과 `min_consumer` 및 `min_producer`(생산자와 소비자 각각이 필요로 하는)에 따라 `producer`나 `consumer` 둘 중 하나로 취급되는 메인 데이터 버전을 포함합니다. 구체적으로, 다음과 같습니다.

- `GraphDef` 버전으로 `TF_GRAPH_DEF_VERSION`, `TF_GRAPH_DEF_VERSION_MIN_CONSUMER`, `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`가 있습니다.
- 체크포인트 버전으로 `TF_CHECKPOINT_VERSION`, `TF_CHECKPOINT_VERSION_MIN_CONSUMER`, `TF_CHECKPOINT_VERSION_MIN_PRODUCER`가 있습니다.

### 기존 연산에 새로운 속성을 기본값으로 추가

다음의 가이드를 따르면 일련의 연산이 변하지 않았을 때만 상위 호환성이 있게 됩니다.

1. 상위 호환성이 필요하다면, `SavedModelBuilder`클래스의 `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables`와 `tf.saved_model.SavedModelBuilder.add_meta_graph` 메서드를 사용하거나 `tf.estimator.Estimator.export_saved_model`을 사용하는 모델을 내보내는 동안 `strip_default_attrs`를 `True`로 설정합니다.
2. 이렇게 하면 모델을 생성/내보낼 때 기본 값 속성이 제거됩니다. 그러면 기본값이 사용될 때 내보낸 `tf.MetaGraphDef`에 새 op 속성이 포함되지 않습니다.
3. 이 컨트롤을 사용하면 오래된 소비자(예를들면, 훈련 바이너리에 뒤쳐진 바이너리를 제공하는)가 모델을 불러오기를 계속할 수 있게 하고 모델 서비스 중단을 막을 수 있습니다.

### GraphDef 버전업

이 섹션은 `GraphDef` 형식에 다른 타입의 변동사항을 만들기 위한 버저닝 방법을 설명합니다.

#### 하나의 연산 추가하기

`GraphDef` 버전을 바꾸지 않고 소비자와 생산자에 동시에 새로운 연산을 추가합니다. 이러한 종류의 변동사항은 자동적으로 하위 호환성이 있고 기존 생산자 스크립트가 갑자기 새로운 기능을 사용하지는 않을 것이기 때문에 상위 호환 계획에 영향을 주지 않습니다.

#### 연산을 추가하고 이를 사용하기 위해 기존 Python 래퍼로 바꾸기

1. 새로운 소비자 기능을 구현하고 `GraphDef` 버전을 올립니다.
2. 이전에 동작하지 않았던 새로운 기능을 사용하는 래퍼를 만들 수 있습니다. 래퍼는 지금 업데이트 가능합니다.
3. Python 래퍼를 변경하여 새로운 기능을 사용합니다. 이 연산을 사용하지 않는 모델은 고장나지 않아야 하므로 `min_consumer`를 올리지 마세요.

#### 연산의 기능을 제거하거나 제한하기

1. 금지된 연산이나 기능을 사용하지 않기 위해 모든 생산자 스크립트(TensorFlow 자체가 아닌)를 고정합니다.
2. `GraphDef`버전을 올리고 새 버전의 GraphDef나 그 이상에서 제거된 연산이나 기능을 금지하는 새로운 소비자 기능을 구현하세요. 가능하다면 TensorFlow에서 금지된 기능으로 `GraphDefs`를 만들지 않도록 합니다. 이를 위해 [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009)를 추가하세요.
3. 하위 호환성을 목적으로 주 릴리즈를 기다립니다.
4. (2)에서의 GraphDef 버전에서 `min_producer`를 올리고 기능을 완전히 제거합니다.

#### 연산의 기능 바꾸기

1. `SomethingV2`와 같이 비슷한 연산의 이름을 추가하며 이를 위한 절차를 거치고 기존 Python 래퍼가 이를 사용할 수 있도록 전환하세요. Python 래퍼를 변경할 때 상위 호환성 확인을 위해 [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py)에 있는 제안 사항을 확인하세요.
2. 예전의 연산을 제거합니다(하위 호환성 때문에 주 버전이 변경될 때만 발생).
3. 예전의 연산을 사용하는 소비자를 배제하기 위해 `min_consumer`를 올리고 예전 연산에 `SomethingV2`를 위한 별칭을 달아줍니다. 그리고 기존 Python 래퍼가 사용할 수 있도록 변환하는 절차를 거치세요.
4. `SomethingV2`를 제거하는 절차를 거칩니다.

#### 안전하지 않은 소비자 버전 금지하기

1. `GraphDef` 버전을 충돌시키고 나쁜 버전을 모든 새로운 GraphDef의 `bad_consumers`에 추가합니다. 가능하면 특정한 연산이나 비슷한 것들을 포함하는 GraphDef에만 `bad_consumers`를 추가합니다.
2. 기존 소비자가 나쁜 버전인 경우, 최대한 빠르게 제거합니다.

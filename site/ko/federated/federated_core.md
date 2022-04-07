# 페더레이션된 코어

이 문서는 [Federated Learning](federated_learning.md)(페더레이션 학습)의 토대가 되는 미래 TFF의 코어 레이어와 미래의 비 학습 페더레이션 알고리즘을 소개합니다.

Federated Core에 대한 간단한 소개를 위해 다음 튜토리얼를 읽어 보세요. 기본 개념 중 일부를 예제로 소개하고 간단한 페더레이션 평균화 알고리즘의 구성을 단계별로 보여줍니다.

- [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](tutorials/custom_federated_algorithms_1.ipynb).

- [Custom Federated Algorithms, Part 2: Implementing Federated Averaging](tutorials/custom_federated_algorithms_2.ipynb).

또한, 페더레이션 학습에 Federated Core API(FC API)를 사용하면 이 레이어를 설계할 때 선택한 일부 사항에 대한 중요한 맥락을 이해할 수 있으므로 [페더레이션 학습](federated_learning.md)과 [이미지 분류](tutorials/federated_learning_for_image_classification.ipynb) 및 [텍스트 생성](tutorials/federated_learning_for_text_generation.ipynb)에 관한 관련 튜토리얼에 익숙해지기를 권장합니다.

## 개요

### 목표, 의도된 용도 및 범위

Federated Core(FC)는 분산 계산을 구현하기 위한 프로그래밍 환경, 즉 각각 중요한 처리를 로컬에서 수행하고 네트워크를 통해 통신하여 작업을 조정하는 여러 컴퓨터(휴대 전화, 태블릿, 내장 기기, 데스크톱 컴퓨터, 센서, 데이터베이스 서버 등)를 포함하는 계산 환경으로 이해할 수 있습니다.

*분산*이라는 용어는 매우 일반적이며, TFF는 가능한 모든 유형의 분산 알고리즘을 대상으로 하지 않으므로 이 프레임워크에서 표현할 수 있는 알고리즘의 유형을 설명하기 위해 덜 일반적인 용어인 *페더레이션 계산(federated computation)*을 사용하길 선호합니다.

완전히 공식적인 방식으로 *페더레이션 계산*이라는 용어를 정의하는 것은 이 문서의 범위를 벗어나지만, 새로운 분산 학습 알고리즘을 설명하는 [연구 간행물](https://arxiv.org/pdf/1602.05629.pdf)에서 의사 코드로 표현될 수 있는 알고리즘 유형으로 생각할 수 있습니다.

FC의 목표는 간단히 말하자면, 의사 코드가 *아니라* 다양한 대상 환경에서 실행 가능한 프로그램 논리를 의사 코드와 같은 추상화 수준에서 유사하게 컴팩트하게 표현하는 것입니다.

FC가 표현하는 알고리즘 종류의 특징을 정의하는 핵심은 시스템 참가자들의 행동이 집합적으로 설명된다는 것입니다. 따라서 데이터를 로컬로 변환하는 *각 기기*와 결과를 *브로드캐스팅*하고 *수집*하거나 *집계*하는 중앙 집중식 코디네이터를 통해 작업을 조정하는 기기에 대해 이야기하는 경향이 있습니다.

TFF는 단순한 *클라이언트-서버* 아키텍처를 뛰어넘을 수 있도록 설계되었지만, 일괄 처리(collective processing)의 개념은 기본입니다. 이는 페더레이션 학습에서의 TFF의 기원이 클라이언트 기기의 제어 아래 유지되고 개인 정보 보호를 위해 중앙 위치로 간단히 다운로드할 수 없는 잠재적으로 중요한 데이터에 대한 계산을 지원하도록 처음부터 설계된 기술에 있기 때문입니다. 이러한 시스템에서 각 클라이언트는 시스템의 결과(모든 참가자에게 일반적으로 가치가 있는 결과)를 계산하는 데 데이터 및 처리 능력을 제공하지만, 각 클라이언트의 개인 정보 및 익명성을 유지하기 위해 노력합니다.

따라서 분산 컴퓨팅을 위한 대부분의 프레임워크는 개별 참가자의 관점, 즉 개별 지점 간 메시지 교환 수준에서 수신 및 발신 메시지를 통한 참가자의 로컬 상태 전환의 상호 의존성 관점에서 처리를 표현하도록 설계된 반면, TFF의 Federated Core는 *전역* 시스템 차원의 관점에서 시스템의 동작을 설명하도록 설계되었습니다(예를 들어, [MapReduce](https://research.google/pubs/pub62/)와 유사).

따라서 일반적인 용도의 분산 프레임워크는 구성 요소로서 *보내기* 및 *받기*와 같은 연산을 제공할 수 있지만, FC는 간단한 분산 프로토콜을 캡슐화하는 `tff.federated_sum`, `tff.federated_reduce` 또는 `tff.federated_broadcast`와 같은 구성 요소를 제공합니다.

## 언어

### Python 인터페이스

TFF uses an internal language to represent federated computations, the syntax of which is defined by the serializable representation in [computation.proto](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). Users of FC API generally won't need to interact with this language directly, though. Rather, we provide a Python API (the `tff` namespace) that wraps arounds it as a way to define computations.

특히, TFF는 `tff.federated_computation`과 같은 Python 함수 데코레이터를 제공하여 데코레이팅된 함수의 본문을 추적하고 TFF의 언어로 페더레이션 계산 논리의 직렬화된 표현을 생성합니다. `tff.federated_computation`으로 데코레이팅된 함수는 이러한 직렬화된 표현의 캐리어 역할을 하며, 캐리어를 다른 계산 본문에 구성 요소로 포함하거나 호출될 때 요청 시 실행할 수 있습니다.

여기에 한 가지 예제가 있습니다. 더 많은 예제는 [사용자 정의 알고리즘](tutorials/custom_federated_algorithms_1.ipynb) 튜토리얼에서 찾을 수 있습니다.

```python
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

즉시 모드가 아닌 TensorFlow에 익숙한 독자는 TensorFlow 그래프를 정의하는 Python 코드 섹션에서 `tf.add` 또는 `tf.reduce_sum`과 같은 함수를 사용하는 Python 코드를 작성하는 것과 유사하다는 것을 알게 될 것입니다. 코드는 Python으로 기술적으로 표현되었지만, 그 아래에 `tf.Graph`의 직렬화 표현을 구성하는 것이 목적이며, TensorFlow 런타임에 의해 내부적으로 실행되는 것은 Python 코드가 아닌 그래프입니다. 마찬가지로, `tff.federated_mean`을 `get_average_temperature`로 표시되는 페더레이션 계산에 *federated op*를 삽입하는 것으로 생각할 수 있습니다.

FC에서 언어를 정의하는 이유 중 일부는 위에서 언급했듯이 페더레이션 계산이 분산 집단 동작을 지정하므로 동작의 논리가 로컬이 아니라는 사실과 관련이 있습니다. 예를 들어, TFF는 네트워크의 다른 위치에 존재할 수 있는 연산자, 입력 및 출력을 제공합니다.

따라서 분산성의 개념을 포착하는 언어와 유형 시스템이 필요합니다.

### 유형 시스템

Federated Core는 다음 유형의 범주를 제공합니다. 이들 유형을 설명할 때 편리한 유형이나 계산 및 연산자의 유형을 설명하는 간단한 표기법을 소개할 뿐만 아니라 유형 생성자를 안내합니다.

첫째, 기존 주류 언어에서 볼 수 있는 유형과 개념적으로 유사한 유형의 범주는 다음과 같습니다.

- **텐서 유형**(`tff.TensorType`): TensorFlow에서와 마찬가지로 `dtype`과 `shape`이 있습니다. 유일한 차이점은 이 유형의 객체는 TensorFlow 그래프에서 TensorFlow ops의 출력을 나타내는 Python의 `tf.Tensor` 인스턴스로 제한되지 않으며, 예를 들어 분산 집계 프로토콜의 출력으로 생성될 수 있는 데이터의 단위를 포함할 수 있다는 것입니다. 따라서 TFF 텐서 유형은 단순히 Python 또는 TensorFlow에서 해당 유형의 구체적인 물리적 표현의 추상 버전입니다.

    TFF's `TensorTypes` can be stricter in their (static) treatment of shapes than TensorFlow. For example, TFF's typesystem treats a tensor with unknown rank as assignable *from* any other tensor of the same `dtype`, but not assignable *to* any tensor with fixed rank. This treatment prevents certain runtime failures (e.g., attempting to reshape a tensor of unknown rank into a shape with incorrect number of elements), at the cost of greater strictness in what computations TFF accepts as valid.

    텐서 유형의 간단한 표기법은 `dtype` 또는 `dtype[shape]`입니다. 예를 들어, `int32` 및 `int32[10]`은 각각 정수 및 int 벡터의 유형입니다.

- **시퀀스 유형**(`tff.SequenceType`): TensorFlow에서 `tf.data.Dataset`의 구체적 개념에 해당하는 TFF의 추상화입니다. 시퀀스의 요소는 순차적인 방식으로 소비될 수 있으며 복잡한 유형을 포함할 수 있습니다.

    시퀀스 유형의 간단한 표현은 `T*`이며, `T`는 요소의 유형입니다. 예를 들어, `int32*`는 정수 시퀀스를 나타냅니다.

- **명명된 튜플 유형**(`tff.StructType`): 이름이 있거나 이름이 지정되지 않은, 사전 정의된 수의 특정 유형의 *요소*를 갖는 튜플 및 사전과 유사한 구조를 구성하는 TFF의 방법입니다. 중요한 것은 TFF의 명명된 튜플 개념은 Python의 인수 튜플과 같은 추상 요소, 즉 전부는 아니지만 일부는 이름이 지정되고, 일부는 위치와 관련된 요소의 모음을 포함합니다.

    명명된 튜플의 간단한 표기법은 `<n_1=T_1, ..., n_k=T_k>`입니다. `n_k`는 선택적 요소 이름이고, `T_k`는 요소 유형입니다. 예를 들어, `<int32,int32>`는 명명되지 않은 정수 쌍에 대한 간단한 표기법이고, `<X=float32,Y=float32>`는 평면의 점을 나타낼 수 있는 `X`와 `Y`라는 이름의 부동 소수점 쌍에 대한 간단한 표기법입니다. 튜플은 중첩될 수 있고 다른 유형과 혼합될 수 있습니다. 예를 들어, `<X=float32,Y=float32>*`는 일련의 점에 대한 간단한 표기법입니다.

- **함수 유형**(`tff.FunctionType`): TFF는 함수가 [일급 값](https://en.wikipedia.org/wiki/First-class_citizen)으로 취급되는 함수형 프로그래밍 프레임워크입니다. 함수는 최대 하나의 인수와 정확히 하나의 결과를 갖습니다.

    함수의 간결한 표기법은 `(T -> U)`입니다. `T`는 인수의 유형이고, `U`는 결과의 유형이거나 인수가 없는 경우(인수 없는 함수는 주로 Phython 레벨에서만 존재하는 중복 제거 개념이지만), `( -> U)`입니다. 예를 들어, `(int32* -> int32)`는 정수 시퀀스를 단일 정수 값으로 줄이는 함수 유형에 대한 표기법입니다.

다음 유형은 TFF 계산의 분산 시스템 측면을 해결합니다. 이러한 개념은 다소 TFF에 고유하므로 추가 주석 및 예제는 [사용자 정의 알고리즘](tutorials/custom_federated_algorithms_1.ipynb) 튜토리얼을 참조하세요.

- **배치 유형**: 이 유형은 이 유형의 상수로 생각할 수 있는 2개의 리터럴 `tff.SERVER` 및 `tff.CLIENTS` 형식 이외의 공용 API에 아직 노출되지 않은 유형입니다. 내부적으로 사용되지만, 향후 릴리스에서는 공개 API에 도입될 예정입니다. 이 유형의 간단한 표현은 `placement`입니다.

    *배치*는 특정 역할을 수행하는 일련의 시스템 참가자를 나타냅니다. 초기 릴리스는 *클라이언트* 및 *서버*의 두 그룹으로 구성된 클라이언트-서버 계산을 대상으로 합니다(후자는 싱글톤 그룹으로 생각할 수 있음). 그러나 보다 정교한 아키텍처에서는 다중 계층 시스템의 중간 집계기와 같이 다른 유형의 집계를 수행하거나 서버 또는 클라이언트에서 사용하는 유형과 다른 유형의 데이터 압축/압축 해제를 사용하는 다른 역할이 있을 수 있습니다.

    배치의 개념을 정의하는 기본 목적은 *페더레이션 유형*을 정의하기 위한 기초입니다.

- **페더레이션 유형**(`tff.FederatedType`): 페더레이션 유형의 값은 특정 배치(예: `tff.SERVER` 또는 `tff.CLIENTS`)에서 정의된 시스템 참가자 그룹이 호스팅하는 값입니다. 페더레이션 유형은 *배치* 값(즉, [종속 유형](https://en.wikipedia.org/wiki/Dependent_type)), *멤버 구성 요소*의 유형(각 참가자가 로컬에서 호스팅하는 콘텐츠의 종류) 및 모든 참가자가 로컬로 같은 항목을 호스팅하고 있는지를 지정하는 추가 비트 `all_equal`에 의해 정의됩니다.

    그룹(배치) `G`가 각각 호스팅하는 유형 `T`의 항목(멤버 구성 요소)을 포함하는 페더레이션 유형의 값에 대한 간단한 표기법은 각각 `all_equal` 비트가 설정되었거나 설정되지 않은 `T@G` 또는 `{T}@G`입니다.

    예를 들면, 다음과 같습니다.

    - `{int32}@CLIENTS`는 클라이언트 기기당 하나씩 잠재적으로 다른 정수 세트로 구성된 *페더레이션 값*을 나타냅니다. 네트워크의 여러 위치에 나타나는 여러 데이터 항목을 포함하는 단일 *페더레이션 값*에 대해 이야기하고 있습니다. "네트워크" 차원을 가진 일종의 텐서(tensor)로 생각할 수 있지만, TFF에서 페더레이션 값의 멤버 구성 요소에 대한 [임의 액세스](https://en.wikipedia.org/wiki/Random_access)를 허용하지 않는다는 점에서 이 비유는 완벽하지 않습니다.

    - `{<X=float32,Y=float32>*}@CLIENTS`는 *페더레이션 데이터세트*를 나타내며, 클라이언트 기기당 하나의 시퀀스인 여러 `XY` 좌표 시퀀스로 구성된 값입니다.

    - `<weights=float32[10,5],bias=float32[5]>@SERVER`는 서버에서 가중치 및 바이어스 텐서의 명명된 튜플를 나타냅니다. 중괄호를 삭제했으므로 `all_equal` 비트가 설정되어 있음을 나타냅니다. 즉, (이 값을 호스팅하는 클러스터에 있을 수 있는 서버 복제본의 수와 관계없이) 단일 튜플만 있습니다.

### 구성 요소

Federated Core의 언어는 몇 가지 추가 요소가 포함된 [람다 미적분](https://en.wikipedia.org/wiki/Lambda_calculus)의 형태입니다.

현재 공개 API에 노출되어 있는 다음과 같은 프로그래밍 추상화를 제공합니다.

- **TensorFlow** 계산( `tff.tf_computation`): `tff.tf_computation` 데코레이터를 사용하여 TFF에서 재사용 가능한 컴포넌트로 래핑된 TensorFlow 코드 섹션입니다. 항상 함수형 유형을 가지고 있으며 TensorFlow의 함수와 달리 구조적 매개변수를 취하거나 시퀀스 유형의 구조적 결과를 반환할 수 있습니다.

    다음은 `tf.data.Dataset.reduce` 연산자를 사용하여 정수의 합계를 계산하는 유형 `(int32* -> int)`의 TF 계산 예입니다.

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

- **intrinsics(내장 기능)** 또는 *페더레이션 연산자* (`tff.federated_...`): 대부분의 FC API를 구성하는`tff.federated_sum` 또는 `tff.federated_broadcast`와 같은 함수 라이브러리이며, 대부분 TFF와 함께 사용되는 분산 통신 연산자를 나타냅니다.

    이들 연산자를 *intrinsics*라고 부르는 이유는 [내장 함수](https://en.wikipedia.org/wiki/Intrinsic_function)와 다소 비슷하지만, TFF가 이해하고 하위 레벨 코드로 컴파일되는 확장 가능한 개방형 연산자 세트이기 때문입니다.

    이들 연산자의 대부분은 페더레이션 유형의 매개변수와 결과를 가지며, 대부분은 다양한 종류의 데이터에 적용할 수 있는 템플릿입니다.

    예를 들어, `tff.federated_broadcast`는 함수 유형 `T@SERVER -> T@CLIENTS`의 템플릿 연산자로 생각할 수 있습니다.

- **람다 식**( `tff.federated_computation`): TFF의 람다 식은 Python의 `lambda` 또는 `def`와 같습니다. 매개변수 이름과 이 매개변수에 대한 참조를 포함하는 본문(표현식)으로 구성됩니다.

    Python 코드에서는 `tff.federated_computation`로 Pyhon 함수를 데코레이팅하고 인수를 정의하여 만들 수 있습니다.

    다음은 앞서 언급한 람다 식의 예입니다.

    ```python
    @tff.federated_computation(tff.type_at_clients(tf.float32))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

- **배치 리터럴**: 현재는 간단한 클라이언트-서버 계산을 정의할 수 있는 `tff.SERVER` 및 `tff.CLIENTS`만 있습니다.

- **함수 호출**(`__call__`): 함수 유형을 가진 모든 것은 표준 Python `__call__` 구문을 사용하여 호출할 수 있습니다. 호출은 표현식이며, 유형은 호출된 함수의 결과와 같은 유형입니다.

    예를 들면, 다음과 같습니다.

    - `add_up_integers(x)`는 인수 `x`에서 앞서 정의된 TensorFlow 계산의 호출을 나타냅니다. 이 표현식의 유형은 `int32`입니다.

    - `tff.federated_mean(sensor_readings)`은 `sensor_readings`에서 페더레이션 평균화 연산자의 호출을 나타냅니다. 이 표현식의 유형은 `float32@SERVER`(위 예제의 컨텍스트를 가정함)입니다.

- **튜플** 형성 및 요소 **선택**: `tff.federated_computation`로 데코레이팅된 함수의 본문에 나타나는 형식 `[x, y]`, `x[y]` 또는 `x.y`의 Python 표현식입니다.

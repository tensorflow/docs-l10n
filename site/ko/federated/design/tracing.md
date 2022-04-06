# 추적

[TOC]

Python 함수에서 [AST](compilation.md#ast)를 구성하는 프로세스를 추적합니다.

TODO(b/153500547): 추적 시스템의 개별 구성 요소를 설명하고 연결합니다.

## 페더레이션 계산 추적하기

상위 수준에서 페더레이션 계산을 추적하는 3가지 구성 요소가 있습니다.

### 인수 압축하기

내부적으로 TFF 계산에는 0 또는 1개의 인수만 있습니다. [computations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) 데코레이터에 제공되는 인수는 TFF 연산에 대한 인수의 유형 서명을 설명합니다. TFF는 이 정보를 사용하여 Python 함수의 인수를 단일 [structure.Struct](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/common_libs/structure.py)로 압축하는 방법을 결정합니다.

참고: `Struct`를 단일 데이터 구조로 사용하여 Python `args`와 `kwargs`를 나타내는 것은 `Struct`에서 명명된 필드와 명명되지 않은 필드를 모두 허용하는 이유입니다.

자세한 내용은 [function_utils.create_argument_unpacking_fn](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/computation/function_utils.py)을 참조하세요.

### 함수 추적하기

`federated_computation`을 추적할 때 사용자의 함수는 [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py)를 각 인수에 대한 대리 대체로 사용하여 호출됩니다. `Value`는 일반적인 Python dunder 메서드(예: `__ getattr__`)를 구현하여 원래 인수 형식의 동작을 에뮬레이션하려고 합니다.

구체적으로, 정확히 하나의 인수가 있을 때 추적은 다음과 같이 수행됩니다.

1. 인수를 나타내는 적절한 형식 서명을 사용하여 [building_blocks.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py)에서 지원되는 [value_impl.ValueImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)를 생성합니다.

2. `ValueImpl`에 대한 함수를 호출합니다. 이로 인해 Python 런타임이 ValueImpl에 의해 구현된 `ValueImpl` 메서드를 호출하여 dunder 메서드를 AST 구성으로 변환합니다. 각 dunder 메서드는 AST를 구성하고 해당 AST가 지원하는 `ValueImpl`을 반환합니다.

예를 들면:

```python
def foo(x):
  return x[0]
```

여기서 함수의 매개변수는 튜플이고, 함수 본문에서 0번째 요소가 선택됩니다. 그러면 `ValueImpl`에서 재정의된 Python의 `__getitem__` 메서드가 호출됩니다. 가장 간단한 경우, `ValueImpl.__getitem__`의 구현은 <code>__getitem__</code>의 호출을 나타내는 <a>building_blocks.Selection</a>을 구성하고 새로운 `Selection`에서 지원되는 `ValueImpl`을 반환합니다.

각 dunder 메서드가 `ValueImpl`을 반환하여 재정의된 dunder 메서드 중 하나를 호출하는 함수의 본문에서 모든 연산을 스탬프 처리하므로 추적이 계속됩니다.

### AST 생성하기

함수를 추적한 결과는 [building_blocks.Lambda](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)로 패키징되며, `parameter_name` 및 `parameter_type`이 압축된 인수를 나타내기 위해 생성된 [building_block.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py)에 매핑됩니다. 결과 `Lambda`는 사용자의 Python 함수를 완전히 나타내는 Python 객체로 반환됩니다.

## TensorFlow 계산 추적하기

TODO(b/153500547): TensorFlow 계산을 추적하는 프로세스를 설명합니다.

## 추적 중 예외에서 오류 메시지 정리하기

TFF 히스토리의 어느 한 지점에서, 사용자의 계산을 추적하는 프로세스에는 사용자의 함수를 호출하기 전에 여러 래퍼 함수를 통과하는 과정이 포함되었습니다. 이것은 다음과 같은 오류 메시지를 생성하는 바람직하지 않은 영향을 미쳤습니다.

```
Traceback (most recent call last):
  File "<user code>.py", in user_function
    @tff.federated_computation(...)
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<user code>", in user_function
    <some line of user code inside the federated_computation>
  File "<tff code>.py", tff_function
  ...
  File "<tff code>.py", tff_function
    <raise some error about something the user did wrong>
FederatedComputationWrapperTest.test_stackframes_in_errors.<locals>.DummyError
```

이 트레이스백에서 사용자 코드(실제로 버그가 포함된 줄)의 최종 줄을 찾는 것은 매우 어렵습니다. 이로 인해 사용자가 이러한 문제를 TFF 버그로 보고해야 하는 불편이 있었습니다.

오늘날 TFF에는 이러한 호출 스택에 추가 TFF 함수가 없는지 확인해야 하는 수고로움이 있습니다. 이것이 TFF의 추적 코드에서 생성기를 사용하는 이유이며, 종종 다음과 같은 패턴으로 사용합니다.

```
# Instead of writing this:
def foo(fn, x):
  return 5 + fn(x + 1)

print(foo(user_fn, 20))

# TFF uses this pattern for its tracing code:
def foo(x):
  result = yield x + 1
  yield result + 5

fooer = foo(20)
arg = next(fooer)
result = fooer.send(user_fn(arg))
print(result)
```

이 패턴을 사용하면 사용자의 코드(위의 `user_fn`)를 호출 스택의 최상위 수준에서 호출할 수 있으며 인수, 출력 및 스레드-로컬 컨텍스트까지도 래핑 함수로 조작할 수 있습니다.

이 패턴의 일부 간단한 버전은 "이전" 및 "이후" 함수로 더 간단하게 대체할 수 있습니다. 예를 들어, 위의 `foo`는 다음으로 대체될 수 있습니다.

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

이 패턴은 "이전"과 "이후" 부분 간에 공유 상태가 필요하지 않은 경우에 적합합니다. 그러나 복잡한 상태 또는 컨텍스트 관리자와 관련된 더 복잡한 경우는 다음과 같이 표현하기가 번거로울 수 있습니다.

```
# With the `yield` pattern:
def in_ctx(fn):
  with create_ctx():
    yield
    ... something in the context ...
  ...something after the context...
  yield

# WIth the `before` and `after` pattern:
def before():
  new_ctx = create_ctx()
  new_ctx.__enter__()
  return new_ctx

def after(ctx):
  ...something in the context...
  ctx.__exit__()
  ...something after the context...
```

후자의 예에서는 어떤 코드가 컨텍스트 내에서 실행되는지가 훨씬 명확하지 않으며, 이전 섹션과 이후 섹션에서 더 많은 상태 비트가 공유되면 상황이 훨씬 불명확해집니다.

"사용자 오류 메시지에서 TFF 함수 숨기기"라는 일반적인 문제에 대한 몇 가지 다른 솔루션이 시도되었습니다. 구체적으로, 예외를 포착하고 다시 발생시키기(스택에 사용자 코드를 호출한 코드도 포함하지 않고 가장 낮은 수준의 사용자 코드만 포함하는 예외를 생성할 수 없기 때문에 실패함), 예외를 포착하고 해당 트레이스백을 필터링된 트레이스백으로 교체하기(이는 CPython 전용이며 Python 언어에서 지원되지 않음) 및 예외 처리기 교체하기(`sys.excepthook`이 `absltest`에서 사용되지 않고 다른 프레임워크에 의해 재정의되었기 때문에 실패함) 등과 같은 시도가 있었습니다. 결국, 생성기 기반의 제어 반전을 통해 TFF 구현에서 약간의 복잡성에도 불구하고 최상의 최종 사용자 경험을 제공할 수 있었습니다.

# Pytype

[Pytype](https://github.com/google/pytype)은 Python 코드의 유형을 확인하고 추론하는 Python용 정적 분석기입니다.

## 이점과 해결 과제

Pytype을 사용하면 많은 이점이 있습니다. 자세한 내용은 https://github.com/google/pytype을 참조하세요. 그러나 Pytype에서 형식 주석을 해석하고 Pytype에 의해 오류가 발생하는 방식은 때로 TensorFlow Federated 가독성에 불편을 초래합니다.

- 데코레이터

Pytype은 주석을 달고 있는 함수에 견주어 주석을 확인합니다. 해당 함수가 데코레이션되면 이러한 동일한 주석이 더 이상 적용되지 않을 수 있는 새 함수가 생성됩니다. TensorFlow와 TensorFlow Federated는 모두 데코레이션된 함수의 입력과 출력을 극적으로 변환하는 데코레이터를 사용합니다. 즉, `@tf.function`, `@tff.tf_computation` 또는 `@tff.federated_computation`으로 데코레이션된 함수는 pytype으로 분석할 때 예상치 못하게 작동할 수 있습니다.

예를 들어:

```
def decorator(fn):

  def wrapper():
    fn()
    return 10  # Anything decorated with this decorator will return a `10`.

  return wrapper


@decorator
def foo() -> str:
  return 'string'


@decorator
def bar() -> int:  # However, this annotation is incorrect.
  return 'string'
```

`foo` 및 `bar` 함수의 반환 형식은 `str`이어야 하는데, 이러한 함수는 문자열을 반환하기 때문이고 여기서 함수가 데코레이션되었는지 여부는 관계가 없습니다.

Python 데코레이터에 대한 자세한 내용은 https://www.python.org/dev/peps/pep-0318/을 참조하세요.

- `getattr()`

Pytype은 속성이 [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 함수를 사용하여 제공되는 클래스를 구문 분석하는 방법을 모릅니다. TensorFlow Federated는 `tff.Struct`, `tff.Value` 및 `tff.StructType`과 같은 클래스에서 `getattr()`을 사용하며 이러한 클래스는 Pytype에서 올바르게 분석되지 않습니다.

- 패턴 매칭

Pytype은 Python 3.10 이전 버전에서 패턴 매칭을 잘 처리하지 못합니다. TensorFlow Federated는 성능상의 이유로 사용자 정의 형식 가드(즉, [`isinstance`](https://docs.python.org/3/library/functions.html#isinstance) 이외의 형식 가드)를 많이 사용하며 Pytype은 이러한 형식 가드를 해석할 수 없습니다. 이 문제는 `typing.cast`를 삽입하거나 로컬에서 Pytype을 비활성화하여 해결할 수 있습니다. 하지만 사용자 정의 형식 가드의 사용은 TensorFlow Federated의 일부에서 널리 퍼져 있어 이 두 가지 해결책 모두 Python 코드를 읽기 어렵게 만듭니다.

참고: Python 3.10에는 [사용자 정의 형식 가드](https://www.python.org/dev/peps/pep-0647/)에 대한 지원이 추가되어 Python 3.10 이상에서는 이 문제를 해결할 수 있습니다. 즉, 이 버전이 TensorFlow Federated가 지원되는 최소 Python 버전입니다.

## TensorFlow Federated에서 Pytype 사용

TensorFlow Federated는 Python 주석과 Pytype 분석기를 사용**합니다**. 그러나 *때로는* Python 주석을 사용하지 않거나 Pytype을 비활성화하는 것이 도움이 됩니다. 로컬에서 Pytype을 비활성화할 경우 Python 코드를 읽기가 더 어려워지면 [특정 파일에 대한 모든 pytype 검사를 비활성화](https://google.github.io/pytype/faq.html#how-do-i-disable-all-pytype-checks-for-a-particular-file)하는 것이 좋습니다.

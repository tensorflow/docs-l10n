# TensorFlow 코드 스타일 가이드

## Python 스타일

TensorFlow가 4개 대신 2개의 공백을 사용하는 것을 제외하고 [PEP 8 Python 스타일 가이드](https://www.python.org/dev/peps/pep-0008/)를 따르세요. [Google Python 스타일 가이드](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)를 따르고 [pylint](https://www.pylint.org/)를 사용하여 Python 변경 사항을 확인하세요.

### pylint

`pylint`를 설치하려면:

```bash
$ pip install pylint
```

TensorFlow 소스 코드 루트 디렉터리에서 `pylint`로 파일을 확인하려면:

```bash
$ pylint --rcfile=tensorflow/tools/ci_build/pylintrc tensorflow/python/keras/losses.py
```

### 지원되는 Python 버전

지원되는 Python 버전은 TensorFlow [설치 가이드](https://www.tensorflow.org/install)를 참조하세요.

공식 및 커뮤니티 지원 빌드에 대해서는 TensorFlow [연속 빌드 상태](https://github.com/tensorflow/tensorflow/blob/master/README.md#continuous-build-status)를 참조하세요.

## C++ 코딩 스타일

TensorFlow C++ 코드에 대한 변경 사항은 [Google C++ 스타일 가이드](https://google.github.io/styleguide/cppguide.html) 및 [TensorFlow 특정 스타일 세부 정보](https://github.com/tensorflow/community/blob/master/governance/cpp-style.md)를 준수해야 합니다. `clang-format`을 사용하여 C/C++ 변경 사항을 확인하세요.

Ubuntu 16+에 설치하려면 다음을 수행합니다.

```bash
$ apt-get install -y clang-format
```

다음을 사용하여 C/C++ 파일의 형식을 확인할 수 있습니다.

```bash
$ clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
$ diff <my_cc_file> /tmp/my_cc_file.cc
```

## 다른 언어

- [Google Java 스타일 가이드](https://google.github.io/styleguide/javaguide.html)
- [Google JavaScript 스타일 가이드](https://google.github.io/styleguide/jsguide.html)
- [Google Shell 스타일 가이드](https://google.github.io/styleguide/shell.xml)
- [Google Objective-C 스타일 가이드](https://google.github.io/styleguide/objcguide.html)

## TensorFlow 규칙 및 특수한 사용

### Python 연산

TensorFlow *연산*은 주어진 입력 텐서가 출력 텐서를 반환하거나 그래프를 빌드할 때 그래프에 op를 추가하는 함수입니다.

- 첫 번째 인수는 텐서여야 하며 그 뒤에 기본 Python 매개변수가 있어야 합니다. 마지막 인수는 기본값이 `None`인 `name`입니다.
- 텐서 인수는 단일 텐서이거나 반복 가능한 텐서여야 합니다. 즉, "텐서 또는 텐서 목록"이 너무 광범위합니다. `assert_proper_iterable`을 참조하세요.
- 텐서를 인수로 사용하는 연산은 C++ 연산을 사용하는 경우 텐서가 아닌 입력을 텐서로 변환하기 위해 `convert_to_tensor`를 호출해야 합니다. 인수는 여전히 설명서에서 특정 dtype의 `Tensor` 객체로 설명됩니다.
- 각 Python 연산에는 `name_scope`가 있어야 합니다. 아래와 같이 op의 이름을 문자열로 전달합니다.
- 연산에는 각 값의 유형과 의미를 모두 설명하는 Args 및 Returns 선언을 포함한 광범위한 Python 주석이 포함되어야 합니다. 설명에서 가능한 형상, dtype 또는 순위가 지정되어야 합니다. 자세한 내용은 설명서를 참조하세요.
- 사용 편의성을 높이려면 예제 섹션에 op의 입력/출력 사용 예를 포함합니다.
- `tf.Tensor.eval` 또는 `tf.Session.run`을 명시적으로 사용하지 마세요. 예를 들어, Tensor 값에 의존하는 로직을 작성하려면 TensorFlow 제어 플로우를 사용합니다. 또는 즉시 실행이 활성화된 경우에만 연산이 실행되도록 제한합니다(`tf.executing_eagerly()`).

예:

```python
def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
          output_collections=(), name=None):
  """My operation that adds two tensors with given coefficients.

  Args:
    tensor_in: `Tensor`, input tensor.
    other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
    my_param: `float`, coefficient for `tensor_in`.
    other_param: `float`, coefficient for `other_tensor_in`.
    output_collections: `tuple` of `string`s, name of the collection to
                        collect result of this op.
    name: `string`, name of the operation.

  Returns:
    `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

  Example:
    >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
              output_collections=['MY_OPS'], name='add_t1t2')
    [2.3, 3.4]
  """
  with tf.name_scope(name or "my_op"):
    tensor_in = tf.convert_to_tensor(tensor_in)
    other_tensor_in = tf.convert_to_tensor(other_tensor_in)
    result = my_param * tensor_in + other_param * other_tensor_in
    tf.add_to_collection(output_collections, result)
    return result
```

사용법:

```python
output = my_op(t1, t2, my_param=0.5, other_param=0.6,
               output_collections=['MY_OPS'], name='add_t1t2')
```

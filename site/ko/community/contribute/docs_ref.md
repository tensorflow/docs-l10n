# TensorFlow API 설명서에 기여하기

## 테스트 가능한 docstring

TensorFlow는 [DocTest](https://docs.python.org/3/library/doctest.html)를 사용하여 Python docstring에서 코드 조각을 테스트합니다. 이 조각은 실행 가능한 Python 코드여야 합니다. 테스트가 가능하도록 `>>>`(세 개의 왼쪽 꺾쇠 괄호)가 들어간 줄을 추가하세요. 예를 들어, 다음은 [array_ops.py](https://www.tensorflow.org/code/tensorflow/python/ops/array_ops.py) 소스 파일의 `tf.concat` 함수에서 발췌한 내용입니다.

```
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.
  ...

  >>> t1 = [[1, 2, 3], [4, 5, 6]]
  >>> t2 = [[7, 8, 9], [10, 11, 12]]
  >>> concat([t1, t2], 0)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12]], dtype=int32)>

  <... more description or code snippets ...>

  Args:
    values: A list of `tf.Tensor` objects or a single `tf.Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

    Returns:
      A `tf.Tensor` resulting from concatenation of the input tensors.
  """

  <code here>
```

참고: TensorFlow DocTest는 TensorFlow 2 및 Python 3을 사용합니다.

### DocTest로 코드를 테스트 가능하게 만들기

현재, 많은 docstring에서 백틱 (```)을 사용하여 코드를 식별합니다. DocTest로 코드를 테스트할 수 있게 하려면 다음과 같이 합니다.

- 백틱(```)을 제거하고 각 줄 앞에 왼쪽 꺾쇠 괄호(>>>)를 사용합니다. 연속된 줄 앞에 (...)를 사용합니다.
- tensorflow.org에서 올바르게 렌더링하려면 Markdown 텍스트와 DocTest 조각을 구분하는 새로운 줄을 추가합니다.

### 사용자 정의

TensorFlow는 내장된 doctest 로직에 몇 가지 사용자 정의를 사용합니다.

- 부동 소수점 값을 텍스트로 비교하지 않습니다. 부동 소수점 값은 *liberal `atol` 및 `rtol` tolerences*와 함께 `allclose`를 사용하여 텍스트에서 추출하고 비교합니다. 그러면 다음과 같은 장점이 있습니다.
    - 더 명확한 문서 - 작성자가 소수 자릿수를 모두 포함할 필요가 없습니다.
    - 보다 강력한 테스트 - 기본 구현의 수치적 변경으로 인해 doctest가 실패하지 않습니다.
- 작성자가 줄에 대한 출력을 포함하는 경우에만 출력을 확인합니다. 따라서 작성자는 일반적으로 인쇄되지 않도록 관련 없는 중간 값을 캡처할 필요가 없으므로 보다 명확한 문서를 작성할 수 있습니다.

### Docstring 고려 사항

- *전체*: doctest의 목표는 문서를 제공하고 문서가 작동하는지 확인하는 것입니다. 이것은 단위 테스트와 다릅니다. 따라서 다음이 권장됩니다.

    - 예제를 간단하게 유지합니다.
    - 길거나 복잡한 출력을 피합니다.
    - 가능하면 올림 숫자를 사용합니다.

- *출력 형식*: 조각의 출력은 출력을 생성하는 코드 바로 아래에 있어야 합니다. 또한 docstring의 출력은 코드가 실행된 후의 출력과 정확히 같아야 합니다. 위의 예를 참조하세요. 또한 DocTest 설명서에서 [이 부분](https://docs.python.org/3/library/doctest.html#warnings)을 확인하세요. 출력이 80줄 제한을 초과하면 추가 출력을 새 줄에 넣을 수 있으며 DocTest는 이를 인식합니다. 예를 들어 아래의 여러 줄 블록을 참조하세요.

- *글로벌*: <code><code data-md-type="codespan">tf</code></code>, `np` 및 `os` 모듈은 TensorFlow의 DocTest에서 항상 사용할 수 있습니다.

- *기호 사용*: DocTest에서 같은 파일에 정의된 기호에 직접 액세스할 수 있습니다. 현재 파일에 정의되지 않은 기호를 사용하려면 `xxx` 대신 TensorFlow의 공용 API `tf.xxx`를 사용합니다. 아래 예에서 볼 수 있는 바와 같이 <code>random.normal</code>은 <code>tf.random.normal</code>을 통해 액세스할 수 있습니다. 그 이유는 <code>random.normal</code>이 `NewLayer`에서 보이지 않기 때문입니다.

    ```
    def NewLayer():
      “””This layer does cool stuff.

      Example usage:

      >>> x = tf.random.normal((1, 28, 28, 3))
      >>> new_layer = NewLayer(x)
      >>> new_layer
      <tf.Tensor: shape=(1, 14, 14, 3), dtype=int32, numpy=...>
      “””
    ```

- *부동 소수점 값*: TensorFlow doctest는 결과 문자열에서 부동 소수점 값을 추출하고 합리적인 허용 오차(`atol=1e-6` , `rtol=1e-6`)로 `np.allclose`를 사용하여 비교를 수행합니다. 이런 식으로 작성자는 docstring이 지나치게 정밀해 수치 문제로 실패가 발생하는 상황을 걱정할 필요가 없습니다. 간단히 예상 값을 붙여넣기만 하면 됩니다.

- *비결정적 출력*: 불확실한 부분에 생략 부호(`...`)를 사용하면 DocTest가 해당 하위 문자열을 무시합니다.

    ```
    >>> x = tf.random.normal((1,))
    >>> print(x)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=..., dtype=float32)>
    ```

- *여러 줄 블록*: DocTest는 한 줄과 여러 줄이 있는 문의 차이에 대해 엄격합니다. 아래의 (...) 사용법에 유의하세요.

    ```
    >>> if x > 0:
    ...   print("X is positive")
    >>> model.compile(
    ...   loss="mse",
    ...   optimizer="adam")
    ```

- *예외*: 발생한 예외를 제외하고 예외 세부 사항은 무시됩니다. 자세한 내용은 [이 내용](https://docs.python.org/3/library/doctest.html#doctest.IGNORE_EXCEPTION_DETAIL)을 참조하세요.

    ```
    >>> np_var = np.array([1, 2])
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
    ```

### 로컬 머신에서 테스트하기

docstring에서 코드를 로컬로 테스트하는 방법에는 두 가지가 있습니다.

- 클래스/함수/메서드의 docstring만 변경하는 경우 해당 파일의 경로를 [tf_doctest.py](https://www.tensorflow.org/code/tensorflow/tools/docs/tf_doctest.py)에 전달하여 테스트할 수 있습니다. 예를 들면 다음과 같습니다.

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">python tf_doctest.py --file=</code>
    </pre>

    설치된 버전의 TensorFlow를 사용하여 실행이 이루어집니다. 다음과 같이 테스트 중인 코드와 같은 코드가 실행되도록 합니다.

    - 최신 [tf-nightly](https://pypi.org/project/tf-nightly/) `pip install -U tf-nightly`를 사용합니다.
    - 풀 요청의 기반을 [TensorFlow](https://github.com/tensorflow/tensorflow) 마스터 분기의 최근 풀로 재지정합니다.

- 클래스/함수/메서드의 코드와 docstring을 변경하는 경우, [소스에서 TensorFlow를 빌드](../../install/source.md)해야 합니다. 소스에서 빌드할 준비가 되면 테스트를 실행할 수 있습니다.

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest</code>
    </pre>

    또는

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest -- --module=ops.array_ops</code>
    </pre>

    `--module`은 `tensorflow.python`에 상대적입니다.

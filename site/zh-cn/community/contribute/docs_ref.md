# 为 TensorFlow API 文档做贡献

## 可测试的文档字符串

TensorFlow 使用 [DocTest](https://docs.python.org/3/library/doctest.html) 来测试 Python 文档字符串中的代码段。代码段必须是可执行的 Python 代码。要启用测试，请在代码行前添加 `>>>`（三个左尖括号）。例如，下面的代码摘自 [array_ops.py](https://www.tensorflow.org/code/tensorflow/python/ops/array_ops.py) 源文件中的 `tf.concat` 函数：

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

注：TensorFlow DocTest 使用 TensorFlow 2 和 Python 3。

### 使用 DocTest 让代码可测试

目前，许多文档字符串都使用反引号 (```) 来标识代码。要使用 DocTest 使代码可测试，请执行以下操作：

- 删除反引号 (```) ，并在每行前使用左括号 (>>>)。在续行前使用 (...)。
- 添加换行符以将 DocTest 代码段与 Markdown 文本分开，从而在 tensorflow.org 上正确呈现。

### 自定义设置

TensorFlow 对内置的 doctest 逻辑使用了一些自定义设置：

- 它不会将浮点值作为文本来比较：浮点值是从文本中提取的，将使用具有*自由 `atol` 和 `rtol` 容差*的 `allclose` 进行比较。这样可以实现：
    - 更清晰的文档 - 作者无需包括所有小数位。
    - 更可靠的测试 - 底层实现中的数值更改永远不会导致 doctest 失败。
- 如果作者包括一行的输出，则仅会检查该输出。这可以使文档更加清晰，因为作者通常不需要捕获无关的中间值来防止它们被打印。

### 文档字符串注意事项

- *总体*：doctest 的目标是提供文档，并确认该文档有效。这与单元测试不同。因此：

    - 确保示例简单。
    - 避免较长或复杂的输出。
    - 如果可能，请使用整数。

- *输出格式*：代码段的输出需要直接位于生成输出的代码下方。另外，文档字符串中的输出必须与执行代码后的输出完全相同。参见上面的示例。另请参阅 DocTest 文档中的[此部分](https://docs.python.org/3/library/doctest.html#warnings)。如果输出超过 80 行的限制，则可以将多余的输出放在新行上，DocTest 会识别出来。例如，查看下面的多行块。

- *全局*：<code><code data-md-type="codespan">tf</code></code>、`np` 和 `os` 模块始终在 TensorFlow 的 DocTest 中可用。

- *使用符号*：在 DocTest 中，您可以直接访问在同一文件中定义的符号。要使用当前文件中未定义的符号，请使用 TensorFlow 的公共 API `tf.xxx` 而不是 `xxx`。如下面的示例中所示，可以通过 <code>tf.random.normal</code> 访问 <code>random.normal</code>。这是因为 <code>random.normal</code> 在 `NewLayer` 中不可见。

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

- *浮点值*：TensorFlow doctest 从结果字符串中提取浮点值，并使用具有合理容差 (`atol=1e-6`, `rtol=1e-6`) 的 `np.allclose` 进行比较。这样，作者便无需担心过于精确的文档字符串会由于数字问题而导致失败。只需粘贴期望值即可。

- *非确定性输出*：对不确定的部分使用省略号 (`...`)，DocTest 将忽略该子字符串。

    ```
    >>> x = tf.random.normal((1,))
    >>> print(x)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=..., dtype=float32)>
    ```

- *多行块*：DocTest 对于单行和多行语句之间的区别非常严格。请注意下面 (...) 的用法：

    ```
    >>> if x > 0:
    ...   print("X is positive")
    >>> model.compile(
    ...   loss="mse",
    ...   optimizer="adam")
    ```

- *异常*：除非是引发的异常，否则将忽略异常详细信息。有关更多详细信息，请参阅[此部分](https://docs.python.org/3/library/doctest.html#doctest.IGNORE_EXCEPTION_DETAIL)。

    ```
    >>> np_var = np.array([1, 2])
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
    ```

### 在本地计算机上测试

有两种方式可以在本地测试文档字符串中的代码：

- 如果仅更改类/函数/方法的文档字符串，则可以通过将该文件的路径传递到 [tf_doctest.py](https://www.tensorflow.org/code/tensorflow/tools/docs/tf_doctest.py) 来进行测试。例如：

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">python tf_doctest.py --file=</code>
    </pre>

    将使用您安装的 TensorFlow 版本来运行此代码。为了确保您正在运行与测试相同的代码：

    - 使用最新的 [tf-nightly](https://pypi.org/project/tf-nightly/) `pip install -U tf-nightly`
    - 将您的拉取请求衍合到 [TensorFlow](https://github.com/tensorflow/tensorflow) master 分支的最新拉取上。

- 如果要更改类/函数/方法的代码和文档字符串，则需要[从源代码构建 TensorFlow](../../install/source.md)。一旦设置为从源代码构建，就可以运行测试：

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest</code>
    </pre>

    或

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest -- --module=ops.array_ops</code>
    </pre>

    `--module` 与 `tensorflow.python` 相关。

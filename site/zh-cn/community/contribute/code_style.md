# TensorFlow 代码样式指南

## Python 样式

遵循 [PEP 8 Python 样式指南](https://www.python.org/dev/peps/pep-0008/)，但 TensorFlow 使用 2 个空格而不是 4 个空格。请遵循 [Google Python 样式指南](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)，并使用 [pylint](https://www.pylint.org/) 检查您的 Python 更改。

### pylint

要安装 `pylint`，请运行以下代码：

```bash
$ pip install pylint
```

要使用 TensorFlow 源代码根目录中的 `pylint` 检查文件，请运行以下代码：

```bash
$ pylint --rcfile=tensorflow/tools/ci_build/pylintrc tensorflow/python/keras/losses.py
```

### 支持的 Python 版本

有关支持的 Python 版本，请参阅 TensorFlow [安装指南](https://www.tensorflow.org/install)。

有关官方和社区支持的构建，请参阅 TensorFlow [持续构建状态](https://github.com/tensorflow/tensorflow/blob/master/README.md#continuous-build-status)。

## C++ 编码样式

对 TensorFlow C++ 代码的变更应符合 [Google C++ 样式指南](https://google.github.io/styleguide/cppguide.html)和 [TensorFlow 特定样式详细信息](https://github.com/tensorflow/community/blob/master/governance/cpp-style.md)。使用 `clang-format` 检查您的 C/C++ 变更。

要在 Ubuntu 16+ 上安装，请运行以下命令：

```bash
$ apt-get install -y clang-format
```

您可以使用以下命令检查 C/C++ 文件的格式：

```bash
$ clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
$ diff <my_cc_file> /tmp/my_cc_file.cc
```

## 其他语言

- [Google Java 样式指南](https://google.github.io/styleguide/javaguide.html)
- [Google JavaScript 样式指南](https://google.github.io/styleguide/jsguide.html)
- [Google Shell 样式指南](https://google.github.io/styleguide/shell.xml)
- [Google Objective-C 样式指南](https://google.github.io/styleguide/objcguide.html)

## TensorFlow 惯例和特殊用法

### Python 运算

TensorFlow *运算*是一种给定输入张量、返回输出张量（或在构建计算图时向计算图添加运算）的函数。

- 第一个参数应当是张量，然后是基本的 Python 参数。最后一个参数是默认值为 `None` 的 `name`。
- 张量参数应当是单个张量或者多个张量的可迭代对象。也就是说，“张量或张量列表”过于宽泛。请参见 `assert_proper_iterable`。
- 如果使用张量作为参数的运算正在使用 C++ 运算，则应调用 `convert_to_tensor` 将非张量输入转换为张量。请注意，参数在文档中仍被描述为特定 dtype 的 `Tensor` 对象。
- 每个 Python 运算都应具有一个 `name_scope`。如下所示，以字符串形式传递运算的名称。
- 运算应包含带参数和返回声明的大量 Python 注释，这些注释说明了每个值的类型和含义。应在说明中指定可能的形状、dtype 或秩。请参阅文档详细信息。
- 为提高可用性，“示例”部分中包括带运算的输入/输出的用法示例。
- 避免显式使用 `tf.Tensor.eval` 或 `tf.Session.run`。例如，要编写依赖于张量值的逻辑，请使用 TensorFlow 控制流。或者，将运算限制为仅在启用 Eager Execution 时 (`tf.executing_eagerly()`) 才运行。

示例：

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

用法：

```python
output = my_op(t1, t2, my_param=0.5, other_param=0.6,
               output_collections=['MY_OPS'], name='add_t1t2')
```

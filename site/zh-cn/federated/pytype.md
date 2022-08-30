# Pytype

[Pytype](https://github.com/google/pytype) 是一个针对 Python 的静态分析器，用于检查和推断 Python 代码的类型。

## 优势与挑战

使用 Pytype 有很多优点，有关详细信息，请参阅 https://github.com/google/pytype for more information。然而，对于 TensorFlow Federated 可读性来说，Pytype 解释类型批注的方式以及由 Pytype 产生的错误有时不太方便。

- 修饰器

Pytype 会根据它们所注解的函数检查注解；如果函数被修饰，则会创建一个新函数，其中可能不再应用相同的注解。TensorFlow 和 TensorFlow Federated 都使用修饰器，这些修饰器可以显著转换所修饰函数的输入和输出；这意味着使用 `@tf.function`、`@tff.tf_computation` 或 `@tff.federated_computation` 修饰的函数在使用 Pytype 时可能会表现出意外行为。

例如：

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

函数 `foo` 和 `bar` 的返回类型应为 `str`，因为这些函数会返回一个字符串，无论这些函数是否经过修饰都是如此。

有关 Python 修饰器的详细信息，请参阅 https://www.python.org/dev/peps/pep-0318/。

- `getattr()`

Pytype 不知道如何解析使用 [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 函数提供特性的类。TensorFlow Federated 会在 `tff.Struct`、`tff.Value` 和 `tff.StructType` 等类中使用`getattr()`，这些类无法被 Pytype 正确分析。

- 模式匹配

在 Python3.10 之前，Pytype 无法很好地处理模式匹配。出于性能原因，TensorFlow Federated 大量使用用户定义的类型保护（即，除 [`isinstance`](https://docs.python.org/3/library/functions.html#isinstance) 之外的类型保护），而 Pytype 无法解释这些类型保护。这可以通过插入 `typing.cast` 或在本地禁用 Pytype 来解决；然而，由于使用用户定义的类型保护在 TensorFlow Federated 的某些部分非常普遍，这两种解决方式最终都会使 Python 代码更难阅读。

注：Python3.10 添加了对[用户定义的类型保护](https://www.python.org/dev/peps/pep-0647/)的支持，因此，这个问题可以在 Python 3.10 成为 Python TensorFlow Federated 支持的最低版本之后解决。

## Pytype 在 TensorFlow Federated 中的使用

TensorFlow Federated **确实** 使用 Python 注解和 Pytype 分析器。但是，不使用 Python 注解或禁用 Pytype *有时*会很有帮助。如果在本地禁用 Pytype 会使 Python 代码更难阅读，则最好[对某个特定文件禁用所有 Pytype](https://google.github.io/pytype/faq.html#how-do-i-disable-all-pytype-checks-for-a-particular-file)。

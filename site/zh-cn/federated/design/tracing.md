# 跟踪

[TOC]

跟踪从 Python 函数构造 [AST](compilation.md#ast) 的过程。

TODO(b/153500547)：描述并链接跟踪系统的各个组件。

## 跟踪联合计算

从较高层面而言，跟踪联合计算包含三个部分。

### 打包参数

在内部，TFF 计算拥有零个或仅有一个参数。提供给 [computations.federated_computation](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/api/computations.py) 装饰器的参数描述了 TFF 计算的参数的类型签名。TFF 使用此信息来确定如何将 Python 函数的参数打包到单个 [structure.Struct](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/common_libs/structure.py) 中。

注：由于使用 `Struct` 作为用于表示 Python `args` 和 `kwargs` 的单一数据结构，因此 `Struct` 也同时接受命名字段和未命名字段。

有关详细信息，请参阅 [function_utils.wrap_as_zero_or_one_arg_callable](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/utils/function_utils.py)。

### 跟踪函数

跟踪 `federated_computation` 时，可以将 [value_impl.Value](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/federated_context/value_impl.py) 用作各个参数的替代来调用用户的函数。`Value` 会尝试通过实现常见的 Python dunder 方法（例如 `__getattr__`）来模拟原始参数类型的行为。

具体而言，只有一个参数时，将通过以下方式执行跟踪：

1. 构造由 [building_blocks.Reference](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 支持的 [value_impl.ValueImpl](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/value_impl.py)，使用适当的类型签名表示参数。

2. 在 `ValueImpl` 上调用函数。这样，Python 运行时会调用由 `ValueImpl` 实现的 dunder 方法，这会将那些 dunder 方法转换为 AST 构造。每个 dunder 方法都会构造 AST 并返回该 AST 支持的 `ValueImpl`。

例如：

```python
def foo(x):
  return x[0]
```

在这里，函数的参数为元组，在函数体中选择第 0 个元素。这会调用 Python 的 `__getitem__` 方法，该方法在 `ValueImpl` 上被重写。在最简单的情况下，实现 `ValueImpl.__getitem__` 会构造 [building_blocks.Selection](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 以表示调用 `__getitem__` 并返回由此新的 `Selection` 支持的 `ValueImpl`。

由于每个 dunder 方法都返回一个 `ValueImpl`，而在函数体中每完成一个运算就会调用一个重写的 dunder 方法，因此将会持续跟踪。

### 构造 AST

跟踪该函数的结果会被打包到 [building_blocks.Lambda](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 中，其 `parameter_name` 和 `parameter_type` 会映射至创建的 [building_block.Reference](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py) 以表示打包的参数。随后，会将生成的 `Lambda` 作为能够完全表示用户 Python 函数的 Python 对象返回。

## 跟踪 TensorFlow 计算

TODO(b/153500547)：描述跟踪 TensorFlow 计算的过程。

## 跟踪期间从异常中清除错误消息

在 TFF 历史记录中的某个时刻，跟踪用户计算的过程涉及在调用用户函数之前先传递许多封装容器函数。这带来了产生如下所示错误消息的不良影响：

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

很难在此回溯中找到用户代码的底行（实际包含错误的那一行）。这导致用户将这些问题报告为 TFF 错误，并且通常使用户的操作更加困难。

如今，为了确保这些调用堆栈没有多余的 TFF 函数，TFF 遇到了一些麻烦。这就是在 TFF 的跟踪代码中使用生成器的原因。使用的生成器模式通常如下所示：

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

这种模式允许在调用堆栈的顶层调用用户的代码（上面的 `user_fn`），同时还允许通过封装函数来操作其参数、输出甚至线程局部上下文。

此模式的一些简化版本可以更简单地替换为“before”和“after”函数。例如，上面的 `foo` 可以替换为：

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

如果在“before”和“after”部分之间不需要共享状态，应当首选此模式。但是，更复杂的情况涉及到复杂的状态或上下文管理器，可能难以用这种方式来表达：

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

在后一个示例中，哪些代码在上下文内运行显得不那么清晰，而在“before”和“after”部分间共享更多状态位时，情况会变得更加模糊。

我们尝试了“从用户错误消息中隐藏 TFF 函数”这种一般问题的其他几种解决方案，包括：捕获并重新引发异常（由于无法创建异常而失败，此异常的堆栈仅包含最低级别的用户代码，而不包括调用它的代码）；捕获异常并将其回溯替换为筛选的回溯（筛选的回溯存在侵入性，特定于 CPython，并且不受 Python 语言支持），然后替换异常处理程序（由于 `sys.excepthook` 不由 `absltest` 使用并且会被其他框架重写而失败）。最后，基于生成器的控制反转以一定的 TFF 实现复杂性为代价，提供了最佳的最终用户体验。

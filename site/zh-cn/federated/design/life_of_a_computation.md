# 计算的生命周期

[TOC]

## 在 TFF 中执行 Python 函数

本示例旨在着重展示如何将 Python 函数转换为 TFF 计算以及如何通过 TFF 评估该计算。

**从用户角度：**

```python
tff.backends.native.set_local_python_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1. 编写 *Python* 函数。

2. 使用 `@tff.tf_computation` 装饰 *Python* 函数。

    注：目前的重点仅在于对 Python 函数进行装饰，暂不深究装饰器本身的细节；[下文](#tf-vs-tff-vs-python)将予以详细说明。

3. 设置 TFF [上下文](context.md)。

4. 调用 *Python* 函数。

**从 TFF 角度：**

**解析** Python 时，`@tff.tf_computation` 装饰器将[跟踪](tracing.md) Python 函数并构造 TFF 计算。

**调用**装饰的 Python 函数时，即调用 TFF 计算，TFF 将在设置的[上下文](context.md)中[编译](compilation.md)和[执行](execution.md)计算。

## TF、TFF 与 Python 间的对比

```python
tff.backends.native.set_local_python_execution_context()

@tff.tf_computation(tf.int32)
def add_one(x):
  return x + 1

@tff.federated_computation(tff.type_at_clients(tf.int32))
def add_one_to_all_clients(values):
  return tff.federated_map(add_one, values)

values = [1, 2, 3]
values = add_one_to_all_clients(values)
values = add_one_to_all_clients(values)
>>> [3, 4, 5]
```

TODO(b/153500547)：描述 TF、TFF 与 Python 间对比情况的示例。

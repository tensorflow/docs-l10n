# 上下文

[TOC]

## `Context`

[context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) 或 [context_base.AsyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) 是可以[构造](tracing.md)、[编译](compilation.md)或[执行](execution.md) [AST](compilation.md#ast) 的环境。

此 API 可定义在**不**使用[执行器](execution.md#executor)执行时应使用的**低级抽象**；[参考](backend.md#reference)后端在此级别集成。

### `ExecutionContext`

[execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/execution_context.py) 是使用编译函数来编译计算并使用[执行器](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)来执行计算的 [context_base.SyncContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) 或 [context_base.AsyncContext](execution.md#executor)。

此 API 定义了在使用[执行器](execution.md#executor)执行时应使用的**高级抽象**；[原生](backend.md#native)后端在此级别集成。

### `FederatedComputationContext`

[federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) 是用于构造联合计算的上下文。此上下文用于跟踪使用 [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) 装饰器装饰的 Python 函数。

### `TensorFlowComputationContext`

[tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) 是用于构造 TensorFlow 计算的上下文。此上下文用于对使用 <a>tensorflow_computation.tf_computation</a> 装饰器装饰的 Python 函数执行序列化。

## `ContextStack`

[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) 是用于与[上下文](#context)堆栈进行交互的数据结构。

您可以通过以下方式设置 TFF 将用于[构造](tracing.md)、[编译](compilation.md)或[执行](execution.md) [AST](compilation.md#ast) 的上下文：

- 调用 [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) 以设置默认上下文。此 API 常用于安装将编译或执行计算的上下文。

- 调用 [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) 以获取当前的上下文堆栈，然后调用 [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) 以将上下文临时安装到堆栈顶部。例如，在跟踪装饰的函数的同时，[federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) 和 [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) 装饰器可将相应的上下文推送到当前上下文堆栈。

### `ContextStackImpl`

[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) 是作为常用线程局部堆栈实现的 [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)。

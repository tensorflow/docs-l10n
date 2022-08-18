# 上下文

[TOC]

## `Context`

[context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) 是一种可以[构造](tracing.md)、[编译](compilation.md)或[执行](execution.md) [AST](compilation.md#ast) 的环境。

此 API 可定义在**不**使用[执行器](execution.md#executor)执行时应使用的**低级抽象**；[参考](backend.md#reference)后端在此级别集成。

### `ExecutionContext`

[execution_context.ExecutionContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/execution_contexts/sync_execution_context.py) 是使用编译函数来编译计算并使用[执行器](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py)来执行计算的 [context_base.Context](execution.md#executor)。

此 API 可定义在使用[执行器](execution.md#executor)执行时应使用的**高级抽象**；[原生](backend.md#native)和 [IREE](backend.md#iree) 后端在此级别集成。

### `FederatedComputationContext`

A [federated_computation_context.FederatedComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation_context.py) is a [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) that constructs federated computations. This context is used trace Python functions decorated with the [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) decorator.

### `TensorFlowComputationContext`

A [tensorflow_computation_context.TensorFlowComputationContext](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation_context.py) is a [context_base.Context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_base.py) that constructs TensorFlow computations. This context is used to serialize Python functions decorated with the [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) decorator.

## `ContextStack`

[context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) 是用于与[上下文](#context)堆栈进行交互的数据结构。

您可以通过以下方式设置 TFF 将用于[构造](tracing.md)、[编译](compilation.md)或[执行](execution.md) [AST](compilation.md#ast) 的上下文：

- 调用 [set_default_context.set_default_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/set_default_context.py) 以设置默认上下文。此 API 常用于安装将编译或执行计算的上下文。

- Invoking [get_context_stack.get_context_stack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/get_context_stack.py) to get the current context stack and then invoking [context_stack_base.ContextStack.install](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py) to temporarily install a context onto the top of the stack. For example, the [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) and [tensorflow_computation.tf_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/tensorflow_context/tensorflow_computation.py) decorators push the corresponding contexts onto the current context stack while the decorated function is being traced.

### `ContextStackImpl`

[context_stack_impl.ContextStackImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_impl.py) 是作为常用线程局部堆栈实现的 [context_stack_base.ContextStack](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/context_stack/context_stack_base.py)。

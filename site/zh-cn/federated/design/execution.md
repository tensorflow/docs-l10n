# 执行

[目录]

[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) 软件包中包含核心 [Executor](#executor) 类和[运行时](#runtime)相关的功能。

## 运行时

运行时是一个逻辑概念，用于描述执行计算的系统。

### TFF 运行时

TFF 运行时通常处理 [AST](compilation.md#ast) 的执行，并将数学计算的执行委派给[外部运行时](#external-runtime)，如 [TensorFlow](#tensorflow)。

### 外部运行时

外部运行时是 TFF 运行时将执行委派给的任何系统。

#### TensorFlow

[TensorFlow](https://www.tensorflow.org/) 是一个开源的机器学习平台。如今，TFF 运行时将数学计算委派给 TensorFlow，这通过可组合成一个层次结构（称为[执行栈](#execution-stack)）的 [eager_tf_executor.EagerTFExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/eager_tf_executor.py) 来实现。

## `Executor`

[executor_base.Executor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_base.py) 是一个抽象接口，它定义了用于执行 [AST](compilation.md#ast) 的 API。[executors](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors) 软件包中包含此接口的具体实现集合。

## `ExecutorFactory`

[executor_factory.ExecutorFactory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_factory.py) 是一个抽象接口，它定义了用于构造 [Executor](#executor) 的 API。这些 factory 会延迟构造 Executor 并管理 Executor 的生命周期；延迟构造 Executor 的动机是推断执行时的客户端数量。

## 执行栈

执行栈是 [Executor](#executor) 的层次结构。[executor_stacks](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 模块包含用于构造和组成特定执行栈的逻辑。

### 本地执行栈

[executor_stacks.local_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 函数构造一个本地执行栈，该执行栈对一定数量的客户端执行 [AST](compilation.md#ast)。

### 远程执行栈

[executor_stacks.remote_executor_factory](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/executors/executor_stacks.py) 函数构造一个远程执行栈，该执行栈对一些服务执行 [AST](compilation.md#ast)。

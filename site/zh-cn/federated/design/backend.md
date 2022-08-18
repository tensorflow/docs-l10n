# 后端

[TOC]

后端是用于[构造](tracing.md)、[编译](compilation.md)和[执行](execution.md) [AST](compilation.md#ast) 的 [Context](context.md#context) 中的[编译器](compilation.md#compiler)和[运行时](execution.md#runtime)的组合，这意味着后端构成了评估 AST 的环境。

[后端](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends)软件包包含可以扩展 TFF 编译器和/或 TFF 运行时的后端；这些扩展可以在相应的后端中找到。

如果后端的[运行时](execution.md#runtime)已实现为[执行堆栈](execution.md#execution-stack)，那么后端可以构造一个 [ExecutionContext](context.md#executioncontext) 来为 TFF 提供评估 AST 的环境。在这种情况下，后端使用高级抽象与 TFF 集成。但是，如果运行时*未*实现为执行堆栈，后端将需要构造一个 [Context](context.md#context) 并使用低级抽象与 TFF 集成。

```dot
<!--#include file="backend.dot"-->
```

**蓝色**节点由 TFF [核心](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core)提供。

**绿色**、**红色**、**黄色**和**紫色**节点分别由[原生](#native)、[mapreduce](#mapreduce)、[iree](#iree) 和[参考](#reference)后端提供。

**虚线**节点由外部系统提供。

**实线**箭头表示关系，**虚线**箭头表示继承。

## 原生

[原生](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native)后端由 TFF 编译器和 TFF 运行时组成，以便以合理高效和可调试的方式编译和执行 AST。

### 原生形式

原生形式是一个 AST，它在拓扑上被分类为 TFF 内部函数的有向无环图 (DAG)，并对这些内部函数的依赖项进行了一些优化。

### Compiler

[compiler.transform_to_native_form](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/compiler.py) 函数将 AST 编译为[原生形式](#native-form)。

### 运行时

原生后端不包含对 TFF 运行时的后端特定扩展，而是可以直接使用[执行堆栈](execution.md#execution-stack)。

### Context

原生上下文是使用原生编译器（或无编译器）和 TFF 运行时构造的 [ExecutionContext](context.md#executioncontext)，例如：

```python
executor = eager_tf_executor.EagerTFExecutor()
factory = executor_factory.create_executor_factory(lambda _: executor)
context = execution_context.ExecutionContext(
    executor_fn=factory,
    compiler_fn=None)
set_default_context.set_default_context(context)
```

但是，有一些常见的配置：

[execution_context.set_local_python_execution_context](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/native/execution_context.py) 函数使用原生编译器和[本地执行堆栈](execution.md#local-execution-stack)构造 `ExecutionContext`。

## MapReduce

[mapreduce](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce) 后端包含构造可以在类似 MapReduce 的运行时上执行的形式所需的数据结构和编译器。

### `MapReduceForm`

[forms.MapReduceForm](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py) 是一种数据结构，定义了可以在类似 MapReduce 的运行时上执行的逻辑表示。此逻辑被组织为 TensorFlow 函数的集合，有关这些函数的性质的更多信息，请参阅[形式](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/forms.py)模块。

### Compiler

The [compiler](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/compiler.py) module contains [Building Block](compilation.md#building-block) and [TensorFlow Computation](compilation.md#tensorflow-computation) transformations required to compile an AST to a [MapReduceForm](#canonicalform).

[form_utils](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/mapreduce/form_utils.py) 模块包含 MapReduce 后端的编译器并且会构造 [MapReduceForm](#canonicalform)。

### 运行时

MapReduce 运行时不由 TFF 提供，而应由外部类似 MapReduce 的系统提供。

### Context

TFF 不提供 MapReduce 上下文。

## IREE

[IREE](https://github.com/google/iree) 是 [MLIR](https://mlir.llvm.org/) 的一个实验性编译器后端。

[iree](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree) 后端包含执行 AST 所需的数据结构、编译器和运行时。

### Compiler

[编译器](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/compiler.py)模块包含将 AST 编译为可以使用 [executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) 执行的形式所需的转换。

### 运行时

[executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) 是一个[执行器](execution.md#executor)，它通过委托给 IREE 运行时来执行计算。此执行器可与来自 TFF 运行时的其他[执行器](execution.md#executor)组合在一起，以构造代表 IREE 运行时的[执行堆栈](execution.md#execution-stack)。

### Context

iree 上下文是使用 iree 编译器和[执行堆栈](execution.md#execution-stack)构造的 [ExecutionContext](context.md#executioncontext)，其中的 [executor.IreeExecutor](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/backends/iree/executor.py) 委托给外部 IREE 运行时。

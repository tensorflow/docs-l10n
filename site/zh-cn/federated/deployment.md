# 部署

除了定义计算之外，TFF 还提供了执行计算的工具。鉴于我们主要关注点在于模拟，因此提供的接口和工具较为通用。本文档概述了部署到各种类型的平台的选项。

注意：本文档仍在构建中。

## 概述

TFF 计算有两种主要的部署模式：

- **原生后端**。如果能够解读 [`computation.proto`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto) 中定义的 TFF 计算的语法结构，我们可以将后端称作*原生*。原生后端不必支持所有语言构造或内部函数。原生后端必须实现标准的 TFF *执行程序*接口之一（例如供 Python 代码使用的 [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)），或者实现在 [`executor.proto`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/executor.proto)（作为 gRPC 端点公开）中定义的与语言无关的版本。

    支持上述接口的原生后端可以代替默认参考运行时进行交互使用，例如，运行笔记本或实验脚本。大多数原生后端将以*解释模式*运行，即它们将按照定义的方式处理计算定义，并以递增方式执行该定义，但并非总是如此。原生后端还可以*转换*（ *编译*或 JIT 编译）部分计算，以提高性能或简化其结构。它的一个常见用例是减少在计算中出现的联合算子的集合，这样，转换的后端数据流的某些部分就不必公开给整个集合。

- **非原生后端**。与原生后端相比，非原生后端无法直接解释 TFF 计算结构，并要求将其转换为后端可以理解的其他*目标表示*。此类后端的一个显著示例是 Hadoop 集群或用于静态数据流水线的类似平台。为了将计算部署到此类后端，必须先对其进行*转换*（或*编译*）。根据设置，此过程可以对用户透明地完成（即，可以将非原生后端包装在标准执行程序接口（例如在后台执行转换的 [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor)）中），也可以将其公开为一种工具，该工具允许用户手动将一个或一组计算转换为可由特定类的后端理解的目标表示。在 [`tff.backends`](https://www.tensorflow.org/federated/api_docs/python/tff/backends) 命名空间中可以找到支持特定类型非原生后端的代码。在撰写本文时，非原生后端的唯一支持类型是一类能够执行单轮 MapReduce 的系统。

## 原生后端

即将提供更多详细信息。

## 非原生后端

### MapReduce

即将提供更多详细信息。

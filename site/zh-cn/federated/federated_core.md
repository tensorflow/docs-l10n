# Federated Core

本文档介绍 TFF 的核心层，它是[联合学习](federated_learning.md)以及未来可能的非学习联合算法的基础。

有关 Federated Core 的简要介绍，请阅读以下教程。这些教程通过示例介绍了某些基本概念，并逐步演示了一个简单联合平均算法的构造。

- [自定义联合算法，第 1 部分：Federated Core 简介](tutorials/custom_federated_algorithms_1.ipynb)。

- [自定义联合算法，第 2 部分：实现联合平均](tutorials/custom_federated_algorithms_2.ipynb)。

另外，建议您熟悉[联合学习](federated_learning.md)以及与[图像分类](tutorials/federated_learning_for_image_classification.ipynb)和[文本生成](tutorials/federated_learning_for_text_generation.ipynb)相关的教程，因为对于联合学习，Federated Core API (FC API) 的使用为我们在设计该层时所作的选择提供了重要的背景信息。

## 概述

### 目标、预期用途和使用范围

对联合核心 (FC) 的最佳理解是将其当作一种实现分布式计算的编程环境。所谓分布式计算，就是由多种设备（手机、平板电脑、嵌入式设备、桌面计算机、传感器、数据库服务器等）分别在本地执行不常用处理，并通过网络通信来协调工作的一种计算机制。

*分布式*是一个非常通用的词汇，但 TFF 并非旨在支持所有可能的分布式算法类型，所以我们更倾向于使用一个不那么宽泛的词：*联合计算*，以便描述可在此框架中表示的算法类型。

虽然以非常正式的方式定义*联合计算*超出了本文档的讨论范围，但是，不妨想一想在介绍一种新分布式学习算法的[研究论文](https://arxiv.org/pdf/1602.05629.pdf)中以伪代码表示的的算法类型。

简而言之，FC 的目标就是以相当于伪代码级别的抽象来实现编程逻辑的类似紧凑表示，但这种编程逻辑并*非*伪代码，而是可在各种目标环境中执行的程序。

根据 FC 的设计，用于表达这些算法类型的关键定义特征是以集合方式描述系统参与者的行为。因此，我们倾向于讨论在本地转换数据的*各个设备*，以及通过*广播*、*收集*或*聚合*结果来协调工作的集中式协调器。

虽然在设计时，TFF 超越了简单的*客户端-服务器*架构，但集合处理的概念仍然是基础。这是由于 TFF 源自联合学习——一种最初设计为支持潜在敏感数据计算的技术。出于隐私保护，这种敏感数据仍受客户端设备控制，并且不能简单地下载到集中位置。此类系统中的每一个客户端都会为系统计算结果（一种我们通常认为对所有参与者都有意义的结果）提供数据和处理能力，同时还努力保持每个客户端的隐私性和匿名性。

因此，虽然大多数分布式计算的框架设计为从各个参与者（即，在各个点对点消息交换的级别上）的角度表达处理，并利用传入和传出消息表达参与者本地状态转换的相互依赖性，但 TFF 的联合核心设计为从*全局*系统级角度（例如，类似于 [MapReduce](https://research.google/pubs/pub62/)）描述系统的行为。

因此，虽然适用于一般目的的分布式框架可能以构建块形式提供*发送*和*接收*运算，但 FC 提供的是封装简单分布式协议的 `tff.federated_sum`、`tff.federated_reduce` 或 `tff.federated_broadcast` 等构建块。

## 语言

### Python 接口

TFF 使用内部语言表示联合计算，其语法由 [computation.proto](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/proto/v0/computation.proto) 中的可序列化表示形式进行定义。不过，FC API 用户通常不需要直接与该语言交互。这相当于我们提供了一个将代码包装起来的 Python API（`tff` 命名空间），作为定义计算的方式。

具体而言，TFF 提供 `tff.federated_computation` 之类的 Python 函数装饰器，用于跟踪装饰函数的主体，并使用 TFF 的语言生成联合计算逻辑的序列化表示形式。使用 `tff.federated_computation` 装饰的函数作为此类序列化表示形式的载体，可将其作为构建模块嵌入另一个计算的主体中，或者在调用时按需求执行。

下面仅举一个例子；在[自定义算法](tutorials/custom_federated_algorithms_1.ipynb)教程中可以找到更多示例。

```python
@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

如果读者熟悉非 Eager TensorFlow，则会发现这种方法类似于在定义 TensorFlow 计算图的 Python 代码部分使用 `tf.add` 或 `tf.reduce_sum` 之类的函数编写 Python 代码。虽然这种代码在技术上使用 Python 表达，但是，其目的是为了在底层构造 `tf.Graph` 的可序列化表示形式，并且它是计算图（而不是 Python 代码），由 TensorFlow 运行时在内部执行。同样，用户可以把 `tff.federated_mean` 看作将*联合运算*插入通过 `get_average_temperature` 表示的联合计算。

通过 FC 定义语言的部分原因是，如上所述，联合计算指定了分布式集合行为，因此，它们的逻辑是非本地的。例如，TFF 提供算子，其输入和输出可能位于网络上的不同位置。

这需要一门语言和一个类型系统来理解分布的概念。

### 类型系统

Federated Core 提供了以下几种类型。在描述这些类型时，我们会指出类型构造函数并采用紧凑表示法，因为这是一种便于描述计算和算子类型的方式。

首先，以下是在概念上与现有主流语言相似的几种类型：

- **张量类型** (`tff.TensorType`)。就像在 TensorFlow 中一样，这些类型有 `dtype` 和 `shape`。唯一的区别是这种类型的对象不仅限于在 TensorFlow 计算图中表示 TensorFlow 运算输出的 Python 的 `tf.Tensor` 实例，而是也可能包括可产生的数据单位，例如，作为分布聚合协议的输出。因此，TFF 张量类型是 Python 或 TensorFlow 中此类类型的具体物理表示形式的抽象版本。

    张量类型的紧凑表示法为 `dtype` 或 `dtype[shape]`。例如，`int32` 和 `int32[10]` 分别是整数和整数向量的类型。

- **序列类型** (`tff.SequenceType`)。这些是 TFF 中等效于 TensorFlow 中 `tf.data.Dataset` 的具体概念的抽象。用户可以按顺序使用序列的元素，并且可以包含复杂的类型。

    序列类型的紧凑表示法为 `T*`，其中 `T` 是元素的类型。例如，`int32*` 表示整数序列。

- **命名元组类型** (`tff.StructType`)。这些是 TFF 使用指定类型构造具有预定义数量*元素*的元组或字典式结构（无论命名与否）的方式。重要的一点是，TFF 的命名元组概念包含等效于 Python 参数元组的抽象，即元组的元素集合中有一部分（并非全部）是命名元素，还有一部分是位置元素。

    命名元组的紧凑表示法为 `<n_1=T_1, ..., n_k=T_k>`，其中 `n_k` 是可选元素名称，`T_k` 是元素类型。例如，`<int32,int32>` 是一对未命名整数的紧凑表示法，`<X=float32,Y=float32>` 是命名为 `X` 和 `Y`（可能代表平面上的一个点）的一对浮点数的紧凑表示法。元组可以嵌套，也可以与其他类型混用，例如，`<X=float32,Y=float32>*` 可能是一系列点的紧凑表示法。

- **函数类型** (`tff.FunctionType`)。TFF 是一个函数式编程框架，其中函数被视为[第一类值](https://en.wikipedia.org/wiki/First-class_citizen)。函数最多有一个参数，并且只有一个结果。

    这些函数的紧凑表示法为 `(T -> U)`，其中 `T` 为参数类型，`U` 为结果类型；或者，如果没有参数（虽然无参数函数是一个大部分情况下仅在 Python 级别存在的过时概念），则可以表示为 `( -> U)`。例如，`(int32* -> int32)` 表示一种将整数序列缩减为单个整数值的函数类型。

以下类型解决 TFF 计算的分布系统方面的问题。由于这些概念在一定程度上是特定于 TFF 的，因此，我们建议您参考[自定义算法](tutorials/custom_federated_algorithms_1.ipynb)教程，了解附加注释和示例。

- **布局类型**。除了 2 个文字形式的 `tff.SERVER` 和 `tff.CLIENTS`（可将其视为这种类型的常量）外，这种类型还没有在公共 API 中公开。它仅供内部使用，但是，将在以后的公共 API 版本中引入。该类型的紧凑表示法为 `placement`。

    *布局*表示扮演特定角色的系统参与者的集合。最初的版本是为了解决客户端-服务器计算的问题，其中有 2 组参与者：*客户端*和*服务器*（可将后者视为单一实例组）。但是，在更复杂的架构中，可能还有其他角色，如多层系统中的中间聚合器。这种聚合器可能执行不同类型的聚合，或者使用不同类型的数据压缩/解压缩，而不是服务器或客户端使用的类型。

    定义布局概念的主要目的是作为定义*联合类型*的基础。

- **联合类型** (`tff.FederatedType`)。联合类型的值是由特定布局（如 `tff.SERVER` 或 `tff.CLIENTS`）定义的一组系统参与者托管的值。联合类型通过*布局*值（因此，它是一种[依赖类型](https://en.wikipedia.org/wiki/Dependent_type)）, *成员组成要素*（每个参与者在本地托管的内容类型），以及指定所有参与者是否在本地托管同一项目的附加部分 `all_equal` 进行定义。

    对于包含 `T` 类型项目（成员组成）的值的联合类型，如果每个项目由组（布局）`G` 托管，则其紧凑表示法为 `T@G` 或 `{T}@G`，分别设置或不设置 `all_equal` 位。

    例如：

    - `{int32}@CLIENTS` 表示包含一组可能不同的整数（每个客户端设备一个）的*联合值*。请注意，我们讨论的单一*联合值*包含网络中多个位置出现的多个数据项。一种理解方式是将其当作具有“网络”维度的张量，虽然这种类比并不完美，因为 TFF 不允许[随机存取](https://en.wikipedia.org/wiki/Random_access)联合值的成员组成要素。

    - `{<X=float32,Y=float32>*}@CLIENTS` 表示一个*联合数据集*，这是一个包含多个 `XY` 坐标序列（每个客户端设备一个序列）的值。

    - `<weights=float32[10,5],bias=float32[5]>@SERVER` 表示服务器上的权重和偏差张量的命名元组。我们省略了花括号，这表示已设置 `all_equal` 位，其中只有一个元组（不管托管该值的集群中有多少个服务器副本）。

### 构建模块

Federated Core 的语言是一种 [λ 演算](https://en.wikipedia.org/wiki/Lambda_calculus)，另外还有几个额外的元素。

它提供了当前在公共 API 中公开的以下编程抽象：

- **TensorFlow** 计算 (`tff.tf_computation`)。TFF 中有一些使用 `tff.tf_computation` 装饰器包装为可重用组件的 TensorFlow 代码部分。这些代码一般都是函数式类型，但是与 TensorFlow 中的函数不同，它们可以接受结构化参数或返回序列类型的结构化结果。

    下面是一个示例，即使用 `tf.data.Dataset.reduce` 算子来计算整数和的 `(int32* -> int)` 类型的 TF 计算：

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

- **内联函数**或*联合算子* (`tff.federated_...`)。这是构成大部分 FC API 的函数库，如 `tff.federated_sum` 或 `tff.federated_broadcast`，其中大多数表示与 TFF 一起使用的分布通信算子。

    我们将其称为*内联函数*是因为它们与[内联函数](https://en.wikipedia.org/wiki/Intrinsic_function)有一定的相似性，属于一种能够被 TFF 理解，并且可编译成底层代码的开放式可扩展算子集。

    这些算子中的大部分都有联合类型的参数和结果，并且大部分都是可以应用到各种类型的数据的模板。

    例如，您可以将 `tff.federated_broadcast` 理解为函数式类型 `T@SERVER -> T@CLIENTS` 的一个模板算子。

- **λ 表达式** (`tff.federated_computation`)。TFF 中的 λ 表达式等效于 Python 中的 `lambda` 或 `def`；它包含参数名称，以及包含对该参数的引用的主体（表达式）。

    在 Python 代码中，使用 `tff.federated_computation` 装饰 Python 函数并定义一个参数即可创建这些表达式。

    下面是我们之前讲过的一个 λ 表达式示例：

    ```python
    @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

- **布局文字**。目前，仅允许使用 `tff.SERVER` 和 `tff.CLIENTS` 定义简单的客户端-服务器计算。

- **函数调用** (`__call__`)。使用标准 Python `__call__` 语法可以调用具有函数式类型的任何对象。这种调用就是一个表达式，其类型与被调用的函数的结果类型相同。

    例如：

    - `add_up_integers(x)` 表示在参数 `x` 上调用之前定义的 TensorFlow 计算。此表达式的类型为 `int32`。

    - `tff.federated_mean(sensor_readings)` 表示在 `sensor_readings` 上调用联合平均算子。此表达式的类型为 `float32@SERVER`（假设在上述示例的上下文中）。

- 构成**元组**并**选择**其元素。出现在使用 `tff.federated_computation` 装饰的函数主体中的 `[x, y]`、`x[y]` 或 `x.y` 形式的 Python 表达式。

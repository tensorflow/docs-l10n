# 联合学习

## 概述

本文档介绍有助于完成联合学习任务（例如使用 TensorFlow 中实现的现有机器学习模型进行联合训练或评估）的接口。在设计这些接口时，我们的主要目标是试验联合学习，而无需了解底层工作原理，以及在各种现有模型和数据上评估实现的联合学习算法。我们建议您为该平台做贡献。TFF 的设计考虑到了可扩展性和可组合性，我们欢迎大家的贡献；我们期待看到您的成果！

该层提供的接口包括以下三个主要部分：

- **模型**。可用于包装现有模型，以便与 TFF 一起使用的类和帮助函数。包装模型可以像调用单个包装函数（如 `tff.learning.from_keras_model`）或为了实现全定制而定义 `tff.learning.Model` 接口的子类一样简单。

- **联合计算构建器**。使用现有模型为训练或评估构造联合计算的帮助函数。

- **数据集**。可以在 Python 中下载和访问的预装数据集合，用于模拟联合学习场景。虽然联合学习的设计旨在使用不能简单地在一个集中位置下载的分散数据，但是，在研究和开发阶段，使用可在本地下载和操作的数据开展初始实验通常非常方便，对可能不熟悉这种方法的开发者来说更是如此。

除了已经分组到 `tff.simulation` 中的研究数据集和其他与模拟相关的功能外，这些接口主要在 `tff.learning` 命名空间中定义。该层使用 [Federated Core (FC)](federated_core.md)（还提供了运行时环境）提供的底层接口实现。

在继续之前，我们建议您先回顾一下有关[图像分类](tutorials/federated_learning_for_image_classification.ipynb)和[文本生成](tutorials/federated_learning_for_text_generation.ipynb)的教程，因为这些教程使用具体示例介绍了本文中所述的大部分概念。如果您想了解有关 TFF 工作原理的详细信息，可以浏览[自定义算法](tutorials/custom_federated_algorithms_1.ipynb)教程，将其作为用于表达联合计算逻辑的底层接口的简介，并学习 `tff.learning` 接口的现有实现。

## 模型

### 架构假设

#### 序列化

TFF 旨在支持各种分布学习场景，在这些场景中，您编写的机器学习模型代码可能在大量具有不同功能的异构化客户端上执行。一方面，尽管在某些应用中，这些客户端可能是功能强大的数据库服务器，但我们的平台打算支持的许多重要用途涉及资源有限的移动和嵌入式设备。我们不能假设这些设备能够托管 Python 运行时；在当前状况下，我们只能假设它们能够托管本地 TensorFlow 运行时。因此，我们在 TFF 中作出的一个基本架构假设是，您的模型代码必须可序列化为 TensorFlow 计算图。

您仍可以（且应该）按照最新的最佳做法（比如使用 Eager 模式）开发 TF 代码。但是，最终代码必须可序列化（例如，对于 Eager 模式代码，可包装为 `tf.function`）。这样可以确保在执行时必需的任何 Python 状态或控制流都可以进行序列化（可能需要利用 [Autograph](https://www.tensorflow.org/guide/autograph)）。

目前，TensorFlow 并不完全支持序列化和反序列化 Eager 模式 TensorFlow。因此，TFF 中的序列化目前遵循 TF 1.0 模式，其中所有代码必须在 TFF 控制的 `tf.Graph` 中构造。这意味着 TFF 目前不能使用已构造的模型；实际上，该模型定义逻辑打包在返回 `tff.learning.Model` 的无参数函数中。随后，TFF 会调用此函数来确保序列化该模型的所有组件。此外，作为一个强类型环境，TFF 需要一些额外的*元数据*，例如您的模型输入类型的规范。

#### 聚合

我们强烈建议大部分用户使用 Keras 构造模型，请参阅下面的 [Keras 转换器](#converters-for-keras)部分。这些包装器自动处理模型更新的聚合以及为模型定义的任何指标。但是，了解如何为一般 `tff.learning.Model` 处理聚合仍非常有用。

一般来说，联合学习中至少有两个聚合层：本地设备端聚合和跨设备聚合（或称联合聚合）：

- **本地聚合**。这种级别的聚合是指跨各个客户端所拥有的多批次样本进行的聚合。它既适用于在本地训练模型时循序演化的模型参数（变量），也适用于您计算的统计数据（如平均损失、准确率和其他指标）。在各个客户端的每个本地数据流上迭代时，您的模型会再次在本地更新这些数据。

    这种级别的聚合由您的模型代码负责执行，并使用标准 TensorFlow 构造完成。

    一般处理结构如下：

    - 该模型先构造 `tf.Variable` 来存放聚合（如处理的批次数量或样本数量），每个批次之和或每个样本的损失等。

    - TFF 在您的 `Model` 上按顺序对客户端数据的后续批次多次调用 `forward_pass` 方法，从而以副作用形式让您更新存放各种聚合的变量。

    - 最后，TFF 在您的模型上调用 `report_local_outputs` 方法，从而让您的模型将它收集的所有汇总统计数据编译成将由客户端导出的一组紧凑的指标。例如，您的模型代码此时可以将损失之和除以处理的样本数量，从而导出平均损失等。

- **联合聚合**。这种级别的聚合是指跨系统中多个客户端（设备）的聚合。同样，它既适用于按客户端求平均值的模型参数（变量），也适用于您的模型作为本地聚合结果导出的指标。

    这种级别的聚合由 TFF 负责执行。但是，作为模型创建者，您可以控制此过程（下文中有详细介绍）。

    一般处理结构如下：

    - 初始模型（以及训练所需要的任何参数）由服务器分发给将参与一轮训练或评估的客户端子集。

    - 对于每个客户端，在本地数据批次流上反复调用您的模型代码（无论是独立还是并行），从而产生一组新的模型参数（训练时），以及一组新的本地指标，如上所述（这是本地聚合）。

    - TFF 运行分布聚合协议来累积和聚合整个系统中的模型参数以及本地导出的指标。该逻辑使用 TFF 自有的*联合计算*语言（不是在 TensorFlow 中），在模型的 `federated_output_computation` 中以声明方式进行表达。有关聚合 API 的详细信息，请参阅[自定义算法](tutorials/custom_federated_algorithms_1.ipynb)教程。

### 抽象接口

这种基本*构造函数* + *元数据*接口由接口 `tff.learning.Model` 表示，如下所述：

- 构造函数、<code>forward_pass</code> 和 <code>report_local_outputs</code> 方法应相应地构造模型变量、前向传递以及要报告的统计数据。如上所述，这些方法构造的 TensorFlow 必须可序列化。

- `input_spec` 属性以及返回可训练、不可训练和本地变量的 3 个属性表示元数据。TFF 使用此信息确定如何将您的模型的各个部分连接到联合优化算法，同时定义内部类型签名来帮助验证构造的系统的正确性（这样，当数据与模型预期要使用的数据不相符时，您的模型不能在这些数据上进行实例化）。

此外，抽象接口 `tff.learning.Model` 会公开属性 `federated_output_computation`。该属性与前述 `report_local_outputs` 属性相结合，可让您控制聚合汇总统计数据的过程。

在[图像分类](tutorials/federated_learning_for_image_classification.ipynb)教程的第二部分，以及我们用于在 [`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/model_examples.py) 中进行测试的示例模型中，您可以找到有关如何定义自定义 `tff.learning.Model` 的示例。

### Keras 转换器

TFF 需要的几乎所有信息都可以通过调用 `tf.keras` 接口获得，因此，如果您有一个 Keras 模型，则可以利用 `tff.learning.from_keras_model` 来构造 `tff.learning.Model`。

请注意，TFF 仍需要您提供构造函数——无参数*模型函数*，如下所示：

```python
def model_fn():
  keras_model = ...
  return tff.learning.from_keras_model(keras_model, sample_batch, loss=...)
```

除了模型本身外，您还需要提供一批示例数据，以便 TFF 用于确定您的模型输入的类型和形状。这样可以确保 TFF 能够正确实例化将实际出现在客户端设备上的数据的模型（因为我们假设当您构造要序列化的 TensorFlow 时，该数据一般不可用）。

我们的[图像分类](tutorials/federated_learning_for_image_classification.ipynb)和[文本生成](tutorials/federated_learning_for_text_generation.ipynb)教程为 Keras 包装器的使用提供了图解说明。

## 联合计算构建器

`tff.learning` 软件包为执行学习相关任务的 `tff.Computation` 提供了多个构建器；我们希望将来能够扩充此类计算的集合。

### 架构假设

#### 执行

运行联合计算包括两个不同阶段。

- **编译**：首先，TFF 将联合学习算法*编译*成整个分布计算的抽象序列化表示形式。这时就会发生 TensorFlow 序列化，但是为了支持更高效的执行，可能会发生其他转换。我们将编译器产生的序列化表示形式称为*联合计算*。

- **执行**：TFF 提供了各种方法来*执行*这些计算。目前，只有通过本地模拟才能支持执行（例如，在使用模拟分散数据的笔记本中）。

TFF 的 Federated Learning API 生成的联合计算（如使用[联合模型平均](https://arxiv.org/abs/1602.05629)或联合评估的训练算法）包括很多元素，最主要的是：

- 模型代码的序列化形式，以及联合学习框架构造的用于驱动模型训练/评估循环的其他 TensorFlow 代码（如构造优化器，应用模型更新，在 `tf.data.Dataset` 上迭代，计算指标，以及在服务器上应用聚合更新等等）。

- *客户端*与*服务器*之间通信的声明式规范（通常是客户端设备上各种形式的*聚合*，以及从服务器到所有客户端的*广播*），以及这种分布式通信如何与 TensorFlow 代码的“客户端-本地”或“服务器-本地”执行交错。

以这种序列化形式表示的*联合计算*采用独立于平台的内部语言（与 Python 不同）进行表达，但是，要使用 Federated Learning API，您无需关注这种表示形式的详细信息。在您的 Python 代码中，计算以类型 `tff.Computation` 的对象形式表示，在大多数情况下，您可以将其当作不透明的 Python `callable`。

在本教程中，您会像调用常规 Python 函数一样调用这些将在本地执行的联合计算。但是，TFF 旨在以一种与执行环境的大部分方面无关的方式表示联合计算，从而使其可能被部署到以下设备：比方说，运行 `Android` 的设备组，或者是数据中心内的集群。同样，它的主要结果仍然是关于[序列化](#serialization)的典型假设。特别是当您调用下述 `build_...` 方法之一时，计算将被完全序列化。

#### 建模状态

TFF 是一种函数式编程环境，但是，联合学习中许多相关的过程是有状态的。例如，涉及多轮联合模型平均的训练循环就是我们可能会分类为*有状态过程*的一个例子。在该过程中，逐轮次发展的状态包括正在训练的模型参数集，与优化器（如动量向量）相关的其他状态可能也是如此。

由于 TFF 是函数式编程环境，因此，在 TFF 中，将有状态的处理作为计算进行建模。该计算接受当前状态以作为输入，然后提供更新的状态以作为输出。为了完整定义有状态的处理，您还需要指定初始状态的来源（否则无法启动该处理）。这通过帮助类 `tff.templates.IterativeProcess` 的定义获取，它具有分别与初始化和迭代对应的 2 个属性：`initialize` 和 `next`。

### 可用构建器

目前，TFF 为联合训练和评估提供了两个生成联合计算的构建器函数：

- `tff.learning.build_federated_averaging_process` 使用一个*模型函数*和一个*客户端优化器*，并返回一个有状态的 `tff.templates.IterativeProcess`。

- `tff.learning.build_federated_evaluation` 使用一个 *model 函数*，并为模型的联合评估返回一个单一的联合计算，因为评估没有状态。

## 数据集

### 架构假设

#### 客户端选择

在典型的联合学习场景中，我们有*大量*（可能是数亿计）客户端设备，但是，在特定时刻，其中可能只有少部分处于活动状态且可供训练（例如，可能仅限已插入电源且不在按流量计费的网络上的客户端，或者是处于空闲状态的客户端）。 通常，可参与训练或评估的客户端集不在开发者的控制范围内。此外，由于协调数百万计的客户端不太现实，因此，典型的一轮训练或评估往往只包含一小部分可用客户端，而这些客户端可能是通过[随即抽样](https://arxiv.org/pdf/1902.01046.pdf)获得的。

这样做的主要结果是，根据设计，联合计算的表示方式与参与者的具体集合无关；所有处理以抽象匿名*客户端*组上的聚合运算形式表示，并且对于不同的训练轮次，该组可能不同。因此，计算与具体参与者的实际绑定（以及与它们馈送给计算的具体数据的实际绑定）在计算本身之外进行建模。

为了模拟联合学习代码的实际部署，通常需要编写如下训练循环：

```python
trainer = tff.learning.build_federated_averaging_process(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  state, metrics = trainer.next(state, data_for_this_round)
```

为了实现这一点，在模拟中使用 TFF 时，将联合数据视为 Python `list`，为每个参与客户端设备使用一个元素来表示该设备的本地 `tf.data.Dataset`。

### 抽象接口

为了使模拟联合数据集的处理实现标准化，TFF 提供了一个抽象接口 `tff.simulation.datasets.ClientData`。通过该接口，用户可以枚举客户端集，并构造包含特定客户端的数据的 `tf.data.Dataset`。在 Eager 模式下，这些 `tf.data.Dataset` 可以作为输入直接馈送给生成的联合计算。

需要注意的是，访问客户端标识的能力是数据集为了在模拟中使用才提供的一个功能，在这种情况下，可能需要对客户端的特定子集的数据进行训练的能力（例如，为了模拟不同类型客户端的日可用性）。编译的计算和底层运行时*不*涉及客户端身份的任何概念。一旦选择来自客户端特定子集的数据作为输入（例如，在对 `tff.templates.IterativeProcess.next` 的调用中），则其中不再出现客户端标识。

### 可用数据集

为了在模拟中使用，我们已指定将命名空间 `tff.simulation.datasets` 专门用于实现 `tff.simulation.datasets.ClientData` 接口的数据集，并为其提供 2 个数据集作为种子，从而支持[图像分类](tutorials/federated_learning_for_image_classification.ipynb)和[文本生成](tutorials/federated_learning_for_text_generation.ipynb)教程。我们希望您为平台贡献自己的数据集。

# TensorFlow Quantum

TensorFlow Quantum (TFQ) 是一个用于[量子机器学习](concepts.md)的 Python 框架。作为一种应用框架，TFQ 允许量子算法研究员和 ML 应用研究员在 TensorFlow 内充分利用 Google 的量子计算框架。

TensorFlow Quantum 侧重于*量子数据*和构建*混合量子经典模型*。它提供了可将在 <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> 中设计的量子算法和逻辑与 TensorFlow 相融合的工具。要有效使用 TensorFlow Quantum，需要对量子计算有基本的了解。

要开始使用 TensorFlow Quantum，请参阅[安装指南](install.md)并通读一些可运行的[笔记本教程](./tutorials/hello_many_worlds.ipynb)。

## 设计

TensorFlow Quantum 实现了将 TensorFlow 与量子计算硬件集成所需的组件。为此，TensorFlow Quantum 引入了两个数据类型基元：

- *量子电路* - 表示 TensorFlow 中 Cirq 定义的量子电路。创建大小不同的电路批次，类似于不同的实值数据点的批次。
- *Pauli 和* - 表示 Cirq 中定义的 Pauli 算子张量积的线性组合。像电路一样，创建大小不同的算子批次。

利用这些基元来表示量子电路，TensorFlow Quantum 提供以下运算：

- 从电路批次的输出分布中采样。
- 基于电路批次计算 Pauli 和批次的期望值。TFQ 实现了与反向传播兼容的梯度计算。
- 模拟电路和状态批次。虽然在现实世界中直接检查整个量子电路的所有量子态振幅的效率极低，但状态模拟可以帮助研究员了解量子电路如何将状态映射到接近精确的精度水平。

在[设计指南](design.md)中阅读有关 TensorFlow Quantum 实现的更多信息。

## 报告问题

使用 <a href="https://github.com/tensorflow/quantum/issues" class="external">TensorFlow Quantum 问题跟踪器</a>报告错误或提交功能请求。

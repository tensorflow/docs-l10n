# TensorFlow Quantum 设计

TensorFlow Quantum (TFQ) 专为解决 NISQ 时代的量子机器学习问题而设计。它将量子计算基元（如构建量子电路）引入 TensorFlow 生态系统。使用 TensorFlow 构建的模型和运算使用这些基元来创建功能强大的量子经典混合系统。

利用 TFQ，研究员可以使用量子数据集、量子模型和经典控制参数来构造 TensorFlow 计算图。这些在单个计算图中均表示为张量。TensorFlow 运算可获得引发经典概率事件的量子测量结果。训练使用标准 [Keras](https://www.tensorflow.org/guide/keras/overview) API 完成。`tfq.datasets` 模块允许研究员试验新奇有趣的量子数据集。

## Cirq

<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> 是 Google 的量子编程框架。它提供了所有基本运算（例如量子位、门、电路和测量），以在量子计算机或模拟量子计算机上创建、修改和调用量子电路。TensorFlow Quantum 使用这些 Cirq 基元来扩展 TensorFlow，以便执行批次计算、模型构建和梯度计算。为有效使用 TensorFlow Quantum，与 Cirq 结合使用是一个好主意。

## TensorFlow Quantum 基元

TensorFlow Quantum 实现了将 TensorFlow 与量子计算硬件集成所需的组件。为此，TFQ 引入了两个数据类型基元：

- *量子电路*：表示 TensorFlow 中 <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> 定义的量子电路 (`cirq.Circuit`)。创建大小不同的电路批次，类似于不同的实值数据点的批次。
- *Pauli 和*：表示 Cirq 中定义的 Pauli 算子张量积的线性组合 (`cirq.PauliSum`)。像电路一样，创建大小不同的算子批次。

### 基本运算

TensorFlow Quantum 使用 `tf.Tensor` 中的量子电路基元实现了处理这些电路并产生有意义输出的运算。

TensorFlow 运算使用优化的 C++ 编写。这些运算从电路采样，计算期望值，然后输出给定​​电路产生的状态。编写灵活而高效的运算很有挑战：

1. 各个电路的大小不同。对于模拟电路，您无法创建静态运算（如 `tf.matmul` 或 `tf.add`），然后用不同的数字替换大小不同的电路。这些运算必须支持具有静态大小的 TensorFlow 计算图所不允许的动态大小。
2. 量子数据可以引出完全不同的电路结构。这是在 TFQ 运算中支持动态大小的另一个原因。量子数据可以表示底层量子态的结构变化，这种变化由原始电路的修改来表示。由于新的数据点会在运行时交换，TensorFlow 计算图在构建后无法修改，因此需要支持这些变化的结构。
3. `cirq.Circuits` 与计算图相似，因为它们也是一系列运算，有些可能包含符号/占位符。使其与 TensorFlow 尽可能兼容十分重要。

由于性能方面的原因，Eigen（许多 TensorFlow 运算中使用的 C++ 库）不适合量子电路模拟。作为替代，<a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">量子超越经典实验</a>中使用的电路模拟器将用作验证器，并扩展为 TFQ 运算的基础（均使用 AVX2 和 SSE 指令编写）。使用物理量子计算机创建具有相同函数签名的运算。在模拟和物理量子计算机之间切换就像更改一行代码一样轻松。这些运算位于 <a href="https://github.com/tensorflow/quantum/blob/master/tensorflow_quantum/core/ops/circuit_execution_ops.py" class="external"><code>circuit_execution_ops.py</code></a> 中。

### 层

TensorFlow Quantum 层使用 `tf.keras.layers.Layer` 接口将采样、期望和状态计算公开给开发者。创建用于经典控制参数或读出运算的电路层非常方便。此外，您还可以创建具有高度复杂性的层，以支持批次电路、批次控制参数值以及执行批次读出运算。有关示例，请参阅 `tfq.layers.Sample`。

### 微分器

与许多 TensorFlow 运算不同，量子电路中的可观测对象没有相对容易计算的梯度公式。这是因为传统计算机只能从在量子计算机上运行的电路中读取样本。

为解决此问题，`tfq.differentiators` 模块提供了几种标准微分技术。用户还可以定义自己的方法来计算梯度——在基于样本的期望计算的“真实世界”环境中以及在解析精确世界中。在解析/精确环境中，有限差分之类的方法通常是最快的（挂钟时间）。尽管比较慢（挂钟时间），但更实用的方法（如<a href="https://arxiv.org/abs/1811.11184" class="external">参数偏移</a>或<a href="https://arxiv.org/abs/1901.05374" class="external">随机方法</a>）通常更有效。将 `tfq.differentiators.Differentiator` 实例化并使用 `generate_differentiable_op` 将其附加到现有运算，或者传递给 `tfq.layers.Expectation` 或 `tfq.layers.SampledExpectation` 的构造函数。要实现自定义微分器，请继承 `tfq.differentiators.Differentiator` 类。要定义用于采样或状态向量计算的梯度运算，请使用 `tf.custom_gradient`。

### 数据集

随着量子计算领域的发展，将会出现更多的量子数据和模型组合，从而使结构化比较变得愈加困难。`tfq.datasets` 模块用作量子机器学习任务的数据源。它可确保对模型和性能进行结构比较。

借助大量社区贡献，我们希望推动 `tfq.datasets` 模块不断发展，从而使研究更加透明和可重现。在量子控制、费米子模拟、近相变分类、量子传感等方面精心策划的问题都可以作为出色的候选资源添加到 `tfq.datasets` 中。要提出新的数据集，请提交 <a href="https://github.com/tensorflow/quantum/issues">GitHub 问题</a>。

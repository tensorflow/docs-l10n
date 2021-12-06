# 常见问题解答

## TensorFlow Federated 能否在生产环境中使用，例如在手机上？

目前不能。尽管我们在设计 TFF 时考虑到了在实际设备中的部署，但是目前，我们还尚未提供任何用于此目的的工具。当前版本旨在用于实验用途，例如表达新的联合算法，或使用随附的模拟运行时来通过您自己的数据集尝试联合学习。

我们预期，随着围绕 TFF 的开源生态系统的不断发展，将提供针对物理部署平台的运行时。

## 如何使用 TFF 对大型数据集进行实验？

TFF 初始版本中包含的默认运行时仅适用于小型实验，例如我们在教程中介绍的实验，其中（所有模拟客户端的）全部数据同时装入一台计算机的内存中，并且整个实验均在 Colab 笔记本中本地运行。

我们的近期未来路线图包括一个高性能运行时，可用于处理包含大型数据集和大量客户端的实验。

## 如何确保 TFF 中的随机性符合我的期望？

由于 TFF 已将联合计算纳入其核心，因此 TFF 的作者将不负责控制进入 TensorFlow `Session` 或在这些会话中调用 `run` 的位置和方式。如果设置了种子，则随机性的语义可以取决于 TensorFlow `Session` 的进入和退出。我们建议使用 TensorFlow 2 样式随机性，例如从 TF 1.14 开始使用 `tf.random.experimental.Generator`。这会使用 `tf.Variable` 管理其内部状态。

为了帮助管理期望，TFF 允许对其序列化的 TensorFlow 设置运算级种子，但不能设置计算图级种子。这是因为运算级种子的语义在 TFF 设置中应该更清楚：每次调用包装为 `tf_computation` 的函数时，都会生成确定性序列，并且只有在此调用内，伪随机数生成器所作的任何保证才成立。请注意，这与在 Eager 模式下调用 `tf.function` 的语义不太一样；每次调用 `tf_computation` 时，TFF 都会高效地进入和退出唯一的 `tf.Session`，而在 Eager 模式下反复调用函数则类似于在同一会话内在输出张量上反复调用 `sess.run`。

## 如何做贡献？

请参阅[自述文件](../README.md)和[贡献者准则](../CONTRIBUTING.md)。

## FedJAX 和 TensorFlow Federated 之间是什么关系？

TensorFlow Federated (TFF) 是一个成熟的联合学习和分析框架，旨在简化不同算法和功能的组合过程，并且支持跨不同的模拟和部署场景移植代码。TFF 提供可扩展的运行时并通过其标准 API 支持许多隐私、压缩和优化算法。此外，TFF 还支持[多种类型的 FL 研究](https://www.tensorflow.org/federated/tff_for_research)，[google-research 仓库](https://github.com/google-research/federated)中收集了来自已发表的 Google 论文的示例集合。

相比之下，[FedJAX](https://github.com/google/fedjax) 是一个基于 Python 和 JAX 的轻量级模拟库，专注于用于研究目的的联合学习算法的易用性和快速原型设计。TensorFlow Federated 和 FedJAX 作为单独的项目开发，没有代码可移植性的预期。

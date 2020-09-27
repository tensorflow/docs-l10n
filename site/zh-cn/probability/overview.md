# TensorFlow Probability

TensorFlow Probability 是 TensorFlow 中用于概率推理和统计分析的库。TensorFlow Probability 是 TensorFlow 生态系统的一部分，提供了概率方法与深度网络的集成、使用自动微分的基于梯度的推理，并能扩展到包含硬件加速 (GPU) 和分布式计算的大型数据集和大型模型。

要开始使用 TensorFlow Probability，请参阅[安装指南](./install)并查看 [Python 笔记本教程](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/){:.external}。

## 组件

我们的概率机器学习工具采用如下结构：

### 第 0 层：TensorFlow

*数值运算*（尤其是 `LinearOperator` 类）使无矩阵实现成为可能，这类实现可以利用特定结构（对角、低秩等）实现高效的计算。它由 TensorFlow Probability 团队构建和维护，已成为核心 TensorFlow 中 [`tf.linalg`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/linalg) 的一部分。

### 第 1 层：统计构建块

- *分布* ([`tfp.distributions`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions))：一个包含批次和[广播](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.broadcasting.html){:.external}语义的概率分布和相关统计数据的大型集合。
- *Bijector* ([`tfp.bijectors`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors))：随机变量的可逆和可组合转换。Bijector 提供了类别丰富的变换分布，包括[对数正态分布](https://en.wikipedia.org/wiki/Log-normal_distribution){:.external}等经典示例以及[掩码自回归流](https://arxiv.org/abs/1705.07057){:.external}等复杂的深度学习模型。

### 第 2 层：模型构建

- 联合分布（例如 [`tfp.distributions.JointDistributionSequential`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/joint_distribution_sequential.py)）：一个或多个可能相互依赖的分布上的联合分布。有关使用 TFP 的 `JointDistribution` 进行建模的介绍，请查看[此 colab](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Modeling_with_JointDistribution.ipynb)
- *概率层* ([`tfp.layers`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers))：对其表示的函数具有不确定性的神经网络层，扩展了 TensorFlow 层。

### 第 3 层：概率推理

- *马尔可夫链蒙特卡洛方法* ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc))：通过采样来近似积分的算法。包括[汉密尔顿蒙特卡洛算法](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo){:.external}、随机游走梅特罗波利斯－黑斯廷斯算法以及构建自定义过渡内核的能力。
- *变分推理* ([`tfp.vi`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi))：通过优化来近似积分的算法。
- *优化器* ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer))：随机优化方法，扩展了 TensorFlow 优化器。包括[随机梯度朗之万动力学](http://www.icml-2011.org/papers/398_icmlpaper.pdf){:.external}。
- *蒙特卡洛* ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/monte_carlo))：用于计算蒙特卡洛期望的工具。

TensorFlow Probability 正在积极开发，接口可能变化。

## 示例

除了导航中列出的 [Python 笔记本教程](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/){:.external}外，还提供了一些示例脚本：

- [变分自编码器](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py) - 使用隐代码和变分推理的表示学习。
- [向量量化自编码器](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vq_vae.py) - 使用向量量化的离散表示学习。
- [贝叶斯神经网络](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/bayesian_neural_network.py) - 对其权重具有不确定性的神经网络。
- [贝叶斯逻辑回归](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py) - 二元分类的贝叶斯推断。

## 报告问题

使用 [TensorFlow Probability 问题跟踪器](https://github.com/tensorflow/probability/issues)报告错误或提交功能请求。

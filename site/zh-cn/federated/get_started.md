# TensorFlow Federated

TensorFlow Federated (TFF) 平台包含两层：

- [联合学习 (FL)](federated_learning.md)：将现有 Keras 或非 Keras 机器学习模型插入 TFF 框架的高级接口。无须学习联合学习算法的详细内容，您就可以执行基本任务，如联合训练或评估。
- [Federated Core (FC)](federated_core.md)：通过将 TensorFlow 与强类型函数式编程环境中的分布式通信算子相结合，可简明表示自定义联合算法的底层接口。

首先，请阅读以下教程。这些教程利用实际示例来引导您学习主要 TFF 概念和 API。此外，请务必按照[安装说明](install.md)配置要与 TFF 一起使用的环境。

- [图像分类联合学习](tutorials/federated_learning_for_image_classification.ipynb)介绍了 Federated Learning (FL) API 的主要部分，并在类 MNIST 联合数据上演示了如何使用 TFF 模拟联合学习。
- [文本生成联合学习](tutorials/federated_learning_for_text_generation.ipynb)进一步演示了如何使用 TFF 的 FL API 为语言建模任务优化序列化预训练模型。
- [自定义联合算法，第 1 部分：联合核心简介](tutorials/custom_federated_algorithms_1.ipynb)和[第 2 部分：实现联合平均算法](tutorials/custom_federated_algorithms_2.ipynb)介绍了 Federated Core API (FC API) 提供的主要概念和接口，并演示了如何实现一个简单的联合平均训练算法，以及如何执行联合评估。

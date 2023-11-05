# TensorFlow 联合教程

这些[基于 Colab](https://colab.research.google.com/) 的教程将使用实际示例向您介绍主要的 TFF 概念和 API。请参阅 [TFF 指南](../get_started.md)获取参考文档。

注：TFF 当前需要 Python 3.9 或更高版本，但 [Google Colaboratory](https://research.google.com/colaboratory/) 的托管运行时当前使用 Python 3.7，因此为了运行这些笔记本，您需要使用[自定义本地运行时](https://research.google.com/colaboratory/local-runtimes.html)。

**联合学习入门**

- [图像分类联合学习](federated_learning_for_image_classification.ipynb)介绍了 Federated Learning (FL) API 的主要部分，并在类 MNIST 联合数据上演示了如何使用 TFF 模拟联合学习。
- [文本生成联合学习](federated_learning_for_text_generation.ipynb)进一步演示了如何使用 TFF 的 FL API 为语言建模任务优化序列化预训练模型。
- [调整推荐的学习聚合](tuning_recommended_aggregators.ipynb)展示了如何将 `tff.learning` 中的基本 FL 计算与提供稳健性、差分隐私、压缩等功能的专用聚合结合使用。
- [矩阵分解的联合重建](federated_reconstruction_for_matrix_factorization.ipynb)引入了部分本地联合学习，其中某些客户端参数从不在服务器上聚合。本教程演示了如何使用 Federated Learning API 训练部分局部矩阵分解模型。

**联合分析入门**

- [隐私频繁项](private_heavy_hitters.ipynb)显示了如何使用 `tff.analytics.heavy_hitters` 构建联合分析计算来发现隐私频繁项。

**编写自定义联合计算**

- [构建您自己的联合学习算法](building_your_own_federated_learning_algorithm.ipynb)以联合平均为例，展示了如何使用 TFF Core API 来实现联合学习算法。
- [组合学习算法](composing_learning_algorithms.ipynb) 展示了如何使用 TFF Learning API 轻松实施新的联合学习算法，特别是联合平均的变体。
- [使用 TFF 优化器的自定义联合算法](custom_federated_algorithm_with_tff_optimizers.ipynb)展示了如何使用 `tff.learning.optimizers` 构建联合平均的自定义迭代过程。
- [自定义联合算法，第 1 部分：Federated Core 简介](custom_federated_algorithms_1.ipynb)和[第 2 部分：实现联合平均](custom_federated_algorithms_2.ipynb)介绍了 Federated Core API (FC API) 提供的关键概念和接口。
- [实现自定义聚合](custom_aggregators.ipynb)解释了 `tff.aggregators` 模块背后的设计原则以及实现从客户端到服务器的值的自定义聚合的最佳做法。

**模拟最佳做法**

- [使用加速器 (GPU) 进行 TFF 模拟](simulations_with_accelerators.ipynb)展示了如何将 TFF 的高性能运行时与 GPU 配合使用。

- [使用 ClientData](working_with_client_data.ipynb) 提供了将 TFF 的基于 [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData) 的模拟数据集集成到 TFF 计算中的最佳做法。

**中级和高级教程**

- [随机噪声生成](random_noise_generation.ipynb)指出了在分散计算中使用随机性的一些微妙之处，并提出了最佳做法和推荐模式。

- [使用 tff.federated_select 向特定客户端发送不同的数据](federated_select.ipynb)介绍了 `tff.federated_select` 算子，并提供了一个向不同客户端发送不同数据的自定义联合算法的简单示例。

- [通过 federated_select 和稀疏聚合实现客户端高效大型模型联合学习](sparse_federated_learning.ipynb)展示了如何使用 TFF 来训练非常大的模型，其中每个客户端设备使用 `tff.federated_select` 和稀疏聚合仅下载和更新模型的一小部分。

- [用于联合学习研究的 TFF：模型和更新压缩](tff_for_federated_learning_research_compression.ipynb)演示了如何在 TFF 中使用基于 [tensor_encoding API](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) 构建的自定义聚合。

- [使用 TFF 中的差分隐私的联合学习](federated_learning_with_differential_privacy.ipynb)演示了如何使用 TFF 训练具有用户级差分隐私的模型。

- [TFF 中对 JAX 的支持](../tutorials/jax_support.ipynb)展示了如何在 TFF 中使用 [JAX](https://github.com/google/jax) 计算，演示了如何将 TFF 设计为能够与其他前端和后端 ML 框架进行互操作。

# Keras：TensorFlow 的高级 API

Keras 是 TensorFlow 平台的高级 API。它提供了一个易于理解的高效接口，非常适合解决机器学习 (ML) 问题，重点关注现代深度学习。Keras 涵盖了机器学习工作流的每个步骤，从数据处理到超参数调优，再到部署。它的开发重点是实现快速实验。

借助 Keras，您可以完全利用 TensorFlow 的可扩缩性和跨平台能力。您可以在 TPU Pod 或大型 GPU 集群上运行 Keras，而且可以导出 Keras 模型以在浏览器中或移动设备上运行。此外，您还可以通过 Web API 提供 Keras 模型。

Keras 用于通过实现以下目标来减少认知负荷：

- 提供简单一致的接口。
- 最大限度减少常见用例所需的操作数量。
- 提供清晰、可操作的错误消息。
- 遵循逐步展示复杂性的原则：轻松上手，并且可以通过边用边学来完成高级工作流。
- 帮助您编写简洁、易读的代码。

## 哪些人应当使用 Keras

简而言之，每个 TensorFlow 用户都应当默认使用 Keras API。无论您是工程师、研究员还是 ML 从业者，都应该从 Keras 开始。

有一些用例（例如，在 TensorFlow 上构建工具或开发您自己的高性能平台）需要低级 [TensorFlow Core API](https://www.tensorflow.org/guide/core)。但是，如果您的用例不属于 [Core API 应用](https://www.tensorflow.org/guide/core#core_api_applications)之一，您应该更愿意使用 Keras。

## Keras API 组件

Keras 的核心数据结构是[层](https://keras.io/api/layers/)和[模型](https://keras.io/api/models/)。层是简单的输入/输出转换，模型是层的有向无环图 (DAG)。

### 层

`tf.keras.layers.Layer` 类是 Keras 中的基本抽象。`Layer` 封装了状态（权重）和一些计算（在 `tf.keras.layers.Layer.call` 方法中定义）。

由层创建的权重可以是可训练的，也可以是不可训练的。层支持以递归方式组合：如果将层实例分配为另一个层的特性，则外层将开始跟踪内层创建的权重。

您也可以使用层来处理数据预处理任务，如归一化和文本向量化。预处理层可以在训练期间或之后直接包含到模型中，这会使模型便于移植。

### 模型

模型是将层分组在一起并且可以在数据上进行训练的对象。

最简单的模型类型是[`Sequential` 模型](https://www.tensorflow.org/guide/keras/sequential_model)，它是层的线性堆叠。对于更复杂的架构，可以使用 [Keras 函数式 API](https://www.tensorflow.org/guide/keras/functional_api)（允许构建任意层的计算图），也可以[使用子类化从头开始编写模型](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)。

`tf.keras.Model` 类具有内置的训练和评估方法：

- `tf.keras.Model.fit`：将模型训练固定数量的周期。
- `tf.keras.Model.predict`：生成输入样本的输出预测。
- `tf.keras.Model.evaluate`：返回模型的损失和指标值； 通过 `tf.keras.Model.compile` 方法配置。

您可以利用这些方法访问以下内置训练功能：

- [回调](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)。您可以利用内置回调实现提前停止、模型检查点和 [TensorBoard](https://www.tensorflow.org/tensorboard) 监控。您还可以[实现自定义回调](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks)。
- [分布式训练](https://www.tensorflow.org/guide/keras/distributed_training)。您可以将训练轻松扩展到多个 GPU、TPU 或设备。
- 步熔合。使用 `tf.keras.Model.compile` 中的 `steps_per_execution` 参数，您可以在单次 `tf.function` 调用中处理多个批次，从而显著提高 TPU 上的设备利用率。

有关如何使用 `fit` 的详细概述，请参阅[训练和评估指南](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)。要了解如何自定义内置训练和评估循环，请参阅[自定义 `fit()` 的功能](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)。

### 其他 API 和工具

Keras 提供了许多其他用于深度学习的 API 和工具，包括以下各项：

- [优化器](https://keras.io/api/optimizers/)
- [指标](https://keras.io/api/metrics/)
- [损失](https://keras.io/api/losses/)
- [数据加载实用工具](https://keras.io/api/data_loading/)

有关可用 API 的完整列表，请参阅 [Keras API 参考](https://keras.io/api/)。要了解有关其他 Keras 项目和计划的更多信息，请参阅 [Keras 生态系统](https://keras.io/getting_started/ecosystem/)。

## 后续步骤

要开始将 Keras 与 TensorFlow 结合使用，请查看以下主题：

- [序贯模型](https://www.tensorflow.org/guide/keras/sequential_model)
- [函数式 API](https://www.tensorflow.org/guide/keras/functional)
- [使用内置方法进行训练和评估](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [通过子类化创建新的层和模型](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [序列化和保存](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [使用预处理层](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [自定义 fit() 的功能](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [从头开始编写训练循环](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [使用 RNN](https://www.tensorflow.org/guide/keras/rnn)
- [了解遮盖和填充](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [编写自己的回调](https://www.tensorflow.org/guide/keras/custom_callback)
- [迁移学习和微调](https://www.tensorflow.org/guide/keras/transfer_learning)
- [多 GPU 和分布式处理/a0}](https://www.tensorflow.org/guide/keras/distributed_training)

要详细了解 Keras，请参阅 [keras.io](http://keras.io) 上的以下主题：

- [关于 Keras](https://keras.io/about/)
- [面向工程师的 Keras 简介](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [面向研究人员的 Keras 简介](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Keras API 参考](https://keras.io/api/)
- [Keras 生态系统](https://keras.io/getting_started/ecosystem/)

# 在 TFX 中使用其他 ML 框架

作为平台的 TFX 是框架中立的，可以与其他 ML 框架（例如 JAX、scikit-learn）一起使用。

对于模型开发者而言，这意味着他们无需重写在另一个 ML 框架中实现的模型代码，而是可以在 TFX 中按原样重用大量训练代码，并从其他功能 TFX 和其余 TensorFlow 生态系统服务中受益。

TFX 流水线 SDK 和 TFX 中的大多数模块（例如流水线编排器）对 TensorFlow 没有任何直接依赖，但有一些方面是面向 TensorFlow 的，例如数据格式。考虑到特定建模框架的需求，TFX 流水线可用于在任何其他基于 Python 的 ML 框架中训练模型。这包括 Scikit-learn、XGBoost 和 PyTorch 等。配合使用标准 TFX 组件和其他框架的一些注意事项包括：

- **ExampleGen** 在 TFRecord 文件中输出 [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)。它是训练数据的通用表示，下游组件使用 [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md) 将其读取为内存中的 Arrow/RecordBatch，而这可以被进一步转换为 `tf.dataset`、`Tensors` 或其他格式。tf.train.Example/TFRecord 以外的有效负载/文件格式正在考虑中，但对于 TFXIO 用户来说，它应该是一个黑盒。
- **Transform** 可以用来生成转换后的训练示例，而不管使用什么框架进行训练，但如果模型格式不是 `saved_model`，用户将无法将转换图嵌入到模型中。在这种情况下，模型预测需要采用转换后的特征而不是原始特征，并且用户可以先运行转换作为预处理步骤，然后再在提供服务时调用模型预测。
- **Trainer** 支持[通用训练](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer)，因此用户可以使用任何 ML 框架训练他们的模型。
- **Evaluator** 默认仅支持 `saved_model`，但用户可以提供一个 UDF 来生成模型评估的预测。

在非基于 Python 的框架中训练模型需要在 Docker 容器中隔离自定义训练组件，这是在 Kubernetes 等容器化环境中运行的流水线的一部分。

## JAX

[JAX](https://github.com/google/jax) 是 Autograd 和 XLA 二者相结合，用于高性能的机器学习研究。[Flax](https://github.com/google/flax) 是一个用于 JAX 的神经网络库和生态系统，旨在实现灵活性。

使用 [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf)，我们能够将经过训练的 JAX/Flax 模型转换为 `saved_model` 格式，该格式可以在进行通用训练和模型评估时在 TFX 中无缝使用。有关详细信息，请查看此[示例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py)。

## scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) 是 Python 编程语言的机器学习库。我们有一个 e2e [示例](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins)，其中包含 TFX-Addons 中的定制培训和评估。

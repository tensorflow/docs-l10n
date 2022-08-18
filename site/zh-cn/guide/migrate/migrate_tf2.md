# TF1.x -&gt; TF2 迁移概述

TensorFlow 2 在几个方面与 TF1.x 完全不同。您仍然可以针对 TF2 二进制安装运行未修改的 TF1.x 代码（[contrib 除外](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)），如下所示：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

但是，这*不是*运行 TF2 行为和 API，并且可能无法使用为 TF2 编写的代码按预期工作。如果您没有在 TF2 行为激活的情况下运行，那么实际上是在 TF2 安装上运行 TF1.x。阅读 [TF1 与 TF2 行为指南](./tf1_vs_tf2.ipynb)，详细了解 TF2 与 TF1.x 有何不同。

本指南概述了将 TF1.x 代码迁移到 TF2 的过程。这让您能够利用新的和未来的功能改进，并使您的代码更简单、更高效且更易于维护。

如果您正在使用 `tf.keras` 的高级 API 并专门使用 `model.fit` 进行训练，那么您的代码应该或多或少与 TF2 完全兼容，但下列注意事项除外：

- TF2 为 Keras 优化器设置了新的[默认学习率](../../guide/effective_tf2.ipynb#optimizer_defaults)。
- TF2 [可能已经更改](../../guide/effective_tf2.ipynb#keras_metric_names)记录指标的“名称”。

## TF2 迁移过程

在迁移之前，通过阅读[指南](./tf1_vs_tf2.ipynb)了解 TF1.x 和 TF2 之间的行为和 API 差异。

1. 运行自动化脚本，将您使用的一些 TF1.x API 转换为 `tf.compat.v1`。
2. 移除旧的 `tf.contrib` 符号（检查 [TF Addons](https://github.com/tensorflow/addons) 和 [TF-Slim](https://github.com/google-research/tf-slim)）。
3. 让您的 TF1.x 模型前向传递在 TF2 中运行并启用 Eager Execution。
4. 将用于训练循环和保存/加载模型的 TF1.x 代码升级到 TF2 等效项。
5. （可选）将兼容 TF2 的 `tf.compat.v1` API 迁移到惯用的 TF2 API。

以下各部分进一步阐述了上述步骤。

## 运行符号转换脚本

这会在重写代码符号以针对 TF 2.x 二进制文件运行时执行初始传递，但不会使您的代码符合 TF 2.x 的惯例，也不会自动使您的代码与 TF2 行为兼容。

您的代码很可能仍会使用 `tf.compat.v1` 端点来访问占位符、会话、集合和其他 TF1.x 样式的功能。

阅读[指南](./upgrade.ipynb)以详细了解使用符号转换脚本的最佳做法。

## 移除 `tf.contrib` 的用法

`tf.contrib` 模块已被淘汰，它的几个子模块已被集成到核心 TF2 API 中。其他子模块现在被分拆到其他项目中，如 [TF IO](https://github.com/tensorflow/io) 和 [TF Addons](https://www.tensorflow.org/addons/overview)。

大量较旧的 TF1.x 代码使用 [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) 库，此库与 TF1.x 一起打包为 `tf.contrib.layers`。将 Slim 代码迁移到 TF2 时，将 Slim API 用法切换为指向 [tf-slim pip 软件包](https://pypi.org/project/tf-slim/)。然后，阅读[模型映射指南](https://tensorflow.org/guide/migrate/model_mapping#a_note_on_slim_and_contriblayers)，了解如何转换 Slim 代码。

或者，如果您使用 Slim 预训练模型，可以考虑尝试来自 `tf.keras.applications` 的 Keras 预训练模型或 [TF Hub](https://tfhub.dev/s?tf-version=tf2&q=slim) 上从原始 Slim 代码导出的 TF2 `SavedModel`。

## 让 TF1.x 模型前向传递在启用 TF2 行为的情况下运行

### 跟踪变量和损失

[TF2 不支持全局集合。](./tf1_vs_tf2.ipynb#no_more_globals)

TF2 中的 Eager Execution 不支持基于 `tf.Graph` 集合的 API。这会影响您构造和跟踪变量的方式。

对于新的 TF2 代码，您需要使用 `tf.Variable` 而不是 `v1.get_variable`，并使用 Python 对象而不是 `tf.compat.v1.variable_scope` 来收集和跟踪变量。通常是下列对象之一：

- `tf.keras.layers.Layer`
- `tf.keras.Model`
- `tf.Module`

使用 `Layer`、`Module` 或 `Model` 对象的 `.variables` 和 `.trainable_variables` 特性聚合变量列表（如 `tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`）。

这些 `Layer` 和 `Model` 类会实现多种其他属性，因此不需要全局集合。它们的 `.losses` 属性可以代替 `tf.GraphKeys.LOSSES` 集合。

阅读[模型映射指南](./model_mapping.ipynb)，详细了解如何使用 TF2 代码建模填充码将基于 `get_variable` 和 `variable_scope` 的现有代码嵌入到 `Layers`、`Models` 和 `Modules` 中。这样一来，无需大量重写即可在启用 Eager Execution 的情况下执行前向传递。

### 适应其他行为变更

如果[模型映射指南](./model_mapping.ipynb)本身不足以让您的模型前向传递运行其他可能更详细的行为变更，请参阅 [TF1.x 与 TF2 行为](./tf1_vs_tf2.ipynb)指南以了解其他行为变更以及如何适应它们。另外，请查看[通过子类化制作新层和模型指南](https://tensorflow.org/guide/keras/custom_layers_and_models.ipynb)以了解详情。

### 验证您的结果

有关如何（以数字方式）验证模型在启用 Eager Execution 时正确运行的简单工具和指南，请参阅[模型验证指南](./validate_correctness.ipynb)。与[模型映射指南](./model_mapping.ipynb)结合使用时，您会发现这特别有用。

## 升级训练、评估和导入/导出代码

使用 `v1.Session` 样式的 `tf.estimator.Estimator` 和其他基于集合的方式构建的 TF1.x 训练循环与 TF2 的新行为不兼容。务必迁移所有 TF1.x 训练代码，因为将其与 TF2 代码结合使用可能会导致意外行为。

您可以从多种策略中进行选择来实现此目的。

最高级别的方式是使用 `tf.keras`。Keras 中的高级函数负责管理许多您自己编写训练循环时容易遗漏的低级细节。例如，这些函数会自动收集正则化损失，并在调用模型时设置 `training=True` 参数。

要了解如何迁移 `tf.estimator.Estimator` 代码以使用 [vanilla](./migrating_estimator.ipynb#tf2_keras_training_api) 和[自定义](./migrating_estimator.ipynb#tf2_keras_training_api_with_custom_training_step) `tf.keras` 训练循环，请参阅 [Estimator 迁移指南](./migrating_estimator.ipynb)。

自定义训练循环让您可以更好地控制模型，例如跟踪各个层的权重。阅读关于[从头开始构建训练循环](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)的指南，了解如何使用 `tf.GradientTape` 检索模型权重并使用它们来更新模型。

### 将 TF1.x 优化器转换为 Keras 优化器

`tf.compat.v1.train` 中的优化器（如 [Adam 优化器](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer)和[梯度下降优化器](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer)）在 `tf.keras.optimizers` 中具有等效项。

下表总结了如何将这些旧版优化器转换为 Keras 等效项。除非需要额外的步骤（例如[更新默认学习率](../../guide/effective_tf2.ipynb#optimizer_defaults)），否则可以直接将 TF1.x 版本替换为 TF2 版本。

请注意，转换优化器[可能会使旧的检查点不兼容](./migrating_checkpoints.ipynb)。

<table>
  <tr>
    <th>TF1.x</th>
    <th>TF2</th>
    <th>额外步骤</th>
  </tr>
  <tr>
    <td>`tf.v1.train.GradientDescentOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>无</td>
  </tr>
  <tr>
    <td>`tf.v1.train.MomentumOptimizer`</td>
    <td>`tf.keras.optimizers.SGD`</td>
    <td>包含 `momentum` 参数</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdamOptimizer`</td>
    <td>`tf.keras.optimizers.Adam`</td>
    <td>将 `beta1` 和 `beta2` 参数重命名为 `beta_1` 和 `beta_2`</td>
  </tr>
  <tr>
    <td>`tf.v1.train.RMSPropOptimizer`</td>
    <td>`tf.keras.optimizers.RMSprop`</td>
    <td>将 `decay` 参数重命名为 `rho`</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdadeltaOptimizer`</td>
    <td>`tf.keras.optimizers.Adadelta`</td>
    <td>无</td>
  </tr>
  <tr>
    <td>`tf.v1.train.AdagradOptimizer`</td>
    <td>`tf.keras.optimizers.Adagrad`</td>
    <td>无</td>
  </tr>
  <tr>
    <td>`tf.v1.train.FtrlOptimizer`</td>
    <td>`tf.keras.optimizers.Ftrl`</td>
    <td>移除 `accum_name` 和 `linear_name` 参数</td>
  </tr>
  <tr>
    <td>`tf.contrib.AdamaxOptimizer`</td>
    <td>`tf.keras.optimizers.Adamax`</td>
    <td>将 `beta1` 和 `beta2` 参数重命名为 `beta_1` 和 `beta_2`</td>
  </tr>
  <tr>
    <td>`tf.contrib.Nadam`</td>
    <td>`tf.keras.optimizers.Nadam`</td>
    <td>将 `beta1` 和 `beta2` 参数重命名为 `beta_1` 和 `beta_2`</td>
  </tr>
</table>

注：在 TF2 中，所有 ε（数值稳定性常数）现在默认为 `1e-7`，而不是 `1e-8`。在大多数用例中，这种差异可以忽略不计。

### 升级数据输入流水线

您可以通过多种方式将数据馈送给 `tf.keras` 模型。这些方式接受 Python 生成器和 Numpy 数组作为输入。

推荐使用 `tf.data` 软件包将数据馈送给模型，其中包含一组用于处理数据的高性能类。属于 `tf.data` 的 `dataset` 高效、表现力强，可与 TF2 良好地集成。

它们可以直接传递给 `tf.keras.Model.fit` 方法。

```python
model.fit(dataset, epochs=5)
```

它们可以直接通过标准 Python 进行迭代：

```python
for example_batch, label_batch in dataset:
    break
```

如果您仍在使用 `tf.queue`，则现在仅支持将它们作为数据结构，而不能作为输入流水线。

您还应迁移所有使用 `tf.feature_columns` 的特征预处理代码。阅读[迁移指南](./migrating_feature_columns.ipynb)来了解详情。

### 保存和加载模型

TF2 使用基于对象的检查点。阅读[检查点迁移指南](./migrating_checkpoints.ipynb)，详细了解如何迁移基于名称的 TF1.x 检查点。另外，请阅读核心 TensorFlow 文档中的[检查点指南](https://www.tensorflow.org/guide/checkpoint)。

保存的模型没有重大兼容性问题。要了解如何将 TF1.x 中的 `SavedModel` 迁移到 TF2，请阅读 <a href="./saved_model.ipynb" data-md-type="link">`SavedModel` 指南</a>。一般而言：

- TF1.x saved_model 可以在 TF2 中运行。
- 如果所有运算均受到支持，TF2 saved_model 可以在 TF1.x 中运行。

另请参阅 `SavedModel` 迁移指南中的 [`GraphDef` 部分](./saved_model.ipynb#graphdef_and_metagraphdef)，详细了解如何使用 `Graph.pb` 和 `Graph.pbtxt` 对象。

## （可选）迁移 `tf.compat.v1` 符号

`tf.compat.v1` 模块包含完整的 TF1.x API 及其原始语义。

即使按照上述步骤操作并最终获得与所有 TF2 行为完全兼容的代码，也可能会多次提及 `compat.v1` API 恰好与 TF2 兼容。您应当避免将这些旧的 `compat.v1` API 用于您编写的任何新代码，尽管它们将继续在您已经编写的代码中运行。

但是，可以选择将现有用法迁移到非旧版 TF2 API。各个 `compat.v1` 符号的文档字符串通常会解释如何将它们迁移到非旧版 TF2 API。此外，[模型映射指南中关于增量迁移到惯用 TF2 API 的部分](./model_mapping.ipynb#incremental_migration_to_native_tf2)也可能对此有所帮助。

## 资源和延伸阅读

如前面所述，最好将所有 TF1.x 代码迁移到 TF2。阅读 TensorFlow 指南的[迁移到 TF2 部分](https://tensorflow.org/guide/migrate)中的指南以了解详情。

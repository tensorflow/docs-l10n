<!--* freshness: { owner: 'kempy' reviewed: '2021-03-09' } *-->

# 可重用 SavedModel

## 简介

TensorFlow Hub 用于托管 TensorFlow 2 的 SavedModel 等资产。开发者可以使用 `obj = hub.load(url)` 将这些资产重新加载到 Python 程序中 [[了解详情](tf2_saved_model)]。返回的 `obj` 为 `tf.saved_model.load()` 的结果（请参阅 TensorFlow 的 [SavedModel 指南](https://www.tensorflow.org/guide/saved_model)）。此对象可以具有任意特性，包括 tf.functions、tf.Variables（从其预训练值初始化）、其他资源，以及递归地包含更多此类对象。

本页将介绍在 TensorFlow Python 程序中*重用*资源而需要通过加载的 `obj` 实现的接口。符合此类接口条件的 SavedModel 称为*可重用 SavedModel*。

“重用”意味着围绕 `obj` 构建更大的模型，并且应当能够进行微调。微调意味着进一步训练作为周围模型一部分的加载 `obj` 中的权重。损失函数和优化器由周围模型确定；`obj` 仅定义输入到输出激活的映射（“前向传递”），可能包括诸如随机失活或批次归一化之类的技术。

按照上述概念，对于所有要重用的 SavedModel，**TensorFlow Hub 团队均建议实现可重用 SavedModel 接口**。`tensorflow_hub` 库中的许多实用工具（特别是 `hub.KerasLayer`）都需要 SavedModel 来实现。

### 与 SignatureDef 的关系

就 tf.functions 和其他 TF2 功能而言，此接口与 SavedModel 的签名不同，后者自 TF1 时期便可用，并沿用到 TF2 中用于推断（例如将 SavedModels 部署到 TF Serving 或 TF Lite）。推断的签名的表达力不够充分，不支持微调，而 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) 为重用模型提供了更自然和更具表达力的 [Python API](https://www.tensorflow.org/tutorials/customization/performance)。

### 与模型构建库的关系

可重用 SavedModel 仅使用 TensorFlow 2 基元，与任何特定的模型构建库（例如 Keras 或 Sonnet）无关。这有助于在模型构建库间重用，而无需依赖原始模型构建代码。

将可重用 SavedModel 加载到任何给定模型构建库或从这些模型构建库保存 SavedModel，都需要进行一些调整。对于 Keras，[hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) 提供加载，Keras 中以 SavedModel 格式保存的内置功能已针对 TF2 重新设计，旨在提供此接口的超集（请参阅 2019 年 5 月的 [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md)）。

### 与任务特定的“常用 SavedModel API”的关系

此页面上的接口定义允许任何数量和类型的输入和输出。[TF Hub 的常用 SavedModel API](common_saved_model_apis/index.md) 使用用法惯例针对特定任务优化了此通用接口，使模型可以轻松地互换。

## 接口定义

### 特性

可重用 SavedModel 为 TensorFlow 2 SavedModel，`obj = tf.saved_model.load(...)` 将返回具有以下特性的对象：

- `__call__`：必选。实现模型计算（“前向传递”）的 tf.function，遵循以下规范。

- `variables`：tf.Variable 对象列表，列出了 `__call__` 的任何可能调用使用的所有变量，包括可训练和不可训练变量。

    如果为空，则可省略此列表。

    注：为方便记录，在加载 TF1 SavedModel 以表示其 `GLOBAL_VARIABLES` 集合时，此名称与 `tf.saved_model.load(...)` 合成的特性一致。

- `trainable_variables`：`v.trainable` 对于所有元素均为 true 的 tf.Variable 对象列表。这些变量必须是 `variables` 的子集。它们是微调对象时要训练的变量。SavedModel 创建者在此可以选择省略一些最初可训练的变量，以表明在微调期间不应修改这些变量。

    如果为空，则可省略此列表，特别是当 SavedModel 不支持微调时。

- `regularization_losses`：tf.function 列表，其中的每个函数均接受零输入并返回单个标量浮点张量。为了进行微调，建议 SavedModel 用户将这些函数作为附加正则化项包括在损失中（在最简单的情况下，无需进一步缩放）。通常，它们用于表示权重正则化器。（由于缺少输入，这些 tf.function 无法表达激活正则化器。）

    如果为空，则可省略此列表，特别是当 SavedModel 不支持微调或不希望规定权重正则化时。

### `__call__` 函数

恢复的 SavedModel `obj` 具有 `obj.__call__` 特性，为恢复的 tf.function，支持按以下方式调用 `obj`。

Synopsis（伪代码）：

```python
outputs = obj(inputs, trainable=..., **kwargs)
```

#### 参数

参数如下。

- 有一个位置必选参数，其中包含 SavedModel 的一批输入激活。类型为以下三者之一：

    - 单个张量，针对单个输入；
    - 张量列表，针对未命名输入的有序序列；
    - 张量字典，由一组特定输入名称键控。

    （此接口的未来修订版本可能支持更多的常规嵌套。）SavedModel 创建者选择其中一种类型，并选择张量形状和数据类型。如果适用，不应定义形状的某些维度（尤其是批次大小）。

- 有一项可选的关键字参数 `training`，该参数接受 Python 布尔值 `True` 或 `False`。默认值为 `False`。如果模型支持微调，并且其计算在两者（例如，随机失活和批次归一化）之间有所不同，则可以使用此参数实现该区别。否则，可以不使用此参数。

    `__call__` 不必接受张量值 `training` 参数。如果有必要，则由调用者使用 `tf.cond()` 在它们之间进行调度。

- SavedModel 创建者可以选择接受更多具有特定名称的可选 `kwargs`。

    - 对于张量值参数，SavedModel 创建者可定义其允许的数据类型和形状。`tf.function` 接受在使用 tf.TensorSpec 输入跟踪的参数上使用 Python 默认值。此类参数可用于自定义 `__call__` 中涉及的数字超参数（例如，随机失活率）。

    - 对于 Python 值参数，SavedModel 创建者可定义其允许的值。此类参数可用作标志，在跟踪函数中进行离散选择（但要注意跟踪记录的组合爆炸式增长）。

恢复的 `__call__` 函数必须提供对所有允许的参数组合的跟踪记录。在 `True` 与 `False` 之间切换 `training` 不得改变参数的容许性。

#### 结果

调用 `obj` 所得的 `outputs` 可以是：

- 单个张量，针对单个输出；
- 张量列表，针对未命名输出的有序序列；
- 张量字典，由一组特定输出名称键控。

（此接口的未来修订版本可能支持更多的常规嵌套。）返回类型可能会因 Python 值关键字参数而异。这样可以考虑产生额外输出的标志。SavedModel 创建者定义输出数据类型和形状及其对输入的依赖。

### 命名可调用对象

按照上述方式，可重用 SavedModel 可以通过放入命名子对象（例如 `obj.foo`、`obj.bar` 等）来提供多个模型部分。每个子对象提供一个 `__call__` 方法，并支持特定于该模型部分的变量等方面的特性。对于上述示例，将包括 `obj.foo.__call__`、`obj.foo.variables` 等。

请注意，此接口*不*涵盖直接将裸 tf.function 添加为 `tf.foo` 的方式。

可重用 SavedModel 的用户只应处理一级嵌套 (`obj.bar`，而非 `obj.bar.baz`)。（此接口的未来修订版本可能支持更深层的嵌套，并且可能放弃对顶级对象本身可调用的要求。）

## 结束语

### 与进程中 API 的关系

本文档介绍了由 tf.function 和 tf.Variable 等基元组成的 Python 类的接口，这些基元通过 `tf.saved_model.save()` 和 `tf.saved_model.load()` 执行序列化和反序列化。但是，传递给 `tf.saved_model.save()` 的原始对象已经具有该接口。适配该接口可以在单个 TensorFlow 程序中实现跨建模 API 交换模型部分。

# TensorFlow 版本兼容性

本文面向需要在不同版本的 TensorFlow 之间向后兼容（针对代码或者数据）的用户，以及想要修改 TensorFlow 并同时保持兼容性的开发者。

## 语义化版本控制 2.0

TensorFlow 的公开 API 遵循语义化版本控制 2.0 ([semver](http://semver.org))。每个版本的 TensorFlow 都采用 `MAJOR.MINOR.PATCH` 形式。例如，TensorFlow 版本 1.2.3 具有 `MAJOR` 版本 1、`MINOR` 版本 2 和 `PATCH` 版本 3。每个数字的更改具有以下含义：

- **MAJOR**：可能向后不兼容的更改。与之前的主要版本兼容的代码和数据不一定与新版本兼容。不过，在一些情况下，现有 TensorFlow 计算图和检查点可以迁移到新版本；请参阅[计算图和检查点的兼容性](#compatibility_of_graphs_and_checkpoints)，详细了解数据兼容性。

- **MINOR**：向后兼容的功能、速度提升等。与之前的次要版本兼容*且*仅依赖于非实验性公开 API 的代码和数据将继续保持兼容。有关公开 API 的详细信息，请参阅[兼容范围](#what_is_covered)。

- **PATCH**：向后兼容的错误修复。

例如，与版本 0.12.1 相比，版本 1.0.0 引入了向后*不兼容*的更改。不过，版本 1.1.1 向后*兼容*版本 1.0.0。<a name="what_is_covered"></a>

## 兼容范围

只有 TensorFlow 的公开 API 在次要和补丁版本中可向后兼容。公开 API 包括以下各项：

- `tensorflow` 模块及其子模块中所有记录的 [Python](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/api_docs/python) 函数和类，但以下符号除外：

    - private 符号：任何函数、类等，其名称以 `_` 开头
    - 实验性和 `tf.contrib` 符号，请参阅[下文](#not_covered)了解详细信息。

    请注意，`examples/` 和 `tools/` 目录中的代码无法通过 `tensorflow` Python 模块到达，因此不在兼容性保证范围内。

    如果符号可通过 `tensorflow` Python 模块或其子模块到达，但未记录，那么它也**不**属于公开 API。

- 兼容性 API（在 Python 中，为 `tf.compat` 模块）。在主要版本中，我们可能会发布实用工具和其他端点来帮助用户过渡到新的主要版本。这些 API 符号已弃用且不受支持（即，我们不会添加任何功能，除了修复一些漏洞外，也不会修复错误），但它们在我们的兼容性保证范围内。

- [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)。

- 下列协议缓冲区文件：

    - [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    - [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    - [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    - [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    - [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    - [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    - [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    - [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    - [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    - [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>

## *不*兼容范围

TensorFlow 的某些部分可能随时以向后不兼容的方式更改。包括以下部分：

- **实验性 API**：为了方便开发，我们将一些明确标记为实验性的 API 符号排除在兼容性保证范围外。特别是，以下符号不在任何兼容性保证范围内：

    - `tf.contrib` 模块或其子模块中的任何符号；
    - 名称包含 `experimental` 或 `Experimental` 的任何符号（模块、函数、参数、属性、类或常量）；或
    - 完全限定名称包含模块或类（其自身为实验性）的任何符号。这类符号包括称为 `experimental` 的任何协议缓冲区的字段和子消息。

- **其他语言**：Python 和 C 以外的其他语言中的 TensorFlow API，例如：

    -  [C++](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/install/lang_c.md)（通过 [`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc) 中的头文件公开）。
    - [Java](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/install/lang_java.md)
    - [Go](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/install/lang_go.md)
    - [JavaScript](https://tensorflow.google.cn/js)

- **复合运算的详细信息**：Python 中的许多 public 函数都会在计算图中展开为多个基元运算，这些详细信息将成为任何计算图（以 `GraphDef` 形式保存到磁盘中）的一部分。对于次要版本，这些详细信息可能会发生变化。特别是，检查计算图之间精确匹配度的回归测试在次要版本中可能会中断，即使计算图的行为应保持不变且现有检查点仍然有效。

- **浮点数值详细信息**：运算计算的特定浮点值可能会随时变化。用户应当仅依赖于近似准确率和数值稳定性，而不是计算的特定位数。次要版本和补丁版本中数值公式的变化应当产生相当或更高的准确率，需要说明的是，机器学习中特定公式的准确率提高可能会导致整个系统的准确率降低。

- **随机数**：计算的特定随机数可能会随时变化。用户应当仅依赖于近似正确的分布和统计强度，而不是计算的特定位数。请参阅[随机数生成](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/guide/random_numbers.ipynb)指南，了解详细信息。

- **分布式 Tensorflow 中的版本偏差**：不支持在一个集群中运行两个不同版本的 TensorFlow。无法保证线路协议的向后兼容性。

- **错误**：如果当前实现被明显中断（即，如果当前实现与文档冲突或已知且定义明确的预期行为由于错误未正确实现），我们保留实施向后不兼容行为（即使不是 API）更改的权利。例如，如果优化器声明要实现已知的优化算法，但由于错误而不匹配该算法，我们将修复优化器。修复可能会损坏依赖错误行为实现收敛的代码。我们会在版本说明中注明这些更改。

- **未使用的 API**：我们保留对未发现使用记录（通过 GitHub 搜索审查 TensorFlow 使用情况）的 API 实施向后不兼容更改的权利。在执行任何此类更改之前，我们会在 [announce@ 邮寄名单](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce)上公布更改意图，提供有关如何解决任何中断问题的说明（如适用），并等待两个星期，让我们的社区有机会分享反馈。

- **错误行为**：我们会将错误替换为非错误行为。例如，我们会更改函数以计算结果，而不是引发错误，即使该错误已有记录。我们还保留更改错误消息文本的权利。此外，错误类型可能会变化，除非特定错误条件的异常类型已在文档中指定。

<a name="compatibility_of_graphs_and_checkpoints"></a>

## SavedModel（计算图和检查点）的兼容性

SavedModel 是 TensorFlow 程序中使用的首选序列化格式。SavedModel 包括两部分：编码为 `GraphDefs` 的一个或多个计算图以及一个检查点。计算图说明了要运行的运算的数据流，检查点包含计算图中变量的已保存张量值。

许多 TensorFlow 用户创建 SavedModel，并使用新版本的 TensorFlow 加载和执行这些 SavedModel。按照 [semver](https://semver.org)，使用一个版本的 TensorFlow 编写的 SavedModel 可以使用主要版本相同的新版本 TensorFlow 进行加载和评估。

我们还为*受支持的* SavedModel 提供其他保证。我们将 TensorFlow 主要版本 `N` 中使用**仅非弃用、非实验性、非兼容性 API** 创建的 SavedModel 称为<em data-md-type="emphasis">版本 `N` 中受支持的 SavedModel</em>。TensorFlow 主要版本 `N` 中支持的任何 SavedModel 都可以使用 TensorFlow 主要版本 `N+1` 加载和执行。不过，构建或修改此类模型所需的功能可能不再提供，因此该保证仅适用于未修改的 SavedModel。

我们会尽可能长期保留向后兼容性，使序列化文件在很长一段时间内可用。

### GraphDef 兼容性

计算图通过 `GraphDef` 协议缓冲区进行序列化。为了方便对计算图进行向后不兼容更改，每个 `GraphDef` 都具有与 TensorFlow 版本独立的版本号。例如，`GraphDef` 版本 17 已弃用 `inv` 运算而改用 `reciprocal`。语义为：

- 每个版本的 TensorFlow 都支持 `GraphDef` 版本的间隔。此间隔将在补丁版本之间保持一致，且仅会在次要版本之间增大。仅针对 TensorFlow 的主要版本（并且仅符合 SavedModel 保证的版本支持）停止支持 `GraphDef` 版本。

- 新创建的计算图会被分配最新的 `GraphDef` 版本号。

- 如果给定版本的 TensorFlow 支持计算图的 `GraphDef` 版本，它在加载和评估时的行为将与用于生成它的 TensorFlow 版本的行为一致（除了上面提到的浮点数值详细信息和随机数），不受 TensorFlow 主要版本的影响。特别是，与一个版本的 TensorFlow（例如 SavedModel）中的检查点文件兼容的 GraphDef 将保持与后续版本中的该检查点兼容，前提是该 GraphDef 受支持。

    请注意，这仅适用于 GraphDef（和 SavedModel）中的序列化计算图：读取检查点的 *Code* 可能无法读取运行不同版本 TensorFlow 的相同代码生成的检查点。

- 如果 `GraphDef` *上*限在（次要）版本中增加到 X，*下*限增加到 X 至少需要 6 个月。例如（我们在此处使用假想的版本号）：

    - TensorFlow 1.2 可能支持 `GraphDef` 版本 4 至 7。
    - TensorFlow 1.3 可以添加 `GraphDef` 版本 8 且支持版本 4 至 8。
    - 至少 6 个月后，TensorFlow 2.0.0 可以停止支持版本 4 至 7，仅支持版本 8。

    请注意，因为 TensorFlow 主要版本的发布周期通常间隔 6 个月以上，上述详细说明的受支持 SavedModel 的保证将比 6 个月的 GraphDef 保证更稳定。

最后，在停止支持 `GraphDef` 版本时，我们会尝试提供工具，用于将计算图自动转换为较新的受支持 `GraphDef` 版本。

## 扩展 TensorFlow 时的计算图和检查点兼容性

本部分仅与对 `GraphDef` 格式进行不兼容更改相关，例如，添加运算、移除运算或更改现有运算的功能。上一部分应当能够满足大多数用户的需求。

<a id="backward_forward"></a>

### 向后和部分向前兼容性

我们的版本管理方案有三个要求：

- **向后兼容性**，支持加载使用旧版本的 TensorFlow 创建的计算图和检查点。
- **向前兼容性**，支持计算图或检查点的生产者在使用者之前升级到新版本 TensorFlow 的情况。
- 支持以不兼容的方式演进 TensorFlow，例如，移除运算、添加特性和移除特性。

请注意，`GraphDef` 版本机制与 TensorFlow 版本独立，`GraphDef` 格式的向后不兼容更改仍受语义化版本控制限制。这意味着，只可以在 TensorFlow 的 `MAJOR` 版本之间（例如 `1.7` 至 `2.0`）移除或更改功能。而且，补丁版本之间（例如，`1.x.1` 至 `1.x.2`）会强制向前兼容。

为了实现向后和向前兼容性，并了解何时在格式中强制执行更改，计算图和检查点包含表明它们生成时间的元数据。以下几部分详细介绍了 TensorFlow 实现和演进 `GraphDef` 版本的指南。

### 独立的数据版本方案

计算图和检查点具有不同的数据版本。两种数据格式以不同的速度互相演进以及从 TensorFlow 演进。两种版本控制系统在 [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) 中定义。每次添加新版本时，都会向标题中添加说明，详细说明更改内容和日期。

### 数据生产者和使用者

我们将数据版本信息分为以下种类：

- **生产者**：生成数据的二进制文件。生产者具有版本 (`producer`) 及其兼容的最小使用者版本 (`min_consumer`)。
- **使用者**：使用数据的二进制文件。使用者具有版本 (`consumer`) 及其兼容的最小生产者版本 (`min_producer`)。

每一项版本化数据都有一个 [`VersionDef versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto) 字段，该字段记录了生成数据的 `producer`、兼容的 `min_consumer` 和禁止的 `bad_consumers` 版本列表。

默认情况下，当生产者生成一些数据时，数据将继承该生产者的 `producer` 和 `min_consumer` 版本。如果已知特定使用者版本包含错误并且必须加以避免，则可以设置 `bad_consumers`。如果以下条件均成立，使用者可以接受一项数据：

- `consumer` >= 数据的 `min_consumer`
- 数据的 `producer` >= 使用者的 `min_producer`
- `consumer` 不在数据的 `bad_consumers` 范围内

由于生产者和使用者来自同一个 TensorFlow 代码库，[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) 包含的主数据版本将被视为 `producer` 或 `consumer`，具体取决于上下文以及 `min_consumer` 和 `min_producer`（分别为生产者和使用者所需）。具体而言，

- 对于 `GraphDef` 版本，我们有 `TF_GRAPH_DEF_VERSION`、`TF_GRAPH_DEF_VERSION_MIN_CONSUMER` 和 `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`。
- 对于检查点版本，我们有 `TF_CHECKPOINT_VERSION`、`TF_CHECKPOINT_VERSION_MIN_CONSUMER` 和 `TF_CHECKPOINT_VERSION_MIN_PRODUCER`。

### 向现有运算添加一个包含默认值的新特性

按照下面的指导操作可为您提供向前兼容性，前提是运算集未发生变化：

1. 如果需要向前兼容性，请将 `strip_default_attrs` 设置为 `True`，同时使用 `SavedModelBuilder` 类的 `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables` 和 `tf.saved_model.SavedModelBuilder.add_meta_graph` 方法或者 `tf.estimator.Estimator.export_saved_model`导出模型
2. 这会在生成/导出模型时排除使用默认值的特性。这样有助于确保在使用默认值时，导出的 `tf.MetaGraphDef` 不包含新运算特性。
3. 进行此控制可以让过期的使用者（例如，滞后于训练二进制文件提供二进制文件） 继续加载模型并防止模型提供出现中断。

### 演进 GraphDef 版本

本部分将介绍如何使用此版本控制机制对 `GraphDef` 格式进行不同类型的更改。

#### 添加运算

同时向使用者和生产者添加新运算，并且不更改任何 `GraphDef` 版本。这种类型的更改会自动向后兼容，并且不会影响向前兼容性计划，因为现有生产者脚本不会突然使用新功能。

#### 添加运算并将现有的 Python 包装器切换为使用该运算

1. 实现新使用者功能并增大 `GraphDef` 版本。
2. 如果可以使包装器仅在之前不奏效的情况中使用新功能，现在可以更新包装器。
3. 将 Python 包装器更改为使用新功能。请不要增大 `min_consumer`，因为不使用此运算的模型不应中断。

#### 移除或限制运算的功能

1. 修复所有生产者脚本（而非 TensorFlow 自身），使之不使用禁止的运算或功能。
2. 增大 `GraphDef` 版本，并为新版本及更高版本的 GraphDef 实现新使用者功能，以禁止移除的运算或功能。如果可以，使 TensorFlow 停止使用禁止的功能生成 `GraphDefs`。为此，请添加 [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009)。
3. 等待主要版本以实现向后兼容。
4. 从 (2) 起增大 GraphDef 版本的 `min_producer`，并完全移除该功能。

#### 更改运算的功能

1. 添加名为 `SomethingV2` 或类似名称的类似新运算，然后添加新运算并将现有 Python 包装器切换为使用该运算。为确保向前兼容性，更改 Python 包装器时请使用 [compat.py](https://tensorflow.google.cn/code/tensorflow/python/compat/compat.py) 中建议的检查。
2. 移除旧运算（由于向后兼容性，只能随主要版本更改执行此操作）。
3. 增大 `min_consumer` 以排除使用旧运算的使用者，作为 `SomethingV2` 的别名重新添加旧运算，然后将现有的 Python 包装器切换为使用该运算。
4. 移除 `SomethingV2`。

#### 禁止单个不安全的使用者版本

1. 为所有新的 GraphDef 增大 `GraphDef` 版本并将差版本添加到 `bad_consumers` 中。如果可以，仅为包含特定运算或类似运算的 GraphDef 将差版本添加到 `bad_consumers` 中。
2. 如果现有使用者具有差版本，请尽快将这些使用者移除。

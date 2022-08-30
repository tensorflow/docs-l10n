# TensorFlow Data Validation：检查和分析数据

一旦数据输入到 TFX 流水线，就可以使用 TFX 组件进行分析和转换。您甚至可以在训练模型之前使用这些工具。

分析和转换数据的原因很多：

- 在数据中查找问题。常见问题包括：
    - 缺失数据，例如具有空值的特征。
    - 标签被视为特征，以至您的模型在训练期间窥探到正确答案。
    - 值超出预期范围的特征。
    - 数据异常。
    - 迁移学习模型的预处理与训练数据不匹配。
- 设计更高效的特征集。例如，您可以识别：
    - 信息量特别大的特征。
    - 冗余特征。
    - 尺度上差异较大，以至可能减慢学习速度的特征。
    - 具有很少或没有独特预测性信息的特征。

TFX 工具既可以帮助发现数据错误，又可以帮助进行特征工程。

## TensorFlow Data Validation

- [概述](#overview)
- [基于架构的样本验证](#schema_based_example_validation)
- [训练-应用偏差检测](#skewdetect)
- [漂移检测](#drift_detection)

### 概述

TensorFlow Data Validation 可识别训练数据和应用数据中的异常，并且可以通过检查数据自动创建架构。可以将组件配置为检测数据中不同类别的异常。它可以：

1. 通过将数据统计信息与对用户预期进行编码的架构加以比较来执行有效性检查。
2. 通过比较训练数据和应用数据中的样本来检测训练-应用偏差。
3. 通过查看一系列数据来检测数据漂移。

我们独立记录以下每个功能：

- [基于架构的样本验证](#schema_based_example_validation)
- [训练-应用偏差检测](#skewdetect)
- [漂移检测](#drift_detection)

### 基于架构的样本验证

TensorFlow Data Validation 通过将数据统计信息与架构进行比较来识别输入数据中的任何异常。架构会对输入数据预期满足的属性（例如数据类型或分类值）进行编码，并且可由用户修改或替换。

Tensorflow Data Validation 通常会在 TFX 流水线的上下文中多次调用：(i) 对于从 ExampleGen 获得的每个拆分，(ii) 对于 Transform 使用的所有预转换数据，以及 (iii) 对于 Transform 生成的所有转换后数据。在 Transform (ii-iii) 的上下文中调用时，可以通过定义 [`stats_options_updater_fn`](tft.md) 来设置统计选项和基于架构的约束。这在验证非结构化数据（例如文本特征）时特别有用。有关示例，请参阅[用户代码](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py)。

#### 高级架构特征

本部分介绍可以帮助完成特殊设置的更高级架构配置。

##### 稀疏特征

在样本中对稀疏特征进行编码通常会引入多个特征，这些特征预期对于所有样本都具有相同的价。例如稀疏特征：

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

将使用单独的索引和值特征进行编码：

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

同时存在以下限制：所有样本的索引和值特征的价应当匹配。通过定义 sparse_feature，可以在架构中使这种限制显性化：

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

稀疏特征定义需要一个或多个索引和一个值特征，它们引用架构中存在的特征。显式定义稀疏特征使 TFDV 能够检查所有引用特征的价是否匹配。

一些用例在特征之间引入了相似的价限制，但不一定对稀疏特征进行编码。使用稀疏特征应当可以解除对您的限制，但并不是理想的方法。

##### 架构环境

默认情况下，验证假定流水线中的所有样本都遵循单个架构。在某些情况下，有必要引入轻微的架构变化，例如，在训练过程中需要使用用作标签的特征（并且应予以验证），但这些特征会在应用过程中丢失。环境可用于表达此类要求，尤其是 `default_environment()`、`in_environment()` 和 `not_in_environment()`。

例如，假设训练需要使用名为“LABEL”的特征，但预计该特征将在应用时丢失。这可以表示为：

- 在架构中定义两个不同的环境：["SERVING", "TRAINING"] 并将“LABEL”仅与环境“TRAINING”关联。
- 将训练数据与环境“TRAINING”关联，将应用数据与环境“SERVING”关联。

##### 架构生成

输入数据架构被指定为 TensorFlow [Schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto) 的实例。

开发者可以依靠 TensorFlow Data Validation 的自动架构构造功能，而不必从头开始手动构造架构。具体来说，TensorFlow Data Validation 基于根据流水线中可用的训练数据计算的统计信息自动构造初始架构。用户只需查看此自动生成的架构，根据需要进行修改，将其纳入版本控制系统，然后将其显式推送到流水线中供进一步验证。

TFDV 包含 `infer_schema()` 以自动生成架构。例如：

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

这将根据以下规则触发自动架构生成：

- 如果已经自动生成架构，则按原样使用它。

- 否则，TensorFlow Data Validation 会检查可用的数据统计信息并为该数据计算合适的架构。

*注：自动生成的架构是一种尽力而为的架构，仅会尝试推断数据的基本属性。用户应根据需要对其进行检查和修改。*

### 训练-应用偏差检测<a name="skewdetect"></a>

#### 概述

TensorFlow Data Validation 可以检测训练数据和应用数据之间的分布偏差。当训练数据的特征值分布与应用数据的特征值分布明显不同时，会发生分布偏差。分布偏差的主要原因之一是使用完全不同的语料库来训练数据生成，以克服所需语料库中缺少初始数据的问题。另一个原因是错误的采样机制，即只选择了应用数据的子样本来进行训练。

##### 示例场景

注：例如，为了补偿代表性不足的数据切片，如果使用有偏采样而未适当地对下采样的样本进行向上加权，则训练数据与应用数据之间的特征值分布会发生人为偏差。

请参阅 [TensorFlow Data Validation 入门指南](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)，了解有关配置训练-应用偏差检测的信息。

### 漂移检测

支持在数据的连续跨度之间（即跨度 N 和跨度 N+1 之间）进行漂移检测（例如训练数据的不同天数之间）。对于分类特征，我们用[切比雪夫距离](https://en.wikipedia.org/wiki/Chebyshev_distance)表示漂移；对于数值特征，我们用近似 [JS 散度](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)表示漂移。您可以设置阈值距离，以便在漂移高于可接受范围时收到警告。设置正确的距离通常是一个迭代过程，需要领域知识和实验。

请参阅 [TensorFlow Data Validation 入门指南](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift)，了解有关配置漂移检测的信息。

## 使用可视化检查数据

TensorFlow Data Validation 提供了用于可视化特征值分布的工具。通过使用 [Facets](https://pair-code.github.io/facets/) 在 Jupyter 笔记本中检查这些分布，您可以发现数据的常见问题。

![特征统计信息](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/feature_stats.png?raw=true)

### 确定可疑的分布

您可以使用 Facets Overview 显示来确定数据中的常见错误，以查找特征值的可疑分布。

#### 不平衡数据

不平衡特征是一种一个值占主导地位的特征。不平衡特征可以自然发生，但如果特征始终具有相同的值，则可能存在数据错误。要在 Facets Overview 中检测不平衡特征，请从“Sort by”下拉列表中选择“Non-uniformity”。

最不平衡的特征将在每个特征类型列表的顶部列出。例如，以下屏幕截图在“Numeric Features”列表的顶部显示了一个全为零的特征，以及另一个高度不平衡的特征：

![不平衡数据的呈现](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/tfx/guide/images/uniform.png?raw=true)

#### 均匀分布的数据

均匀分布的特征是一种所有可能的值都以接近相同的频率出现的特征。与不平衡数据一样，这种分布可以自然发生，但也可能由数据错误引起。

要在 Facets Overview 中检测均匀分布的特征，请从“Sort by”下拉列表中选择“Non-uniformity”，然后选中“Reverse order”复选框：

![均匀数据直方图](https://github.com/tensorflow/docs-l10n/blob/master/site/en-snapshot/tfx/guide/images/uniform_cumulative.png?raw=true)

如果唯一值不超过 20 个，则使用条形图表示字符串数据；如果唯一值超过 20 个，则使用累积分布图表示字符串数据。因此，对于字符串数据，均匀分布可能会显示为像上方一样的条形图，也可能显示为像下方一样的直线：

![折线图：均匀数据的累积分布](images/uniform_cumulative.png)

##### 可以产生均匀分布数据的错误

下面列出了一些可以产生均匀分布数据的常见错误：

- 使用字符串表示日期等非字符串数据类型。例如，对于日期时间特征，您将具有许多唯一值，其表示形式为“2017-03-01-11-45-03”。这些唯一值将均匀分布。

- 包含诸如“行号”之类的索引作为特征。这里同样有许多唯一值。

#### 缺失数据

要检查某个特征是否完全缺失值，请执行以下操作：

1. 从“Sort by”下拉列表中选择“Amount missing/zero”。
2. 选中“Reverse order”复选框。
3. 查看“missing”列，以了解特征存在缺失值的实例的百分比。

数据错误也可能导致特征值不完整。例如，您可能预期某个特征的值列表始终具有三个元素，但发现有时它只有一个元素。要检查不完整的值或特征值列表没有预期数量的元素的其他情况，请执行以下操作：

1. 从右侧的“Chart to show”下拉菜单中选择“Value list length”。

2. 查看每个特征行右侧的图表。此图表显示了特征的值列表长度范围。例如，下方屏幕截图中突出显示的行显示了具有一些零长度值列表的特征：

![Facets Overview 用零长度的特征值列表显示特征](images/zero_length.png)

#### 特征之间的尺度差异较大

如果您的特征在尺度上差异很大，模型可能难以学习。例如，如果某些特征的范围是 0 到 1，而其他特征的范围是 0 到 1,000,000,000，则尺度上会存在较大的差异。比较各个特征的“max”和“min”列，以找到大幅变化的尺度。

考虑归一化特征值以减少这些较大的差异。

#### 带无效标签的标签

TensorFlow 的 Estimator 对它们作为标签接受的数据类型存在限制。例如，二元分类器通常仅与 {0, 1} 标签一起使用。

在 Facets Overview 中查看标签值，并确保它们符合 [Estimator 的要求](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md)。

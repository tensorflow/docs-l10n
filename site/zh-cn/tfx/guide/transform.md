# Transform TFX 流水线组件

Transform TFX 流水线组件可对从 [ExampleGen](examplegen.md) 组件发出的 tf.Examples 执行特征工程（使用由 [SchemaGen](schemagen.md) 组件创建的数据架构）并发出 SavedModel 以及有关转换前和转换后数据的统计信息。执行时，SavedModel 将接受从 ExampleGen 组件发出的 tf.Examples 并发出转换后的特征数据。

- 使用：从 ExampleGen 组件发出的 tf.Examples，以及从 SchemaGen 组件创建的数据架构。
- 发出：SavedModel 到 Trainer 组件，转换前后的统计信息。

## 配置 Transform 组件

在 `preprocessing_fn` 编写完成后，需要在 Python 模块中对其进行定义，随后将该模块作为输入提供给 Transform 组件。Transform 会加载该模块，并查找和使用名为 `preprocessing_fn` 的函数来构造预处理流水线。

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

此外，您可能希望为基于 [TFDV](tfdv.md) 的转换前或转换后的统计数据计算提供选项。为此，请在同一模块中定义 `stats_options_updater_fn`。

## Transform 和 TensorFlow Transform

Transform 广泛使用 [TensorFlow Transform](tft.md) 对数据集执行特征工程。TensorFlow Transform 是一个出色的工具，可在特征数据进入模型并用于训练过程之前对其进行转换。常见的特征转换包括：

- **嵌套**：通过查找从高维空间到低维空间的有意义映射，将稀疏特征（如词汇生成的整数 ID）转换为密集特征。有关嵌套的介绍，请参见[机器学习速成课程中的“嵌套”单元](https://developers.google.com/machine-learning/crash-course/embedding)。
- **词汇生成**：通过创建将每个唯一值映射到 ID 编号的词汇，将字符串或其他非数字特征转换为整数。
- **归一化值**：转换数字特征，使其全部落入相似范围内。
- **分桶化**：通过将值分配到离散的存储分区，将连续值特征转换为分类特征。
- **丰富文本特征**：从原始数据（如词例、N 元语法、实体、情感等）生成特征，以丰富特征集。

TensorFlow Transform 可为上述及其他多种转换提供支持：

- 基于您的最新数据自动生成词汇。

- 在将数据发送到模型之前，对数据执行任意转换。TensorFlow Transform 会将转换构建到模型的 TensorFlow 计算图中，因此可在训练和推断时执行相同的转换。您可以定义引用数据全局属性的转换，例如某个特征在所有训练实例中的最大值。

您可以在运行 TFX 之前以任意方式转换数据。但是，如果在 TensorFlow Transform 中执行转换，则转换将成为 TensorFlow 计算图的一部分。这种方式有助于避免训练/应用偏差。

建模代码内部的转换使用 FeatureColumns。使用 FeatureColumns，您可以定义分桶化、使用预定义词汇的整数化，或任何其他无需查看数据即可定义的转换。

与之相比，TensorFlow Transform 执行转换则需要完全传递数据以计算事先未知的值。例如，词汇生成就需要完全传递数据。

注：这些计算在后台 [Apache Beam](https://beam.apache.org/) 中实现。

除了使用 Apache Beam 来计算值以外，TensorFlow Transform 还支持将这些值嵌入到 TensorFlow 计算图中，随后可供加载到训练计算图中。例如，在对特征进行归一化时，`tft.scale_to_z_score` 函数将计算特征的平均值和标准差，并在 TensorFlow 计算图中计算该函数减去平均值并除以标准差的表示。通过发出 TensorFlow 计算图（而不仅仅是统计信息），TensorFlow Transform 有效简化了编写预处理流水线的过程。

由于预处理以计算图形式表示，因此可在服务器上进行，从而保证了训练与应用之间的一致性。这种一致性消除了训练/应用偏差问题的其中一种来源。

TensorFlow Transform 支持用户使用 TensorFlow 代码指定其预处理流水线。这意味着流水线与 TensorFlow 计算图采用相同的构造方式。如果计算图中仅使用 TensorFlow 运算，则流水线将为接受批量输入并返回批量输出的纯映射。此类流水线相当于在使用 `tf.Estimator` API 时将计算图置于 `input_fn` 内。为了指定诸如计算分位数等完全传递运算，TensorFlow Transform 提供了名为 `analyzers` 的特殊函数，这种函数与 TensorFlow 运算类似，但实际上却指定将由 Apache Beam 执行的延迟计算，并将插入计算图的输出指定为常量。普通的 TensorFlow 运算将接受单一批次作为输入，仅对该批次执行计算并发出一个批次，而 `analyzer` 则会对所有批次执行全局归约（在 Apache Beam 中实现）并返回结果。

通过将普通的 TensorFlow 运算与 TensorFlow Transform 分析器相结合，用户可以创建复杂的流水线来对其数据进行预处理。例如，`tft.scale_to_z_score` 函数接受输入张量并返回平均值为 `0`、方差为 `1` 的归一化张量。实现方法是，该函数在后台调用 `mean` 和 `var` 分析器，从而在计算图中有效生成与输入张量的平均值和方差相等的常量。然后将使用 TensorFlow 运算来减去平均值并除以标准差。

## TensorFlow Transform `preprocessing_fn`

TFX Transform 组件可通过处理与数据读写相关的 API 调用，并将输出 SavedModel 写入磁盘来简化 Transform 的使用。作为 TFX 用户，您只需定义一个名为 `preprocessing_fn` 的函数。在 `preprocessing_fn` 中，您需要定义用于操作张量的输入字典以生成张量的输出字典的一系列函数。您可以使用 [TensorFlow Transform API](/tfx/transform/api_docs/python/tft) 查找诸如 scale_to_0_1 和 compute_and_apply_vocabulary 之类的辅助函数，或使用常规的 TensorFlow 函数，如下所示。

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
```

### 理解 preprocessing_fn 的输入

`preprocessing_fn` 描述了对张量（即，`Tensor`、`SparseTensor` 或 `RaggedTensor`）的一系列运算。为了正确定义 `preprocessing_fn`，有必要了解数据如何表示为张量。`preprocessing_fn` 的输入由架构确定。[`Schema` proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto#L72) 最终会被转换为用于数据解析的“特征规范”（有时称为“解析规范”），请参阅{ a8}此处有关转换逻辑的更多详细信息。

## 使用 TensorFlow Transform 处理字符串标签

通常，用户会希望使用 TensorFlow Transform 来生成词汇并应用该词汇将字符串转换为整数。执行这一工作流时，在模型中构造的 `input_fn` 将输出整数化字符串。但标签是一个例外，因为要使模型能够将输出（整数）标签重新映射到字符串，模型需要 `input_fn` 来输出字符串标签以及标签可能值的列表。例如，如果标签为 `cat` 和 `dog`，则 `input_fn` 的输出应为这些原始字符串，并且需要将键 `["cat", "dog"]` 作为参数传递至 Estimator（请在下文了解详细信息）。

为了处理字符串标签到整数的映射，您应使用 TensorFlow Transform 来生成词汇。我们在下面的代码段中对此进行了演示：

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

上方的预处理函数接受原始输入特征（也将作为预处理函数输出的一部分返回）并对其调用 `tft.vocabulary`。这会为 `education` 生成可在模型中访问的词汇。

该示例还展示了如何转换标签，然后为转换后的标签生成词汇。特别是，它接受原始标签 `education`，并将除前 5 个标签（按频率）之外的所有标签转换为 `UNKNOWN`，而不将标签转换为整数。

在模型代码中，必须为分类器提供由 `tft.vocabulary` 生成的词汇作为 `label_vocabulary` 参数。方法是先使用辅助函数以列表形式读取该词汇。以下代码段展示了该方法。请注意，示例代码使用了上文所讨论的转换后的标签，但是我们在此展示的是使用原始标签的代码。

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## 配置转换前和转换后的统计信息

如上所述，Transform 组件会调用 TFDV 来计算转换前和转换后的统计信息。TFDV 会将可选的 [StatsOptions](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/statistics/stats_options.py) 对象作为输入。用户可能希望配置此对象以启用某些附加统计信息（例如 NLP 统计信息）或设置经过验证的阈值（例如最小/最大词例频率）。为此，请在模块文件中定义一个 `stats_options_updater_fn`。

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

转换后的统计信息通常受益于用于预处理特征的词汇的知识。对于每个 TFT 生成的词汇，将词汇名称到路径的映射提供给 StatsOptions（因此为 TFDV）。此外，可以通过 (i) 直接修改 StatsOptions 中的 `vocab_paths` 字典或 (ii) 使用 `tft.annotate_asset` 添加外部创建的词汇的映射。

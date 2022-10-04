# ExampleGen TFX 流水线组件

ExampleGen TFX 流水线组件可将数据提取到 TFX 流水线。它使用外部文件/服务来生成样本，供其他 TFX 组件读取。它还提供一致且可配置的分区，并为 ML 最佳做法打乱数据集的顺序。

- 使用：来自外部数据源（如 CSV、`TFRecord`、Avro、Parquet 和 BigQuery）的数据
- 发出：`tf.Example` 记录、`tf.SequenceExample` 记录或 proto 格式，取决于载荷格式

## ExampleGen 和其他组件

ExampleGen 为使用 [TensorFlow Data Validation](tfdv.md) 库的组件（如 [SchemaGen](schemagen.md)、[StatisticsGen](statsgen.md) 和 [Example Validator](exampleval.md)）提供数据。它还为使用 [TensorFlow Transform](tft.md) 库的 [Transform](transform.md) 提供数据，并最终在推断期间为部署目标提供数据。

## 数据源和格式

目前，TFX 的标准安装包含了以下数据源和格式的完整 ExampleGen 组件：

- [CSV](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/csv_example_gen)
- [tf.Record](https://github.com/tensorflow/tfx/tree/master/tfx/components/example_gen/import_example_gen)
- [BigQuery](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_big_query/example_gen)

还可以使用自定义执行器为以下数据源和格式开发 ExampleGen 组件：

- [Avro](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py)
- [Parquet](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/parquet_executor.py)

请参阅源代码中的用法示例和[此讨论](/tfx/guide/examplegen#custom_examplegen)，以获取有关如何使用和开发自定义执行器的更多信息。

注：在大多数情况下，从 `base_example_gen_executor` 继承优于从 `base_executor` 继承。因此，最好遵循执行器源代码中的 Avro 或 Parquet 示例。

此外，以下数据源和格式可作为[自定义组件](/tfx/guide/understanding_custom_components)示例使用：

- [Presto](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/presto_example_gen)

### 提取 Apache Beam 支持的数据格式

Apache Beam 支持从[多种数据源和格式](https://beam.apache.org/documentation/io/built-in/)提取数据（[见下文](#additional_data_formats)）。这些功能可用来为 TFX 创建自定义 ExampleGen 组件，一些现有的 ExampleGen 组件就展示了这一点。

## 如何使用 ExampleGen 组件

对于受支持的数据源（目前为 CSV 文件、具有 `tf.Example`、`tf.SequenceExample` 和 proto 格式的 TFRecord 文件，以及 BigQuery 查询的结果），ExampleGen 流水线组件可以直接在部署中使用，而且几乎不需要自定义。例如：

```python
example_gen = CsvExampleGen(input_base='data_root')
```

或如下所示，直接使用 `tf.Example` 导入外部 TFRecord：

```python
example_gen = ImportExampleGen(input_base=path_to_tfrecord_dir)
```

## 跨度、版本和拆分

跨度是训练样本的分组。如果数据保留在文件系统上，则每个跨度可能会存储在单独的目录中。跨度的语义没有硬编码到 TFX 中；一个跨度可能对应一天的数据、一个小时的数据，或对您的任务有意义的其他分组。

每个跨度可以保存多个版本的数据。举例来说，如果您从一个跨度中移除一些样本以清理质量较差的数据，则可能会产生该跨度的新版本。默认情况下，TFX 组件会在跨度内的最新版本上运行。

跨度内的每个版本可以进一步细分为多个拆分。拆分跨度的最常见用例是将其拆分为训练数据和评估数据。

![跨度和拆分](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tfx/guide/images/spans_splits.png?raw=true)

### 自定义输入/输出拆分

注：此功能仅在 TFX 0.14 之后可用。

要自定义 ExampleGen 将输出的训练/评估拆分比率，请为 ExampleGen 组件设置 `output_config`。例如：

```python
# Input has a single split 'input_dir/*'.
# Output 2 splits: train:eval=3:1.
output = proto.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=3),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1)
             ]))
example_gen = CsvExampleGen(input_base=input_dir, output_config=output)
```

请注意在此示例中如何设置 `hash_buckets`。

对于已拆分的输入源，请为 ExampleGen 组件设置 `input_config`：

```python

# Input train split is 'input_dir/train/*', eval split is 'input_dir/eval/*'.
# Output splits are generated one-to-one mapping from input splits.
input = proto.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
example_gen = CsvExampleGen(input_base=input_dir, input_config=input)
```

对于基于文件的样本生成器（例如 CsvExampleGen 和 ImportExampleGen），`pattern` 是 glob 相对文件模式，它映射到具有由输入基础路径给定的根目录的输入文件。对于基于查询的样本生成器（例如 BigQueryExampleGen 和 PrestoExampleGen），`pattern` 是 SQL 查询。

默认情况下，整个输入基础目录会被视为单个输入拆分，并以 2:1 的比率生成训练和评估输出拆分。

请参阅 [proto/example_gen.proto](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)，了解 ExampleGen 的输入和输出拆分配置。并请参考[下游组件指南](#examplegen_downstream_components)，了解如何利用下游的自定义拆分。

#### 拆分方法

使用 `hash_buckets` 拆分方法时，可以使用特征对样本进行分区（无需使用整个记录）。如果存在特征，ExampleGen 将使用该特征的指纹作为分区键。

此特征可用于维护有关样本某些属性的稳定拆分：例如，如果选择“user_id”作为分区特征名称，用户将始终被放入同一个拆分中。

对“特征”的含义以及如何将“特征”与指定名称匹配的解释取决于 ExampleGen 实现和样本的类型。

对于现成的 ExampleGen 实现：

- 如果它生成 tf.Example，则“特征”表示 tf.Example.features.feature 中的条目。
- 如果它生成 tf.SequenceExample，则“特征”表示 tf.SequenceExample.context.feature 中的条目。
- 仅支持 int64 和字节特征。

在下列情况下，ExampleGen 会引发运行时错误：

- 样本中不存在指定的特征名称。
- 空特征：`tf.train.Feature()`。
- 不支持的特征类型，例如浮点特征。

要基于样本中的特征输出训练/评估拆分，请为 ExampleGen 组件设置 `output_config`。例如：

```python
# Input has a single split 'input_dir/*'.
# Output 2 splits based on 'user_id' features: train:eval=3:1.
output = proto.Output(
             split_config=proto.SplitConfig(splits=[
                 proto.SplitConfig.Split(name='train', hash_buckets=3),
                 proto.SplitConfig.Split(name='eval', hash_buckets=1)
             ],
             partition_feature_name='user_id'))
example_gen = CsvExampleGen(input_base=input_dir, output_config=output)
```

请注意在此示例中如何设置 `partition_feature_name`。

### 跨度

注：此功能仅在 TFX 0.15 之后可用。

可以通过在[输入 glob 模式](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)中使用“{SPAN}”规范来检索跨度：

- 此规范会匹配数字并将数据映射到相关的 SPAN 号。例如，“data_{SPAN}-*.tfrecord”将收集“data_12-a.tfrecord”、“date_12-b.tfrecord”之类的文件。
- 可以选择在映射时使用整数的宽度来指定该规范。例如，“data_{SPAN:2}.file”会映射到“data_02.file”和“data_27.file”等文件（分别作为 Span-2 和 Span-27 的输入），但不会映射到“data_1.file”或“data_123.file”。
- 如果缺少 SPAN 规范，则假定跨度始终为“0”。
- 如果指定了 SPAN，流水线将处理最新的跨度，并将跨度号存储在元数据中。

例如，假设有以下输入数据：

- “/tmp/span-1/train/data”
- “/tmp/span-1/eval/data”
- “/tmp/span-2/train/data”
- “/tmp/span-2/eval/data”

输入配置如下所示：

```python
splits {
  name: 'train'
  pattern: 'span-{SPAN}/train/*'
}
splits {
  name: 'eval'
  pattern: 'span-{SPAN}/eval/*'
}
```

触发流水线时，它将进行如下处理：

- 将“/tmp/span-2/train/data”处理为训练拆分
- 将“/tmp/span-2/eval/data”处理为评估拆分

其中，跨度号为“2”。如果稍后“/tmp/span-3/...”准备就绪，只需再次触发流水线，它将提取跨度“3”进行处理。下面展示了关于使用跨度规范的代码示例：

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

可以使用 RangeConfig 检索特定跨度，这将在下面详细介绍。

### 日期

注：此功能仅在 TFX 0.24.0 之后可用。

如果您的数据源按照日期在文件系统中进行组织，则 TFX 支持直接将日期映射到跨度号。表示从日期到跨度的映射有三种规范：{YYYY}、{MM} 和 {DD}：

- 如果指定了其中任何一个规范，则这三个规范应全部存在于[输入 glob 模式](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)中：
- 可以专门指定 {SPAN} 规范或这组日期规范。
- 计算日历日期，其中年份为 YYYY，月份为 MM，月份中的日期为 DD，那么，跨度号就会被计算为自 Unix 纪元以来的天数（即 1970-01-01）。例如，“log-{YYYY}{MM}{DD}.data”与文件“log-19700101.data”匹配，并将其用作 Span-0 的输入，并将“log-20170101.data”用作 Span-17167 的输入。
- 如果指定了这组日期规范，则流水线将处理最新的日期，并将相应的跨度号存储在元数据中。

例如，假设有以下按日历日期组织的输入数据：

- “/tmp/1970-01-02/train/data”
- “/tmp/1970-01-02/eval/data”
- “/tmp/1970-01-03/train/data”
- “/tmp/1970-01-03/eval/data”

输入配置如下所示：

```python
splits {
  name: 'train'
  pattern: '{YYYY}-{MM}-{DD}/train/*'
}
splits {
  name: 'eval'
  pattern: '{YYYY}-{MM}-{DD}/eval/*'
}
```

触发流水线时，它将进行如下处理：

- 将“/tmp/1970-01-03/train/data”处理为训练拆分
- 将“/tmp/1970-01-03/eval/data”处理为评估拆分

其中，跨度号为“2”。如果稍后“/tmp/1970-01-04/...”准备就绪，只需再次触发流水线，它将提取跨度“3”进行处理。下面展示了关于使用跨度规范的代码示例：

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='{YYYY}-{MM}-{DD}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='{YYYY}-{MM}-{DD}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### 版本

注：此功能仅在 TFX 0.24.0 之后可用。

可以通过在[输入 glob 模式](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)中使用“{VERSION}”规范来检索版本：

- 此规范匹配数字，并将数据映射到 SPAN 下的相关 VERSION 号。请注意，版本规范可以与跨度或日期规范组合使用。
- 也可以按照与 SPAN 规范相同的方式，用宽度指定此规范。例如，“span-{SPAN}/version-{VERSION:4}/data-*”。
- 如果缺少 VERSION 规范，则会将版本设置为 None。
- 如果同时指定了 SPAN 和 VERSION，则流水线将处理最新跨度的最新版本，并将版本号存储在元数据中。
- 如果指定了 VERSION，但未指定 SPAN（或日期规范），则将引发错误。

例如，假设有以下输入数据：

- “/tmp/span-1/ver-1/train/data”
- “/tmp/span-1/ver-1/eval/data”
- “/tmp/span-2/ver-1/train/data”
- “/tmp/span-2/ver-1/eval/data”
- “/tmp/span-2/ver-2/train/data”
- “/tmp/span-2/ver-2/eval/data”

输入配置如下所示：

```python
splits {
  name: 'train'
  pattern: 'span-{SPAN}/ver-{VERSION}/train/*'
}
splits {
  name: 'eval'
  pattern: 'span-{SPAN}/ver-{VERSION}/eval/*'
}
```

触发流水线时，它将进行如下处理：

- 将“/tmp/span-2/ver-2/train/data”处理为训练拆分
- 将“/tmp/span-2/ver-2/eval/data”处理为评估拆分

其中，跨度号为“2”，版本号为“2”。如果稍后“/tmp/span-2/ver-3/...”准备就绪，只需再次触发流水线，它将提取跨度“2”和版本“3”进行处理。下面展示了关于使用版本规范的代码示例：

```python
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN}/ver-{VERSION}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN}/ver-{VERSION}/eval/*')
            ])
example_gen = CsvExampleGen(input_base='/tmp', input_config=input)
```

### 范围配置

注：此功能仅在 TFX 0.24.0 之后可用。

TFX 支持在基于文件的 ExampleGen 中使用范围配置（用于描述不同 TFX 实体的范围的抽象配置）来检索和处理特定的跨度。要检索特定的跨度，请为基于文件的 ExampleGen 组件设置 `range_config`。例如，假设有以下输入数据：

- “/tmp/span-01/train/data”
- “/tmp/span-01/eval/data”
- “/tmp/span-02/train/data”
- “/tmp/span-02/eval/data”

为了专门检索和处理跨度为“1”的数据，除了输入配置外，我们还要指定范围配置。请注意，ExampleGen 只支持单跨度静态范围（以指定对特定单个跨度的处理）。因此，对于 StaticRange，start_span_number 必须等于 end_span_number。使用提供的跨度和零填充的宽度信息（如果提供），ExampleGen 将用所需的跨度号替换所提供的拆分模式中的 SPAN 规范。使用示例如下所示：

```python
# In cases where files have zero-padding, the width modifier in SPAN spec is
# required so TFX can correctly substitute spec with zero-padded span number.
input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='span-{SPAN:2}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='span-{SPAN:2}/eval/*')
            ])
# Specify the span number to be processed here using StaticRange.
range = proto.RangeConfig(
                static_range=proto.StaticRange(
                        start_span_number=1, end_span_number=1)
            )

# After substitution, the train and eval split patterns will be
# 'input_dir/span-01/train/*' and 'input_dir/span-01/eval/*', respectively.
example_gen = CsvExampleGen(input_base=input_dir, input_config=input,
                            range_config=range)
```

如果使用的是日期规范而非 SPAN 规范，则范围配置也可以用于处理特定日期。例如，假设有按日历日期组织的下列输入数据：

- “/tmp/1970-01-02/train/data”
- “/tmp/1970-01-02/eval/data”
- “/tmp/1970-01-03/train/data”
- “/tmp/1970-01-03/eval/data”

为了专门检索和处理 1970 年 1 月 2 日的数据，我们需要执行以下操作：

```python
from  tfx.components.example_gen import utils

input = proto.Input(splits=[
                proto.Input.Split(name='train',
                                            pattern='{YYYY}-{MM}-{DD}/train/*'),
                proto.Input.Split(name='eval',
                                            pattern='{YYYY}-{MM}-{DD}/eval/*')
            ])
# Specify date to be converted to span number to be processed using StaticRange.
span = utils.date_to_span_number(1970, 1, 2)
range = proto.RangeConfig(
                static_range=range_config_pb2.StaticRange(
                        start_span_number=span, end_span_number=span)
            )

# After substitution, the train and eval split patterns will be
# 'input_dir/1970-01-02/train/*' and 'input_dir/1970-01-02/eval/*',
# respectively.
example_gen = CsvExampleGen(input_base=input_dir, input_config=input,
                            range_config=range)
```

## 自定义 ExampleGen

如果当前可用的 ExampleGen 组件不符合您的需求，您可以创建自定义 ExampleGen，这将使您能够从不同的数据源或不同的数据格式中读取数据。

### 基于文件的 ExampleGen 自定义（实验性）

首先，使用自定义 Beam PTransform 扩展 BaseExampleGenExecutor，前者提供从训练/评估输入拆分到 TF 样本的转换。例如，[CsvExampleGen 执行器](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/executor.py)提供从输入 CSV 拆分到 TF 样本的转换。

然后，使用上述执行器创建组件（如在 [CsvExampleGen 组件](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/csv_example_gen/component.py)中所做）。或者，将自定义执行器传递到标准 ExampleGen 组件中，如下所示。

```python
from tfx.components.base import executor_spec
from tfx.components.example_gen.csv_example_gen import executor

example_gen = FileBasedExampleGen(
    input_base=os.path.join(base_dir, 'data/simple'),
    custom_executor_spec=executor_spec.ExecutorClassSpec(executor.Executor))
```

现在，我们还支持使用此[方法](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/)读取 Avro 和 Parquet 文件。

### 其他数据格式

Apache Beam 支持通过 Beam I/O 转换读取多种[其他数据格式](https://beam.apache.org/documentation/io/built-in/)。您可以使用类似 [Avro 示例](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/custom_executors/avro_executor.py#L56)的模式，利用 Beam I/O 转换创建自定义 ExampleGen 组件。

```python
  return (pipeline
          | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
          | 'ToTFExample' >> beam.Map(utils.dict_to_example))
```

在撰写本文时，当前支持的 Beam Python SDK 格式和数据源包括：

- Amazon S3
- Apache Avro
- Apache Hadoop
- Apache Kafka
- Apache Parquet
- Google Cloud BigQuery
- Google Cloud BigTable
- Google Cloud Datastore
- Google Cloud Pub/Sub
- Google Cloud Storage (GCS)
- MongoDB

请查看 [Beam 文档](https://beam.apache.org/documentation/io/built-in/)，了解最新列表。

### 基于查询的 ExampleGen 自定义（实验性）

首先，使用自定义 Beam PTransform 扩展 BaseExampleGenExecutor，前者可从外部数据源读取。然后，通过扩展 QueryBasedExampleGen 创建一个简单的组件。

这可能需要也可能不需要额外的连接配置。例如，[BigQuery 执行器](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_big_query/example_gen/executor.py)使用默认的 beam.io 连接器进行读取，该连接器可提取连接配置详细信息。[Presto 执行器](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/presto_component/executor.py)需要自定义 Beam PTransform 和[自定义连接配置 protobuf](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto) 作为输入。

如果自定义 ExampleGen 组件需要连接配置，请创建一个新的 protobuf，然后通过 custom_config（目前为可选执行参数）将其传入。以下是如何使用已配置组件的示例。

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips')
```

## ExampleGen 下游组件

下游组件支持自定义拆分配置。

### StatisticsGen

默认行为是对所有拆分执行统计信息生成。

要排除任何拆分，请为 StatisticsGen 组件设置 `exclude_splits`。例如：

```python
# Exclude the 'eval' split.
statistics_gen = StatisticsGen(
             examples=example_gen.outputs['examples'],
             exclude_splits=['eval'])
```

### SchemaGen

默认行为是基于所有拆分生成架构。

要排除任何拆分，请为 SchemaGen 组件设置 `exclude_splits`。例如：

```python
# Exclude the 'eval' split.
schema_gen = SchemaGen(
             statistics=statistics_gen.outputs['statistics'],
             exclude_splits=['eval'])
```

### ExampleValidator

默认行为是根据架构验证输入样本中所有拆分的统计信息。

要排除任何拆分，请为 ExampleValidator 组件设置 `exclude_splits`。例如：

```python
# Exclude the 'eval' split.
example_validator = ExampleValidator(
             statistics=statistics_gen.outputs['statistics'],
             schema=schema_gen.outputs['schema'],
             exclude_splits=['eval'])
```

### Transform

默认行为是分析并从“训练”拆分中生成元数据，然后转换所有拆分。

要指定分析拆分和转换拆分，请为 Transform 组件设置 `splits_config`。例如：

```python
# Analyze the 'train' split and transform all splits.
transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=_taxi_module_file,
      splits_config=proto.SplitsConfig(analyze=['train'],
                                               transform=['train', 'eval']))
```

### Trainer 和 Tuner

默认行为是根据“训练”拆分进行训练，并根据“评估”拆分进行评估。

要指定训练拆分和评估拆分，请为 Trainer 组件设置 `train_args` 和 `eval_args`。例如：

```python
# Train on the 'train' split and evaluate on the 'eval' split.
Trainer = Trainer(
      module_file=_taxi_module_file,
      examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=proto.TrainArgs(splits=['train'], num_steps=10000),
      eval_args=proto.EvalArgs(splits=['eval'], num_steps=5000))
```

### Evaluator

默认行为是提供根据“评估”拆分计算的指标。

要计算有关自定义拆分的评估统计信息，请为 Evaluator 组件设置 `example_splits`。例如：

```python
# Compute metrics on the 'eval1' split and the 'eval2' split.
evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      example_splits=['eval1', 'eval2'])
```

有关更多详细信息，请参阅 [CsvExampleGen API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/CsvExampleGen)、[FileBasedExampleGen API 实现](https://github.com/tensorflow/tfx/blob/master/tfx/components/example_gen/component.py)和 [ImportExampleGen API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportExampleGen)。

# StatisticsGen TFX 流水线组件

StatisticsGen TFX 流水线组件根据训练数据和应用数据来生成特征统计信息，以供其他流水线组件使用。StatisticsGen 使用 Beam 来扩展为大型数据集。

- 使用：由 ExampleGen 流水线组件创建的数据集。
- 发出：数据集统计信息。

## StatisticsGen 和 TensorFlow Data Validation

StatisticsGen 广泛使用 [TensorFlow Data Validation](tfdv.md) 来根据您的数据集生成统计信息。

## 使用 StatsGen 组件

StatisticsGen 流水线组件通常非常易于部署，而且几乎不需要自定义。典型代码如下所示：

```python
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## 将 StatsGen 组件与架构一起使用

当流水线第一次运行时，StatisticsGen 的输出将用于推断架构。不过，在随后的运行中，您可能具有手动选择的架构，其中包含有关数据集的附加信息。通过将此架构提供给 StatisticsGen，TFDV 可以根据数据集的已声明属性提供更多有用的统计信息。

在此设置中，您将使用由 ImporterNode 导入的精选架构调用 StatisticsGen，代码如下所示：

```python
from tfx import components
from tfx.types import standard_artifacts

...

user_schema_importer = components.ImporterNode(
    instance_name='import_user_schema',
    source_uri=user_schema_dir, # directory containing only schema text proto
    artifact_type=standard_artifacts.Schema)

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```

### 创建精选架构

TFX 中的 `Schema` 是 TensorFlow Metadata <a href="https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto" data-md-type="link">`Schema` proto</a> 的一个实例。这可以从头开始以[文本格式](https://googleapis.dev/python/protobuf/latest/google/protobuf/text_format.html)创作。但是，将 `SchemaGen` 生成的推断架构用作起点要容易得多。执行 `SchemaGen` 组件后，架构将位于以下路径的流水线根目录下：

```
<pipeline_root>/SchemaGen/schema/<artifact_id>/schema.pbtxt
```

其中，`<artifact_id>` 表示 MLMD 中此版本架构的唯一 ID。随后，可以修改此架构 proto 以传达有关无法可靠推断的数据集的信息，这样，`StatisticsGen` 的输出便会更加有用，而且 [`ExampleValidator`](https://www.tensorflow.org/tfx/guide/exampleval) 组件中执行的验证也会更加严格。

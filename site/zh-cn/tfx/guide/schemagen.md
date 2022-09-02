# SchemaGen TFX 流水线组件

一些 TFX 组件使用*架构*来描述输入数据。架构是 [schema.proto](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto) 的一个实例。它可以指定特征值的数据类型、是否在所有示样本中都必须存在特征、允许的值范围以及其他属性。SchemaGen 流水线组件将通过从训练数据中推断类型、类别和范围来自动生成架构。

- 使用：来自 StatisticsGen 组件的统计信息
- 发出：数据架构 proto

下面的代码摘自一个架构 proto：

```proto
...
feature {
  name: "age"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
feature {
  name: "capital-gain"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
...
```

以下 TFX 库使用架构：

- TensorFlow Data Validation
- TensorFlow Transform
- TensorFlow Model Analysis

在典型的 TFX 流水线中，SchemaGen 会生成一个架构，供其他流水线组件使用。但是，自动生成的架构为最大努力，并且仅尝试推断数据的基本属性。开发者应根据需要对其进行审阅和修改。

可以使用 ImportSchemaGen 组件将修改后的架构带回流水线中。可以移除用于初始架构生成的 SchemaGen 组件，并且所有下游组件都可以使用 ImportSchemaGen 的输出。还建议使用导入的架构添加 [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) 以连续检查训练数据。

## SchemaGen 和 TensorFlow Data Validation

SchemaGen 广泛使用 [TensorFlow Data Validation](tfdv.md) 来推断架构。

## 使用 SchemaGen 组件

### 用于初始架构生成

SchemaGen 流水线组件通常非常易于部署，而且几乎不需要自定义。典型代码如下所示：

```python
schema_gen = tfx.components.SchemaGen(
    statistics=stats_gen.outputs['statistics'])
```

有关更多详细信息，请参阅 [SchemaGen API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/SchemaGen)。

### 用于已审阅的架构导入

将 ImportSchemaGen 组件添加到流水线，以将已审阅的架构定义带入流水线。

```python
schema_gen = tfx.components.ImportSchemaGen(
    schema_file='/some/path/schema.pbtxt')
```

`schema_file` 应该是文本 protobuf 文件的完整路径。

有关更多详细信息，请参阅 [ImportSchemaGen API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen)。

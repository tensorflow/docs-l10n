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

在典型的 TFX 流水线中，SchemaGen 会生成一个将由其他流水线组件使用的架构。

注：自动生成的架构是一种尽力而为的架构，仅会尝试推断数据的基本属性。开发者应根据需要对其进行检查和修改。

## SchemaGen 和 TensorFlow Data Validation

SchemaGen 广泛使用 [TensorFlow Data Validation](tfdv.md) 来推断架构。

## 使用 SchemaGen 组件

SchemaGen 流水线组件通常非常易于部署，而且几乎不需要自定义。典型代码如下所示：

```python
from tfx import components

...

infer_schema = components.SchemaGen(
    statistics=compute_training_stats.outputs['statistics'])
```

# ExampleValidator TFX 流水线组件

ExampleValidator 流水线组件可识别训练和应用数据中的异常。它可以检测数据中不同类的异常。例如，它可以：

1. 通过将数据统计信息与编码用户期望的架构进行比较来执行有效性检查。
2. 通过比较训练和应用数据检测训练-应用偏差。
3. 通过查看一系列数据来检测数据漂移。

ExampleValidator 流水线组件通过将 StatisticsGen 流水线组件所计算的数据统计信息与架构进行比较，来识别样本数据中的任何异常。推断的架构会对输入数据预期要满足的属性进行编码，并且开发者可以修改这些属性。

- 使用：来自 SchemaGen 组件的架构和来自 StatisticsGen 组件的统计信息。
- 发出：验证结果

## ExampleValidator 和 TensorFlow Data Validation

ExampleValidator 广泛使用 [TensorFlow Data Validation](tfdv.md) 来验证输入的数据。

## 使用 ExampleValidator 组件

ExampleValidator 流水线组件通常非常易于部署，而且几乎不需要自定义。典型代码如下所示：

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

有关更多详细信息，请参阅 [ExampleValidator API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator)。

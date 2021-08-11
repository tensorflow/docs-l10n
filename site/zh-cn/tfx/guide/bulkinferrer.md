# BulkInferrer TFX 流水线组件

BulkInferrer TFX 组件可以对无标签数据执行批量推断。生成的 InferenceResult ([tensorflow_serving.apis.prediction_log_pb2.PredictionLog](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto)) 包含原始特征和预测结果。

BulkInferrer 使用以下内容：

- [SavedModel](https://www.tensorflow.org/guide/saved_model.md) 格式的训练模型。
- 包含特征的无标签 tf.Examples。
- （可选）来自 [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) 组件的验证结果。

BulkInferrer 发出以下内容：

- [InferenceResult](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)

## 使用 BulkInferrer 组件

BulkInferrer TFX 组件用于基于无标签的 tf.Examples 执行批量推断。通常将其部署在 [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator.md) 组件之后以使用验证的模型执行推断，或者部署在 [Trainer](https://www.tensorflow.org/tfx/guide/trainer.md) 组件之后以直接在导出的模型上执行推断。

目前，它可以执行内存中模型推断和远程推断。远程推断要求模型托管在 Cloud AI Platform 上。

典型的代码如下所示：

```python
bulk_inferrer = BulkInferrer(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    data_spec=bulk_inferrer_pb2.DataSpec(),
    model_spec=bulk_inferrer_pb2.ModelSpec()
)
```

有关更多详细信息，请参阅 [BulkInferrer API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/BulkInferrer)。

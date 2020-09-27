# Pusher TFX 流水线组件

Pusher 组件用于在模型训练或再训练期间将经过验证的模型推送到[部署目标](index.md#deployment_targets)。在部署之前，Pusher 依靠源于其他验证组件的一个或多个推荐来决定是否推送模型。

- 如果新训练的模型“足够好”，可以推送到生产环境，则 [Evaluator](evaluator) 会推荐该模型。
- （可选，但建议参考）如果模型在生产环境中是一个机械可应用模型，则 [InfraValidator](infra_validator) 会推荐该模型。

Pusher 组件使用 [SavedModel](/guide/saved_model) 格式的训练模型，并生成相同的 SavedModel 以及版本控制元数据。

## 使用 Pusher 组件

Pusher 流水线组件通常非常易于部署，而且几乎不需要自定义，因为所有工作均由 Pusher TFX 组件完成。典型代码如下所示：

```python
from tfx import components

...

pusher = components.Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

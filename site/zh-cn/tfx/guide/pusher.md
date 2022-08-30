# Pusher TFX 流水线组件

Pusher 组件用于在模型训练或再训练期间将经过验证的模型推送到[部署目标](index.md#deployment_targets)。在部署之前，Pusher 依靠源于其他验证组件的一个或多个推荐来决定是否推送模型。

- 如果新训练的模型“足够好”，可以推送到生产环境，则 [Evaluator](evaluator) 会推荐该模型。
- （可选，但建议参考）如果模型在生产环境中是一个机械可应用模型，则 [InfraValidator](infra_validator) 会推荐该模型。

Pusher 组件使用 [SavedModel](/guide/saved_model) 格式的训练模型，并生成相同的 SavedModel 以及版本控制元数据。

## 使用 Pusher 组件

Pusher 流水线组件通常非常易于部署，而且几乎不需要自定义，因为所有工作均由 Pusher TFX 组件完成。典型代码如下所示：

```python
pusher = Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=tfx.proto.PushDestination(
    filesystem=tfx.proto.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

### 推送从 InfraValidator 生成的模型。

（从版本 0.30.0 开始）

InfraValidator 还可以生成包含<a>带预热模型</a>的 <code>InfraBlessing</code> 工件。可以像`Model` 工件一样由 Pusher 对其进行推送。

```python
infra_validator = InfraValidator(
    ...,
    # make_warmup=True will produce a model with warmup requests in its
    # 'blessing' output.
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)

pusher = Pusher(
    # Push model from 'infra_blessing' input.
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

有关更多详细信息，请参阅 [Pusher API 参考](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher)。

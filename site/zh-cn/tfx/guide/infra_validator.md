# InfraValidator TFX 流水线组件

InfraValidator 是一个 TFX 组件，用作将模型推送到生产环境之前的预警层。名称“infra”验证器源自它在实际模型应用“基础架构 (infrastructure)”中对模型进行验证这一事实。如果 [Evaluator](evaluator.md) 是为了保证模型的性能，InfraValidator 就是为了保证模型机械上的精细，并防止推送不良模型。

## 工作原理

InfraValidator 会获取模型，使用该模型启动沙盒化模型服务器，然后查看它是否可以成功加载并选择性查询。基础结构验证结果将以与 [Evaluator](evaluator.md) 相同的方式在 `blessing` 输出中生成。

InfraValidator侧重于模型服务器二进制文件（例如 [TensorFlow Serving](serving.md)）与要部署的模型之间的兼容性。尽管名为“infra”验证器，但正确配置环境是**用户的责任**，并且基础结构验证器仅与用户配置环境中的模型服务器交互，以查看其是否正常工作。正确配置此环境将确保基础结构验证的通过或失败会指示模型是否可用于生产应用环境。这意味着部分（但不限于）以下内容：

1. InfraValidator 使用与生产环境中相同的模型服务器二进制文件。这是基础结构验证环境必须收敛到的最低级别。
2. InfraValidator 使用与生产环境中相同的资源（例如，CPU、内存和加速器的分配数量和类型）。
3. InfraValidator 使用与生产环境中相同的模型服务器配置。

用户可以根据情况选择 InfraValidator 与生产环境的相同程度。从技术上讲，可以在本地 Docker 环境中对模型进行基础结构验证，然后在完全不同的环境（例如 Kubernetes 集群）中应用，而不会出现问题。但是，InfraValidator 不会检查此差异。

### 运算模式

根据配置，会以下面的其中一种模式完成基础结构验证：

- `LOAD_ONLY` 模式：检查模型是否已成功加载到应用基础架构中。**或者**
- `LOAD_AND_QUERY` 模式：`LOAD_ONLY` 模式外加发送一些样本请求以检查模型是否能够应用推断。InfraValidator 不关心预测正确与否。只关心请求是否成功。

## 使用方法

通常，InfraValidator 会在 Evaluator 组件旁边进行定义，它的输出将馈送到 Pusher。如果 InfraValidator 失败，模型将不会被推送。

```python
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=tfx.proto.EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

### 配置 InfraValidator 组件。

您可以利用三种 proto 配置 InfraValidator。

#### `ServingSpec`

`ServingSpec` 是 InfraValidator 最关键的配置。它定义如下内容：

- 要运行<u>什么</u>类型的模型服务器
- 在<u>哪里</u>运行该模型服务器

对于模型服务器类型（称为应用二进制文件），我们支持：

- [TensorFlow Serving](serving.md)

注：InfraValidator 允许指定同一个模型服务器类型的多个版本，以便在不影响模型兼容性的情况下升级模型服务器版本。例如，用户可以同时使用 `2.1.0` 和 `latest` 版本测试 `tensorflow/serving` 镜像，以确保模型也与最新的 `tensorflow/serving` 版本兼容。

目前支持以下应用平台：

- 本地 Docker（应事先安装 Docker）
- Kubernetes（仅提供对 KubeflowDagRunner 的有限支持）

通过指定 `ServingSpec` 的 [`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof) 块来于应用二进制文件和应用平台进行选择。例如，要使用在 Kubernetes 集群上运行的 TensorFlow Serving 二进制文件，应设置 `tensorflow_serving` 和 `kubernetes` 字段。

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(
        tensorflow_serving=tfx.proto.TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=tfx.proto.KubernetesConfig()
    )
)
```

要进一步配置 `ServingSpec`，请查看 [protobuf 定义](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)。

#### `ValidationSpec`

可选配置，用于调整基础结构验证标准或工作流。

```python
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=tfx.proto.ServingSpec(...),
    validation_spec=tfx.proto.ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

所有 ValidationSpec 字段都有一个可靠的默认值。请在 [protobuf 定义](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto)中查看更多详细信息。

#### `RequestSpec`

可选配置，用于在 `LOAD_AND_QUERY` 模式下运行基础结构验证时指定如何构建样本请求。为了使用 `LOAD_AND_QUERY` 模式，需要在组件定义中同时指定 `request_spec` 执行属性以及 `examples` 输入通道。

```python
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=tfx.proto.ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
        local_docker=tfx.proto.LocalDockerConfig(),
    ),
    request_spec=tfx.proto.RequestSpec(
        # InfraValidator will look at how "classification" signature is defined
        # in the model, and automatically convert some samples from `examples`
        # artifact to prediction RPC requests.
        tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

### 生成带预热的 SavedModel

（从版本 0.30.0 开始）

由于 InfraValidator 使用真实请求验证模型，它可以轻松地将这些验证请求重用为 SavedModel 的[预热请求](https://www.tensorflow.org/tfx/serving/saved_model_warmup)。InfraValidator 提供了一个选项 (`RequestSpec.make_warmup`) 来导出带预热的 SavedModel。

```python
infra_validator = InfraValidator(
    ...,
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)
```

然后，输出的 `InfraBlessing` 工件将包含一个带预热的 SavedModel，和 `Model` 工件一样，也可以由 [Pusher](pusher.md) 推送。

## 局限性

当前的 InfraValidator 尚未完成，且存在一定的局限性。

- 只能验证 TensorFlow [SavedModel](/guide/saved_model) 模型格式。

- 在 Kubernetes 上运行 TFX 时，流水线应由 `KubeflowDagRunner` 在 Kubeflow Pipelines 内执行。模型服务器将在与 Kubeflow 所用的相同的 Kubernetes 集群和命名空间中启动。

- InfraValidator 主要侧重于 [TensorFlow Serving](serving.md) 的部署，虽然对于 [TensorFlow Lite](/lite)、[TensorFlow.js](/js) 或其他推断框架的部署仍然有用，但不太准确。

- `LOAD_AND_QUERY` 模式对 [Predict](/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def) 方法签名（TensorFlow 2 中唯一的可导出方法）的支持有限。InfraValidator 需要 Predict 签名才能将序列化的 [`tf.Example`](/tutorials/load_data/tfrecord#tfexample) 作为唯一输入进行使用。

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    - 请查看 [Penguin 示例](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local_infraval.py)中的示例代码，了解此签名如何与 TFX 中的其他组件进行交互。

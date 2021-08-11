# Apache Beam 和 TFX

[Apache Beam](https://beam.apache.org/) 提供了一个框架，用于运行在各种执行引擎上运行的数据批处理和流处理作业。一些 TFX 库使用 Beam 运行任务，实现了跨计算集群的高度可扩展性。Beam 包含对各种执行引擎或“运行程序”的支持，其中包括在单个计算节点上运行的直接运行程序，这对于开发、测试或小型部署而言非常实用。Beam 提供了一个抽象层，使 TFX 无需修改代码便可在任何支持的运行程序上运行。TFX 使用 Beam Python API，因此仅适用于 Python API 支持的运行程序。

## 部署和可扩展性

随着工作负载要求的增加，Beam 可以扩展到跨大型计算集群的超大型部署。它的可扩展性仅受限于底层运行程序的可扩展性。大型部署中的运行程序通常将部署到诸如 Kubernetes 或 Apache Mesos 之类的容器编排系统中，实现应用部署、扩展和管理的自动化。

有关 Apache Beam 的更多信息，请参阅 [Apache Beam](https://beam.apache.org/) 文档。

对于 Google Cloud 用户，推荐使用 [Dataflow](https://cloud.google.com/dataflow)，它通过自动扩缩资源，动态工作再平衡，与其他 Google Cloud 服务深度集成，内置安全性，以及监控。

## 自定义 Python 代码和依赖关系

在 TFX 流水线中使用 Beam 的一个明显的复杂性是处理自定义代码和/或需要从其他 Python 模块获取的依赖关系。这可能会在以下示例中成为问题：

- preprocessing_fn 需要引用用户自己的 Python 模块
- 用于 Evaluator 组件的自定义提取程序
- 从 TFX 组件子类化的自定义模块

TFX 依赖于 Beam 对[管理 Python 流水线依赖关系](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)的支持来处理 Python 依赖项。目前有两种管理方式：

1. 以源代码包的形式提供 Python 代码和依赖关系
2. [仅限 Dataflow] 将容器映像用作工作进程

下文将讨论这些问题。

### 以源代码包的形式提供 Python 代码和依赖关系

建议以下用户使用此方式：

1. 熟悉 Python 打包，并且
2. 仅使用 Python 源代码（即，没有 C 模块或共享库）。

请按照[管理 Python 流水线依赖关系](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)的其中一个路径，使用以下 beam_pipeline_args 中的一项来提供此功能：

- --setup_file
- --extra_package
- --requirements_file

注意：在上述任何情况下，请务必将相同版本的 `tfx` 列为依赖项。

### [仅限 Dataflow] 将容器映像用作工作进程

TFX 0.26.0 及更高版本具有针对 Dataflow 工作进程使用[自定义容器映像](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images)的实验性支持。

为了使用此功能，您必须：

- 构建预装了 `tfx` 和用户自定义代码与依赖项的 Docker 映像。
    - 对于 (1) 使用 `tfx>=0.26` 和 (2) 使用 Python 3.7 开发流水线的用户，最简单的方式是扩展官方 `tensorflow/tfx` 映像的相应版本：

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

- 将构建的映像推送到可以由 Dataflow 使用的项目访问的容器映像注册中心。
    - Google Cloud 用户可以考虑使用 [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build)，它可以很好地将上述步骤自动化。
- 提供以下 `beam_pipeline_args`：

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562)：当其为 Dataflow 的默认值时，移除 use_runner_v2**

**TODO(b/179738639)：创建有关如何在 https://issues.apache.org/jira/browse/BEAM-5440 之后在本地测试自定义容器的文档。**

## Beam 流水线参数

一些 TFX 组件依赖 Beam 进行分布式数据处理。可以通过 `beam_pipeline_args` 对它们进行配置，该参数在流水线创建期间指定：

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

TFX 0.30 及更高版本添加了一个接口，`with_beam_pipeline_args`，用于扩展每个组件的流水线级别 Beam 参数：

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```

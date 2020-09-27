# 了解 TFX 流水线

MLOps 是一种应用 DevOps 做法来帮助自动化、管理和审核机器学习 (ML) 工作流的做法。ML 工作流包括以下步骤：

- 准备、分析和转换数据。
- 训练和评估模型。
- 将训练的模型部署到生产中。
- 跟踪 ML 工件并了解其依赖项。

以特殊方式管理这些步骤可能既困难又耗时。

TFX 提供了一个工具包，可帮助您在各种编排器上编排 ML 流程，从而简化 MLOps 的实现过程，这些编排器包括：Apache Airflow、Apache Beam 和 Kubeflow Pipelines。通过将工作流实现为 TFX 流水线，您可以：

- 使 ML 流程自动化，从而允许您定期重新训练、评估和部署模型。
- 利用分布式计算资源来处理大型数据集和工作负载。
- 通过运行具有不同超参数集的流水线来提高实验速度。

本指南介绍了解 TFX 流水线所需的核心概念。

## 工件

TFX 流水线中步骤的输出称为**工件**。工作流中的后续步骤可能会使用这些工件作为输入。这样，TFX 便允许您在工作流步骤之间传输数据。

例如，`ExampleGen` 标准组件发出序列化样本，另一些组件（例如 `StatisticsGen` 标准组件）则将这些样本用作输入。

必须使用在 [ML Metadata](mlmd) 存储中注册的**工件类型**对工件进行强类型化。详细了解 [ML Metadata 中使用的概念](mlmd#concepts)。

工件类型具有名称并定义其属性的架构。工件类型名称在 ML Metadata 存储中必须唯一。TFX 提供了几种[标准工件类型](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external }，这些类型描述了复杂的数据类型和值类型，例如：字符串、整数和浮点数。您可以[重用这些工件类型](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py){: .external }，或者定义派生自 [`Artifact`](https://github.com/tensorflow/tfx/blob/master/tfx/types/artifact.py){: .external } 的自定义工件类型。

## 参数

参数是在执行流水线之前已知的流水线输入。借助参数，您可以通过配置而不是代码来更改流水线或部分流水线的行为。

例如，您可以使用参数来运行具有不同超参数集的流水线，而无需更改流水线的代码。

利用参数，您可以更轻松地使用不同的参数集来运行流水线，从而提高实验速度。

详细了解 [RuntimeParameter 类](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/data_types.py){: .external }。

## 组件

**组件**是 ML 任务的实现，您可以将其用作 TFX 流水线中的步骤。组件包括：

- 组件规范，用于定义组件的输入和输出工件以及组件的必需参数。
- 执行器，用于实现代码以执行 ML 工作流中的步骤，例如提取和转换数据或训练和评估模型。
- 组件接口，用于包装组件规范和执行器以在流水线中使用。

TFX 提供了可在流水线中使用的几个[标准组件](index#tfx_standard_components)。如果这些组件不能满足您的需求，您可以构建自定义组件。[详细了解自定义组件](understanding_custom_components)。

## 流水线

TFX 流水线是 ML 工作流的可移植实现，可以在各种编排器上运行，例如：Apache Airflow、Apache Beam 和 Kubeflow Pipelines。流水线由组件实例和输入参数组成。

组件实例生成工件作为输出，并且通常依赖于上游组件实例生成的工件作为输入。组件实例的执行顺序通过创建工件依赖项的有向无环图 (DAG) 来确定。

例如，考虑一个执行以下操作的流水线：

- 使用自定义组件直接从专有系统中提取数据。
- 使用 StatisticsGen 标准组件为训练数据计算统计信息。
- 使用 SchemaGen 标准组件创建数据架构。
- 使用 ExampleValidator 标准组件检查训练数据是否存在异常。
- 使用 Transform 标准组件对数据集执行特征工程。
- 使用 Trainer 标准组件训练模型。
- 使用 Evaluator 组件评估训练的模型。
- 如果模型通过评估，则流水线会使用自定义组件将训练的模型排入专有部署系统队列中。

![](images/tfx_pipeline_graph.svg)

为了确定组件实例的执行顺序，TFX 会分析工件依赖项。

- 数据提取组件没有任何工件依赖项，因此它可以是计算图中的第一个节点。
- StatisticsGen 依赖于数据提取生成的*样本*，因此必须在数据提取后执行。
- SchemaGen 依赖于 StatisticsGen 创建的*统计信息*，因此必须在 StatisticsGen 后执行。
- ExampleValidator 依赖于 StatisticsGen 创建的*统计信息*和 SchemaGen 创建的*架构*，因此必须在 StatisticsGen 和 SchemaGen 后执行。
- Transform 依赖于数据提取生成的*样本*和 SchemaGen 创建的*架构*，因此必须在数据提取和 SchemaGen 后执行。
- Trainer 依赖于数据提取生成的*样本*、SchemaGen 创建的*架构*以及 Transform 生成的*已保存模型*，因此只能在数据提取、SchemaGen 和 Transform 后执行。
- Evaluator 依赖于数据提取生成的*样本*和 Trainer 生成的*已保存模型*，因此必须在数据提取和 Trainer 后执行。
- 自定义部署器依赖于 Trainer 生成的*已保存模型* 和 Evaluator 创建的*分析结果*，因此部署器必须在 Trainer 和 Evaluator 后执行。

根据此分析，编排器的工作方式如下：

- 依次运行数据提取、StatisticsGen、SchemaGen 组件实例。
- ExampleValidator 和 Transform 组件可以并行运行，因为它们共享输入工件依赖项，并且不依赖于彼此的输出。
- 在 Transform 组件完成后，Trainer、Evaluator 和自定义部署器组件实例依次运行。

详细了解如何[构建 TFX 流水线](build_tfx_pipeline)。

## TFX 流水线模板

TFX 流水线模板通过提供可针对用例自定义的预构建流水线，使流水线开发变得更加容易。

详细了解如何[自定义 TFX 流水线模板](build_tfx_pipeline#build-a-pipeline-using-a-template)。

## 流水线运行

运行是流水线的单次执行。

## 编排器

编排器是一个您可以在其中执行流水线运行的系统。TFX 支持众多编排器，例如：[Apache Airflow](airflow)、[Apache Beam](beam_orchestrator) 和 [Kubeflow Pipelines](kubeflow)。TFX 还使用术语 *DagRunner* 来指代支持编排器的实现。

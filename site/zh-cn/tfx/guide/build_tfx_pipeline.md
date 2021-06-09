# 构建 TFX 流水线

注：有关 TFX 流水线的概念视图，请参阅[理解 TFX 流水线](understanding_tfx_pipelines)。

注：想在深入了解详细信息之前构建您的第一个流水线？请从[使用模板构建流水线](#build_a_pipeline_using_a_template)开始。

## 使用 `Pipeline` 类

TFX 流水线使用 [`Pipeline` 类](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py) {: .external } 进行定义。以下示例将演示如何使用 `Pipeline` 类。

<pre class="devsite-click-to-copy prettyprint">pipeline.Pipeline(
    pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;,
    pipeline_root=&lt;var&gt;pipeline-root&lt;/var&gt;,
    components=&lt;var&gt;components&lt;/var&gt;,
    enable_cache=&lt;var&gt;enable-cache&lt;/var&gt;,
    metadata_connection_config=&lt;var&gt;metadata-connection-config&lt;/var&gt;,
)
</pre>

替换以下内容：

- <var>pipeline-name</var>：此流水线的名称。流水线名称必须唯一。

    TFX 会使用流水线名称在 ML Metadata 中查询组件输入工件。重用流水线名称可能会导致意外行为。

- <var>pipeline-root</var>：此流水线输出的根路径。根路径必须是编排器具有读取和写入访问权限的目录的完整路径。在运行时，TFX 使用流水线根目录生成组件工件的输出路径。此目录可以是本地目录，也可以位于受支持的分布式文件系统（如 Google Cloud Storage 或 HDFS）。

- <var>components</var>：组成此流水线工作流的组件实例的列表。

- <var>enable-cache</var>：（可选）布尔值，指示此流水线是否使用缓存来加速流水线执行。

- <var>metadata-connection-config</var>：（可选）ML Metadata 的连接配置。

## 定义组件执行计算图

组件实例会生成工件作为输出，并且通常依赖于上游组件实例生成的工件作为输入。通过创建工件依赖项的有向无环图 (DAG) 来确定组件实例的执行顺序。

例如，`ExampleGen` 标准组件可以从 CSV 文件提取数据并输出序列化的样本记录。`StatisticsGen` 标准组件接受这些样本记录作为输入并生成数据集统计信息。在此示例中，`StatisticsGen` 的实例必须遵循 `ExampleGen`，因为 `SchemaGen` 取决于 `ExampleGen` 的输出。

### 基于任务的依赖关系

注：通常不建议使用基于任务的依赖关系。通过使用工件依赖项定义执行计算图，您可以利用 TFX 的自动工件沿袭跟踪和缓存功能。

您还可以使用组件的 [`add_upstream_node` 和 `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external } 方法定义基于任务的依赖关系。您可以通过 `add_upstream_node` 指定当前组件必须在指定组件之后执行。或者通过 `add_downstream_node` 指定当前组件必须在指定组件之前执行。

## 流水线模板

要快速设置流水线并查看所有部件是如何装配在一起的，最简单的方式是使用模板。模板的使用方法在[本地构建 TFX 流水线](build_local_pipeline)中进行了介绍。

## 缓存

TFX 流水线缓存使您的流水线可以跳过在先前的流水线运行中使用相同输入集执行过的组件。如果启用了缓存，流水线会尝试将每个组件的签名、组件和输入集与此流水线先前的组件执行进行匹配。如果存在匹配项，流水线将使用先前运行中的组件输出。如果无匹配，则执行组件。

如果流水线使用非确定性组件，请勿使用缓存。例如，如果为流水线创建一个组件来创建随机数，启用缓存会使此组件执行一次。在此示例中，后续运行会使用首次运行的随机数，而不是生成随机数。

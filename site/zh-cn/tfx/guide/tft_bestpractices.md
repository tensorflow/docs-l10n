<devsite-mathjax config="TeX-AMS-MML_SVG"></devsite-mathjax>

# 预处理数据以进行机器学习：选项和建议

本文档是由两部分组成的系列文章中的第一部分，该系列文章探讨了机器学习 (ML) 的数据工程和特征工程主题，重点介绍监管学习任务。第一部分介绍在 Google Cloud 上的机器学习流水线中预处理数据的最佳做法。该文档重点介绍如何使用 TensorFlow 和开源 [TensorFlow Transform](https://github.com/tensorflow/transform){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" } (`tf.Transform`) 库来准备数据、训练模型，并应用该模型用于预测。本文档重点介绍了预处理数据以进行机器学习时遇到的挑战，并介绍了在 Google Cloud 上有效地执行数据转换的选项和方案。

本文档假定您熟悉 [BigQuery](https://cloud.google.com/bigquery/docs){: .external }、[Dataflow](https://cloud.google.com/dataflow/docs){: .external }、[Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external } 和 TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview) API。

第二个文档[使用 Google Cloud 进行机器学习数据预处理](../tutorials/transform/data_preprocessing_with_cloud)提供了分步教程，说明如何实现 `tf.Transform` 流水线。

## 简介

机器学习可帮助您在数据中自动发现复杂但可能有用的模式。这些模式被精简到机器学习模型中，然后可以在新数据点上使用 – 这个过程称为“预测”或“推断”。

构建机器学习模型的过程涉及多个步骤，每个步骤都有自已的技术和概念挑战。这篇由两部分组成的系列文章重点介绍监管学习任务以及选择、转换和增强源数据以创建目标变量的强大预测信号的过程。这些操作结合了领域知识与数据科学技术。这些操作是[特征工程](https://developers.google.com/machine-learning/glossary/#feature_engineering){: .external }的本质。

实际机器学习模型的训练数据集大小很容易达到或超过 1 TB。因此，您需要大规模数据处理框架，才能采用分布式方式高效地处理这些数据集。当您使用机器学习模型进行预测时，您必须在新数据点上应用训练数据时所使用的转换。通过应用这一转换，您可以将实时数据集以模型预期的方式呈现给机器学习模型。

本文档讨论了下列不同粒度级别的特征工程操作面临的挑战：实例级、全通和时间窗口聚合。本文档还介绍了在 Google Cloud 上执行数据转换以进行机器学习的选项和方案。

本文档还概述了 [TensorFlow Transform](https://github.com/tensorflow/transform){: .external } (`tf.Transform`)，tf.Transform 是一个 TensorFlow 库，允许您通过数据预处理流水线定义实例级和全通数据转换。这些流水线使用 [Apache Beam](https://beam.apache.org/){: .external } 执行，它们还会创建工件，让您在预测的时候应用与应用模型时同样的转换。

## 预处理数据以进行机器学习

本部分介绍数据预处理操作和数据就绪的各阶段，还讨论了预处理操作的类型及其粒度。

### 数据工程与特征工程比较

预处理数据以进行机器学习同时涉及数据工程和特征工程。数据工程是将“原始数据”转换为“已就绪数据”的过程。然后，特征工程对已就绪数据进行调整，创建机器学习模型所期望的特征。这些术语具有以下含义：

**原始数据**（或仅称为**数据**）：以源形式提供的数据，没有事先为机器学习做任何准备。在此上下文中，数据可能以原始形式提供（在数据湖中）或以转换后形式提供（在数据仓库中）。数据仓库中的已转换数据可能已从其原始形式转换为可用于分析的形式。但在此上下文中，原始数据指的是数据尚未专门为机器学习任务做准备。如果数据从最终调用机器学习模型进行预测的流式传输系统发送，则也会被视为原始数据。

**已就绪数据**：形式可用于处理机器学习任务的数据集：数据源已经过解析、联接并以表格形式存储。已就绪数据已聚合并汇总至正确的粒度 - 例如，数据集中的每行代表一个唯一的客户，每列代表客户的摘要信息，如过去六周的总支出。在已就绪数据表中，已删除不相关的列，并且已过滤掉无效记录。监管学习任务存在目标特征。

**工程化特征**：数据集的特征根据模型需要进行了调整 - 即，通过对已就绪数据集中的列执行某些机器学习专有的操作以及在训练和预测期间为模型创建新特征来创建特征，如后面的[预处理操作](#preprocessing_operations)中所述。这样的操作示例有数值标尺列（将值缩放到 0 到 1 之间）、剪辑值以及采用[独热编码](https://developers.google.com/machine-learning/glossary/#one-hot_encoding){: .external }方式编码的分类特征。

下面的图 1 展示了准备预处理数据所涉及的步骤：

<figure id="data-flow-raw-prepared-engineered-features">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-data-preprocessing-flow.svg"
    alt="Flow diagram showing raw data moving to prepared data moving to engineered features.">
  <figcaption><b>Figure 1.</b> The flow of data from raw data to prepared data to engineered
features to machine learning.</figcaption>
</figure>

实际上，来自同一来源的数据经常处于不同的就绪阶段。例如，数据仓库中某表格的一个字段可能直接用作工程化特征；同时，同一表中的另一个字段可能需要经过转换才能成为工程化特征。同样，同一数据预处理步骤中可能会结合使用数据工程和特征工程操作。

### 预处理操作

数据预处理包括多项操作。每项操作都是为了帮助机器学习构建更好的预测模型。本文档不详细讨论这些预处理操作的详情，但本部分会简要介绍某些操作。

对于结构化数据，数据预处理操作包括以下步骤：

- **数据清理**：从原始数据中移除或更正存在损坏值或无效值的记录，以及移除缺少大量列的记录。
- **实例选择和分区**从输入数据集中选择数据点以创建[训练、评估（验证）和测试集](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets){: .external }。该过程包括用于可重复随机采样、少数类过度采样和分层分区的技术。
- **特征调整**：提高机器学习特征的质量，包括缩放和归一化数值、输入缺失值、剪辑离群值以及调整存在偏差分布的值。
- **特征转换**：（通过[分桶](https://developers.google.com/machine-learning/glossary/#bucketing){: .external }）将数值特征转换为分类特征，并（通过独热编码、[计数学习](https://dl.acm.org/doi/10.1145/3326937.3341260){: .external }、稀疏特征嵌入等技术）将分类特征转换为数值表示法。某些模型仅能处理数值或分类特征，而有的模型可以处理混合类型特征。即使是模型能够处理这两种类型，但也可受益于同一特征的不同表示法（数值和分类）。
- **特征提取**：使用 [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis){: .external }、[嵌入向量](https://developers.google.com/machine-learning/glossary/#embeddings){: .external }提取和[哈希处理](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f){: .external }等技术创建维度较低且更有效的数据表示法，进而减少特征的数量。
- **特征选择**：选择一部分输入特征用于训练模型，并通过使用[过滤器或封装容器方法](https://en.wikipedia.org/wiki/Feature_selection){: .external }忽略不相关的特征或冗余特征。如果特征缺少大量值，则特征选择也可以直接丢弃特征。
- **特征构造**：使用[多项式展开](https://en.wikipedia.org/wiki/Polynomial_expansion){: .external }（通过使用单变量数学函数）或[特征组合](https://developers.google.com/machine-learning/glossary/#feature_cross){: .external }（用来捕获特征交互）之类的典型技术创建新特征。特征也可以使用机器学习使用场景领域的业务逻辑来构造。

当您使用图片、音频或文本文档等非结构化数据时，深度学习已将数据合入到模型架构中以取代基于相关领域知识的特征工程。[卷积层](https://developers.google.com/machine-learning/glossary/#convolutional_layer){: .external }会自动预先处理特征。构建适当的模型架构需要对数据有一些经验知识，此外，还需要进行一些预处理，例如：

- 对于文本文档：需进行[词干提取和词形还原](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){: .external }、[TF-IDF](https://en.wikipedia.org/wiki/Tf%e2%80%93idf){: .external } 计算、[n-gram](https://en.wikipedia.org/wiki/N-gram){: .external } 提取及嵌入向量查找。
- 对于图片：需进行裁剪、大小调整、剪裁、高斯模糊处理和 Canary 过滤器操作。
- 对于所有类型的数据（包括文本和图片）：需进行[迁移学习](https://developers.google.com/machine-learning/glossary/#transfer_learning){: .external }，将完全训练的模型中除最后层以外的所有层视为一个特征工程步骤。

### 预处理粒度

本部分讨论数据转换类型的粒度，说明了为什么在使用用于训练数据的转换准备新数据点进行预测的时候，这一点至关重要。

预处理和转换操作可以根据操作粒度来分类，如下所示：

- **训练和预测期间的实例级转换**。这些是明确的转换，这种情况下，转换只需要来自同一实例的值。例如，实例级转换可能包括将特征的值裁剪到某个阈值、以多项式方式展开另一个特征、将两个特征相乘，或者比较两个特征来创建布尔标志。

    这些转换在训练和预测期间必须以相同方式应用，因为模型将用经过转换的特征进行训练，而不是用原始输入值来训练。如果未以相同方式转换数据，则模型的表现会很差，因为它得到的数据中有训练时未遇到过的值分布。如需了解详情，请参阅[预处理挑战](#preprocessing_challenges)部分中的训练-应用偏差讨论。

- **训练期间采用全通转换，但预测期间采用实例级转换**。在这种情况下，转换是有状态的，因为它们使用一些预先计算的统计信息来执行转换。在训练期间，您要分析整个训练数据，以计算用于在预测时转换训练数据、评估数据和新数据的数值，例如最小值、最大值、平均值和方差等。

    例如，要对数值特征进行归一化以进行训练，您需要计算整个训练数据的平均值 (μ) 及其标准差 (σ)。这种计算称为“全通”或“分析”操作。当您应用模型进行预测时，需要对新数据点的值进行归一化，以避免出现训练-应用偏差。因此，在训练期间计算的 μ 和 σ 值用于调整特征值，这就是以下简单的实例级操作：

    <div> $$ value_{scaled} = (value_{raw} - \mu) \div \sigma $$</div>

    全通转换包括下列操作：

    - 使用根据训练数据集计算的“最小值”和“最大值”对数值特征进行 MinMax 缩放。
    - 使用根据训练数据集计算的 μ 和 σ 对数值特征进行标准缩放（z-score 归一化）。
    - 使用分位数对数值特征进行分桶。
    - 使用中位数（数值特征）或模式（分类特征）输入缺失值。
    - 通过提取输入分类特征的所有不同值（词汇表），将字符串（标称值）转换为整数（索引）。
    - 计算某个字词（特征值）在所有文档（实例）中的出现次数以计算 TF-IDF。
    - 计算输入特征的 PCA 以将数据投射到较低的维度空间（具有线性相关特征）。

    您只应使用训练数据来计算 μ、σ、最小值和最大值等统计信息。如果您加入测试和评估数据来进行这些操作，就会[泄露](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742){: .external }用来训练模型的评估和测试数据信息。这样做会影响测试和评估结果的可靠性。为了确保对所有数据集应用一致的转换，您可以使用根据训练数据计算的同一些统计信息来转换测试和评估数据。

- **在训练和预测期间进行历史聚合**。这涉及到创建业务聚合、派生和标志作为预测任务的输入信号，例如创建[新近度、频率和货币 (RFM)](https://en.wikipedia.org/wiki/RFM_(market_research)){: .external } 指标供客户构建倾向模型。这些类型的特征可以预计算并存储在特征存储区中，以便在模型训练、批量打分和在线预测期间使用。您还可以在训练和预测之前对这些聚合执行其他特征工程（例如转换和调整）。

- **在训练期间进行历史聚合，但在预测期间进行实时聚合**。此方法包括汇总随时间变化的实时值来创建特征。在此方法中，要聚合的实例是通过时间窗口子句定义的。例如，如果您要根据路线在过去 5 分钟、过去 10 分钟、过去 30 分钟以及其他间隔的流量指标来估算出租车行程时间，可以使用此方法。您也可以使用该方法根据过去 3 分钟内计算的温度和振动值的移动平均值来预测引擎零件是否出现故障。尽管这些聚合可以离线准备以用于训练，但是要在模型使用期间从数据流中实时计算。

    更确切地说，在准备训练数据时，如果聚合值不在原始数据中，则该值将在数据工程阶段创建。原始数据通常以 `(entity, timestamp, value)`的格式存储在数据库中。在前面的示例中，entity 是出租车路线的路段标识符以及引擎故障的引擎零件标识符。您可以使用限定时间窗口的操作来计算 `(entity, time_index, aggregated_value_over_time_window)`，并使用聚合特征作为模型训练的输入。

    将模型应用于实时（在线）预测时，会期望以派生自聚合值的特征作为输入。因此，您可以使用 Apache Beam 之类的流式处理技术，利用流入系统的实时数据点计算聚合值。流处理技术在新数据点到达时，根据时间窗口聚合实时数据。您还可以在训练和预测之前对这些聚合执行其他特征工程（例如转换和调整）。

## Google Cloud 上的机器学习流水线{: id="machine_learning_pipeline_on_gcp" }

本部分讨论使用代管式服务在 Google Cloud 上训练和提供 TensorFlow 机器学习模型的典型端到端流水线的核心组件。此外，还讨论了在何种情形下可以实现不同类别的数据预处理操作，以及实现此类转换时可能遇到的常见挑战。[tf.Transform 的工作原理](#how_tftransform_works)部分介绍了“TensorFlow 转换”库如何帮助应对这些挑战。

### 概要架构

下图（图 2）显示了用于训练和应用 TensorFlow 模型的典型机器学习流水线的高层架构。图中的标签 A、B 和 C 指的是流水线中可以预处理数据的不同位置。这些步骤的详细信息将在下一部分提供。

<figure id="high-level-architecture-for-training-and-serving">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-ml-training-serving-architecture.svg"
    alt="Architecture diagram showing stages for processing data.">
  <figcaption><b>Figure 2.</b> High-level architecture for ML training and
    serving on Google Cloud.</figcaption>
</figure>

此流水线包括以下步骤：

1. 导入原始数据后，表格数据存储在 BigQuery 中，其他数据（如图片、音频和视频）存储在 Cloud Storage 中。本系列文章的第二部分以 BigQuery 中存储的表格数据为例。
2. 使用 Dataflow 规模化执行数据工程（准备）和特征工程。这种执行将生成可供机器学习使用的训练、评估和测试集，它们存储在 Cloud Storage 中。理想情况下，这些数据集存储为 [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) 文件，这是用于 TensorFlow 计算的最优格式。
3. 将 TensorFlow 模型[训练程序包](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container){: .external }提交给 Vertex AI Training，后者会使用在前面步骤中预处理过的数据来训练模型。此步骤会输出经过训练的 TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model)，并将其导出到 Cloud Storage。
4. 以采用 REST API 的服务形式将经过训练的 TensorFlow 模型部署到 Vertex AI Prediction，以便用于在线预测。此模型也可用于批量预测作业。
5. 将模型部署为 REST API 后，客户端应用和内部系统可以调用此 API，具体方法是发送带有某些数据点的请求并接收来自模型的包含预测结果的响应。
6. 如要编排和自动执行此流水线，可以使用 [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction){: .external } 作为调度器来调用数据准备、模型训练和模型部署步骤。

您还可以使用 [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/){: .external } 来存储输入特征以进行预测。例如，您可以定期使用最新的原始数据创建工程化特征，并将其存储在 Vertex AI Feature Store 中。客户端应用从 Vertex AI Feature Store 中提取所需的输入特征，并将其发送到模型以接收预测。

### 在何处执行预处理

在图 2 中，标签 A、B 和 C 显示，数据预处理操作可以在 BigQuery、Dataflow 或 TensorFlow 中进行。以下各部分分别介绍了这些选项的工作原理。

#### 选项 A：BigQuery{: id="option_a_bigquery"}

通常，BigQuery 中会针对以下操作实现逻辑：

- 采样：从数据中随机选择一个子集。
- 过滤：移除不相关或无效的实例。
- 分区：拆分数据以生成训练、评估和测试集。

BigQuery SQL 脚本可用作 Dataflow 预处理流水线的源查询，这是图 2 中的数据处理步骤。例如，如果系统在加拿大使用，并且数据仓库包含来自世界各地的事务，则过滤以获取仅限加拿大范围的训练数据最好在 BigQuery 中完成。BigQuery 中的特征工程简单且可扩缩，并且支持实现实例级聚合和历史聚合特征转换。

但是，只有在您使用模型进行批量预测（评分）或者在线预测期间特征在 BigQuery 中预计算，但在 Vertex AI Feature Store 中存储时，我们才建议您使用 BigQuery 进行特征工程。如果您计划部署模型以进行在线预测，并且在线特征存储区中没有工程化特征，则必须复制 SQL 预处理操作以转换由其他系统生成的原始数据点。换句话说，您需要将该逻辑实现两次：一次是在 SQL 中预处理 BigQuery 中的训练数据，一次是在使用模型的应用逻辑中预处理用于预测的在线数据点。

例如，如果您的客户端应用采用 Java 编写，则需要在 Java 中重新实现该逻辑。由于实现差异，这样做可能会引起错误，如本文档后面的[预处理挑战](#preprocessing_challenges)中的训练-应用偏差部分所述。此外，维护两种不同的实现也会产生额外的开销。只要在 SQL 中更改了逻辑以预处理训练数据，就需要相应地更改 Java 实现才能在应用模型时预处理数据。

如果您的模型只用于批量预测（例如，使用 Vertex AI [批量预测](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions){: .external }），并且您用于评分的数据来自 BigQuery，则可以在 BigQuery SQL 脚本中实现这些预处理操作。在这种情况下，您可以使用同一预处理 SQL 脚本来准备训练数据和评分数据。

全通有状态转换不适合在 BigQuery 中实现。如果使用 BigQuery 进行全通转换，则需要辅助表来存储有状态转换所需的数值，例如用来缩放数值特征的平均值和方差。此外，在 BigQuery 上使用 SQL 实现全通转换会增加 SQL 脚本的复杂性，并在训练和评分 SQL 脚本之间产生复杂的依赖关系。

#### 选项 B：Dataflow{: id="option_b_cloud_dataflow"}

如图 2 所示，您可以在 Apache Beam 中实现需要大量计算的预处理操作，并使用 Dataflow 规模化运行它们。Dataflow 是一种用于批处理和流式数据处理的全代管式自动扩缩服务。与 BigQuery 不同，使用 Dataflow 时，您还可以使用外部专用库进行数据处理。

Dataflow 可以执行实例级转换以及历史和实时聚合特征转换。特别是，如果您的机器学习模型需要 `total_number_of_clicks_last_90sec` 之类的输入特征，则 Apache Beam [窗口化函数](https://beam.apache.org/documentation/programming-guide/#windowing){: .external }可以实时汇总（流式处理）事件数据（例如点击事件）的时间窗口值，并据此计算这些特征。先前在讨论[转换粒度](#preprocessing_granularity)时，我们称之为“在训练期间进行历史聚合，但在预测期间进行实时聚合”。

下面的图 3 显示了 Dataflow 在处理流式数据以实现接近实时的预测时起到的作用。

<figure id="high-level-architecture-for-stream-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-streaming-data-with-dataflow-architecture.svg"
    alt="Architecture for using stream data for prediction.">
  <figcaption><b>Figure 3.</b> High-level architecture using stream data
    for prediction in Dataflow.</figcaption>
</figure>

如图 3 所示，在处理期间，名为*数据点*的事件注入到 [Pub/Sub](https://cloud.google.com/pubsub/docs){: .external } 中。Dataflow 使用这些数据点，根据随时间进行的聚合来计算特征，然后调用已部署的机器学习模型 API 进行预测。预测随后会被发送到出站 Pub/Sub 队列。从 Pub/Sub 起，预测可以由监控或控制等下游系统使用，也可以推送回（例如以通知形式）原始请求客户端。预测结果也可以存储在 [Cloud Bigtable](https://cloud.google.com/bigtable/docs){: .external } 等低延迟数据存储区中，以进行实时提取。还可以使用 Cloud Bigtable 来累积和存储这些实时聚合，以便在预测需要时能够查询。

同一 Apache Beam 实现可用于批处理来自 BigQuery 等离线数据存储区的训练数据，以及用于流式处理实时数据以提供在线预测。

在其他典型架构（例如图 2 所示的架构）中，客户端应用直接调用已部署的模型 API 进行在线预测。在这种情况下，如果在 Dataflow 中实现预处理操作以准备训练数据，则这些操作不会应用于直接进入模型的预测数据。因此，在应用模型以进行线预测期间，应将此类转换集成到模型中。

通过规模化计算所需的统计信息，Dataflow 可用于执行全通转换。但是，这些统计信息需要存储在某处，以便在预测期间用于转换预测数据点。使用 TensorFlow Transform (`tf.Transform`) 库，您可以将这些统计信息直接嵌入模型中，而不是将其存储在其他位置。[tf.Transform 的工作原理](#how_tftransform_works)稍后将介绍此方法。

#### 选项 C：TensorFlow{: id="option_c_tensorflow"}

如图 2 所示，您可以在 TensorFlow 模型自身中实现数据预处理和转换操作。如图所示，您为了训练 TensorFlow 模型所实现的预处理在模型导出并部署以用于预测时成为模型不可或缺的一部分。TensorFlow 模型中的转换可以通过以下方式之一完成：

- 在 `input_fn` 函数和 `serving_fn` 函数中实现所有实例级转换逻辑。`input_fn` 函数使用 [`tf.data.Dataset` API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 准备数据集以训练模型。`serving_fn` 函数接收并准备数据以进行预测。
- 使用 [Keras 预处理层](https://keras.io/guides/preprocessing_layers/){: .external }或[创建自定义层](https://keras.io/guides/making_new_layers_and_models_via_subclassing/){: .external }，将转换代码直接放在 TensorFlow 模型中。

[`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators) 函数中的转换逻辑代码定义了用于在线预测的 SavedModel 的应用接口。如果您在 `serving_fn` 函数的转换逻辑代码中实现曾用于准备训练数据的转换，则可确保在应用时将相同的转换应用于新预测数据点。

但是，由于 TensorFlow 模型独立或小批量处理每个数据点，因此您无法使用所有数据点计算聚合。因此，无法在 TensorFlow 模型中实现全通转换。

### 预处理挑战

下面是实现数据预处理的主要挑战：

- **训练-应用偏差**。[训练-应用偏差](https://developers.google.com/machine-learning/guides/rules-of-ml/#training-serving_skew){: .external }指的是训练期间和应用期间的效果（预测性能）差异。这种偏差可能由您在训练中及应用流水线中处理数据的方式之间的差异造成。例如，如果您的模型是根据对数转换特征训练的，但是在应用期间为其提供的是原始特征，则预测输出可能不准确。

    如果转换成为模型本身的一部分，处理实例级转换可能会变得非常简单，如前面的[选项 C：TensorFlow](#option_c_tensorflow) 中所述。在这种情况下，模型应用接口（[`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators) 函数）接受原始数据，而模型会在计算输出之前先在内部转换此数据。这些转换与应用于原始训练和预测数据点的转换相同。

- **全通转换**。您不能在 TensorFlow 模型中实现全通转换（例如扩缩和归一化转换）。在全通转换中，必须基于训练数据计算一些统计信息（例如用来缩放数值特征的 `max` 和 `min` 值），如[选项 B：Dataflow](#option_b_dataflow) 中所述。这些值必须存储在某处，以便在应用模型进行预测期间用于将新的原始数据点转换为实例级转换，从而避免造成训练-应用偏差。您可以使用 TensorFlow Transform (`tf.Transform`) 库将统计信息直接嵌入 TensorFlow 模型中。[tf.Transform 的工作原理](#how_tftransform_works)稍后将介绍此方法。

- **预先准备数据以提高训练效率**。在模型中实现实例级转换可能会降低训练过程的效率。这种降低是因为在每个周期将相同的转换重复应用于相同的训练数据。假设您的原始训练数据有 1000 个特征，并且您应用混合的实例级转换来生成 10000 个特征。如果您在模型中实现这些转换，然后向模型提供原始训练数据，则这 10000 个操作会在每个实例上应用 *N* 次，其中 *N* 是周期数。此外，如果您使用的是加速器（GPU 或 TPU），它们会在 CPU 执行这些转换时处于空闲状态，这会导致这些昂贵的加速器得不到有效利用。

    理想情况下，在训练之前通过使用[选项 B：Dataflow](#option_b_dataflow) 中所述的方法来转换训练数据，在这种情况下，这 10000 个转换操作仅在每个训练实例上应用一次。转换后的训练数据随后会提供给模型。不会应用更多的转换，并且加速器会一直处于忙碌状态。此外，使用 Dataflow 可帮助您使用全代管式服务规模化预处理大量数据。

    预先准备训练数据可以提高训练效率。但是，在模型外部实现转换逻辑（[选项 A：BigQuery](#option_a_bigquery) 或[选项 B：Dataflow](#option_b_dataflow) 中介绍了具体方法）无法解决训练-应用偏差问题。除非您将工程化特征存储在特征存储区中以用于训练和预测，否则必须在某处实现转换逻辑，以将其应用于用于预测的新数据点，因为模型接口需要转换后的数据。TensorFlow Transform (`tf.Transform`) 库可以帮助您解决此问题，如以下部分所述。

## tf.Transform 的工作原理{:#how_tftransform_works}

`tf.Transform` 库对于需要全通的转换非常有用。`tf.Transform` 库的输出导出为 TensorFlow 计算图，此计算代表实例级转换逻辑以及通过全通转换计算的统计信息，用于训练和应用。在训练和应用阶段使用同一个计算图可以防止出现偏差，因为两个阶段执行的转换操作完全相同。此外，`tf.Transform` 库可以在 Dataflow 上的批处理流水线中规模化运行，以预先准备训练数据并提高训练效率。

下面的图 4 显示了 `tf.Transform` 库如何预处理和转换数据以进行训练和预测。以下各部分介绍了此过程。

<figure id="tf-Transform-preprocessing--transforming-data-for-training-and-prediction">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-behavior-flow.svg"
    alt="Diagram showing flow from raw data through tf.Transform to predictions.">
  <figcaption><b>Figure 4.</b> Behavior of <code>tf.Transform</code> for
    preprocessing and transforming data.</figcaption>
</figure>

### 转换训练和评估数据

可使用 `tf.Transform` Apache Beam API 中实现的转换来预处理原始训练数据，并在 Dataflow 上规模化运行。预处理分以下阶段进行：

- **分析阶段**：在分析阶段，有状态转换所需的统计信息（如平均值、方差和分位数）由全通操作利用训练数据来计算。此阶段会生成一组转换工件，包括 `transform_fn` 计算图。`transform_fn` 计算图是一个 TensorFlow 计算图，它具有实例级操作形式的转换逻辑。它将分析阶段计算的统计信息作为常量包含在内。
- **转换阶段**：在转换阶段，`transform_fn` 计算图应用于原始训练数据，其中计算出的统计信息用于在实例级处理数据记录（例如，扩缩数值列）。

这类两阶段方法可解决执行全通转换时遇到的[预处理挑战](#preprocessing_challenges)。

预处理评估数据时，会使用 `transform_fn` 计算图中的逻辑以及从训练数据的分析阶段中计算的统计信息来仅应用实例级操作。换句话说，您不会将以全通方式分析评估数据计算得出的新统计信息（例如 μ 和 σ）用于对评估数据中的数值特征进行归一化，而是使用从训练数据计算的统计信息，以实例级方式来转换评估数据。

首先使用 Dataflow 规模化准备转换的训练和评估数据，然后才能将其用于训练模型。此批量数据准备过程解决了预先准备数据以提高训练效率的[预处理挑战](#preprocessing_challenges)。如图 4 所示，模型内部接口需要转换后的特征。

### 将转换附加到导出的模型上

如上所述，由 `tf.Transform` 流水线生成的 `transform_fn` 计算图存储为导出的 TensorFlow 计算图。导出的计算图将转换逻辑作为实例级操作包含在内，并将全通转换中计算的所有统计信息作为计算图常量包含在内。在导出经训练的模型以进行应用时，`transform_fn` 计算图会作为其 `serving_fn` 函数的一部分附加到 SavedModel 上。

当模型应用接口应用模型来进行预测时，该接口需要原始格式（即未执行过任何转换）的数据点。但是，模型内部接口需要已转换格式的数据。

`transform_fn` 计算图现在是模型的一部分，它会对传入的数据点应用所有预处理逻辑。它在预测期间使用实例级操作中存储的常量（例如使用 μ 和 σ 来归一化数值特征）。因此，`transform_fn` 计算图将原始数据点转换为已转换的格式。转换后的格式是模型内部接口生成预测结果所需的格式，如图 4 所示。

这种机制解决了训练-应用偏差的[预处理挑战](#preprocessing_challenges)，因为在预测应用期间转换新数据点所用的逻辑（实现），正是转换训练和评估数据所用的同一逻辑（实现）。

## 预处理选项汇总

下表汇总了本文档讨论的数据预处理选项。在下表中，“N/A”代表“不适用”。

<table class="alternating-odd-rows">
<tbody>
<tr>
<th>数据预处理选项</th>
<th>实例级<br>（无状态转换）</th>
<th>
  <p>训练期间全通，应用期间实例级（有状态转换）</p>
</th>
<th>
  <p>训练与应用期间实时（窗口）聚合（流式转换）</p>
</th>
</tr>
<tr>
  <td>
    <p>       <b>BigQuery</b>          (SQL)</p>
  </td>
  <td>
    <p><b>批量评分：良好</b>—在训练和批量评分期间，对数据应用相同的转换实现。</p>
    <p>       <b>在线预测：不推荐</b>—您可以处理训练数据，但会导致训练-应用偏差，因为您使用不同的工具处理应用数据。</p>
  </td>
  <td>
    <p><b>批量评分：不推荐</b>。</p>
    <p><b>在线预测：不推荐</b>。</p>
    <p>虽然您可以将使用 BigQuery 计算的统计信息用于实例级批量/在线转换，但这并不简单，因为您必须维护一个统计信息存储区，以便在训练期间填充数据并在预测期间使用。</p>
  </td>
  <td>
    <p><b>批量评分：不适用</b>—此类聚合是根据实时事件计算的。</p>
    <p>       <b>在线预测：不推荐</b>—您可以处理训练数据，但会导致训练-应用偏差，因为您使用不同的工具处理应用数据。</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam)</p>
  </td>
  <td>
    <p><b>批量评分：良好</b>—在训练和批量评分期间，对数据应用相同的转换实现。</p>
    <p>       <b>在线预测：良好</b>—如果应用时的数据来自 Pub/Sub 并由 Dataflow 使用。否则会导致训练-应用偏差。</p>
  </td>
  <td>
    <p><b>批量评分：不推荐</b>。</p>
    <p><b>在线预测：不推荐</b>。</p>
    <p>虽然您可以将使用 Dataflow 计算的统计信息用于实例级批量/在线转换，但这并不简单，因为您必须维护一个统计信息存储区，以便在训练期间填充数据并在预测期间使用。</p>
  </td>
  <td>
    <p><b>批量评分：不适用</b>—此类聚合是根据实时事件计算的。</p>
    <p>       <b>在线预测：良好</b>—在训练（批处理）和应用（流式处理）期间，对数据应用相同的 Apache Beam 转换。</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam + TFT)</p>
  </td>
  <td>
    <p><b>批量评分：良好</b>—在训练和批量评分期间，对数据应用相同的转换实现。</p>
    <p>       <b>在线预测：推荐</b>—避免训练-应用偏差和预先准备训练数据。</p>
  </td>
  <td>
    <p><b>批量评分：不推荐</b>。</p>
    <p><b>在线预测：不推荐</b>。</p>
    <p>建议采用这两种用法，因为训练期间的转换逻辑和计算的统计信息存储为 TensorFlow 计算图，此计算图附加到导出模型以进行应用。</p>
  </td>
  <td>
    <p><b>批量评分：不适用</b>—此类聚合是根据实时事件计算的。</p>
    <p>       <b>在线预测：良好</b>—在训练（批处理）和应用（流式处理）期间，对数据应用相同的 Apache Beam 转换。</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>TensorFlow</b> <sup>*</sup>       <br>（<code>input_fn</code> 和 <code>serving_fn</code>）</p>
  </td>
  <td>
    <p><b>批量评分：不推荐</b>。</p>
    <p><b>在线预测：不推荐</b>。</p>
    <p>为了提高这两种情况的训练效率，最好事先准备好训练数据。</p>
  </td>
  <td>
    <p>       <b>批量评分：不可行</b>。</p>
    <p>       <b>在线预测：不可行</b>。</p>
  </td>
  <td>
    <p><b>批量评分：不适用</b>—此类聚合是根据实时事件计算的。</p>
<p>       <b>在线预测：不可行</b>。</p>
  </td>
</tr>
</tbody>
</table>

<sup>*</sup> 使用 TensorFlow 时，交叉、嵌入和独热编码等转换应以声明方式作为 `feature_columns` 列执行。

## 后续步骤

- 要实现 `tf.Transform` 流水线并使用 Dataflow 运行该流水线，请参阅本系列文章的第二部分：[使用 TensorFlow Transform 为机器学习预处理数据](https://www.tensorflow.org/tfx/tutorials/transform/data_preprocessing_with_cloud)。
- 学习 Coursera 专项课程，了解如何[在 Google Cloud 上使用 TensorFlow 进行机器学习](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external }。
- 阅读[机器学习规则](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }，了解机器学习工程中的最佳做法。

- 如需查看更多参考架构、图表和最佳做法，请浏览<a href="https://www.tensorflow.org/tfx/guide/solutions" track-type="tutorial" track-name="textLink" track-metadata-position="nextSteps">云架构中心</a>。

# TensorBoard 通用数据 API 和实现

@wchargin, 2019-07-25

**状态**：使用 Python 指定；正在执行多个实现，这些实现处于不同阶段。

## 目的

TensorBoard 曾一直作为直接从磁盘读取数据的本地网络服务器运行。而 [TensorBoard.dev] 则在 App Engine 上运行并读取用户上传到我们的托管服务中的数据。此外，某些与 TensorBoard 类似的 Google 内部服务具有自己的数据库，并直接重新实现 TensorBoard HTTP API。

为了确保 TensorBoard 体验在所有后端之间统一，并让用户在所有环境下都能轻松地使用插件，我们需要一个 Python API（采用多种实现）来存储和访问数据。

本文档针对该 API 提出了读取方面的建议，并提供了针对每种实用后端实现该 API 的高级信息。

## 本体论

大多数 TensorBoard 插件都可以通过一种或多种关系模式自然表达：

- **scalars:** *(run, tag, step) → (value: f32)*
- **images:** *(run, tag, step, image_index) → (image_data: blob)*
- **音频** *(run, tag, step, audio_index) → (wav_data: blob)*
- **text:** *(run, tag, step, text_index) → (text_data: blob)*
- **histograms:** *(run, tag, step, bucket_index) → (lower_bound: f32, count: usize)*
- **PR curves:** *(run, tag, step, threshold_index) → (tp, fp, tn, fn, precision, recall)*
- **网格：**
    - 顶点：*(run, tag, step, vertex_index) → (x: f32, y: f32, z: f32, color: u8)*
    - 面：*(run, tag, step, face_index) → (v1: usize, v2: usize, v3: usize)*
- **graphs:** *(run, tag, kind: enum) → (graph: proto)*
- **超参数：**
    - 实验配置：*() → (experiment_info: proto)*
    - 会话组：*(session_group) → (session_info: proto)*

我们可以允许插件直接定义这些表，并借助关系数据库对其进行自由操作。但是，做出这种重要承诺会限制我们实现这些 API 的灵活性，而我们已经知道我们至少需要三种实现。此外，当出现新颖的插件却无法有效适配这一模型时，那么在使用该模型时难免会存在诸多问题。

相反，我们在上述关系中观察总结出三种常见模式，并赋予它们特权：

- **标量时间序列。**
    - 主要由标量插件使用。
    - 在实验中由运行、标记和步骤键控。
- **张量时间序列。**
    - 包括直方图和 PR 曲线。
    - 在实验中由运行、标记和步骤键控。
    - 每个张量的大小均以一个常量为上界（直方图为 0.5 KB，PR 曲线为 5 KB），小于 Cloud Spanner 单元大小 (10 MiB)。
- **Blob 序列时间序列。**
    - 包括图像、音频、文本、网格、计算图、超参数。
    - 潜在的未来用例：视频。
    - 可能是“概念上的张量”（例如图像、音频），但实际上是以压缩格式（例如 PNG、FLAC）编码。
    - 在实验中由运行、标记、步骤和索引键控。
    - 包括“序列”始终仅包含一个元素或“时间序列”始终仅包含一个步骤的简并用例。
    - 为何是 blob 序列而不是 blob 本身？每个图像和音频摘要在每个步骤均可包含多个图像/音频片段。赋予序列特权可有效实现这一目标，方法是在每个 GCS 对象中仅存储一个 PNG 文件并将其直接提供给浏览器。这种方法几乎没有弊端。
    - 大小可能不一。图像大小可达千字节（MNIST 输入）或千兆字节（[组织切片图]、TensorFlow 计算图和检查点）。
    - 实现可能对其进行特殊加密（例如，使用 GCS [CMEK]）：敏感数据通常需要包含在此类 blob 中，而不是标量或直方图中。
    - 序列长度通常具有较小的上界值。图像和音频的默认长度为 3，且网格始终为单例序列。

请注意，我们将标量与任意张量区分开。标量可以支持任意张量无法执行的某些运算（例如按区间内的最小/最大值进行聚合，或诸如“查找最近的 `accuracy` 至少为 0.9 的运行”之类的查询。标量信息中心也是使用最广泛的信息中心，因此我们想充分利用我们所能实现的任何额外的性能改进：例如，我们不必存储张量形状。

## API 调用

每个存储类都有两种 API 调用：“list metadata”和“get values”。所有 API 调用都要求调用者按拥有数据的插件（例如“images”）进行筛选。所有 API 调用都需要使用“实验 ID”参数和身份验证令牌。有关签名的更多详细信息如下：

- 标量时间序列：目前，与张量时间序列相同。以后可能会学习新的选项。
- 张量时间序列：
    - 支持按运行集和标记集（跨积语义）进行筛选。
    - 支持按步骤范围或特殊的“最近的 k 个步骤”进行筛选。
    - 降采样到提供的步骤数。
    - 返回嵌套映射：
        - 针对数据请求：*(run) → (tag) → list[(step, wall_time, tensor_value)]*；
        - 针对元数据请求：*(run) → (tag) → (max_step, max_wall_time, summary_metadata)*。
- Blob 序列时间序列：
    - 支持按运行集和标记集进行筛选。
    - 支持按步骤范围或特殊的“最近的步骤”进行筛选。
    - 支持按索引范围或特殊的“最近的索引”进行筛选。
    - 降采样到提供的步骤数。
    - 返回嵌套映射：
        - 针对数据请求：*(run) → (tag) → list[(step, wall_time, list[(blob_key)])]*；
        - 针对元数据请求：*(run) → (tag) → (max_step, max_length, summary_metadata)*。

请注意，仅在“步骤”轴上支持降采样。

每个“get values”API 调用都将对响应的最大大小应用上界（计算方法为 *num_runs* × *num_tags* × *sample_size*）。（回想一下，张量时间序列的元素预计较小。）

“blob 键”是用户可访问的绝对网址或不透明键的标记联合。具有不透明键的 blob 可以通过附加的 API 调用来读取：

- Blob 读取：
    - 接受由 blob 序列查询返回的 opaque_key 输入。
    - 返回指定 blob 的内容。
    - 另外，对于想要直接使用 DOM API 呈现 `<img>`/`<audio>`/ 等元素的插件，作为核心 HTTP 端点 `/data/blob/OPAQUE_KEY` 公开。

最后，插件将从顶级 API 调用开始搜索：

- 列表：
    - 接受插件标识符（例如“images”）。
    - 返回 *(run) → (tag) → (kind, summary_metadata)* 的映射，其中 *kind* 用于标识四个存储类之一。

请注意，这些 API 调用旨在由 TensorBoard 插件代码调用，因此会非常紧密地映射到那些预期的访问模式上。未来，我们可能会为最终用户提供其他 API，用于方便特殊浏览以及进行批处理分析或创建自定义信息中心（例如，直接导出到 `pandas` 数据帧以在 Colab 中使用）。这些 API 可以基于本文档所述的 API 实现。

## 存储实现

本部分内容介绍了这些 API 的可行实现。内容旨在提供信息，不可视为规范。

**托管后端**是我们对公共服务预期的稳定状态：

- 上传时，所有数据都被提取到 Cloud Spanner 或 GCS 中。
- 标量时间序列存储在将候选键 *(run_id, tag_id, step)* 映射至挂钟时间和浮点值的 Spanner 表中。
- 张量时间序列存储在 Spanner 表中，该表将候选键 *(run_id, tag_id, step)* 映射至挂钟时间、[打包的张量内容]的字节串以及张量形状和数据类型。
- Blob 序列时间序列存储在 GCS 上，并列在将候选键 *(run_id, tag_id, step, index)* 映射至 GCS 对象键的表中。（即使对于较小对象也是如此。）
- 单独的表存储运行和标记名称以及摘要元数据（由 ID 键控）。
- 可以在 SQL 中实现线性分桶降采样，如 [PR #1022] 中所述。
- 如果需要，[Cloud Spanner 的 `TABLESAMPLE` 算子]将免费提供支持伯努利采样或蓄水池采样的均匀随机降采样。
    - `TABLESAMPLE` 是 [SQL:2011 的一部分]，也可由 PostgreSQL 和 SQL Server 实现，但无法由 SQLite 实现。（sqlite-users 邮寄名单中从未出现过“tablesample”和“downsample”。）Cloud Spanner 不会实现算子的 `REPEATABLE(seed)` 部分。

可以适时基于 TensorBoard 的数据库模式实现**本地 SQLite 后端**：

- 在 [`--db_import` 时]，所有数据都被提取到 SQLite 中。
- 标量/张量时间序列存储在与其 Cloud Spanner 对应项同构的表中。
- Blob 序列时间序列存储在将 *(run_id, tag_id, step, index)* 映射至唯一 blob ID 的表中。
- Blob 数据存储在将 *blob_id* 映射至实际 blob 数据的表中。（此为“GCS”。）
- 单独的表存储运行和标记名称以及摘要元数据（由 ID 键控）。
- 请注意，SQLite blob 的大小限制为 1 GB。我们将向 blob 数据表添加 chunk_index 列的选项设置为开启状态，以便能够存储任意大小的 blob。
- 可以在 SQL 中实现线性分桶降采样，如 [PR #1022](https://github.com/tensorflow/tensorboard/pull/1022) 中所述。
- 均匀随机降采样没有内置函数。如果需要，我们必须选择一个实现：朴素模型（不利于周期性数据或非序列步骤）、 `ORDER BY random()`（不利于重现性；可能很慢）、 `WHERE random() < k`（伯努利采样；不利于重现性；其他方面可能不错）、`xs` 的 `WHERE rowid IN xs`（在客户端上预先计算的索引列表，可能不错？）、与将步骤映射至均匀随机数并限制为前 `k` 个随机值的全局表 `JOIN`（可能不错？）。

作为 Google 内部用户的迁移路径，我们将提供一个后端，与现有 Google 内部存储系统对接以获取实验数据。（Google 员工请参阅本文档的内部版本以了解详细信息。）

同样地，为了与磁盘上现有的 TensorBoard 数据兼容，我们将提供一个后端，使用与当前 TensorBoard（`plugin_event_multiplexer` 堆栈）相同的加载逻辑，并通过这些 API 公开数据。它将成为连通过去与未来的桥梁，并铺设一条持续的迁移路径，有效避免了全局转换所涉及的巨大工作量。

## 考虑的替代方案

### 将关系存储模型公开给插件

在“本体论”部分中已经讨论。我们在探索这种方式时提出了一个建议，允许插件以 *(candidate_key) → (columns)* 的形式声明零个或多个关系；这些是上方“本体论”部分中列出的关系。将基于一组固定类型绘制关系维度：标量类型 `i64`、`f64` 和 `bool`；元数据类型 `run_id`、`tag_id` 和 `step`；以及 blob 类型 `bytes`、`string` 和`large_file`。为每个主特性指定模式后，插件将能够查询：

- “任意值”
- “这些 `values` 之一”
- 对于 step-type 特性：“降采样至 `k`”，相对于 `run_id` 和 `tag_id` 特性进行解释

…并指定在结果集中包含还是排除每个非主特性。例如：

- 标量：
    - 关系 `data`：*(run_id, tag_id, step) → (value: f64)*
    - 查询 `data` (*run_id* in `RUNS`, *tag_id* in `TAGS`, *step* downsample `1000`) → *value*
- 图像：
    - 关系 `data`：*(run_id, tag_id, step: image_index: i64) → (blob: large_file)*
    - 查询 `data` (*run_id* in `RUNS`, *tag_id* in `TAGS`, *step* downsample `100`, *image_index* ANY) → *blob*

如上所述，我们选择不采用此选项，因为它公开了我们势必要支持的诸多灵活性，并且具有很高的概念表面积（例如，自定义域特定的查询语言）。

### 对于本地后端，将 blob 存储在文件系统而非 SQLite 中

Blob 实际上没有大小限制。如果我们选择保留对原始 logdir 的引用（例如，“事件文件路径加上文件中的字节偏移量”），而非将数据复制到我们自己的存储空间中，那么将减少数据重复情况。如果瓶颈源于 SQLite 本身而非底层 I/O，那么这也可以缓解并发请求的压力。

文件将存储在什么位置？目前，TensorBoard 的数据库模式会将所有数据存储在单个数据库中，以单个文件形式存储在磁盘上。这是一种简单方便的模型。如果存储其他文件，意味着我们必须要求将这些文件与其余数据共同存放在同一位置中（这会造成系统难以复制、共享、备份等），或者要求提供一些全局 TensorBoard 数据目录，例如 `${XDG_DATA_HOME}/tensorboard/blobs/`（这种重要承诺将要求后续工作更加慎重）。

请注意，某些 TensorBoard 插件（例如 projector）已经从其数据模型中的绝对文件路径执行读取。这会导致 logdir 不可移植，实属一个痛点。

如何存储文件？无论何种方式，我们最终都将建立某种微型数据库，并承担所有相关的存储任务。例如，如果我们使用像 `${run_id}/${tag_id}/${step}/${index}` 一样的文件路径层次结构，那么读取运行中每个标记的最近 blob 将是标记数和步骤数的二次方程 (!)，因为 `open`(2) 需要线性扫描所含目录中所有文件名的列表。也许我们可以像 Git 的对象存储一样通过散列法和分片法来解决此问题，这会进一步提高复杂性。性能不佳的文件系统；网络文件系统；多用户系统和权限。需要声明的是，实现数据库超出了此项目的范围。

另外，假定的性能提升并不明确：[SQLite 可比文件系统更快]！

### 包括“非时序 blob”存储类

这里提出的三种存储类均为时间序列：标量、张量或 blob 序列。本文档的先前版本曾提出过针对*非时序 blob* 的第四种存储类，在实验中仅由运行和标记键控。其预期目的是用于“运行级元数据”。针对该用例的现有解决方案确实会使用摘要：例如，带有特殊标记名 `_hparams_/session_start_info` 的摘要指定超参数配置。但这一直不太正统。理想情况下，运行级元数据将成为更完整的系统的一部分。例如，它可以用于驱动运行选择（“在所有插件中仅向我展示使用 Adam 优化器的运行”）。首先，摘要就不适用于该数据。

这种存储类可以根据现有的存储类来实现，方法是将非时序 blob 表示为仅在步骤 0 处采样的单例 blob 序列。在 TensorFlow 2.x 中，即使计算图在运行过程中也可能不是静态的（[`trace_export`] 接受 `step` 参数），从某种意义上说，超参数也不一定为静态。鉴于鲜有实用的用例，而且后备效果甚佳，我们放弃了这种存储类。

## 更新日志

- **2020-01-22**：将 blob 序列 `corresponding_max_index`（最新基准的长度）更改为 `max_length`（任何基准的最大长度）以满足实际需求且更为自然。
- **2019-08-07**：基于设计评审反馈进行了修订：移除了非时序 blob（请参阅“考虑的替代方案”部分）。
- **2019-07-25**：初始版本。


[TensorBoard.dev]: https://tensorboard.dev
[组织切片图]: https://iciar2018-challenge.grand-challenge.org/Dataset/
[CMEK]: https://cloud.google.com/storage/docs/encryption/customer-managed-keys
[PR #1022]: https://github.com/tensorflow/tensorboard/pull/1022
[SQL:2011 的一部分]: https://jakewheat.github.io/sql-overview/sql-2003-foundation-grammar.html#sample-clause
[Cloud Spanner 的 `TABLESAMPLE` 算子]: https://cloud.google.com/spanner/docs/query-syntax#tablesample-operator
[打包的张量内容]: https://github.com/tensorflow/tensorflow/blob/48b2094fd691bf2db96096d739afb23ff6807e33/tensorflow/core/framework/tensor.proto#L31-L36
[`--db_import` 时]: https://github.com/tensorflow/tensorboard/blob/0e9a000a1ef484762e743335a6b5754154bd9cdd/tensorboard/plugins/core/core_plugin.py#L334-L341
[SQLite 可比文件系统更快]: https://www.sqlite.org/fasterthanfs.html
[`trace_export`]: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/summary/trace_export
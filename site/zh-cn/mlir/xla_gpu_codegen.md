# XLA 的 MLIR 代码生成

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'timshen' reviewed: '2020-06-16' }
*-->

XLA 在 `HloInstruction` 上进行运算并对此表示执行许多优化，同时在目标设备之间共享其中的许多优化。在某些时候，会计算线性调度，并将内存缓冲区静态地分配给每个值。特定于设备的代码生成在运算时会遍历此序列，并调用“发射器”以生成适合该设备的表示（例如，CPU 上每个 XLA 计算对应的单个 LLVM 函数，或封装 GPU 运算的“Thunk”序列，也可能是针对 GPU 时生成的 PTX）。

作为暂存步骤，我们目前会在 XLA 完成缓冲区分配阶段之后立即拦截该过程，转而在 `lhlo` 方言中发出 MLIR 模块。之后，我们会根据设备使用 MLIR 组件（主要是 Linalg、仿射和 GPU 方言）执行代码生成。

下面是通过使用 `lhlo` 作为代码生成输入来增量迁移 XLA/GPU 的记录计划。

## 任务

 | 主机 | 设备
--- | --- | ---
| 主机 | 设备 --- | --- | --- 输入格式 | HloInstruction*（任务 1） | HloInstruction*（任务 1） 输出格式 | xla::Thunk（任务 2） | LLVM IR（任务 3） | HloInstruction*（任务 1） | HloInstruction*（任务 1）
输出格式 | xla::Thunk（任务 2） | LLVM IR（任务 3）

- **任务 1** 将主机和设备的输入格式从 HloInstruction* 更改为 LHLO。
- **任务 2** 将主机的输出格式从 Thunk 更改为“主机的某种停机坪”（请参见下文）
- **任务 3** 将设备输出从 LLVM IR 迁移到某种形式的 MLIR。它在此项目中为可选项，有关详细信息，请参阅“迁移设备 LLVM IR”部分。

此项目尽可能优先考虑使用启用了 LHLO 发射器的端到端可运行模型。这意味着目标列表将按照优先级进行如下排序：

- 使 XLA/GPU 可与 LHLO 发射器一起运行，而不修改现有的 Thunk 和发射器。
- 在 LHLO 中根据具体情况消除对 HloInstruction* 的引用：
    - 将旧发射器切换为基于 MLIR 的发射器（例如 Linalg），或者
    - 以机械方式转换现有发射器以接受 MLIR 表示（使用 GPU 方言迁移到 Standard）。

## 迁移 Thunk（任务 2）

xla::gpu::Thunk 是一种具有如下特点的数据结构：

- 可从主机调用 (xla::gpu::Thunk::ExecuteOnStream())
- 在其子类中携带各种数据
- 可与 BufferAllocation::Slice 和 StreamExecutor 进行交互
- 可启动内核
- 可调入所有运行时库

成本包括：

- 表示特定于算子的配置数据（例如卷积配置）
- 迁移算子形状和运算对象形状
- 表示 Thunk 树（while、条件等）

迁移作业与 LHLO/发射器迁移无关。在资源有限的情况下，它的优先级位于 LHLO/发射器迁移之后。

关于如何从 LHLO 降级主机端部分，我们有几种选择：

- TFRT
    - （利）可以使用出色的 CUDA 和 HIP 封装容器
    - （利）易于实现库调用（cuDNN、cuBLAS、cuFFT 等），因为 TFRT 算子由 C++ 代码解释
    - （弊）主机端正在开发且未经测试
- 即时编译的 CPU 代码
    - （利）出色的降级能力。创建一些循环和条件即可完成
    - （弊）GPU 方言尚未对 chains/streams/asynchronicity/device 分配进行建模
    - （弊）仅对 CUDA/HIP 运行时提供最小支持（工具包路径、版本、动态加载等）
- 现有的（解释）XLA 运行时

决策：采用 TFRT，但也支持在 TFRT 中即时编译 CPU 代码。

## 迁移设备 LLVM IR（任务 3）

元素发射器通过逐个填充元素来生成目标算子。每个输出元素都取决于来自运算对象的一组元素。所有元素都可以通过组合缓冲区和动态索引进行描述。这样足以描述几乎所有的“数学算子，但出于性能原因，(Cpu|Gpu)ElementalIrEmitter 中仅直接实现了“数学”算子的一个大子集。

ElementalIrEmitter 的独特之处在于：

- XLA/GPU 和 CPU 之间会共享大部分代码
- 它代表了在模型中看到的大部分算子，包括所有元素算子
- 大多数融合完全取决于 ElementalIrEmitter
- 结构简单，因为它描述的是算子元素和运算对象元素之间的数据依赖 DAG
- 它通常可移植且高级（例如，不同于 GPU kReduce 和 GPU kCopy）
- 对动态形状的支持至少对于元素级算子来说很容易

现在，对于所有算子（无论是否以元素为单位进行发射），每个 XLA 算子的结束状态都有几种形式：

1. 设备代码保持为 LLVM IR
2. 将旧发射器重构为类似 LHLO -&gt; MLIR LLVM 方言的形式：
    - （成本）如果我们最终要迁移到 Standard，那将是一劳永逸的工作
    - （收益）简单而机械，可在短期内完成
    - （收益）与 (1) 相比并没有更多收益
3. 将旧发射器重构为类似 LHLO -&gt; MLIR GPU + Standard + 循环的形式：
    - （成本）将现有发射器提升为 Standard 会带来一些挑战。指针和 GEP 需要转换为 MemRef 和 SubView。确保 AMDGPU 的完整性也是一个问题。
    - （成本）XLA/GPU 严重依赖 LLVM 元数据：
        - 用于块/线程索引的 `range`
        - 用于加载/存储的 `align`、`dereferenceable`、`invariant.load`、`alias.scope` 和 `noalias`
        - 用于顺序循环的 `llvm.loop.unroll.disable`、`llvm.loop.unroll.full` 和 `llvm.loop.vectorize.enable`
    - （收益）可为长期。更好的可移植性。
4. 将旧发射器重构为 LHLO -&gt; Linalg，并编写新的 Linalg 发射器
    - （成本）这视情况而定。与之前的选项相比，匹配 XLA 性能的新实现需要通过基准测试 &lt;-&gt; 优化工作流，这对于某些算子而言是一项巨大的开销。
    - （收益）统一堆栈；社区支持；可移植性；更大的优化潜力。

结论：

- 不要选择 (2)。(1) 或 (3) 都比 (2) 好。(2) 的成本比 (1) 高，因为它需要大量的机械重构。如果选择 (1)，我们仍然可以实现使 XLA 拾取 MLIR 发射器这一目标。具体方法为：LHLO -&gt; LLVM IR -&gt; 运行旧的设备发射器。
- ElementalIrEmitter 算子适合选择 (4)，但无法递增。因为以元素发出的所有算子都连接到同一个计算图中，所以无法逐算子执行。此项工作还可以作为几个正在发展的方案（内核生成器、Linalg）的统一点。
- 所有其他算子应选择 (1)。作为延伸目标，它们可能会迁移到 (3) 或 (4)。

## 优先级

虽然上述所有三个任务都可并行化，但在有限的资源下，它们必须被序列化。优先级重点是完成每个任务的可见结果。

优先级为：任务 1（旧发射器的 LHLO）&gt; 任务 2（Thunk）&gt; 任务 3（MLIR 发射器）。

在任务 1 结束时，XLA 的用户可以生成 LHLO（例如内核生成器）并执行它们。编译格式不是可序列化的 MLIR。

在任务 2 结束时，LHLO 降级为合适的可序列化的 MLIR。这样可以进行离线编译。

在任务 3 结束时，所有 XLA 发射器的实现均基于 MLIR。

## 详细设计

### 第 1 步：（任务 1）完成 LHLO 并让旧发射器接受 LHLO

此步骤使所有现有的 XLA/GPU 发射器与 MLIR 算子 进行交互。此步骤是纯重构和 NFC。

此步骤主要是机械步骤，但值得注意的是，未嵌套的 HloComputation 和 LHLO 之间存在以下差异：

- 每个 HloInstruction 都可以直接访问它的运算对象（数据流 DAG）。相反，每个 LHLO 算子只能访问它的运算对象缓冲区（算子和缓冲区之间的二分体）。LHLO 算子必须通过使用-定义链访问它们的运算对象算子。
- 从经验上看，未嵌套的旧发射器几乎从不访问其运算对象。唯一的例外是 kReduce。
- 未嵌套的旧发射器访问 BufferAssignment 只是为了获取切片，而非访问辅助数据结构（如 dataflow_analysis() 或 alias_analysis()）。llvm_ir 会基于切片信息构建自己的 alias_analysis()。

结论是 LHLO 应该能够轻而易举地满足要求。

### 第 2 步：（可选）性能分析支持

**仅当我们开始舍弃某些 XLA Thunk 逻辑（请参见下一个步骤）时才需要执行此步骤。**

在实际打开任何基于 MLIR 的发射器之前，我们需要对基于 MLIR 的发射器进行性能分析。

目前，XLA 会通过调入 StreamExecutor 的计时器来执行自己的性能分析。后台计时器会在内核启动前后插入两个事件，并测量这两个事件之间的同步时间。

要在 MLIR 中支持性能分析，大致有三种方式：

- 端到端地运行性能分析器
- 使用注入的性能分析器为 LHLO 中的每个算子添加一个性能分析算子

“端到端”方式对 MLIR 透明，但存在一个问题，XLA 一开始不使用该方式也是因为这个问题：由性能分析器 (nvprof/...) 收集的库调用无法轻松关联到 HLO 算子。例如，cuDNN 会为每个 HLO 启动多个内核，很难区分哪个内核对应于哪个 HLO。

“注入的性能分析器”方式要求：

- LHLO 将性能分析器作为参数
- 在每个算子之前和之后插入 profile.start/profile.end
- 将 profile.{start,end} 降级到 C++ 实现的传递

无法对 MLIR 生成的算子轻松进行精确的性能分析，因为：

- MLIR 没有计时器，也不依赖 TFRT/StreamExecutor
- MLIR 无法轻松调入具有复杂参数的 C 函数

### 第 3 步：（任务 2）迁移 Thunk

请注意，Thunk 大致分为三种类型：

- KernelThunk，启动内核
- 控制流 Thunk，具有主机控制流逻辑（conditional、while、for 和 sequence）并启动主体内核
- 库 Thunk：cuDNN、cuBLAS、cuFFT、NCCL 等

计划是：

- 使 Thunk 可（反）序列化
- 帮助将 TFRT 改进到能够支持这些语义的状态
- 随着状态的改进，以增量方式迁移各个 Thunk

这些操作项仅进行了部分排序。实际执行顺序/工程并行性将即时进行评估。

### 第 4 步：（任务 3）迁移后的 ElementalIrEmitter

性能分析完成后，我们就可以完成并调整 MLIR 中所有基于 ElementalIrEmitter 的发射器。然后，我们默认将它们打开，假设所有这些基于 MLIR 的发射器都使用一个流。

请注意，迁移 XLA/CPU 的 ElementalIrEmitter 也是有益的，因为它们会共享大部分代码。

完成所有基准测试和性能搜寻（TODO：定义性能奇偶校验）后，我们打开新的基于 MLIR 的元素发射器，然后删除旧的 ElementalIrEmitter。

此步骤还为以后的迁移提供了简单的融合过渡（嵌套算子）。

### 第 5 步：多流支持或删除

我们不能删除[某些发射器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/stream_assignment.cc#L140)，除非我们在 MLIR 中对其提供支持，或者删除该功能。在这种情况下，MLIR 的工作量相对较大，而 XLA 的收益却相对较小。我们应该调查多流 XLA/GPU 的当前用户，并在合理的情况下尝试删除此功能。

### 第 6 步：（任务 3）迁移后的设备算子

此步骤会迁移所有未嵌套的算子，之后我们可以删除所有未嵌套的发射器。

这要求重写/重构 kCopy 和 kReduce。kReduce 已经进行了大量工作，因此需要完成的实际工作量还有待观察。

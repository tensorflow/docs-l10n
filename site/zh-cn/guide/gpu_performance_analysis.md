# 使用 TensorFlow Profiler 优化 TensorFlow GPU 性能

## 概述

本指南将向您展示如何将 TensorFlow Profiler 与 TensorBoard 结合使用，以深入了解您的 GPU 并获得最佳性能，以及在您的一个或多个 GPU 未得到充分利用时进行调试。

如果您是 Profiler 的新用户：

- 请首先参阅包含 Keras 示例的 [TensorFlow Profiler：剖析模型性能](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)笔记本以及 [TensorBoard](https://www.tensorflow.org/tensorboard)。
- 参阅[使用 Profiler 优化 TensorFlow 性能](https://www.tensorflow.org/guide/profiler#profiler_tools)指南，了解可用于在主机 (CPU) 上优化 TensorFlow 性能的各种性能剖析工具和方式。

请谨记，将计算卸载到 GPU 可能并非总是有益，对于小型模型而言格外如此。以下原因可能会导致产生一定开销：

- 主机 (CPU) 与设备 (GPU) 之间的数据传输；以及
- 主机启动 GPU 内核时存在延迟。

### 性能优化工作流

本指南概述了调试性能问题的方式，从单个 GPU 开始讲起，然后逐步延伸到具有多个 GPU 的单个主机。

建议按以下顺序调试性能问题：

1. 优化和调试一个 GPU 上的性能：
    1. 检查输入流水线是否是瓶颈。
    2. 调试一个 GPU 的性能。
    3. 启用混合精度（使用 `fp16` (float16)），可选择启用 [XLA](https://www.tensorflow.org/xla)。
2. 优化和调试多 GPU 单主机上的性能。

例如，如果您使用 TensorFlow [分布策略](https://www.tensorflow.org/guide/distributed_training)在具有多个 GPU 的单个主机上训练模型并注意到 GPU 利用率不够理想，那么您应首先优化和调试一个 GPU 的性能，然后再调试多 GPU 系统。

作为在 GPU 上运行高性能代码的基线，本指南假定您已在使用 `tf.function`。Keras `Model.compile` 和 `Model.fit` API 将在后台自动使用 `tf.function`。使用 `tf.GradientTape` 编写自定义训练循环时，请参阅[使用 tf.function 提升性能](https://www.tensorflow.org/guide/function)以了解如何启用 `tf.function`。

接下来的部分将讨论针对上述每个场景的建议方式，以帮助识别和修正性能瓶颈。

## 1. 优化一个 GPU 上的性能

在理想情况下，您的程序应满足 GPU 利用率高、CPU（主机）与 GPU（设备）之间通信量最低，并且输入流水线无开销的特点。

分析性能的第一步是获取使用一个 GPU 运行的模型的性能剖析文件。

TensorBoard 的 Profiler [概览页面](https://www.tensorflow.org/guide/profiler#overview_page)显示了您的模型在运行性能剖析期间的性能的顶级视图，可以让您了解您的程序与理想场景之间的差距。

![TensorFlow Profiler Overview Page](images/gpu_perf_analysis/overview_page.png "The overview page of the TensorFlow Profiler")

概览页面中需要注意的关键数字包括：

1. 源自实际设备执行的单步用时
2. 设备与主机上执行的运算的百分比
3. 使用 `fp16` 的内核的数量

要实现最佳性能，就意味着要在所有三种情况下最大化这些数字。要深入了解您的程序，您需要熟悉 TensorBoard 的 Profiler [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer)。以下部分介绍了在诊断性能瓶颈时应查找的一些常见 Trace Viewer 模式。

下面是在一个 GPU 上运行的模型跟踪记录视图的图像。在 *TensorFlow Name Scope* 和 *TensorFlow Ops* 部分中，您可以识别模型的不同部分，例如前向传递、损失函数、后向传递/梯度计算和优化器权重更新。您还可以在每个*流*（即 CUDA 流）旁的 GPU 上运行运算。每个流均用于特定任务。在此跟踪记录中，*Stream#118* 用于启动计算内核以及设备到设备的复制。*Stream#119* 用于主机到设备的复制，*Stream#120* 用于设备到主机的复制。

下面的跟踪记录展示了高性能模型的共同特征。

![image](images/gpu_perf_analysis/traceview_ideal.png "An example TensorFlow Profiler trace view")

例如，GPU 计算时间线 (*Stream#118*) 看起来非常“忙碌”，几乎没有间隙。从主机到设备的复制 (*Stream #119*) 和从设备到主机的复制 (*Stream #120*) 次数最少，步骤间的间隙也最小。当您为程序运行 Profiler 时，您可能无法在跟踪记录视图中识别这些理想特征。本指南的其余部分涵盖了常见场景及其修正方式。

### 1. 调试输入流水线

GPU 性能调试的第一步是确定您的程序是否受输入约束。最简单的判断方式是使用 TensorBoard 上的 Profiler 的[输入流水线分析器](https://www.tensorflow.org/guide/profiler#input_pipeline_analyzer)，它提供了输入流水线的用时概览。

![image](images/gpu_perf_analysis/input_pipeline_analyzer.png "TensorFlow Profiler Input-Analyzer")

如果您的输入流水线对单步用时影响显著，您可以采取以下潜在可行操作：

- 您可以使用 `tf.data` 特定[指南](https://www.tensorflow.org/guide/data_performance_analysis)来了解如何调试您的输入流水线。
- 检查输入流水线是否为瓶颈的另一种快速方式是使用不需要任何预处理的随机生成输入数据。[此处](https://github.com/tensorflow/models/blob/4a5770827edf1c3974274ba3e4169d0e5ba7478a/official/vision/image_classification/resnet/resnet_runnable.py#L50-L57)提供了对 ResNet 模型使用此技术的示例。如果输入流水线处于最佳水平，那么您在使用真实数据和生成的随机/人工数据时应体验到相近的性能。在使用合成数据的情况下，唯一的开销将源于输入数据复制，可以对其进行预提取和优化。

此外，请参阅[优化输入数据流水线的最佳做法](https://www.tensorflow.org/guide/profiler#optimize_the_input_data_pipeline)。

### 调试一个 GPU 的性能

导致 GPU 利用率较低的因素有多种。以下是查看 [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 时的一些常见场景以及潜在可行的解决方案。

#### 1. 分析步骤之间的间隙

当您的程序运行性能不佳时，通常可以观察到训练步骤之间的间隙较大。在下面的跟踪记录视图图像中，第 8 步和第 9 步之间存在较大的间隙，这意味着 GPU 在此期间处于空闲状态。

![image](images/gpu_perf_analysis/traceview_step_gaps.png "TensorFlow Profile trace view showing gaps between steps")

如果您的 Trace Viewer 显示步骤之间存在较大间隙，那么可能表明您的程序受输入约束。在这种情况下，您应参阅上一部分中介绍的有关调试输入流水线的内容。

但是，即使输入流水线经过优化，由于 CPU 线程争用，在一个步骤的结尾和另一个步骤的开头处之间仍然可能存在间隙。`tf.data` 会利用后台线程来并行处理流水线。这些线程可能会干扰在每个步骤开始时发生的 GPU 主机端活动，例如复制数据或调度 GPU 运算。

如果您注意到在 GPU 上调度这些运算的主机端存在较大的间隙，您可以设置环境变量 `TF_GPU_THREAD_MODE=gpu_private`。这可以确保 GPU 内核能够从自己的专用线程启动，并且不会排在 `tf.data` 工作之后。

步骤之间的间隙也可能由指标计算、Keras 回调或在主机上运行的 `tf.function` 之外的运算引起。这些运算的性能不及 TensorFlow 计算图内部的运算。此外，其中一些运算还会在 CPU 上运行并从 GPU 来回复制张量。

如果在优化输入流水线后，您仍能在 Trace Viewer 中注意到步骤之间存在间隙，则您应查看步骤之间的模型代码并检查停用回调/指标能否提升性能。这些运算的一些详细信息也会在 Trace Viewer（设备端和主机端）上提供。在这种情况下，建议通过在固定数量的步骤而非每一步之后执行运算来分摊这些运算的开销。在 `tf.keras` API 中使用 `compile` 方法时，设置 `experimental_steps_per_execution` 标志会自动执行此操作。对于自定义训练循环，请使用 `tf.while_loop`。

#### 2. 实现更高的设备利用率

##### 1. 小型 GPU 内核和主机内核启动延迟

主机会将内核加入队列以在 GPU 上运行，但内核实际在 GPU 上执行之前会存在延迟（约 20-40 微秒）。在理想情况下，主机会在 GPU 上将足够多的内核加入队列以便 GPU 能够将大部分时间用于执行，而非等待主机将更多内核加入队列。

TensorBoard 上的 Profiler [概览页面](https://www.tensorflow.org/guide/profiler#overview_page)显示了 GPU 因等待主机启动内核而空闲的时间。在下图中，GPU 因等待内核启动而空闲约 10% 的单步用时。

![image](images/gpu_perf_analysis/performance_summary.png "Summary of performance from TensorFlow Profile")

针对同一程序的 [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 显示了由主机忙于在 GPU 上启动内核而产生了较小的内核间隙。

![image](images/gpu_perf_analysis/traceview_kernel_gaps.png "TensorFlow Profile trace view demonstrating gaps between kernels")

通过在 GPU 上启动大量小型运算（例如标量相加），主机可能无法跟上 GPU 的速度。TensorBoard 中针对同一项性能剖析的 [TensorFlow Stats](https://www.tensorflow.org/guide/profiler#tensorflow_stats) 工具显示 126,224 次 Mul 运算耗时 2.77 秒。因此，每个内核大约为 21.9 微秒，该时间非常短（大约与启动延迟时间相同）并且可能会导致主机内核启动延迟。

![image](images/gpu_perf_analysis/tensorflow_stats_page.png "TensorFlow Profile stats page")

如果您的 [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 显示 GPU 上的运算之间存在许多小的间隙（如上图所示），您可以：

- 串联小型张量并使用矢量化运算，或者使用更大批次大小以使每个启动的内核执行更多工作，这将使 GPU 保持忙碌时间的变长。
- 确保您使用 `tf.function` 创建 TensorFlow 计算图，从而避免在纯 Eager 模式下运行运算。如果您使用的是 `Model.fit`（与使用 `tf.GradientTape` 的自定义训练循环相反），那么 `tf.keras.Model.compile` 将自动为您执行此操作。
- 通过 `tf.function(jit_compile=True)` 或自动聚簇来使用 XLA 融合内核。有关详情，请参阅下面的[启用混合精度和 XLA](#3._enable_mixed_precision_and_xla) 部分，以了解如何启用 XLA 来获得更高的性能。此功能可以提高设备利用率。

##### 2. TensorFlow 运算放置

Profiler [概览页面](https://www.tensorflow.org/guide/profiler#overview_page)显示了主机与设备上放置的运算的百分比（您还可以通过查看 [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 来验证特定运算的放置）。如下图所示，您的主机运算百分比值要比设备运算百分比值小得多。

![image](images/gpu_perf_analysis/opp_placement.png "TF Op Placement")

理想情况下，大多数计算密集型运算都应放置在 GPU 上。

要了解模型中的运算和张量分配给了哪些设备，请将 `tf.debugging.set_log_device_placement(True)` 设置为程序的第一条语句。

请注意，在某些情况下，即使您指定将某一运算放置在特定的设备上，其实现也可能会重写此条件（例如：`tf.unique`）。即使对于单 GPU 训练，指定分布策略（例如 `tf.distribute.OneDeviceStrategy`）也可以使您的设备上的运算放置更加确定。

将大部分运算放置在 GPU 上的原因之一是防止主机和设备之间的内存复制过多（主机和设备之间针对模型输入/输出数据的内存复制是在预期之中的）。下方基于 GPU 流 *#167*、*#168* 和 *#169* 的跟踪记录视图中演示了复制过多的示例。

![image](images/gpu_perf_analysis/traceview_excessive_copy.png "TensorFlow Profile trace view demonstrating excessive H2D/D2H copies")

如果这些复制会阻塞 GPU 内核执行，那么它们有时就会造成性能下降。[Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 中的内存复制运算提供了有关这些复制张量所源自的运算的更多信息，但将 memCopy 与运算相关联可能并不总是那么容易。在这些情况下，不妨查看邻近的运算以检查内存复制是否在每个步骤的同一位置发生。

#### 3. GPU 上更高效的内核

在程序的 GPU 利用率达到可接受水平后，下一步就要考虑通过利用 Tensor Core 或融合运算来提高 GPU 内核的效率了。

##### 1. 利用 Tensor Core

新款 NVIDIA® GPU 搭载了专门的 [Tensor Core](https://www.nvidia.com/en-gb/data-center/tensor-cores/)，可以显著提升符合条件的内核的性能。

您可以使用 TensorBoard 的 [GPU Kernel Stats](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) 来呈现哪些 GPU 内核符合 Tensor Core 条件，以及哪些内核正在使用 Tensor Core。启用 `fp16`（请参阅下面的“启用混合精度”部分）是使程序的通用矩阵乘法 (GEMM) 内核（matmul 运算）利用 Tensor Core 的一种方式。当精度为 fp16 且输入/输出张量维度可被 8 或 16 整除（对于 `int8`）时，GPU 内核便可以高效地使用 Tensor Core。

注：使用 cuDNN v7.6.3 及更高版本时，将在必要时自动填充卷积维度以利用 Tensor Core。

有关针对 GPU 提高内核效率的其他详细建议，请参阅 [NVIDIA® 深度学习性能](https://docs.nvidia.com/deeplearning/performance/index.html#perf-guidelines)指南。

##### 2. 融合运算

使用 `tf.function(jit_compile=True)` 可以融合较小的运算以构成较大的内核，从而显著提升性能。要了解详情，请参阅 [XLA](https://www.tensorflow.org/xla) 指南。

### 3. 启用混合精度和 XLA

完成上述步骤后，启用混合精度和 XLA 是能够进一步提升性能的两个可选步骤。建议的方式是逐一启用它们并验证性能优势是否符合预期。

#### 1. 启用混合精度

TensorFlow [混合精度](https://www.tensorflow.org/guide/keras/mixed_precision)指南介绍了如何在 GPU 上启用 `fp16` 精度。在 NVIDIA® GPU 上启用 [AMP](https://developer.nvidia.com/automatic-mixed-precision) 以使用 Tensor Core，并实现高达 3 倍的整体加速（与在 Volta 和较新的 GPU 架构上使用 `fp32` (float32) 精度相比）。

确保矩阵/张量维度满足调用使用 Tensor Core 的内核的要求。当精度为 fp16 且输入/输出维度可被 8 或 16 整除（对于 int8）时，GPU 内核将使用 Tensor Core。

请注意，使用 cuDNN v7.6.3 及更高版本时，将在必要时自动填充卷积维度以利用 Tensor Core。

请遵循以下最佳做法，以最大限度地提升 `fp16` 精度的性能优势。

##### 1. 使用最优的 fp16 内核

启用 `fp16` 后，您程序的矩阵乘法 (GEMM) 内核应使用利用 Tensor Core 的相应 fp16 版本。但在某些情况下可能不会如此，并且您的程序会回退到低效的实现，使您无法体验到启用 `fp16` 带来的预期加速。

![image](images/gpu_perf_analysis/gpu_kernels.png "TensorFlow Profile GPU Kernel Stats page")

[GPU Kernel Stats](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) 页面中显示了哪些运算符合 Tensor Core 条件，哪些内核实际使用了高效的 Tensor Core。[NVIDIA® 深度学习性能](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores)指南中提供了有关如何利用 Tensor Core 的其他建议。此外，使用 `fp16` 的优势也将体现在以前受制于内存限制的内核中，因为现在的运算时间将减半。

##### 2. 动态与静态损失扩缩

使用 `fp16` 时需要进行损失扩缩，以防止因精度较低而导致的下溢出。损失扩缩包含动态和静态两种类型，这两种类型都在[混合精度指南](https://www.tensorflow.org/guide/keras/mixed_precision)中进行了更详细的说明。您可以使用 `mixed_float16` 策略在 Keras 优化器中自动启用损失扩缩。

注：Keras 混合精度 API 会默认将独立的 Softmax 运算（不属于 Keras 损失函数的运算）评估为 `fp16`，这可能导致出现数值问题和收敛性不佳的情况。将此类运算转换为 `fp32` 可以获得最佳性能。

在尝试优化性能时，请注意，动态损失扩缩会引入在主机上运行的额外条件运算，并导致出现能够在 Trace Viewer 中看到的步骤之间的间隙。另一方面，静态损失扩缩则没有这样的开销，并且在性能方面可能是更好的选择，只是必须注意您需要指定正确的静态损失扩缩值。

#### 2. 使用 tf.function(jit_compile=True) 或自动聚簇启用 XLA

作为使用单个 GPU 获得最佳性能的最后一步，您可以尝试启用 XLA，这将融合运算、提高设备利用率并降低内存占用量。有关如何在程序中使用 `tf.function(jit_compile=True)` 或自动聚簇来启用 XLA 的详细信息，请参阅 [XLA](https://www.tensorflow.org/xla) 指南。

您可以将全局 JIT 级别设置为 `-1`（关闭）、`1` 或 `2`。级别越高就越激进，可能会降低并行度并使用更多内存。如果存在内存限制，请将该值设置为 `1`。请注意，XLA 对于具有可变输入张量形状的模型效果不佳，因为 XLA 编译器每当遇到新形状时都必须继续编译内核。

## 2. 优化多 GPU 单主机上的性能

`tf.distribute.MirroredStrategy` API 可用于将模型训练从一个 GPU 扩展到单个主机上的多个 GPU（要详细了解如何使用 TensorFlow 进行分布式训练，请参阅[使用 TensorFlow 进行分布式训练](https://www.tensorflow.org/guide/distributed_training)、[使用 GPU](https://www.tensorflow.org/guide/gpu) 和[使用 TPU](https://www.tensorflow.org/guide/tpu) 指南，以及 [使用 Keras 进行分布式训练](https://www.tensorflow.org/tutorials/distribute/keras)教程）。

尽管在理想情况下，从一个 GPU 转换至多个 GPU 的扩展性开箱即用，但您有时仍会遇到性能问题。

从使用单个 GPU 进行训练过渡到使用同一主机上的多个 GPU 时，理想情况下您应当能够体验到性能扩展，唯独会因梯度通信而产生额外开销且主机线程利用率会提高。由于这种开销，在从 1 个 GPU 转换为 2 个 GPU 的情况下，您将不会正好获得一倍的提速。

下面的跟踪记录视图显示了在多个 GPU 上训练时的额外通信开销示例。串联梯度、跨副本通信以及在进行权重更新之前予以拆分会产生一些开销。

![image](images/gpu_perf_analysis/traceview_multi_gpu.png "TensorFlow Profile trace view for single host multi GPU scenario")

以下核对清单将帮助您在多 GPU 场景中优化性能时获得更好的性能：

1. 尝试最大化批次大小，这将提高设备利用率并分摊跨多个 GPU 的通信成本。使用[内存性能剖析器](https://www.tensorflow.org/guide/profiler#memory_profile_summary)有助于了解您的程序与达到内存利用率峰值还有多少差距。请注意，虽然较大的批次大小会影响收敛，但这通常会被性能优势所抵消。
2. 从单个 GPU 迁移到多个 GPU 时，同一主机现在必须处理更多输入数据。因此，在完成第 1 步之后，建议重新检查输入流水线的性能并确保它并未成为瓶颈。
3. 检查程序跟踪记录视图中的 GPU 时间线以确定是否存在任何不必要的 AllReduce 调用，因为这会导致所有设备间的同步。在上面显示的跟踪记录视图中，AllReduce 是通过 [NCCL](https://developer.nvidia.com/nccl) 内核完成的，并且针对每个步骤上的梯度，在每个 GPU 上只调用一次 NCCL。
4. 检查可以尽可能避免的非必要 D2H、H2D 和 D2D 复制运算。
5. 检查单步用时以确保每个副本都在执行相同的工作。例如，可能会发生由于主机错误地在某个 GPU（通常为 `GPU0`）上调度了过多的工作，而导致该 GPU 被过度使用的情况。
6. 最后，在跟踪记录视图中检查所有 GPU 的训练步骤，查看是否存在任何按顺序执行的运算。当您的程序包含从一个 GPU 到另一个 GPU 的控制依赖项时，通常会出现这种情况。过去，调试这种情况下的性能都需要根据具体情况加以解决。如果您在您的程序中观察到这种行为，请[提交 GitHub 议题](https://github.com/tensorflow/tensorflow/issues/new/choose)并附上跟踪记录视图的图像。

### 1. 优化梯度 AllReduce

使用同步策略进行训练时，每个设备都会接收一部分输入数据。

计算在模型中的前向和后向传递后，需要对每个设备上计算的梯度进行聚合和归约。此*梯度 AllReduce* 发生在每个设备上的梯度计算之后、优化器更新模型权重之前。

每个 GPU 首先会串联模型层之间的梯度，使用 `tf.distribute.CrossDeviceOps`（`tf.distribute.NcclAllReduce` 为默认值）在 GPU 之间进行通信，然后在逐层归约后返回梯度。

优化器将使用这些归约后的梯度来更新模型的权重。理想情况下，此过程应在所有 GPU 上同时发生，以防止出现任何开销。

AllReduce 的时间应大致等同于以下值：

```
(number of parameters * 4bytes)/ (communication bandwidth)
```

此计算可用于快速检查您在运行分布式训练作业时的性能是否符合预期，或者您是否需要进行进一步的性能调试。您可以从 `Model.summary` 中获知模型中的参数数量。

请注意，每个模型参数的大小均为 4 个字节，因为 TensorFlow 使用 `fp32` (float32) 来进行梯度通信。即使您启用了 `fp16`，NCCL AllReduce 也会使用 `fp32` 参数。

为了充分利用扩缩的优势，单步用时需要远远大于这些开销。实现此目标的一种方式是使用更大的批次大小，因为批次大小会影响单步用时，但不会影响通信开销。

### 2. GPU 主机线程争用

运行多个 GPU 时，CPU 的工作是通过在设备间有效地启动 GPU 内核来保持所有设备均忙碌地运行。

但是，当 CPU 可以在一个 GPU 上调度大量独立运算时，CPU 可以决定使用其大量主机线程来使一个 GPU 保持忙碌，然后以非确定性顺序启动另一个 GPU 的内核。这可能会导致偏差或负面扩缩，从而对性能产生负面影响。

下面的 [Trace Viewer](https://www.tensorflow.org/guide/profiler#trace_viewer) 显示了 CPU 以效率低下的方式交错启动 GPU 内核时的开销。效率低下的原因是 `GPU1` 先是处于空闲状态，然后在 `GPU2` 启动后才开始运行运算。

![image](images/gpu_perf_analysis/traceview_gpu_idle.png "TensorFlow Profile device trace view demonstrating inefficient kernel launch")

主机的跟踪记录视图显示主机首先启动了 `GPU2` 上的内核，然后才启动了 `GPU1` 上的内核（请注意，下面的 `tf_Compute*` 运算并不表示 CPU 线程）。

![image](images/gpu_perf_analysis/traceview_host_contention.png "TensorFlow Profile host trace view demonstrating inefficient kernel launch")

如果您在程序的跟踪记录视图中遇到这种 GPU 内核交错的情况，建议采取以下措施：

- 将 TensorFlow 环境变量 `TF_GPU_THREAD_MODE` 设置为 `gpu_private`。此环境变量会告知主机将 GPU 的线程保持为不公开。
- 默认情况下，`TF_GPU_THREAD_MODE=gpu_private` 会将线程数设置为 2，这在大多数情况下已经足够。但您也可以通过将 TensorFlow 环境变量 `TF_GPU_THREAD_COUNT` 设置为所需的线程数来更改该线程数。

# 使用 TensorBoard Debugger V2 调试 TensorFlow 程序中的数值问题

> *注*：tf.debugging.experimental.enable_dump_debug_info() 是实验性 API，将来可能发生重大变更。

在 TensorFlow 程序期间，有时可能会发生涉及 [NaN](https://en.wikipedia.org/wiki/NaN) 的灾难性事件，从而破坏模型训练过程。此类事件的根本原因通常难以查找，尤其是对于较大和复杂程度较高的模型。为了更轻松地调试此类模型错误，TensorBoard 2.3+（与 TensorFlow 2.3+ 一起）提供了一个名为 Debugger V2 的专用信息中心。在本文中，我们将在使用 TensorFlow 编写的神经网络中，通过解决涉及 NaN 的真实错误来演示如何使用此工具。

本教程中演示的技术适用于其他类型的调试活动（例如在复杂程序中检查运行时张量形状）。本教程重点介绍 NaN，因为它们的发生频率相对较高。

## 观察错误

我们将调试的 TF2 程序的源代码[可在 GitHub 上找到](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/v2/debug_mnist_v2.py)。该示例程序还打包在 TensorFlow pip 软件包（版本 2.3+）中，并且可以通过以下方式调用：

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2
```

此 TF2 程序可创建一个多层感知 (MLP) 并对其进行训练以识别 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 图像。本示例特意使用 TF2 的低级 API 来定义自定义层构造、损失函数和训练循环，因为与使用较为易用但不太灵活的高级 API（如 [tf.keras](https://www.tensorflow.org/guide/keras)）相比，使用此更灵活但更易出错的 API 时，出现 NaN 错误的可能性更高。

程序会在每个训练步骤之后打印测试准确率。我们可以在控制台中看到，在第一个步骤之后，测试准确率卡在了接近随机水平的地方 (~0.1)。这显然不是模型训练所预期的效果：我们希望准确率随着步骤的增加逐渐接近 1.0 (100%)。

```
Accuracy at step 0: 0.216
Accuracy at step 1: 0.098
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
...
```

根据经验作出的猜测是，此问题是由数值不稳定（如 NaN 或无穷大）引起的。但是，我们该如何确定这就是原因，以及如何找到产生数值不稳定的 TensorFlow 运算呢？为了回答这些问题，我们使用 Debugger V2 检测一下这个包含错误的程序。

## 使用 Debugger V2 检测 TensorFlow 代码

[`tf.debugging.experimental.enable_dump_debug_info()`](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info) 是 Debugger V2 的 API 入口点。它使用单行代码来检测 TF2 程序。例如，在程序的开头附近添加以下行会将调试信息写入位于 /tmp/tfdbg2_logdir 的日志目录 (logdir)。调试信息涵盖 TensorFlow 运行时的各个方面。在 TF2 中，它包括 Eager Execution 的完整历史记录、通过 [@tf.function](https://www.tensorflow.org/api_docs/python/tf/function) 执行的计算图构建、计算图的执行、由执行事件生成的张量值，以及这些事件的代码位置（Python 堆栈跟踪）。丰富的调试信息能够帮助用户缩小难以查找的错误的范围。

```py
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
```

`tensor_debug_mode` 参数控制 Debugger V2 从每个 Eager 或计算图内张量中提取哪些信息。“FULL_HEALTH”是一种模式，它会捕获有关每个浮点型张量（例如，常见的 float32 和不太常见的 [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) 数据类型）的以下信息：

- 数据类型
- 秩
- 元素总数
- 浮点型元素可以细分为以下类别：负有限 (`-`)、零 (`0`)、正有限 (`+`)、负无穷 (`-∞`)、正无穷 (`+∞`) 和 `NaN`。

“FULL_HEALTH”模式适用于调试涉及 NaN 和无穷的错误。请参阅下文了解其他受支持的 `tensor_debug_mode`。

`circular_buffer_size` 参数控制保存到 logdir 中的张量事件的数量。默认值为 1000，这样仅会将所检测的 TF2 程序结束前的最后 1000 个张量保存到磁盘。此默认行为会以牺牲调试数据的完整性来减少调试器的开销。如果首选完整性（比如在本文所述的情况下），我们可以通过将参数设置为负值（例如，本文为 -1）来停用循环缓冲区。

debug_mnist_v2 示例通过向 `enable_dump_debug_info()` 传递命令行标记来对它进行调用。要在启用调试检测的情况下再次运行有问题的 TF2 程序，请执行以下代码：

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2 \
    --dump_dir /tmp/tfdbg2_logdir --dump_tensor_debug_mode FULL_HEALTH
```

## 在 TensorBoard 中启动 Debugger V2 GUI

使用调试程序检测运行该程序会在 /tmp/tfdbg2_logdir 下创建一个 logdir。我们可以启动 TensorBoard 并利用以下代码将其指向该 logdir：

```sh
tensorboard --logdir /tmp/tfdbg2_logdir
```

在网络浏览器中，前往 TensorBoard 页面（网址为：http://localhost:6006）。默认情况下，Debugger V2 插件将处于停用状态，因此请从右上角的 Inactive plugins 菜单中选择它。选择后，它应显示如下页面：

![Debugger V2 full view screenshot](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tensorboard/images/debugger_v2_1_full_view.png?raw=true)

## 使用 Debugger V2 GUI 查找 NaN 的根本原因

TensorBoard 中的 Debugger V2 GUI 分为六个版块：

- **Alerts**：位于左上角，包含调试程序从所检测的 TensorFlow 程序的调试数据中检测到的“警报”事件的列表。每个警报都表示需要注意的某种异常。在我们的示例中，此版块用显眼的粉红色突出显示了 499 个 NaN/∞ 事件。这证实了我们的怀疑，即模型无法学习是因为其内部张量值中存在 NaN 和/或无穷。我们稍后将对这些警报进行深入研究。
- **Python Execution Timeline**：位于中上版块的上半部分。它表示运算和计算图的 Eager Execution 的完整历史记录。时间轴上的每个框都标有运算或计算图名称的首字母（例如，“T”代表“TensorSliceDataset”运算，“m”代表“模型”`tf.function`）。我们可以使用时间轴上的导航按钮和滚动条来浏览时间轴。
- **Graph Execution**：位于 GUI 的右上角，是我们调试任务的中心。它包含所有在计算图内进行计算的浮点张量的历史记录（即，由 `@tf-function` 编译）。
- **Graph Structure**（中上版块的下半部分）、**Source Code**（左下版块）和 **Stack Trace**（右下版块）最初为空。这些内容将在我们与 GUI 进行交互时填充。这三个版块也将在我们的调试任务中扮演重要角色。

在了解界面的组织结构之后，让我们采取以下步骤来深入了解 NaN 出现的原因。首先，在 Alerts 版块中点击 **NaN/∞** 警报。这将在 Graph Execution 版块滚动显示 600 个计算图张量，并将焦点放在 #88 上，这是一个由 `Log`（自然对数）运算生成的名为“Log:0”的张量。在二维 float32 张量的 1000 个元素中，以显眼的粉红色突出显示一个 -∞ 元素。这是 TF2 程序的运行时历史记录中的第一个张量，其中包含任何 NaN 或无穷：在它之前计算的张量不包含 NaN 或 ∞；此后计算的许多（实际上是大多数）张量都包含 NaN。我们可以上下滚动 Graph Execution 列表来进行确认。此观察结果强烈表明 `Log` 运算是导致 TF2 程序中数值不稳定的根源。

![Debugger V2: Nan / Infinity alerts and graph execution list](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tensorboard/images/debugger_v2_2_nan_inf_alerts.png?raw=true)

为什么此 `Log` 运算会产生 -∞？要回答此问题，需要检查运算的输入。点击张量名称 (`Log:0`)，在 Graph Structure 部分的 TensorFlow 计算图中，会显示 `Log` 运算附近区域的简单但信息丰富的呈现效果。请注意信息流的方向为从上到下。运算本身在中间以粗体显示。在紧挨着运算的上方，我们可以看到一个占位运算，它为 `Log` 运算提供唯一输入。此 `probs` 占位符生成的张量在 Graph Execution 列表中位于什么位置？使用黄色背景作为视觉辅助，我们可以看到 `probs:0` 张量在 `Log:0` 张量的上方并且隔了三行，即第 85 行。

![Debugger V2: Graph structure view and tracing to input tensor](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tensorboard/images/debugger_v2_3_graph_input.png?raw=true)

更仔细地查看一下第 85 行的 `probs:0` 张量的数值分解，我们就能发现使用者 `Log:0` 产生 -∞ 的原因：在 `probs:0` 的 1000 个元素中，有一个元素的值是 0。-∞ 是计算 0 的自然对数的结果！如果我们能以某种方式确保 `Log` 运算只获得正输入，就能够防止 NaN/∞ 的发生。为此，我们可以在 `probs` 占位张量上应用裁剪（例如，通过使用 [`tf.clip_by_value()`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)）。

我们离解决错误越来越近了，但还没有完成。要应用修复，我们需要知道 Log 运算及其占位输入在 Python 源代码中的位置。Debugger V2 提供了一流的支持，可跟踪计算图运算和执行事件到它们的源代码。当我们在 Graph Executions 中点击 `Log:0` 张量后，Stack Trace 版块会使用 Log 运算创建的原始堆栈跟踪进行填充。堆栈跟踪有点大，因为它包含来自 TensorFlow 内部代码（例如，gen_math_ops.py 和 dumping_callback.py）的许多帧，对于大多数调试任务，我们可以放心地忽略这些帧。我们需要关注的帧是 debug_mnist_v2.py（即，我们实际上正在尝试调试的 Python 文件）中的第 216 行。点击“Line 204”会在 Source Code 版块显示相应代码行的视图。

![Debugger V2: Source code and stack trace](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tensorboard/images/debugger_v2_4_source_code.png?raw=true)

我们终于找到了源代码，该代码从 `probs` 输入创建了有问题的 `Log` 运算。这是我们的自定义分类交叉熵的损失函数，该函数用 `@tf.function` 进行了装饰并随后转换为 TensorFlow 计算图。`probs` 占位运算对应于损失函数的第一个输入参数。`Log` 运算使用 tf.math.log() API 调用进行创建。

要对此错误进行值裁剪修正，可使用如下代码：

```py
  diff = -(labels *
           tf.math.log(tf.clip_by_value(probs), 1e-6, 1.))
```

它将解决此 TF2 程序中的数值不稳定问题，并成功训练 MLP。解决数值不稳定性的另一种可能方式是使用 [`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)。

至此，我们观察了 TF2 模型错误，并提出了修复该错误的代码更改建议。这是在 Debugger V2 工具的帮助下完成的，该工具提供了对所检测的 TF2 程序的 Eager 和计算图执行历史记录的完全可见性，包括张量值的数值摘要，以及运算、张量及其原始源代码之间的关联。

## Debugger V2 的硬件兼容性

Debugger V2 支持主流的训练硬件，包括 CPU 和 GPU。还支持使用 [tf.distributed.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) 的多 GPU 训练。对 [TPU](https://www.tensorflow.org/guide/tpu) 的支持仍处于早期阶段，需要先调用以下代码：

```py
tf.config.set_soft_device_placement(True)
```

然后再调用 `enable_dump_debug_info()`。它对 TPU 还可能有其他限制。如果您在使用 Debugger V2 时遇到问题，请在我们的 [GitHub 议题页面](https://github.com/tensorflow/tensorboard/issues)上报告错误。

## Debugger V2 的 API 兼容性

Debugger V2 在级别相对较低的 TensorFlow 软件堆栈上实现，因此兼容 [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)、[tf.data](https://www.tensorflow.org/guide/data) 以及在 TensorFlow 的较低级别上构建的其他 API。Debugger V2 还向后兼容 TF1，尽管对于 TF1 程序生成的调试 logdir，Eager Execution Timeline 将为空。

## API 使用提示

关于此调试 API 的一个常见问题是，应该在 TensorFlow 代码的哪个位置插入对 `enable_dump_debug_info()` 的调用。通常，应该在 TF2 程序中尽早调用该 API，最好在 Python 导入行之后以及构建计算图和执行开始之前进行调用。这样可以确保全面涵盖所有为模型及其训练提供支持的运算和计算图。

目前支持的 tensor_debug_modes 包括：`NO_TENSOR`、`CURT_HEALTH`、`CONCISE_HEALTH`、`FULL_HEALTH` 和 `SHAPE`。它们从每个张量提取的信息量以及所调试程序的性能开销各不相同。请参阅 <code>enable_dump_debug_info()</code> 文档的<a>“参数”部分</a>。

## 性能开销

调试 API 会增加所检测的 TensorFlow 程序的性能开销。开销因 `tensor_debug_mode`、硬件类型和所检测的 TensorFlow 程序的性质而异。作为参考，在 GPU 上，对于批次大小为 64 的 [Transformer 模型，](https://github.com/tensorflow/models/tree/master/official/nlp/transformer)`NO_TENSOR` 模式会在其训练期间增加 15% 的开销。其他 tensor_debug_modes 的开销百分比更高：对于 `CURT_HEALTH`、`CONCISE_HEALTH`、`FULL_HEALTH` 和 `SHAPE` 模式大约为 50%。在 CPU 上，开销略低。在 TPU 上，开销目前较高。

## 与其他 TensorFlow 调试 API 的关系

请注意，TensorFlow 提供了用于调试的其他工具和 API。您可以在 API 文档页面的 [`tf.debugging.*` 命名空间](https://www.tensorflow.org/api_docs/python/tf/debugging)下浏览此类 API。在这些 API 中，最常用的是 [`tf.print()`](https://www.tensorflow.org/api_docs/python/tf/print)。什么时候应该使用 Debugger V2，什么时候应该使用 `tf.print()` 呢？对于以下情况，使用 `tf.print()` 会很方便：

1. 我们确切知道要打印哪些张量，
2. 我们知道在源代码中插入这些 `tf.print()` 语句的确切位置，
3. 此类张量的数量不是太大。

对于其他情况（例如，检查许多张量值、检查由 TensorFlow 的内部代码生成的张量值，以及像我们在上文中展示的那样搜索数值不稳定的来源），使用 Debugger V2 进行调试速度更快。另外，Debugger V2 提供了一种检查 Eager 和计算图张量的统一方式。它还提供了有关计算图结构和代码位置的信息，而 `tf.print()` 不具备这些功能。

可以用来调试涉及 ∞ 和 NaN 问题的另一个 API 是 [`tf.debugging.enable_check_numerics()`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics)。与 `enable_dump_debug_info()` 不同，`enable_check_numerics()` 不会在磁盘上保存调试信息。相反，它仅会在 TensorFlow 运行时期间监视 ∞ 和 NaN，并在任何运算生成此类不良数值后立即报告错误并附带原始代码位置。与 `enable_dump_debug_info()` 相比，它的性能开销较低，但无法完整追踪程序执行的历史记录，并且没有类似 Debugger V2 的图形用户界面。

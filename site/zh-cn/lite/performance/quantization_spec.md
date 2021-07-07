# TensorFlow Lite 8 位量化规范

以下文档概述了 TensorFlow Lite 的 8 位量化方案的规范，旨在为硬件开发者提供使用量化 TensorFlow Lite 模型进行推断的硬件支持。

## 规范摘要

我们提供的是规范，并且只能在遵守规范时提供部分行为保证。我们也理解不同硬件有其偏好和限制，这可能会在实现规范时造成细微偏差，从而导致实现无法达到比特级精确。但在大多数情况下这或许可以接受（我们将尽我们所知提供一套测试，其中包括我们从几种模型中收集到的按运算容差），机器学习（以及最常见的深度学习）的性质决定了无法提供任何硬性保证。

8 位量化近似于使用以下公式得到的浮点值。

$$real_value = (int8_value - zero_point) \times scale$$

按轴（即卷积运算中的按通道）或按张量权重由 `int8` 补码值表示，范围为 `[-127, 127]`，零点等于 0。按张量激活/输入由 `int8` 补码值表示，范围为 `[-128, 127]`，零点范围为 `[-128, 127]`。

下文记录了特定运算的其他例外情况。

注：过去，我们的量化工具使用的是按张量、非对称、`uint8` 量化。用于 8 位量化的新工具、参考内核和优化内核将使用此规范。

## 有符号整数与无符号整数

TensorFlow Lite 量化将主要优先考虑用于 8 位的 `int8` 工具和内核。这是为了方便由等于 0 的零点表示的对称量化。此外，许多后端还有针对 `int8xint8` 累积的其他优化。

## 按轴与按张量

按张量量化意味着每个完整张量将有一个尺度和/或零点。按轴量化意味着 `quantized_dimension` 中的每个切片将有一个尺度和/或 `zero_point`。量化维度指定尺度和零点所对应的张量形状的维度。例如，张量 `t`（具有 `dims=[4, 3, 2, 1]`，量化参数为：`scale=[1.0, 2.0, 3.0]`、`zero_point=[1, 2, 3]`、`quantization_dimension=1`）将在 `t` 的第二个维度上进行量化：

```
t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3
```

通常，`quantized_dimension` 是卷积权重的 `output_channel`，但从理论上讲，它可以是与内核实现中每个点积相对应的维度，从而在不影响性能的情况下允许更多的量化粒度。这大大提高了准确率。

TFLite 为越来越多的运算提供按轴支持。在撰写本文时，已支持 Conv2d 和 DepthwiseConv2d。

## 对称与非对称

激活是非对称的：它们的零点可以位于有符号 `int8` 范围 `[-128, 127]` 内的任何位置。许多激活在本质上是非对称的，零点是一种相对廉价的方式，可以有效获得额外的二进制位精度。由于激活仅乘以常量权重，因此可以对常量零点值进行大幅优化。

权重是对称的：强制使零点等于 0。权重值会乘以动态输入和激活值。这意味着将权重的零点与激活值相乘时，会不可避免地产生运行时开销。通过强制使零点等于 0，我们可以避免此开销。

数学解释：这类似于 [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) 中的 2.3 节，不同之处在于我们允许将尺度值设为按轴。这很容易泛化，如下所示：

$A$ 是量化激活的 $m x n$ 矩阵。<br>$B$ 是量化权重的 $n x p$ 矩阵。<br>考虑将 $A$, $a_j$ 的第 $j$ 行乘以 $B$, $b_k$ 的第 $k$ 列，两者的长度均为 $n$。量化的整数值和零点值分别是 $q_a$, $z_a$ 和 $q_b$, $z_b$。

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} = \sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) = \sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b - \sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

<!-- Don't change these `\\(` `\\)` to `$`. mathjax fails here with `$`-->

\(\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)}\) 项是不可避免的，因为它执行的是输入值和权重值的点积。

$$\sum_{i=0}^{n} q_{b}^{(i)} z_a$$ 和 $$\sum_{i=0}^{n} z_a z_b$$ 项由常量组成，这些常量在每次推断调用时保持不变，因此可以预先计算。

由于激活会更改每个推断，因此每个推断都需要计算 \(\sum_{i=0}^{n} q_{a}^{(i)} z_b\) 项。通过强制使权重对称，我们可以移除此项的开销。

## Int8 量化算子规范

我们在下面描述了 Int8 TFLite 内核的量化要求：

```
ADD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

AVERAGE_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONCATENATION
  Input ...:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 0)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

DEPTHWISE_CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 3)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

FULLY_CONNECTED
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-tensor
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-tensor
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

L2_NORMALIZATION
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

LOGISTIC
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

MAX_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MUL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

RESHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

RESIZE_BILINEAR
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

SPACE_TO_DEPTH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TANH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

PAD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GATHER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

BATCH_TO_SPACE_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SPACE_TO_BATCH_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TRANSPOSE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MEAN
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUB
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SQUEEZE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LOG_SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (16.0 / 256.0, 127)

MAXIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

ARG_MAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

MINIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LESS
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

PADV2
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GREATER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

GREATER_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

LESS_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SLICE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

NOT_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

QUANTIZE (Requantization)
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
```

## 参考文献

[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

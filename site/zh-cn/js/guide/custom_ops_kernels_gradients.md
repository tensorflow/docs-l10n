# 在 TensorFlow.js 中编写自定义运算、内核和梯度

## 概述

本指南概括了在 TensorFlow.js 中定义自定义运算、内核和梯度的机制，旨在提供对主要概念的概述并指向演示这些概念实际运作情况的代码。

### 本指南的受众

本指南相对比较高级，涉及到 TensorFlow.js 的一些内部结构，可能对以下人群特别有用：

- 对自定义各种数学运算行为感兴趣的 TensorFlow.js 高级用户（例如，替换现有梯度实现的研究人员或需要修补库中缺失功能的用户）
- 构建用于扩展 TensorFlow.js 的库（例如，基于 TensorFlow.js 基元构建的通用线性代数库或者新的 TensorFlow.js 后端）的用户。
- 有兴趣为 TensorFlow.js 贡献新运算并希望大致了解这些机制运作方式的用户。

本文**不是** TensorFlow.js 的通用指南，因为会涉及到内部实现机制。您无需了解这些机制即可使用 TensorFlow.js

您需要能够（或愿意尝试）阅读 TensorFlow.js 源代码才能充分利用本指南。

## 术语

在本指南中，有几个关键术语需要预先说明。

**运算** - 对一个或多个张量进行的数学运算，产生一个或多个张量作为输出。运算是“高级”代码，可以使用其他运算来定义其逻辑。

**内核** - 与特定硬件/平台功能关联的运算的特定实现。内核相对“低级”并且特定于后端。一些运算具有从运算到内核的一对一映射，而另一些运算使用多个内核。

**梯度****/GradFunc** - 用于计算该函数相对于某些输入的导数的**运算/内核**的“反向模式”定义。梯度是“高级”代码（不特定于后端），可以调用其他运算或内核。

**内核注册表** - 从**（内核名称、后端名称）**元组到内核实现的映射。

**梯度注册表** - 从**内核名称到梯度实现**的映射。

## 代码组织

[运算](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops)和[梯度](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients)在 [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core) 中定义。

内核特定于后端，在各自的后端文件夹中定义（例如 [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels)）。

自定义运算、内核和梯度不需要在这些软件包内定义，但是将在它们的实现中经常使用相似符号。

## 实现自定义运算

一种方式是将自定义运算视为返回一些张量输出的 JavaScript 函数，通常以张量作为输入。

- 一些运算可以完全按照现有运算来定义，应该直接导入和调用这些函数。[这是一个示例](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts)。
- 运算的实现也可以分派到后端特定内核。这通过 `Engine.runKernel` 来完成，将在“实现自定义内核”部分中进一步介绍。[这是一个示例](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts)。

## 实现自定义内核

后端特定的内核实现允许针对给定运算提供优化的逻辑实现。内核由调用 [`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F) 的运算调用。内核实现由四个方面定义：

- 内核名称。
- 在其中实现内核的后端。
- 输入：内核函数的张量参数。
- 特性：内核函数的非张量参数。

这是一个[内核实现](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts)示例。用于实现的约定特定于后端，最好通过查看每个特定后端的实现和文档来进行理解。

通常，内核的运算级别低于张量，而且直接读取和写入内存，最终由 tfjs-core 包装成张量。

实现内核后，就可以使用 tfjs-core 中的 [`registerKernel` 函数](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F)将其注册到 TensorFlow.js。您可以为您希望运行内核的每个后端都注册一个内核。注册后，可以使用 `tf.engine().runKernel(...)` 调用内核，TensorFlow.js 将确保分派到当前有效后端中的实现。

## 实现自定义梯度

梯度通常针对给定内核定义（通过对 `tf.engine().runKernel(...)` 的调用中使用的相同内核名称来标识）。这样，tfjs-core 在运行时可以使用注册表来查找任何内核的梯度定义。

实现自定义梯度适用于：

- 添加库中可能不存在的梯度定义
- 替换现有梯度定义以自定义给定内核的梯度计算。

您可以查看[此处的梯度实现](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients)示例。

针对给定调用实现梯度后，可以使用 tfjs-core 中的 [`registerGradient` 函数](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F)将其注册到 TensorFlow.js。

实现可绕过梯度注册表（因此允许以任意方式计算任意函数的梯度）的自定义梯度的另一种方式是使用 [tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad)。

这是一个使用 customGrad 的[库内运算示例](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64)

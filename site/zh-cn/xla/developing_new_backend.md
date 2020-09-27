# 为 XLA 开发新后端

本初级指南适用于希望以有效方式将 TensorFlow 轻松重新定位到其硬件的尝鲜者。本指南没有提供逐步介绍，并假定读者具备 [LLVM](http://llvm.org)、[Bazel](https://bazel.build/) 和 TensorFlow 方面的知识。

XLA 提供了一个新架构或加速器可以实现的抽象接口，供创建用于运行 TensorFlow 计算图的后端。与为新硬件实现每个现有的 TensorFlow 运算相比，重新定位 XLA 的易操作性和可扩展性将显著提高。

以下场景覆盖了大多数实现：

1. XLA 尚未正式支持的现有 CPU 架构，带或不带现有 [LLVM](http://llvm.org) 后端。
2. 非 CPU 类硬件，带现有 LLVM 后端。
3. 非 CPU 类硬件，不带现有 LLVM 后端。

> 注：LLVM 后端既包括正式发布的 LLVM 后端之一，也包括内部开发的自定义 LLVM 后端。

## 场景 1：XLA 尚未正式支持现有的 CPU 架构

在此场景中，请首先查看现有的 [XLA CPU 后端](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/)。通过使用 LLVM，XLA 可以将 TensorFlow 轻松地重新定位到不同的 CPU，因为 CPU 的 XLA 后端之间的主要区别就在于 LLVM 所生成的代码。Google 针对 X64 和 ARM64 架构对 XLA 执行测试。

如果硬件厂商为其硬件提供了 LLVM 后端，那么将该后端链接到使用 XLA 构建的 LLVM 会非常方便。在 JIT 模式下，XLA CPU 后端会为主机 CPU 发出代码。对于提前编译，[`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) 可以提供 LLVM 元组来配置目标架构。

如果厂商未提供现有的 LLVM 后端，但存在另一种代码生成器，则应当可以重用大多数现有 CPU 后端。

## 场景 2：非 CPU 类硬件，带现有 LLVM 后端

可以基于现有的 [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) 和 [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) 类建模新的 [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) 实现，因为它们已发出 LLVM IR。根据硬件的性质，在 LLVM IR 生成方面可能需要执行多项更改，但许多代码都可以与现有后端进行共享。

XLA 的 [GPU 后端](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/)就是一个很好的示例。该 GPU 后端面向非 CPU 类 ISA，因此，其代码生成的某些方面仅适用于 GPU 域。其他类型的硬件（例如 Hexagon 等 DSP，具有上游 LLVM 后端）可以重用部分 LLVM IR 发布逻辑，但其他部分仍不通用。

## 场景 3：非 CPU 类硬件，不带现有 LLVM 后端

如果无法使用 LLVM，则最佳选择是为所需的硬件实现新的 XLA 后端。此选项涉及的工作量最大。需要实现的类如下：

- [`StreamExecutor`](https://www.tensorflow.org/code/tensorflow/stream_executor/stream_executor.h)：对于许多设备而言，并不需要 `StreamExecutor` 的所有方法。有关详细信息，请参阅现有 `StreamExecutor` 实现。
- [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h)：此类可将 HLO 计算的编译封装为 `xla::Executable`。
- [`xla::Executable`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h)：此类用于在平台上启动编译的计算。
- [`xla::TransferManager`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/transfer_manager.h)：此类使后端能够提供特定于平台的机制，用于通过给定的设备内存句柄构造 XLA 文字数据。换言之，它可以帮助封装主机与设备之间的双向数据传输。

# XLA 架构

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;"> <img style="width:50%" src="./images/xlalogo.png"> </div>

## 为什么要构建 XLA？

对于将 XLA 用于 TensorFlow，我们有以下几项目标：

- *提高执行速度*。编译子计算图以减少短暂运算的执行时间，从而消除 TensorFlow 运行时的开销；融合流水线运算以降低内存开销；并针对已知张量形状执行专门优化以支持更积极的常量传播。

- *提高内存使用率*。分析和安排内存使用量，原则上需要消除许多中间存储缓冲区。

- *降低对自定义运算的依赖*。通过提高自动融合的低级运算的性能，使之达到手动融合的自定义运算的性能水平，从而消除对多种自定义运算的需求。

- *减少移动资源占用量*。通过提前编译子计算图并发出可以直接链接到其他应用的对象/头文件对，消除 TensorFlow 运行时。这样，移动推断的资源占用量可降低几个数量级。

- *提高便携性*。使针对新颖硬件编写新后端的工作变得相对容易，在新硬件上运行时，大部分 TensorFlow 程序都能够以未经修改的方式运行。与针对新硬件专门设计各个整体运算的方式相比，这种模式不必重新编写 TensorFlow 程序即可有效利用这些运算。

## XLA 工作原理

XLA 的输入语言称为“HLO IR”或仅为“HLO”（高级运算）。[运算语义](./operation_semantics.md)页面中介绍了 HLO 的语义。可以将 HLO 简单理解为[编译器 IR](https://en.wikipedia.org/wiki/Intermediate_representation)。

XLA 接受在 HLO 中定义的计算图（“计算”）并将其编译为适用于各种架构的机器指令。XLA 采用模块化设计，可以轻松融入其他后端以[针对某些新颖的硬件架构](./developing_new_backend.md)。TensorFlow 源代码树中包含适用于 x64 和 ARM64 架构的 CPU 后端，以及 NVIDIA GPU 后端。

下图显示了 XLA 中的编译过程：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">   <img src="./images/how-does-xla-work.png"> </div>

XLA 提供了多种与目标无关的优化和分析过程（例如 [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。

完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。例如，XLA GPU 后端可以执行特别有利于 GPU 编程模型的运算融合，并确定如何将计算划分为计算流。在此阶段，后端还可能对某些运算或运算组合针对优化库调用执行模式匹配。

下一步是针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 [LLVM](http://llvm.org) 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。

目前，GPU 后端通过 LLVM NVPTX 后端支持 NVIDIA GPU。CPU 后端支持多个 CPU ISA。

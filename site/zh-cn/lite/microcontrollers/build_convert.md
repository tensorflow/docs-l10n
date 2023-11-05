# 构建并转换模型

微控制器具有有限的 RAM 和存储空间，这限制了机器学习模型的规模。此外，面向微控制器的 TensorFlow Lite 目前只支持有限的一部分运算，因此并非所有的模型结构都是可行的。

本文档解释了转换一个 TensorFlow 模型以使其可在微控制器上运行的过程。本文档也概述了可支持的运算，并对于设计与训练一个模型以使其符合内存限制给出了一些指导。

一个端到端的、可运行的建立与转换模型的示例，见于如下的 Jupyter notebook 中： <a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/create_sine_model.ipynb">create_sine_model.ipynb</a>

<a class="button button-primary" href="https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb">train_hello_world_model.ipynb</a>

## 模型转换

要转换一个已训练的 TensorFlow 模型以在微控制器上运行，您应该使用 [TensorFlow Lite 转换器 Python API](https://www.tensorflow.org/lite/models/convert/)。它可以将模型转换成 [`FlatBuffer`](https://google.github.io/flatbuffers/)（缩减模型大小），并进行修改以使用 TensorFlow Lite 运算。

以下的 Python 代码片段展示了如何使用预训练量化进行模型转换：

### 量化

许多微控制器平台没有本地文件系统的支持。从程序中使用一个模型最简单的方式是将其以一个 C 数组的形式包含并编译进你的程序。

以下的 unix 命令会生成一个以 `char` 数组形式包含 TensorFlow Lite 模型的 C 源文件：

```bash
xxd -i converted_model.tflite > model_data.cc
```

其输出类似如下：

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

一旦你已经生成了此文件，你可以将它包含入你的程序。在嵌入式平台上，将数组声明改变为 `const` 类型以获得更好的内存效率是重要的。

有关如何在程序中包含并使用模型的示例，请参阅 <em>Hello World</em> 示例中的 <a><code>evaluate_test.cc</code></a>。

## 模型结构与训练

设计要在微控制器上使用的模型时，必须考虑模型的大小、工作负载和使用的运算。

### 模型规模

模型必须在二进制文件和运行时方面都足够小，才能与程序的其他部分一起装入目标设备的内存。

为了创建一个更小的模型，您可以在架构中使用更少和更小的层。不过，小模型更易面临欠拟合问题。这意味着对于许多问题，明智的做法是尝试并使用可以装入内存的最大模型。但是，使用更大的模型也会导致处理器工作负载增加。

注：在一个 Cortex M3 上，面向微控制器的 TensorFlow Lite 的核心运行时占 16 KB。

### 工作负载

工作负载受到模型大小和复杂度的影响。大而复杂的模型可能会导致更高的占空比，即设备处理器的工作时间延长，空闲时间缩短。根据您的应用，能耗和热量输出增加可能会成为一个问题。

### 运算支持

面向微控制器的 TensorFlow Lite 目前仅支持有限的一部分 TensorFlow 运算，这影响了可以运行的模型架构。我们正致力于在参考实现和针对特定架构的优化方面扩展运算支持。

可以在 [`micro_mutable_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) 文件中查看支持的运算。

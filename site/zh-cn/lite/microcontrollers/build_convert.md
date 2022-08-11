# 构建和转换模型

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

For an example of how to include and use a model in your program, see [`model.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/model.cc) in the *Hello World* example.

## 模型结构与训练

在设计一个面向微控制器的模型时，考虑模型的规模、工作负载，以及用到的运算是非常重要的。

### 模型规模

一个模型必须在二进制和运行时方面都足够小，以使其可以和你程序的其他部分一起符合你目标设备的内存限制。

为了创建一个更小的模型，你可以在你的结构里使用更少和更小的层。然而，小规模的模型更易面临欠拟合问题。这意味着对于许多问题，尝试并使用符合内存限制的尽可能大规模的模型是有意义的。但是，使用更大规模的模型也会导致处理器工作负载的增加。

注：在一个 Cortex M3 上，面向微控制器的 TensorFlow Lite 的核心运行时占 16 KB。

### 工作负载

工作负载受到模型规模与复杂度的影响。大规模、复杂的模型可能会导致更高的占空比，即导致你所用设备处理器的工作时间增长、空闲时间缩短。视你的应用，这种情况所带来的电力消耗与热量输出的增加可能会成为一个问题。

### 运算支持

面向微控制器的 TensorFlow Lite 目前仅支持有限的部分 TensorFlow 运算，这影响了可以运行的模型结构。我们正致力于在参考实现和针对特定结构的优化方面扩展运算支持。

The supported operations can be seen in the file [`all_ops_resolver.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.cc)

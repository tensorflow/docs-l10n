# 创建运算

注：为确保您的 C++ 自定义运算与 TensorFlow 的官方 pip 软件包 ABI 兼容，请遵循[自定义运算仓库](https://github.com/tensorflow/custom-op)中的指南。指南包含端到端代码示例以及用于构建和分发自定义运算的 Docker 镜像。

如果您想创建的运算不在现有 TensorFlow 库的涵盖范围内，我们建议您首先尝试以现有 Python 运算或函数组合的形式使用 Python 编写该运算。如果无法做到这一点，则可以创建自定义 C++ 运算。由于以下几点原因，您可能希望创建自定义 C++ 运算：

- 无法轻易或根本无法将您的运算表示为现有运算的组合。
- 将您的运算表示为现有基元的组合并不高效。
- 您想以手动方式融合未来的编译器难以融合的基元组合。

例如，假设您想要实现诸如“中值池化”之类的功能，与“MaxPool”算子类似，但需要计算滑动窗口期间的中值而不是最大值。可以使用运算组合来实现这一目的（例如，使用 ExtractImagePatches 和 TopK），但在性能或内存效率方面可能不如原生运算那样出色，对于原生运算，您可以利用单个融合运算实现更巧妙的过程。和往常一样，通常有必要首先尝试使用算子组合来表示您想要的运算，只有在这被证实难以实现或效率低下时，才选择添加新运算。

要整合自定义运算，您需要执行以下操作：

1. 在 C++ 文件中注册新运算。运算注册会定义运算功能的接口（规范），此接口与运算的实现无关。例如，运算注册会定义运算的名称及运算的输入和输出，还会定义用于张量形状推断的形状函数。
2. 使用 C++ 实现运算。运算的实现称为内核，它是您在第 1 步中注册的规范的具体实现。可以有多个内核用于不同的输入/输出类型或架构（例如，CPU、GPU）。
3. 创建一个 Python 封装容器（可选）。此封装容器是用于以 Python 创建运算的公共 API。默认封装容器是根据运算注册生成的，用户可以直接使用它或向其中添加内容。
4. 编写一个函数来计算运算的梯度（可选）。
5. 测试运算。为方便起见，我们通常在 Python 中进行测试，但您也可以在 C++ 中测试运算。如果您要定义梯度，可以使用 Python `tf.test.compute_gradient_error` 验证梯度。要了解如何测试 ReLu 之类的算子及其梯度的前向函数，请参阅 [`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py)。

### 前提条件

- 对 C++ 有一定的了解。
- 必须已安装 [TensorFlow 二进制文件](../../install)，或者必须已[下载 TensorFlow 源代码](../../install/source.md)，并且能够构建。

## 定义运算接口

您可以通过将接口注册到 TensorFlow 系统来定义运算的接口。在注册中，您需要指定运算的名称、输入（类型和名称）和输出（类型和名称），以及文档字符串和该运算可能需要的任意[特性](#attrs)。

要了解这一过程的工作原理，假设您想要创建一个接受 `int32` 张量并输出该张量副本（将第一个元素之外的所有其他元素都设置为零）的运算。为此，请先创建一个名为 `zero_out.cc` 的文件，然后添加对 `REGISTER_OP` 宏的调用，该宏可以定义运算的接口：

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`ZeroOut` 运算会将一个包含 32 位整数的张量 `to_zero` 作为输入，并输出一个包含 32 位整数的张量 `zeroed`。该运算还使用形状函数来确保输出张量与输入张量的形状相同。例如，如果输入是形状为 [10, 20] 的张量，则此形状函数会指定输出形状也是 [10, 20]。

注：运算名称必须采用驼峰命名法，并且对于在二进制文件中注册的所有其他运算，该名称必须唯一。

## 实现运算的内核

在定义接口后，您需要为运算提供一个或多个实现。要创建其中一个内核，请先创建一个扩展 `OpKernel` 并重写 `OpKernel` 方法的类。`Compute` 方法提供了一个类型为 `OpKernelContext*` 的 `context` 参数，您可以从中访问输入张量和输出张量等有用信息。

将内核添加到您在上面创建的文件中。内核可能如下所示：

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};
```

实现内核后，您需要将其注册到 TensorFlow 系统。在注册中，您要指定此内核将在哪些不同约束下运行。例如，您可能有一个面向 CPU 的内核，以及一个面向 GPU 的内核。

要针对 `ZeroOut` 运算执行此操作，请将以下代码添加到 `zero_out.cc` 中：

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 重要提示：您的 OpKernel 的实例可能会被同时访问。`Compute` 方法必须为线程安全。使用互斥保护对类成员的任何访问。或者，最好不要通过类成员共享状态！请考虑使用 [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h) 来跟踪运算状态。

### 多线程 CPU 内核

要编写多线程 CPU 内核，可以使用 [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) 中的 Shard 函数。此函数会在配置为用于运算内线程的线程之间对计算函数进行分片（请参阅 [`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto) 中的 intra_op_parallelism_threads）。

### GPU 内核

GPU 内核分为两部分实现：OpKernel 内核，CUDA 内核及其启动代码。

有时，OpKernel 实现在 CPU 和 GPU 内核之间很常见，例如检查输入和分配输出。在这种情况下，建议的实现是：

1. 定义在设备上模板化的 OpKernel 和张量的基元类型。
2. 要对输出进行实际计算，Compute 函数会调用模板化仿函数结构体。
3. 针对 CPUDevice 的仿函数特殊版本在同一文件中定义，但针对 GPUDevice 的仿函数特殊版本在 .cu.cc 文件中定义，因为它将使用 CUDA 编译器进行编译。

下面是一个示例实现。

```c++
// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_
```

```c++
// kernel_example.cc
#include "kernel_example.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Example")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("input_times_two: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ExampleFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
```

```c++
// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * __ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```

## 构建运算库

### 使用系统编译器编译运算（TensorFlow 二进制文件安装）

您应当能够使用 `C++` 编译器（例如系统上可用的 `g++` 或 `clang`）编译 `zero_out.cc`。二进制 PIP 软件包会安装在系统特定位置编译运算所需的头文件和库。不过，TensorFlow Python 库提供了 `get_include` 函数来获取头目录，而 `get_lib` 目录有一个可与之关联的共享对象。下面是这些函数在 Ubuntu 机器上的输出。

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/tensorflow'
```

假设您安装了 `g++`，下面是您可以用于将运算编译成动态库的命令序列。

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

在 macOS 上，构建 `.so` 文件时需要添加附加标志 "-undefined dynamic_lookup"。

> 关于 `>=5` 的 `gcc` 版本的注意事项：gcc 自版本 `5` 起使用新的 C++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx)。TensorFlow 网站上提供的二进制 pip 软件包使用 `gcc4` 构建，该编译器使用旧版 ABI。如果您使用 `gcc>=5` 编译运算库，请在命令行中添加 `-D_GLIBCXX_USE_CXX11_ABI=0`，使库与旧版 ABI 兼容。

### 使用 Bazel 编译运算（TensorFlow 源代码安装）

如果您已安装 TensorFlow 源代码，则可以使用 TensorFlow 的构建系统编译运算。在 [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) 目录中放置一个包含以下 Bazel 构建规则的 BUILD 文件。

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

运行以下命令以构建 `zero_out.so`。

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

要使用 CUDA 内核编译 `Example` 运算，您需要使用 `tf_custom_op_library` 的 `gpu_srcs` 参数。将具有以下 Bazel 构建规则的 BUILD 文件置于 [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) 目录内的新文件夹（例如，“example_gpu”）中。

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

运行以下命令以构建 `kernel_example.so`。

```bash
$ bazel build --config opt //tensorflow/core/user_ops/example_gpu:kernel_example.so
```

注：如上所述，如果您使用 gcc&gt;=5 进行编译，请将 `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` 添加到 Bazel 命令行参数中。

> 注：虽然您可以使用标准的 `cc_library` 规则创建共享库（`.so` 文件），但我们强烈建议您使用 `tf_custom_op_library` 宏。它可以添加一些必需依赖项，并执行检查以确保共享库与 TensorFlow 的插件加载机制兼容。

## 在 Python 中使用运算

TensorFlow Python API 提供了 `tf.load_op_library` 函数来加载动态库并向 TensorFlow 框架注册运算。`load_op_library` 会返回一个 Python 模块，其中包含运算和内核的 Python 封装容器。因此，在构建此运算后，您可以执行以下操作以从 Python 运行它：

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```

请记住，生成的函数将获得一个蛇形名称（以符合 [PEP8](https://www.python.org/dev/peps/pep-0008/)）。因此，如果您的运算在 C++ 文件中命名为 `ZeroOut`，则 Python 函数将称为 `zero_out`。

要使运算成为正则函数并可从 Python 模块中 `import`，在 Python 源文件中调用 `load_op_library` 可能很有用，如下所示：

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## 验证运算是否正常运行

为运算编写测试，您可以验证是否已成功实现运算。请使用以下内容创建 `zero_out_op_test.py` 文件：

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```

然后运行测试（假设您已安装 TensorFlow）：

```sh
$ python zero_out_op_test.py
```

## 将高级功能构建到运算中

现在，您已经知道如何构建一个基本（并受到一些限制的）运算和实现，我们来看看您通常需要构建到运算中的一些更复杂内容。这些包括：

- [条件检查和验证](#conditional-checks-and-validation)
- [运算注册](#op-registration)
    - [特性](#attrs)
    - [特性类型](#attr-types)
    - [多态性](#polymorphism)
    - [输入和输出](#inputs-and-outputs)
    - [向后兼容性](#backwards-compatibility)
- [GPU 支持](#gpu-support)
    - [编译 GPU 设备的内核](#compiling-the-kernel-for-the-gpu-device)
- [在 Python 中实现梯度](#implement-the-gradient-in-python)
- [C++ 中的形状函数](#shape-functions-in-c)

### 条件检查和验证

上面的示例假设运算已应用于任意形状的张量。如果运算只应用于向量，该怎么办？这就需要在上面的 OpKernel 实现中添加检查。

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

上述代码会声明输入是一个向量，如果不是，返回将设置 `InvalidArgument` 状态。[`OP_REQUIRES` 宏](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)采用三个参数：

- `context`，可以是其 `SetStatus()` 方法的 `OpKernelContext` 或 `OpKernelConstruction` 指针（请参阅 [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h)）。
- 条件。例如，[`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h) 中存在用于验证张量形状的函数。
- 错误本身，由 `Status` 对象表示，请参阅 [`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h)。`Status` 包含类型（通常为 `InvalidArgument`，但具体请参见类型列表）和消息。可以在 [`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h) 中找到用于构造错误的函数。

或者，如果您要测试从某个函数返回的 `Status` 对象是否为错误，请使用 [`OP_REQUIRES_OK`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)（如果是，则返回错误）。这两个宏都会在出错时从函数返回。

### 运算注册

#### 特性

运算可以包含特性，特性值在向计算图中添加运算时设置。这些值用于配置运算，用户可以在内核实现中以及运算注册的输入和输出类型中访问它们的值。尽可能使用输入而不是特性，因为输入更灵活。原因在于特性是常量，必须在构造计算图时定义。相比之下，输入是值可以动态变化的张量；也就是说，输入在每一步都可以变化，使用馈送进行设置等。特性用于无法通过输入完成的操作：任何影响签名（输入或输出的数量或类型）或者不能每一步都更改的配置。

您可以在注册运算时定义特性，只需使用 `Attr` 方法指定该特性的名称和类型即可（满足以下格式规范）：

```
<name>: <attr-type-expr>
```

其中 `<name>` 以字母开头，可以由字母数字字符和下划线组成，`<attr-type-expr>` 是[下文所述](#attr-types)形式的类型表达式。

例如，如果您希望 `ZeroOut` 运算保留用户指定的索引，而不是仅保留第 0 个元素，则可以按如下方式注册该运算：

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

（请注意，[特性类型](#attr-types)与用于输入和输出的 `tf.DType` 不同。）

随后，您的内核可以通过 `context` 参数在内核的构造函数中访问此特性：

```c++
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));
  }
  void Compute(OpKernelContext* context) override {
    // ...
  }
 private:
  int preserve_index_;
};
```

然后，您可以在 `Compute` 方法中使用该特性：

```c++
  void Compute(OpKernelContext* context) override {
    // ...

    // We're using saved attr to validate potentially dynamic input
    // So we check that preserve_index is in range
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the requested input value
    output_flat(preserve_index_) = input(preserve_index_);
  }
```

#### 特性类型

特性中支持以下类型：

- `string`：任意字节序列（无需是 UTF8）。
- `int`：有符号整数。
- `float`：浮点数。
- `bool`：True 或 False。
- `type`：[`DataType`](https://www.tensorflow.org/code/tensorflow/core/framework/types.cc) 的（非引用）值之一。
- `shape`：[`TensorShapeProto`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto)。
- `list(<type>)`：`<type>` 列表，其中 `<type>` 是上述类型之一。请注意，`list(list(<type>))` 无效。

另请参阅 [`op_def_builder.cc:FinalizeAttr`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc) 来查看最终列表。

##### 默认值和约束

特性可能具有默认值，并且某些类型的特性可能具有约束。要定义具有约束的特性，您可以使用以下 `<attr-type-expr>`：

`{'<string1>', '<string2>'}`：值必须是值为 `<string1>` 或 `<string2>` 的字符串。使用此语法时，类型的名称 `string` 会被隐式指定。以下代码模拟了一个枚举：

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

`{<type1>, <type2>}`：值的类型为 `type`，必须是 `<type1>` 或 `<type2>` 之一，其中 `<type1>` 和 `<type2>` 是受支持的 `tf.DType`。您未指定特性的类型是 `type`。如果 `{...}` 中包含类型列表，类型会被隐式指定。例如，在以下示例中，特性 `t` 是一个必须为 `int32`、`float` 或 `bool` 的类型：

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

常见类型约束的快捷方式如下：

- `numbertype`：类型 `type` 仅限于数值（非字符串和非布尔）类型。
- `realnumbertype`：类似于 `numbertype`，没有复杂类型。
- `quantizedtype`：类似于 `numbertype` ，但只是量化的数值类型。

这些快捷方式允许的特定类型列表由 [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h) 中的函数（如 `NumberTypes()`）定义。在以下示例中，特性 `t` 必须是数值类型之一：

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

对于此运算：

```python
tf.number_type(t=tf.int32)  # Valid
tf.number_type(t=tf.bool)   # Invalid
```

列表可与其他列表和单一类型组合。以下运算允许特性 `t` 为任意数值类型或布尔类型：

```c++
REGISTER_OP("NumberOrBooleanType")
    .Attr("t: {numbertype, bool}");
```

对于此运算：

```python
tf.number_or_boolean_type(t=tf.int32)  # Valid
tf.number_or_boolean_type(t=tf.bool)   # Valid
tf.number_or_boolean_type(t=tf.string) # Invalid
```

`int >= <n>`：值必须是整型，且大于或等于 `<n>`，其中 `<n>` 是自然数。例如，以下运算注册指定特性 `a` 的值必须至少为 `2`：

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

`list(<type>) >= <n>`：类型为 `<type>` 的列表，其长度大于或等于 `<n>`。例如，以下运算注册指定特性 `a` 是 `int32` 或 `float` 类型的列表，并且必须至少有 3 个值：

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

要设置特性的默认值（使其在生成的代码中可选），请将 `= <default>` 添加到末尾，例如：

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

此外，还可以同时指定约束和默认值：

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

默认值的支持语法是在生成的 GraphDef 定义的 proto 表示法中使用的语法。

以下示例演示了如何为所有类型指定默认值：

```c++
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
```

特别要注意的是，类型为 `type` 的值使用 `tf.DType`。

#### 多态性

##### 类型多态性

对于可以采用不同类型作为输入或生成不同输出类型的运算，您可以在运算注册的[输入或输出类型](#inputs-and-outputs)中指定[特性](#attrs)。通常，您需要为每种受支持的类型注册一个 `OpKernel`。

例如，如果除了 `int32` 之外您还想对 `float` 执行 `ZeroOut` 运算，则运算注册可能如下所示：

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

运算注册现在指定输入的类型必须是 `float` 或 `int32`，并且其输出将是同一类型，因为它们都具有 `T` 类型。

###### 命名

输入、输出和特性通常应当会采用蛇形名称。但是，用作输入类型或在输出类型中使用的特性例外。向计算图中添加运算时，系统可以推断出这些特性，因此这些特性不会显示在运算的函数中。例如，ZeroOut 的最后一个定义将生成如下所示的 Python 函数：

```python
def zero_out(to_zero, name=None):
  """...
  Args:
    to_zero: A `Tensor`. Must be one of the following types:
        `float32`, `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `to_zero`.
  """
```

如果向 `to_zero` 传递 `int32` 张量，`T` 将被自动设置为 `int32`（实际上是 `DT_INT32`）。这些推断特性会采用大写或驼峰式名称。

将此运算与具有决定输出类型的类型特性的运算进行比较：

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

在这种情况下，用户必须指定输出类型，如生成的 Python 中所示：

```python
def string_to_number(string_tensor, out_type=None, name=None):
  """Converts each string in the input Tensor to the specified numeric type.

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.int32`.
      Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
```

###### 类型多态化示例

```c++
#include "tensorflow/core/framework/op_kernel.h"

class ZeroOutInt32Op : public OpKernel {
  // as before
};

class ZeroOutFloatOp : public OpKernel {
 public:
  explicit ZeroOutFloatOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<float>();

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value
    if (N > 0) output_flat(0) = input(0);
  }
};

// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutInt32Op);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutFloatOp);
```

要保留[向后兼容性](#backwards-compatibility)，您应当在将特性添加到现有运算时指定[默认值](#default-values-and-constraints)：

```c++
REGISTER_OP("ZeroOut")
  .Attr("T: {float, int32} = DT_INT32")
  .Input("to_zero: T")
  .Output("zeroed: T")
```

假设您想添加更多类型，例如 `double`：

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

通常，您可以使用 C++ 模板，而无需使用上面所示的冗余代码编写另一个 `OpKernel`。对于每个重载，您仍然有一个内核注册（`REGISTER_KERNEL_BUILDER` 调用）。

```c++
template <typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<T>();

    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // Preserve the first input value
    if (N > 0) output_flat(0) = input(0);
  }
};

// Note that TypeConstraint<int32>("T") means that attr "T" (defined
// in the op registration above) must be "int32" to use this template
// instantiation.
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    ZeroOutOp<double>);
```

如果您有多个重载，则可以将注册放在宏中。

```c++
#include "tensorflow/core/framework/op_kernel.h"

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
```

根据您为其注册内核的类型列表，您或许可以使用 [`tensorflow/core/framework/register_types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h) 提供的宏：

```c++
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T");

template <typename T>
class ZeroOutOp : public OpKernel { ... };

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
```

##### 列表输入和输出

除了能够接受或生成不同类型之外，运算还可以使用或生成数量可变的张量。

在下一个示例中，特性 `T` 存储了一个类型*列表*，并同时用作输入 `in` 和输出 `out` 的类型。输入和输出是该类型的张量列表（输出中的张量数量和类型与输入的相同，因为二者的类型均为 `T`）。

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

您还可以对可在列表中指定的类型施加限制。在下一个示例中，输入是 `float` 和 `double` 张量的列表。例如，运算接受输入类型 `(float, double, float)`，在这种情况下，输出类型也将是 `(float, double, float)`。

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

如果您希望列表中的所有张量都具有相同类型，则可以运行如下命令：

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

此运算接受 `int32` 张量列表，并使用 `int` 特性 `N` 指定列表的长度。

类型也可以是[多态类型](#type-polymorphism)。在下一个示例中，输入是相同（但未指定）类型 (`"T"`) 的张量（长度为 `"N"`）列表，输出是一个具有匹配类型的张量：

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

默认情况下，张量列表的最小长度为 1。您可以[对相应特性使用 `">="` 约束](#default-values-and-constraints)来更改该默认值。在下一个示例中，输入是至少有 2 个 `int32` 张量的列表：

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

相同的语法也适用于 `"list(type)"` 特性：

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### 输入和输出

综上所述，运算注册可以有多个输入和输出：

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

每个输入或输出规范的格式如下：

```
<name>: <io-type-expr>
```

其中 `<name>` 以字母开头，可以包含字母数字字符和下划线。`<io-type-expr>` 是以下类型表达式之一：

- `<type>`，其中 `<type>` 是受支持的输入类型（例如 `float`、`int32`、`string`）。它会指定一个具有给定类型的张量。

    请参阅 `tf.DType`。

    ```c++
    REGISTER_OP("BuiltInTypesExample")
        .Input("integers: int32")
        .Input("complex_numbers: complex64");
    ```

- `<attr-type>`，其中 `<attr-type>` 是类型为 `type` 或 `list(type)`（可能有类型限制）的[特性](#attrs)的名称。此语法允许[多态运算](#polymorphism)。

    ```c++
    REGISTER_OP("PolymorphicSingleInput")
        .Attr("T: type")
        .Input("in: T");

    REGISTER_OP("RestrictedPolymorphicSingleInput")
        .Attr("T: {int32, int64}")
        .Input("in: T");
    ```

    引用类型为 `list(type)` 的特性可让您接受张量序列。

    ```c++
    REGISTER_OP("ArbitraryTensorSequenceExample")
        .Attr("T: list(type)")
        .Input("in: T")
        .Output("out: T");

    REGISTER_OP("RestrictedTensorSequenceExample")
        .Attr("T: list({int32, int64})")
        .Input("in: T")
        .Output("out: T");
    ```

    请注意，输出 `out` 和输入 `in` 中的张量数量和类型相同，因为二者的类型均为 `T`。

- 对于具有相同类型的张量序列：`<number> * <type>`，其中 `<number>` 是类型为 `int` 的[特性](#attrs)的名称。`<type>` 可以是 `tf.DType`，也可以是类型为 `type` 的特性的名称。就第一种情况举例来说，此运算会接受 `int32` 张量列表：

    ```c++
    REGISTER_OP("Int32SequenceExample")
        .Attr("NumTensors: int")
        .Input("in: NumTensors * int32")
    ```

    此运算会接受任何类型的张量列表，只要它们类型都相同即可：

    ```c++
    REGISTER_OP("SameTypeSequenceExample")
        .Attr("NumTensors: int")
        .Attr("T: type")
        .Input("in: NumTensors * T")
    ```

- 对于张量引用：`Ref(<type>)`，其中 `<type>` 是之前的类型之一。

系统将推断输入类型中使用的任何特性。按照惯例，这些推断特性使用大写名称（如 `T` 或 `N`）。否则，输入、输出和特性会具有与函数参数类似的名称（例如 `num_outputs`）。有关详情，请参阅[前面有关命名的部分](#naming)。

有关详情，请参阅 [`tensorflow/core/framework/op_def_builder.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h)。

#### 向后兼容性

假设您已编写一个不错的自定义运算并与其他人共享，客户在使用您的运算时感到满意。不过，您需要以某种方式对运算进行更改。

通常，对现有检入规范的变更必须向后兼容：更改运算规范不得破坏之前根据旧规范构造的序列化 `GraphDef` 协议缓冲区。[此处](./versions.md#compatibility_of_graphs_and_checkpoints)详细介绍了 `GraphDef` 兼容性。

可通过以下几种方式保持向后兼容性。

1. 添加到运算的任何新特性都必须定义默认值，并且在使用该默认值时，运算必须具有原始行为。要将运算从非多态更改为多态，您*必须*为新类型特性提供默认值，以便在默认情况下保留原始签名。例如，如果您的运算是：

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: float")
        .Output("out: float");
    ```

    您可以使用以下代码以向后兼容的方式使该运算变为多态：

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: T")
        .Output("out: T")
        .Attr("T: numerictype = DT_FLOAT");
    ```

2. 您可以安全地放宽对特性的约束。例如，您可以从 `{int32, int64}` 更改为 `{int32, int64, float}` 或 `type`，也可以从 `{"apple", "orange"}` 更改为 `{"apple", "banana", "orange"}` 或 `string`。

3. 只要列表类型的默认值与旧签名匹配，您就可以将单一输入/输出更改为列表输入/输出。

4. 您可以添加新的列表输入/输出（如果默认为空）。

5. 为您创建的任何新运算设置命名空间，方法是在运算名称前添加项目独有的内容作为前缀。这样可以避免您的运算与未来版本的 TensorFlow 中可能包含的任何运算发生冲突。

6. 未雨绸缪！尝试预测运算的未来用途。某些签名变更无法以兼容的方式完成（例如，将相同类型的列表变为类型变化的列表）。

可以在 [`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc) 中找到安全和不安全变更的完整列表。如果您无法以向后兼容的方式更改运算，则创建新的运算，并使用新语义设置新名称。

另请注意，虽然这些变更可以保持 `GraphDef` 兼容性，但生成的 Python 代码可能会以与旧调用者不兼容的方式发生更改。要使 Python API 保持兼容，可以在手写 Python 封装容器中小心地进行更改，并保留旧签名（可能要在末尾添加新的可选参数时除外）。通常，只有在 TensorFlow 更改主要版本时，才可以进行不兼容的变更，并且这些变更必须符合 <a data-md-type="raw_html" href="./versions.md#compatibility_of_graphs_and_checkpoints">`GraphDef` 版本语义</a>。

### GPU 支持

您可以实现不同的 OpKernel 并为 CPU 和 GPU 各注册一个内核，就像您可以[为不同类型注册内核](#polymorphism)一样。[`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/) 中提供了几个支持 GPU 的内核示例。请注意，某些内核在 `.cc` 文件中具有 CPU 版本，在以 `_gpu.cu.cc` 结尾的文件中具有 GPU 版本，并且在 `.h` 文件中具有一些共享的通用代码。

例如，`tf.pad` 在 [`tensorflow/core/kernels/pad_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc) 中具有除 GPU 内核之外的所有代码。GPU 内核位于 [`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc) 中，共享代码是在 [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h) 中定义的模板化类。我们以这种方式组织代码有两个原因：可以让您在 CPU 和 GPU 实现之间共享通用代码，并将 GPU 实现放入单独的文件中，这样它便只能由 GPU 编译器进行编译。

有一点需要注意，即使使用 `pad` 的 GPU 内核版本，它在 CPU 内存中仍然需要 `"paddings"` 输入。要标记在 CPU 上保留输入或输出，请添加对内核注册的 `HostMemory()` 调用，例如：

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### 编译 GPU 设备的内核

要了解使用 CUDA 内核实现运算的示例，请参阅 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc)。`tf_custom_op_library` 会接受 `gpu_srcs` 参数，可以在其中指定包含 CUDA 内核的源文件（`*.cu.cc` 文件）列表。要与 TensorFlow 的二进制文件安装一起使用，必须使用 NVIDIA 的 `nvcc` 编译器编译 CUDA 内核。您可以使用以下命令序列将 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) 和 [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc) 编译成一个可动态加载的库：

```bash
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

可以像往常一样，在 Python 中使用 `tf.load_op_library` 函数加载上面生成的 `cuda_op_kernel.so`。

请注意，如果 CUDA 库未安装在 `/usr/local/lib64` 中，则您需要在上面的第二个 (g++) 命令中明确指定路径。例如，如果 CUDA 安装在 `/usr/local/cuda-8.0` 中，则添加 `-L /usr/local/cuda-8.0/lib64/`。

注：在某些 Linux 设置中，需要在 `nvcc` 编译步骤中添加其他选项。将 `-D_MWAITXINTRIN_H_INCLUDED` 添加到 `nvcc` 命令行可以避免 `mwaitxintrin.h` 出错。

### 在 Python 中实现梯度

给定运算的计算图后，TensorFlow 会使用自动微分（反向传播）添加表示相对于现有运算的梯度的新运算。要使自动微分对新运算生效，您必须注册一个梯度函数，在给定相对于运算输出的梯度时，计算相对于运算输入的梯度。

在数学上，如果运算计算 \(y = f(x)\)，注册的梯度运算会通过链式法则将相对于 \(y\) 的损失 \(L\) 的梯度 \(\partial L/ \partial y\) 转换为相对于 \(x\) 的梯度 \(\partial L/ \partial x\)：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

对于 `ZeroOut`，输入中只有一个条目会影响输出，因此相对于输入的梯度是稀疏的“独热”张量。具体表示方式如下：

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
```

有关使用 `tf.RegisterGradient` 注册梯度函数的详细信息如下：

- 对于具有一个输出的运算，梯度函数将采用 `tf.Operation`、`op` 和 `tf.Tensor` `grad`，并根据张量 `op.inputs[i]`、`op.outputs[i]` 和 `grad` 构建新运算。任何特性的相关信息均可通过 `tf.Operation.get_attr` 找到。

- 如果运算有多个输出，则梯度函数将采用 `op` 和 `grads`，其中 `grads` 是相对于每个输出的梯度列表。梯度函数的结果必须是 `Tensor` 对象（表示相对于每个输入的梯度）列表。

- 如果某个输入（例如用作索引的整数输入）没有明确定义的梯度，则返回的相应梯度应为 `None`。例如，对于采用浮点张量 `x` 和整数索引 `i` 的运算，梯度函数将为 `return [x_grad, None]`。

- 如果运算根本没有有意义的梯度，您通常无需注册任何梯度，并且只要从不需要该运算的梯度，就没有问题。在某些情况下，运算没有明确定义的梯度，但可以参与梯度计算。在这种情况下，您可以使用 `ops.NotDifferentiable` 自动向后传播零。

请注意，在调用梯度函数时，只有运算的数据流图可用，而张量数据本身不可用。因此，必须使用其他 TensorFlow 运算执行所有计算，以在计算图执行时运行。

### C++ 中的形状函数

TensorFlow API 具有一项称为“形状推断”的功能，该功能可以提供有关张量形状的信息，而无需执行计算图。形状推断由在 C++ `REGISTER_OP` 声明中为每个运算类型注册的“形状函数”提供支持，并承担两个角色：声明输入的形状在计算图构造期间是兼容的，并指定输出的形状。

形状函数定义为 `shape_inference::InferenceContext` 类上的运算。例如，在 ZeroOut 的形状函数中：

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` 会声明第一个输出的形状应当设置为第一个输入的形状。如果按输出索引选择输出（如上例所示），则 `set_output` 的第二个参数应当为 `ShapeHandle` 对象。您可以通过空 `ShapeHandle` 对象的默认构造函数创建该对象。索引为 `idx` 的输入的 `ShapeHandle` 对象可以通过 `c->input(idx)` 获得。

有许多常用的形状函数适用于多种运算（例如 `shape_inference::UnchangedShape`），您可以在 [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) 中找到这些函数并按如下方式使用它们：

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

形状函数还可以约束输入的形状。对于[具有向量形状约束的 `ZeroOut`](#conditional-checks-and-validation) 版本，形状函数将如下所示：

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

`WithRank` 调用会验证输入形状 `c->input(0)` 是否具有只有一维的形状（或者，如果输入形状未知，输出形状将为具有一个未知维度的向量）。

如果您的运算是[多态的且包含多个输入](#polymorphism)，则可以使用 `InferenceContext` 成员确定要检查的形状的数量，并使用 `Merge` 验证这些形状是否均兼容（或者，使用可提供对运算特性的访问权限的 `InferenceContext::GetAttr` 来访问表示长度的特性）。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &input));
        TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
      }
      c->set_output(0, output);
      return Status::OK();
    });
```

由于形状推断是可选功能，并且张量的形状可能会动态变化，因此对于任何输入的不完整形状信息，形状函数必须可靠。[`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) 中的 `Merge` 方法允许调用者声明两个形状是相同的，即使其中任一个或两个都没有完整的信息。我们为所有核心 TensorFlow 运算定义了形状函数，并提供了许多不同的用法示例。

`InferenceContext` 类具有许多可用于定义形状函数操作的函数。例如，您可以使用 `InferenceContext::Dim` 和 `InferenceContext::WithValue` 验证特定维度是否具有非常具体的值；您可以使用 `InferenceContext::Add` 和 `InferenceContext::Multiply` 指定输出维度是两个输入维度的和/积。要了解您可以指定的所有不同形状操作，请参阅 `InferenceContext` 类。以下示例将第一个输出的形状设置为 (n, 3)，其中第一个输入的形状为 (n, ...)

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

如果您具有复杂的形状函数，则应考虑添加测试，以验证各种输入形状组合是否会生成预期的输出形状组合。您可以在我们的一些[核心运算测试](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc)中看到有关如何编写这些测试的示例。（`INFER_OK` 和 `INFER_ERROR` 的语法有点晦涩，但在表示测试中的输入和输出形状规范时力求做到紧凑。就目前而言，请查看这些测试中的周围注释来了解形状字符串规范。）

## 为您的自定义运算构建 pip 软件包

要为您的运算构建 `pip` 软件包，请参阅 [tensorflow/custom-op](https://github.com/tensorflow/custom-op) 示例。本指南说明了如何从 TensorFlow pip 软件包构建自定义运算，而不是从源代码构建 TensorFlow。

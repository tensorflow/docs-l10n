# 使用 AOT 编译

## 什么是 tfcompile？

`tfcompile` 是一个可将 TensorFlow 计算图提前 (AOT) 编译为可执行代码的独立工具。它可以缩减二进制文件的总大小，也可以避免一些运行时开销。`tfcompile` 的典型用例是将推断计算图编译为适用于移动设备的可执行代码。

TensorFlow 计算图通常由 TensorFlow 运行时执行。在执行计算图中的每个节点时，均会产生一定的运行时开销。这也会导致二进制文件更大，因为除了计算图本身以外，还需包含 TensorFlow 运行时代码。由 `tfcompile` 生成的可执行代码不会使用 TensorFlow 运行时，而仅仅依赖于计算实际使用的内核。

编译器基于 XLA 框架构建。[tensorflow/compiler](https://www.tensorflow.org/code/tensorflow/compiler/) 下提供了用于将 TensorFlow 桥接到 XLA 框架的代码。

## tfcompile 的功能是什么？

`tfcompile` 接受由 TensorFlow 的 feed 和 fetch 概念标识的子计算图，并生成实现该子计算图的函数。`feeds` 为函数的输入参数，`fetches` 为函数的输出参数。所有输入必须完全由 feed 指定；生成的剪枝子计算图不能包含占位符或变量节点。通常，需要将所有占位符和变量指定为 feed，以确保生成的子计算图不再包含这些节点。生成的函数将打包为 `cc_library`，其中带有导出函数签名的头文件和包含实现的对象文件。用户编写代码以适当调用生成的函数。

## 使用 tfcompile

本部分将详细介绍使用 `tfcompile` 从 TensorFlow 子计算图生成可执行二进制文件的高级步骤。步骤包括：

- 第 1 步：配置要编译的子计算图
- 第 2 步：使用 `tf_library` 构建宏编译子计算图
- 第 3 步：编写代码以调用子计算图
- 第 4 步：创建最终的二进制文件

### 第 1 步：配置要编译的子计算图

标识与生成的函数的输入和输出参数相对应的 feed 和 fetch。然后在 [`tensorflow.tf2xla.Config`](https://www.tensorflow.org/code/tensorflow/compiler/tf2xla/tf2xla.proto) proto 中配置 `feeds` 和 `fetches`。

```textproto
# Each feed is a positional input argument for the generated function.  The order
# of each entry matches the order of each input argument.  Here “x_hold” and “y_hold”
# refer to the names of placeholder nodes defined in the graph.
feed {
  id { node_name: "x_hold" }
  shape {
    dim { size: 2 }
    dim { size: 3 }
  }
}
feed {
  id { node_name: "y_hold" }
  shape {
    dim { size: 3 }
    dim { size: 2 }
  }
}

# Each fetch is a positional output argument for the generated function.  The order
# of each entry matches the order of each output argument.  Here “x_y_prod”
# refers to the name of a matmul node defined in the graph.
fetch {
  id { node_name: "x_y_prod" }
}
```

### 第 2 步：使用 tf_library 构建宏编译子计算图

在此步骤中，会使用 `tf_library` 构建宏来将计算图转换为 `cc_library`。`cc_library` 由对象文件和头文件组成：对象文件包含从计算图生成的代码，通过头文件则可访问生成的代码。`tf_library` 使用 `tfcompile` 将 TensorFlow 计算图编译为可执行代码。

```build
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# Use the tf_library macro to compile your graph into executable code.
tf_library(
    # name is used to generate the following underlying build rules:
    # <name>           : cc_library packaging the generated header and object files
    # <name>_test      : cc_test containing a simple test and benchmark
    # <name>_benchmark : cc_binary containing a stand-alone benchmark with minimal deps;
    #                    can be run on a mobile device
    name = "test_graph_tfmatmul",
    # cpp_class specifies the name of the generated C++ class, with namespaces allowed.
    # The class will be generated in the given namespace(s), or if no namespaces are
    # given, within the global namespace.
    cpp_class = "foo::bar::MatMulComp",
    # graph is the input GraphDef proto, by default expected in binary format.  To
    # use the text format instead, just use the ‘.pbtxt’ suffix.  A subgraph will be
    # created from this input graph, with feeds as inputs and fetches as outputs.
    # No Placeholder or Variable ops may exist in this subgraph.
    graph = "test_graph_tfmatmul.pb",
    # config is the input Config proto, by default expected in binary format.  To
    # use the text format instead, use the ‘.pbtxt’ suffix.  This is where the
    # feeds and fetches were specified above, in the previous step.
    config = "test_graph_tfmatmul.config.pbtxt",
)
```

> 要为此样本生成 GraphDef proto (test_graph_tfmatmul.pb)，请运行 [make_test_graphs.py](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/make_test_graphs.py) 并使用 --out_dir 标记指定输出位置。

典型的计算图包含代表通过训练所学权重的 [`Variables`](https://www.tensorflow.org/guide/variables)，但 `tfcompile` 无法编译包含 `Variables` 的子计算图。[freeze_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py) 工具可使用存储在检查点文件中的值，将变量转换为常量。为方便起见，`tf_library` 宏支持可运行该工具的 `freeze_checkpoint` 参数。有关更多示例，请参阅 [tensorflow/compiler/aot/tests/BUILD](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/BUILD)。

> 已编译的子计算图中的常量将直接编译为生成的代码。要将常量传递给生成的函数（而非对其进行编译），只需将其作为 feed 传递即可。

有关 `tf_library` 构建宏的详细信息，请参阅 [tfcompile.bzl](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile.bzl)。

有关底层 `tfcompile` 工具的详细信息，请参阅 [tfcompile_main.cc](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile_main.cc)。

### 第 3 步：编写代码以调用子计算图

在此步骤中，将使用在上一步骤中由 `tf_library` 构建宏生成的头文件 (`test_graph_tfmatmul.h`) 调用生成的代码。头文件位于与构建软件包相对应的 `bazel-bin` 目录内，并基于在 `tf_library` 构建宏上设置的名称特性进行命名。例如，为 `test_graph_tfmatmul` 生成的头文件将为 `test_graph_tfmatmul.h`。以下是生成的头文件的简化版。在 `bazel-bin` 目录内生成的文件包含其他实用注释。

```c++
namespace foo {
namespace bar {

// MatMulComp represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code.
class MatMulComp {
 public:
  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    ARGS_RESULTS_AND_TEMPS,  // Allocate arg, result and temp buffers
    RESULTS_AND_TEMPS_ONLY,  // Only allocate result and temp buffers
  };

  MatMulComp(AllocMode mode = AllocMode::ARGS_RESULTS_AND_TEMPS);
  ~MatMulComp();

  // Runs the computation, with inputs read from arg buffers, and outputs
  // written to result buffers. Returns true on success and false on failure.
  bool Run();

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument.
  void** args();

  void set_arg0_data(float* data);
  float* arg0_data();
  float& arg0(size_t dim0, size_t dim1);

  void set_arg1_data(float* data);
  float* arg1_data();
  float& arg1(size_t dim0, size_t dim1);

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result.
  void** results();


  float* result0_data();
  float& result0(size_t dim0, size_t dim1);
};

}  // end namespace bar
}  // end namespace foo
```

生成的 C++ 类在 `foo::bar` 命名空间中称为 `MatMulComp`，因为它是在 `tf_library` 宏中指定的 `cpp_class`。所有生成的类都有相似的 API，唯一区别在于处理参数和结果缓冲区的方法。这些方法因缓冲区的数量和类型而异，缓冲区的数量和类型通过 `feed` 和 `fetch` 参数对 `tf_library` 宏指定。

在生成的类中管理三种缓冲区类型：`args` 代表输入，`results` 代表输出，`temps` 代表内部用于执行计算的临时缓冲区。默认情况下，生成的类的每个实例都会为您分配和管理上述所有缓冲区。`AllocMode` 构造函数参数可用于更改此行为。所有缓冲区都与 64 字节边界对齐。

生成的 C++ 类仅是由 XLA 生成的低级代码的包装器。

调用基于 [`tfcompile_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/tfcompile_test.cc) 的生成函数的示例：

```c++
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmul.h" // generated

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());


  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);

  // Set up args and run the computation.
  const float args[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(args + 0, args + 6, matmul.arg0_data());
  std::copy(args + 6, args + 12, matmul.arg1_data());
  matmul.Run();

  // Check result
  if (matmul.result0(0, 0) == 58) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed. Expected value 58 at 0,0. Got:"
              << matmul.result0(0, 0) << std::endl;
  }

  return 0;
}
```

### 第 4 步：创建最终的二进制文件

在此步骤中，将对第 2 步中由 `tf_library` 生成的库以及第 3 步中编写的代码进行结合，从而创建最终的二进制文件。以下为示例 `bazel` BUILD 文件。

```build
# Example of linking your binary
# Also see //tensorflow/compiler/aot/tests/BUILD
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# The same tf_library call from step 2 above.
tf_library(
    name = "test_graph_tfmatmul",
    ...
)

# The executable code generated by tf_library can then be linked into your code.
cc_binary(
    name = "my_binary",
    srcs = [
        "my_code.cc",  # include test_graph_tfmatmul.h to access the generated header
    ],
    deps = [
        ":test_graph_tfmatmul",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)
```

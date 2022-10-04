# XLA 自定义调用

本文档介绍如何编写和使用 XLA“自定义调用”。借助自定义调用，您可以从 XLA 程序调用以 C++ 或 CUDA 等编程语言编写的代码。

警告：自定义调用是一种底层高级用户功能。使用自定义调用时，很容易使用难以调试（甚至难以通知）的方式破坏程序。除非您准备在发生问题时自行调试 XLA，否则不应使用自定义调用，而且在陷入困境时，您应当预计到从 XLA 开发者那里得到的帮助会相对较少。

警告：自定义调用 API/ABI 目前不稳定。我们不打算随意更改它，但它也可能发生变化。下文介绍了一些未来可能进行的变更。

## CPU 上的自定义调用

您可以通过 XLA 的客户端 API 创建代表自定义调用的 HLO 指令。在撰写本文时，尚未通过 TensorFlow 公开此功能。

例如，以下代码使用自定义调用来计算 `A[i] = B[i % 128]

```c++
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla::F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                      /*shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}));
}

void do_custom_call(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* in0 = reinterpret_cast<const float*>(in[0]);
  const float* in1 = reinterpret_cast<const float*>(in[1]);
  for (int i = 0; i < 2048; ++i) {
    out_buf[i] = in0[i % 128] + in1[i];
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "Host");
```

请注意，函数 `do_custom_call` 需要了解其运行所处的缓冲区的大小。在此示例中，我们对大小 128 和 2048 进行硬编码。如果您不想这样做，则可以将大小作为参数传递给调用。

## GPU 上的自定义调用

GPU 自定义调用框架与 CPU 上的框架有所不同。下面是一个 CUDA 示例，它执行与上述 CPU 代码相同的 `A[i] = B[i % 128] + C[i]` 计算。

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
```

首先请注意，GPU 自定义调用函数*仍然是在 CPU 上执行的函数*。我们的 `do_custom_call` CPU 函数负责将 GPU 上的工作加入队列。在这里，它会启动 CUDA 内核，但也可以执行其他操作，例如调用 cublas。

此功能仅适用于 GPU 自定义调用；无法为 CPU 自定义调用发出失败信号。

与 CPU 示例中一样，我们已将输入和输出缓冲区的大小硬编码到我们的自定义调用中。但是，与 CPU 情况不同，将缓冲区大小作为运算对象传递给自定义调用将无法正常工作。通常，我们需要 CPU 上可用的缓冲区大小；例如，在启动内核时，我们需要了解要使用的块/网格大小。但是，如果我们将缓冲区大小作为运算对象传递给自定义调用，它们的值将驻留在 GPU 内存中。随后，我们必须在运算开始时执行一个消耗大量资源的同步设备-主机 memcpy 来读取大小。

为了解决此问题，我们提供了 `opaque` 参数。创建自定义调用时，可以将此参数设置为任意字节串：

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

由于 `xla::Shape` 具有协议缓冲区表示，因此您可以将此序列化 proto 存储在 `opaque` 中，并在 GPU 自定义调用中将其反序列化。不过请注意，尽管 `xla::ShapeProto` 不会经常变化，但有时*确实*会变化。检查 git 日志以查看其过去发生的变化。

## 发出错误信号。

如果您的自定义调用遇到错误，您可以通过在 CPU 上为函数使用以下签名来向 XLA 运行时发出错误信号（而不是崩溃或在输出缓冲区中返回废话）：

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

… 在 GPU 上：

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

您可以使用 `XlaCustomCallStatusSetFailure` 发出失败信号，例如：

```c++
void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status) {
  // ... do some work.

  if (bad_condition) {
    char* error_message = "An error occurred";
    XlaCustomCallStatusSetFailure(status, error_message, strlen(error_message));
    return;
  }

  // ... continue.
}
```

您也可以使用 `XlaCustomCallStatusSetSuccess` 来指示成功，但 `XlaCustomCallStatus` 默认处于成功状态，因此完全忽略它也会指示成功。

使用具有此签名的自定义调用函数时，您必须使用适当的 API 版本集创建相应的 `custom-call` 运算，例如：

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

注：在将来，所有客户端都需要将其自定义调用函数迁移到新的 API 版本，旧版本将被弃用。对于不会失败的自定义调用，您可以简单地添加新的 `XlaCustomCallStatus*` 参数，然后忽略它。

失败时，不会使用任何自定义调用输出；XLA 运行时将终止计算。HLO 计算无法从错误中恢复（例如通过捕获和处理错误）。

## 将元组传递给自定义调用

请考虑以下自定义调用。

```c++
using xla::ShapeUtil;
using xla::F32;
Shape p0_shape = ShapeUtil::MakeTuple({
    ShapeUtil::MakeShape(F32, {32}),
    ShapeUtil::MakeTuple({
        ShapeUtil::MakeShape(F32, {64}),
        ShapeUtil::MakeShape(F32, {128}),
    }),
    ShapeUtil::MakeShape(F32, {256}),
});
xla::XlaOp p0 = xla::Parameter(0, p0_shape, "p0");

Shape out_shape = ShapeUtil::MakeTuple({
  ShapeUtil::MakeShape(F32, {512}),
  ShapeUtil::MakeShape(F32, {1024}),
});
xla::CustomCall(&b, "do_custom_call", /*operands=*/{p0}, out_shape);
```

在 CPU 和 GPU 上，元组在内存中均被表示为指针数组。在 C++ 伪代码中，上述参数 0 的布局如下所示。

```c++
// In-memory layout of parameter 0 from custom-call above.  True on both CPU
// and GPU.
float* subbuf0 = new float[32];
float* subbuf1 = new float[64];
float* subbuf2 = new float[128]
float* subbuf3 = new float[256];

void* subtuple = new void*[2];
(*subtuple)[0] = subbuf1;
(*subtuple)[1] = subbuf2;

void* p0 = new void*[3];
(*p0)[0] = subbuf0;
(*p0)[1] = subtuple;
(*p0)[2] = subbuf3;
```

尽管元组的内存中表示在 CPU 和 GPU 中相同，但是在 CPU 和 GPU 自定义调用的调用惯例中，它们的处理方式有所不同。

### 元组输出作为临时缓冲区

自定义调用的元组输入非常方便，但并非绝对必要。如果我们不支持在自定义调用中使用元组输入，则在将元组传递到自定义调用之前，始终可以使用 get-tuple-element 将它们解包。

另一方面，元组*输出*可让您完成以其他方式无法完成的任务。

具有元组输出的明显原因是，这是自定义调用（或任何其他 XLA 运算）返回多个独立数组的方式。

但不太明显的是，元组输出也是一种为自定义调用提供临时内存的方式。是的，一个*输出*可以代表一个临时缓冲区。考虑一下，一个输出缓冲区具有运算可以写入的属性，并且可以在属性被写入后从缓冲区中读取。这正是您想要的临时缓冲区。

在上面的示例中，假设我们要使用 `F32[1024]` 作为临时缓冲区。随后，我们将像上面那样写入 HLO，这样便永远不会读取自定义调用输出的元组索引 1。

### CPU 自定义调用中的元组

在 CPU 代码中，我们有一个函数 `do_custom_call(const void** ins, void* out)`。`ins` 是一个只包含一个元素的数组，此元素指向 `param0`。通过解除引用该指针可以访问 `param0` 的子缓冲区，而通过解除引用 `out` 则可以访问 `output_tuple` 的子缓冲区。

### GPU 自定义调用中的元组

在 GPU 代码中，我们有一个函数 `do_custom_call(..., void** buffers, ...)`。在这种情况下，`buffers` 是一个包含*六个*设备指针的主机数组，每个指针对应输入/输出中的每个叶缓冲区。为了生成扁平列表，我们遍历参数和输出，并对每个参数的形状执行先序遍历。具体代码如下：

```c++
// Layout of `buffers` parameter to GPU custom call function for custom-call
// above.
buffers[0] == subbuf0
buffers[1] == subbuf1
buffers[2] == subbuf2
buffers[3] == subbuf3
buffers[4] == output_subbuf0
buffers[5] == output_subbuf1
```

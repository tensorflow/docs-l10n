# XLA 사용자 정의 호출

This document describes how to write and use XLA "custom calls". Custom calls let you invoke code written in a programming language like C++ or CUDA from an XLA program.

Warning: Custom calls are a low-level power-user feature. It is easy to break your program in difficult-to-debug (and even difficult-to-notice) ways using custom-calls. You shouldn't use custom calls unless you're prepared to debug XLA yourself when something goes wrong, and you should expect relatively less assistance from XLA developers if you run into trouble.

Warning: The custom-call API/ABI is not currently stable. We don't intend to change it capriciously, but it may change. Some possible future changes are described below.

## CPU에서 사용자 정의 호출

You can create an HLO instruction which represents a custom-call via XLA's client API. This is not exposed via TensorFlow as of writing.

예를 들어, 다음 코드는 사용자 정의 호출을 사용하여 CPU에서 `A[i] = B[i % 128] + C[i]`를 계산합니다(물론, 일반 HLO를 사용하여 이 작업을 수행할 수 있으며, 그렇게 해야 합니다!).

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

Notice that the function `do_custom_call` needs to know the dimensions of the buffers it operates over. In this example we hardcode the sizes 128 and 2048. If you don't want to do this, you can pass the dimensions in as parameters to the call.

## GPU에서 사용자 정의 호출

The GPU custom call framework is somewhat different than that on the CPU. Here is a CUDA example that does the same `A[i] = B[i % 128] + C[i]` computation as the CPU code above.

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

  const int64 block_dim = 64;
  const int64 grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
```

먼저, GPU 사용자 정의 호출 함수는 *여전히 CPU에서 실행되는 함수*라는 점을 알아야 합니다. `do_custom_call` CPU 함수는 GPU에서 작업을 대기열에 추가하는 역할을 합니다. 여기에서는 CUDA 커널을 시작하지만 cublas 호출과 같은 다른 작업을 수행할 수도 있습니다.

`buffers` is an array of pointers which lives on the host, and each element it contains points to device (i.e. GPU) memory. The parameters come first, followed by the output value. This is notably different from the CPU calling convention, which has two params, `ins` and `out`. The main reason we diverge is to make it possible to handle tuple-shaped inputs/outputs efficiently; see the section below.

As in the CPU example, we've hardcoded the input and output buffer sizes into our custom call. However unlike in the CPU case, passing the buffer sizes in as operands to the custom call would not work well. Usually we need the buffer sizes available to us on the CPU; e.g. when launching a kernel, we need to know the block/grid dimensions to use. But if we were to pass the buffer sizes as operands to our custom call, their values would live in GPU memory. We'd then have to do an expensive synchronous device-to-host memcpy at the start of our operation just to read the sizes.

To let you work around this, we provide the `opaque` parameter. You can set this to an arbitrary string of bytes when you create the custom call:

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

`xla::Shape`에는 프로토콜 버퍼 표현이 있으므로 이 직렬화된 proto를 `opaque` 내부에 저장하고 GPU 사용자 정의 호출 내에서 역직렬화할 수 있습니다. 하지만 `xla::ShapeProto`가 자주 변경되지는 않지만 변경이 되기는 *한다*는 점을 알아야 합니다. git 로그를 확인하여 과거에 어떻게 변경되었는지 확인하세요.

## Signalling an error.

If your custom call encounters an error, you can signal the error to the XLA runtime (instead of e.g. crashing or returning nonsense in the output buffers) by using the following signature for your function on CPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

... and on GPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

You can signal failure by using `XlaCustomCallStatusSetFailure`, e.g.:

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

You can also use `XlaCustomCallStatusSetSuccess` to indicate success, but the `XlaCustomCallStatus` is in a success state by default, so ignoring it completely will also indicate success.

When using custom call functions with this signature, you must create the corresponding `custom-call` op with the appropriate API version set, e.g.:

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

NOTE: In the future all clients will be required to migrate their custom call functions to the new API version and the old one will be deprecated. For custom calls that can't fail, you can simply add the new `XlaCustomCallStatus*` parameter and then ignore it.

On failure, none of the custom call outputs will be used; the XLA runtime will terminate the computation. It is not possible for an HLO computation to recover from the error (e.g. by catching and handling it).

## 사용자 정의 호출에 튜플 전달하기

다음 사용자 정의 호출을 고려합니다.

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

CPU와 GPU 모두에서 튜플은 메모리에서 포인터의 배열로 표현됩니다. C++ 의사 코드에서 위의 매개변수 0은 다음과 같이 배치됩니다.

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

튜플의 메모리 내 표현은 CPU와 GPU에서 동일하지만 CPU 및 GPU 사용자 정의 호출의 호출 규칙에서 다르게 처리됩니다.

### Tuple outputs as temp buffers

사용자 정의 호출에 대한 튜플 입력은 편리하지만 반드시 필요한 것은 아닙니다. 사용자 정의 호출에 대한 튜플 입력을 지원하지 않는 경우, 사용자 정의 호출에 전달하기 전에 항상 get-tuple-element를 사용하여 튜플을 압축 해제할 수 있습니다.

한편, 튜플 *출력*은 다른 방법으로는 할 수 없었던 일을 가능하게 해줍니다.

튜플 출력을 갖는 분명한 이유는 이것이 사용자 정의 호출(또는 다른 XLA op)이 여러 독립 배열을 반환하는 방식이기 때문입니다.

이보다는 덜 분명하지만 또 다른 이유로, 튜플 출력은 사용자 정의 호출의 임시 메모리를 제공하는 방법이기도 합니다. *출력*은 임시 버퍼를 나타낼 수 있습니다. 출력 버퍼에는 op가 쓸 수 있는 속성이 있고 쓰여진 후에는 읽을 수 있다는 것을 고려하세요. 이것이 바로 임시 버퍼에서 우리가 바라는 것입니다.

위의 예에서 `F32[1024]`를 임시 버퍼로 사용한다고 가정합니다. 그런 다음, 위와 같이 HLO를 작성하고 사용자 정의 호출 출력의 튜플 인덱스 1을 읽지 않습니다.

### CPU 사용자 정의 호출의 튜플

CPU 코드에는 `do_custom_call(const void** ins, void* out)` 함수가 있습니다. `ins`는 `param0`을 가리키는 하나의 요소만 있는 배열입니다. `param0`의 서브 버퍼는 해당 포인터를 역참조하여 액세스할 수 있으며 `output_tuple`의 서브 버퍼는 `out`을 역참조하여 액세스할 수 있습니다.

### GPU 사용자 정의 호출의 튜플

GPU 코드에는 `do_custom_call(..., void** buffers, ...)` 함수가 있습니다. 이 경우 `buffers`는 입력/출력의 각 리프 버퍼에 대해 하나씩 *6개*의 기기 포인터로 구성된 호스트 배열입니다. 플랫 목록을 생성하기 위해 매개변수와 출력을 반복하고 각각에 대해 형상의 전위 운행(preorder traversal)을 수행합니다. 구체적으로, 다음과 같습니다.

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

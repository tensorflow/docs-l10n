# XLA 사용자 정의 호출

본 문서에는 XLA "사용자 정의 호출"을 작성하고 사용하는 방법이 설명되어 있습니다. 사용자 정의 호출을 사용하면 XLA 프로그램에서 C++ 또는 CUDA와 같은 프로그래밍 언어로 작성된 코드를 호출할 수 있습니다.

경고: 사용자 정의 호출은 낮은 수준의 고급 사용자를 위한 기능입니다. 사용자 정의 호출을 사용하면 디버깅 또는 심지어 알리기 어려운 방식으로 프로그램을 중단하기 쉽습니다. 따라서 문제가 발생했을 때 XLA를 직접 디버깅할 준비가 되어 있지 않다면 사용자 정의 호출을 사용하지 않아야 하며 실제로 문제가 발생했을 때 XLA 개발자로부터 받을 수 있는 지원도 상대적으로 제한된다는 것을 예상해야 합니다.

경고: 사용자 정의 호출 API/ABI는 아직 안정적이지 않습니다. 변덕스럽게 바꿀 의도는 없지만 바뀔 수는 있습니다. 가능한 향후 일부 변경 사항은 아래에 설명되어 있습니다.

## CPU에서 사용자 정의 호출

XLA의 클라이언트 API를 통해 사용자 정의 호출을 나타내는 HLO 명령을 생성할 수 있습니다. 이것은 작성 시점에서 TensorFlow를 통해 노출되지 않습니다.

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

`do_custom_call` 함수는 연산이 이루어지는 버퍼의 크기를 알아야 합니다. 이 예에서는 크기 128과 2048을 하드 코딩합니다. 이렇게 하지 않으려면 크기를 매개변수로 호출에 전달할 수 있습니다.

## GPU에서 사용자 정의 호출

GPU 사용자 정의 호출 프레임워크는 CPU의 프레임워크와 약간 다릅니다. 다음은 위의 CPU 코드와 동일한 `A[i] = B[i % 128] + C[i]` 계산을 수행하는 CUDA 예입니다.

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

`buffers`는 호스트에 있는 포인터의 배열이며, 여기에 포함된 각 요소는 장치(즉, GPU) 메모리를 가리킵니다. 매개변수가 먼저 오고 출력 값이 뒤따릅니다. 이것은 두 개의 매개변수 `ins` 및 `out`이 있는 CPU 호출 규칙과는 확연하게 다릅니다. 분기하는 주된 이유는 튜플 형상의 입력/출력을 효율적으로 처리할 수 있도록 하기 위해서입니다. 아래 섹션을 참조하세요.

CPU 예에서와 같이, 입력 및 출력 버퍼 크기를 사용자 정의 호출에 하드 코딩했습니다. 그러나, CPU의 경우와 달리 버퍼 크기를 피연산자로 사용자 지정 호출에 전달하면 제대로 동작하지 않습니다. 일반적으로, CPU에서 사용할 수 있는 버퍼 크기가 필요합니다. 예를 들어, 커널을 시작할 때 사용할 블록/그리드 크기를 알아야 합니다. 그러나 버퍼 크기를 사용자 지정 호출에 피연산자로 전달하면 해당 값이 GPU 메모리에 저장됩니다. 그러면 연산을 시작할 때 단순히 크기를 읽기 위해 값 비싼 동기식 장치-호스트 memcpy를 수행해야 합니다.

이 문제를 해결할 수 있도록 `opaque` 매개변수를 제공합니다. 사용자 정의 호출을 생성할 때 이를 임의의 바이트 문자열로 설정할 수 있습니다.

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

`xla::Shape`에는 프로토콜 버퍼 표현이 있으므로 이 직렬화된 proto를 `opaque` 내부에 저장하고 GPU 사용자 정의 호출 내에서 역직렬화할 수 있습니다. 하지만 `xla::ShapeProto`가 자주 변경되지는 않지만 변경이 되기는 *한다*는 점을 알아야 합니다. git 로그를 확인하여 과거에 어떻게 변경되었는지 확인하세요.

## 오류 신호 보내기.

사용자 정의 호출에 오류가 발생하면 CPU의 함수에 대해 다음 서명을 사용하여 예를 들어, 충돌 또는 출력 버퍼에서 넌센스를 반환하는 대신 XLA 런타임에 오류 신호를 보낼 수 있습니다.

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

...그리고 GPU의 경우:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

`XlaCustomCallStatusSetFailure`를 사용하여 실패 신호를 보낼 수 있습니다. 예:

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

`XlaCustomCallStatusSetSuccess`를 사용하여 성공 여부를 표시할 수도 있지만 `XlaCustomCallStatus`는 기본으로 성공 상태에 있으므로 이를 무시하면 완전히 성공으로 표시됩니다.

이 서명과 함께 사용자 정의 호출 기능을 사용할 때 적절한 API 버전 세트로 해당하는 `custom-call` op를 생성해야 합니다. 예:

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

참고: 앞으로 모든 클라이언트는 사용자 정의 호출 기능을 새 API 버전으로 마이그레이션해야 하며 이전 버전은 더 이상 사용되지 않습니다. 실패할 수 없는 사용자 지정 호출의 경우 새 `XlaCustomCallStatus*` 매개변수를 추가한 다음 무시하면 됩니다.

실패 시 사용자 정의 호출 출력이 사용되지 않습니다. XLA 런타임은 계산을 종료합니다. HLO 계산을 수행하는 동안 예를 들어, 포착 및 처리하여 오류로부터 복구하는 것은 불가능합니다.

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

### 튜플 출력을 임시 버퍼로 사용

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

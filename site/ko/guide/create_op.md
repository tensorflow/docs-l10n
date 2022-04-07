# op 만들기

참고: C++ 사용자 정의 ops의 ABI가 TensorFlow의 공식 pip 패키지와 호환되도록 하려면 [사용자 정의 op 리포지토리](https://github.com/tensorflow/custom-op)의 가이드를 따르세요. 가이드에는 엔드 투 엔드 코드 예제와 사용자 지정 ops를 작성 및 배포하기 위한 Docker 이미지가 있습니다.

기존 TensorFlow 라이브러리에 포함되지 않는 op를 만들려면 먼저 기존 Python ops 또는 함수의 구성으로 op를 Python으로 작성하는 것이 좋습니다. 가능하지 않다면, 사용자 정의 C++ op를 작성할 수 있습니다. 사용자 정의 C++ op를 작성하는 몇 가지 이유는 다음과 같습니다.

- 기존 ops의 구성으로 작업을 표현하는 것은 쉽지 않거나 불가능합니다.
- 기존 프리미티브의 구성으로 연산을 표현하는 것이 비효율적일 때
- 사용자가 미래의 컴파일러에서 융합이 어려운 프리미티브의 구성을 수동으로 융합하려 할 때

예를 들어, "MaxPool" 연산자와 비슷한 "중앙값 풀링"과 같은 연산을 구현할 때 최대값 대신 슬라이딩 윈도우에 대해 중앙값을 계산한다고 가정합니다. 연산의 구성을 사용하여 이 연산을 수행할 수 있지만(예: ExtractImagePatches 및 TopK 사용), 단일 융합 연산으로 더 똑똑한 연산을 수행할 수 있는 네이티브 연산보다는 성능 또는 메모리의 효율성이 떨어질 수 있습니다. 항상 그렇듯이, 일반적으로 연산자 구성을 사용하여 원하는 것을 표현해 볼 만한데, 가장 어렵고 비효율적일 경우에만 새 연산을 추가하도록 선택하는 것이 좋습니다.

사용자 정의 op를 통합하려면 다음을 수행해야 합니다.

1. C++ 파일에 새 op를 등록합니다. Op 등록에서 op의 구현과는 독립적인 op 기능에 대한 인터페이스(사양)를 정의합니다. 예를 들어,  op 등록에서 op의 이름과 op의 입력 및 출력을 정의합니다. 또한, 텐서 형상 유추에 사용되는 형상 함수를 정의합니다.
2. C++로 op를 구현합니다. op의 구현을 커널이라고 하며 1단계에서 등록한 사양의 구체적인 구현입니다. 다양한 입력/출력 유형 또는 아키텍처(예: CPU, GPU)를 위한 커널이 여러 개 있을 수 있습니다.
3. Python 래퍼를 만듭니다(선택 사항). 이 래퍼는 Python에서 op를 만드는 데 사용되는 공개 API입니다. 기본 래퍼는 op 등록에서 생성되며 직접 사용하거나 추가할 수 있습니다.
4. op의 그래디언트를 계산하는 함수를 작성합니다(선택 사항).
5. op를 테스트합니다. 보통 편의를 위해 Python에서 이 연산을 테스트하지만, C++에서 op를 테스트할 수도 있습니다. 그래디언트를 정의하면 Python `tf.test.compute_gradient_error`을 사용하여 확인할 수 있습니다. Relu 같은 연산자의 전달 함수와 그래디언트를 테스트하는 예제는 [`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py)를 참조하세요.

### 전제 조건

- C++에 어느 정도 익숙해야 합니다.
- [TensorFlow 바이너리](../../install)를 설치했거나 [TensorFlow 소스](../../install/source.md)를 다운로드하여 빌드할 수 있어야 합니다.

## op 인터페이스 정의하기

op를 TensorFlow 시스템에 등록하여 op의 인터페이스를 정의합니다. 등록 시 op의 이름, 해당 입력(유형 및 이름) 및 출력(유형 및 이름), 그리고 docstrings 및 op에 필요한 [attrs](#attrs)를 지정합니다.

작동 원리를 알아보기 위해 `int32`의 텐서를 가져와서 첫 번째 요소를 제외한 모든 요소를 ​​0으로 설정하여 텐서의 복사본을 출력하는 op를 만든다고 가정합니다. 그렇게 하려면, `zero_out.cc`이라는 파일을 작성합니다. 그런 다음, op의 인터페이스를 정의하는 `REGISTER_OP` 매크로에 대한 호출을 추가합니다.

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

이 `ZeroOut` op는 32-bit 정수의 텐서 `to_zero` 하나를 입력으로 사용하고 32-bit 정수의 텐서 `zeroed`를 출력합니다. 또한, op는 형상 함수를 사용하여 출력 텐서가 입력 텐서와 같은 형상이 되도록 합니다. 예를 들어, 입력이 형상[10, 20]의 텐서인 경우, 이 형상 함수는 출력 형상도 [10, 20]로 지정합니다.

참고: op 이름은 CamelCase여야 하며 바이너리에 등록된 다른 모든 op 중에서 고유해야 합니다.

## op의 커널 구현하기

인터페이스를 정의한 후, 하나 이상의 op 구현을 제공합니다. 이들 커널 중 하나를 작성하려면, `OpKernel`을 확장하여 `Compute` 메서드를 대체하는 클래스를 작성합니다. `Compute` 메서드는 유형 `OpKernelContext*`의 `context` 인수를 하나 제공하며, 이 인수에서 입력 및 출력 텐서와 같은 유용한 항목에 액세스할 수 있습니다.

위에서 만든 파일에 커널을 추가합니다. 커널은 다음과 같을 수 있습니다.

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

커널을 구현한 후에는 TensorFlow 시스템에 커널을 등록합니다. 등록 시 이 커널이 실행될 다른 제약 조건을 지정합니다. 예를 들어, CPU용 커널 하나와 GPU용 커널 하나가 있을 수 있습니다.

`ZeroOut` op용 커널을 구현하려면, `zero_out.cc`에 다음을 추가합니다.

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 중요: OpKernel 인스턴스에 동시에 액세스할 수 있습니다. `Compute` 메서드는 스레드로부터 안전해야 합니다. 뮤텍스를 사용하여 클래스 멤버에 대한 액세스를 보호하세요. 또는 더 나은 방법으로, 클래스 멤버를 통해 상태를 공유하지 마세요! op 상태를 추적하기 위해 [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h)를 사용하는 것이 좋습니다.

### 다중 스레드 CPU 커널

다중 스레드 CPU 커널을 작성하기 위해 [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h)의 Shard 함수를 사용할 수 있습니다. 이 함수는 intra-op 스레딩에 사용되도록 구성된 스레드 간에 계산 함수를 분할합니다([`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)의 intra_op_parallelism_threads 참조).

### GPU 커널

GPU 커널은 OpKernel 및 CUDA 커널과 시작 코드의 두 부분으로 구현됩니다.

입력 검사 및 출력 할당과 같이 CPU와 GPU 커널 간에 OpKernel 구현이 공통적으로 사용되는 경우가 있습니다. 이 경우, 제안 구현은 다음과 같습니다.

1. Device 템플릿 형식의 OpKernel과 텐서의 기본 유형을 정의합니다.
2. 출력의 실제 계산을 수행하기 위해 Compute 함수에서 템플릿 형식의 functor 구조체를 호출합니다.
3. CPUDevice에 대한 해당 functor의 전문화는 같은 파일에 정의되어 있지만, GPUDevice에 대한 전문화는 CUDA 컴파일러로 컴파일되므로 .cu.cc 파일에 정의되어 있습니다.

다음은 구현 예입니다.

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
#include "kernel_example.h"
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

## op 라이브러리 빌드하기

### 시스템 컴파일러를 사용하여 op 컴파일하기(TensorFlow 바이너리 설치)

시스템에서 사용 가능한 `g++` 또는 `clang`과 같은 `C++` 컴파일러로 `zero_out.cc`를 컴파일할 수 있습니다. 이진 PIP 패키지는 시스템의 특정 위치에 op를 컴파일하는 데 필요한 헤더 파일과 라이브러리를 설치합니다. 하지만, TensorFlow Python 라이브러리는 헤더 디렉토리를 가져오는 `get_include` 함수를 제공하며, `get_lib` 디렉토리에는 링크할 공유 객체가 있습니다. Ubuntu 머신에서 이들 함수의 출력은 다음과 같습니다.

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/tensorflow'
```

`g++`를 설치했다고 가정하면, 다음은 op를 동적 라이브러리로 컴파일하는 데 사용할 수 있는 명령 시퀀스입니다.

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

macOS에서는 `.so` 파일을 빌드할 때 추가 플래그 "-undefined dynamic_lookup"이 필요합니다.

> `gcc` 버전 `>=5`에 대한 참고 사항: gcc는 버전 `5`부터 새로운 [C++ ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx)를 사용합니다. TensorFlow 웹 사이트에서 사용 가능한 2진 pip 패키지는 이전 ABI를 사용하는 `gcc4`로 빌드되었습니다. `gcc>=5`로 op 라이브러리를 컴파일하는 경우, 명령줄에 `-D_GLIBCXX_USE_CXX11_ABI=0`을 추가하여 라이브러리가 이전 abi와 호환되도록 하세요.

### bazel(TensorFlow 소스 설치)을 사용하여 op 컴파일하기

TensorFlow 소스가 설치되어 있으면, TensorFlow의 빌드 시스템을 사용하여 op를 컴파일할 수 있습니다. [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) 디렉토리에 다음 Bazel 빌드 규칙을 가진 BUILD 파일을 저장합니다.

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

다음 명령을 실행하여 `zero_out.so`를 빌드합니다.

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

CUDA 커널을 사용하여 `Example` 연산을 컴파일하려면 `tf_custom_op_library`의 `gpu_srcs` 매개변수를 사용해야 합니다. 다음 Bazel 빌드 규칙이 있는 BUILD 파일을 [`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) 디렉터리(예: "example_gpu") 내의 새 폴더에 배치합니다.

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

다음 명령을 실행하여 `kernel_example.so`를 빌드합니다.

```bash
$ bazel build --config opt //tensorflow/core/user_ops/example_gpu:kernel_example.so
```

참고: 위에서 설명한 대로 gcc&gt;=5로 컴파일하는 경우, Bazel 명령줄 인수에 `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`을 추가합니다.

> 참고: 표준 `cc_library` 규칙을 사용하여 공유 라이브러리(`.so` 파일)를 만들 수 있지만, `tf_custom_op_library` 매크로를 사용하는 것이 좋습니다. 이 매크로는 필수 종속성을 추가하고 공유 라이브러리가 TensorFlow의 플러그인 로딩 메커니즘과 호환되는지 확인합니다.

## Python에서 op 사용하기

TensorFlow Python API는 `tf.load_op_library` 함수를 제공하여 동적 라이브러리를 로드하고 TensorFlow 프레임워크에 op를 등록합니다. `load_op_library`는 op 및 커널에 대한 Python 래퍼가 포함된 Python 모듈을 반환합니다. 따라서, 일단 op를 빌드하면 다음을 수행하여 Python에서 실행할 수 있습니다.

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```

생성된 함수에는 snake_case 이름이 지정됩니다([PEP8](https://www.python.org/dev/peps/pep-0008/) 준수). 따라서, C++ 파일에서 op의 이름이 `ZeroOut`인 경우, Python 함수의 이름은 `zero_out`입니다.

Python 모듈에서 op를 정규 함수로 `import` 가능하게 하려면, 다음과 같이 Python 소스 파일에 `load_op_library` 호출을 포함하는 것이 유용할 수 있습니다.

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## op가 작동하는지 확인하기

op를 성공적으로 구현했는지 확인하는 좋은 방법은 테스트를 작성하는 것입니다. 다음 내용으로 `zero_out_op_test.py` 파일을 작성합니다.

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

그런 다음 테스트를 실행합니다(tensorflow가 설치되었다고 가정).

```sh
$ python zero_out_op_test.py
```

## Op에 고급 특성 빌드하기

기본 (그리고, 다소 제한적인) op 및 구현을 빌드하는 방법을 살펴보았으므로 일반적으로 op에 빌드하는 데 필요한 조금 더 복잡한 항목을 살펴보겠습니다. 여기에는 다음이 포함됩니다.

- [조건부 검사 및 확인](#conditional-checks-and-validation)
- [Op 등록](#op-registration)
    - [Attrs](#attrs)
    - [Attr 유형](#attr-types)
    - [다형성](#polymorphism)
    - [입력 및 출력](#inputs-and-outputs)
    - [이전 버전과의 호환성](#backwards-compatibility)
- [GPU 지원](#gpu-support)
    - [GPU 기기용 커널 컴파일하기](#compiling-the-kernel-for-the-gpu-device)
- [Python에서 그래디언트 구현하기](#implement-the-gradient-in-python)
- [C++의 형상 함수](#shape-functions-in-c)

### 조건부 검사 및 확인

위의 예제에서는 op가 모든 형상의 텐서에 적용되었다고 가정했습니다. 벡터에만 적용된 경우는 어떻게 해야 할까요? 위의 OpKernel 구현에 검사를 추가해야 합니다.

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

입력이 벡터임을 인증하는 내용이며, 그렇지 않은 경우, `InvalidArgument` 상태를 설정하여 반환합니다. [`OP_REQUIRES` 매크로](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)는 세 가지 인수를 사용합니다.

- `context`는 `SetStatus()` 메서드에 대한 `OpKernelContext` 또는 `OpKernelConstruction` 포인터([`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) 참조)일 수 있습니다.
- 조건. 예를 들어, [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)에 텐서의 형상을 확인하는 함수가 있습니다.
- `Status` 객체로 표시되는 오류 자체는 [`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h)를 참조하세요. `Status`에는 유형(종종 `InvalidArgument`이지만, 유형의 목록 참조)과 메시지가 있습니다. 오류 생성 함수는 [`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)에서 찾을 수 있습니다.

일부 함수에서 반환된 `Status` 객체가 오류인지 테스트하려는 경우, [`OP_REQUIRES_OK`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h)를 사용합니다. 이 두 매크로는 모두 오류 시 함수로부터 반환합니다.

### Op 등록

#### Attrs

Ops는 attr을 가질 수 있으며, op가 그래프에 추가될 때 값이 설정됩니다. 이들 값은 op를 구성하는 데 사용되며 커널 구현 내에서, 그리고 op 등록에서 입력 및 출력 유형으로 해당 값에 액세스할 수 있습니다. 입력이 더 유연하기 때문에 가능하면 attr 대신 입력을 사용하는 것이 좋습니다. attrs는 상수이고 그래프 생성 시 정의해야 하기 때문입니다. 반면에, 입력은 값이 동적일 수 있는 텐서입니다. 즉, 입력은 단계마다 변할 수 있고 피드를 사용하여 설정할 수 있습니다. Attrs은 서명(입력 또는 출력의 수 또는 유형)에 영향을 미치거나 단계별로 변경할 수 없는 구성과 같이 입력으로 구성할 수 없는 연산에 사용됩니다.

op를 등록할 때 `Attr` 메서드를 사용하여 op의 이름과 유형을 지정함으로써 attr를 정의합니다. 다음 형식의 사양이 필요합니다.

```
<name>: <attr-type-expr>
```

`<name>`은 문자로 시작하고 영숫자와 밑줄로 구성될 수 있으며, `<attr-type-expr>`은 [아래 설명](#attr-types)된 형식의 유형 표현식입니다.

예를 들어, `ZeroOut` op가 0번째 요소만이 아닌 사용자 지정 인덱스를 유지하도록 하려면 op를 다음과 같이 등록할 수 있습니다.

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

([속성 유형](#attr-types)의 집합은 입력 및 출력에 사용되는 `tf.DType`과는 다릅니다.)

커널은 `context` 매개변수를 통해 생성자에서 이 attr에 액세스할 수 있습니다.

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

그런 다음 `Compute` 메서드에서 사용할 수 있습니다.

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

#### Attr 유형

다음 유형이 attr에서 지원됩니다.

- `string`: 바이트 시퀀스(UTF8일 필요는 없음)
- `int`: 부호 있는 정수
- `float`: 부동 소수점 숫자
- `bool`: 참 또는 거짓
- `type`: [`DataType`](https://www.tensorflow.org/code/tensorflow/core/framework/types.cc)의 (비참조) 값 중 하나
- `shape`: [`TensorShapeProto`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto)
- `list(<type>)`: `<type>`의 목록, `<type>`은 위의 유형 중 하나입니다. `list(list(<type>))`는 유효하지 않습니다.

명확한 목록은 [`op_def_builder.cc:FinalizeAttr`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc)을 참조하세요.

##### 기본값 및 제약 조건

Attrs는 기본값을 가질 수 있으며, attrs의 일부 유형에는 제약 조건이 있을 수 있습니다. 제약 조건이 있는 attr을 정의하려면, 다음 `<attr-type-expr>`을 사용할 수 있습니다.

`{'<string1>', '<string2>'}`: 값은 `<string1>` 또는 `<string2>` 값을 가진 문자열이어야 합니다. 이 구문을 사용하면 유형 `string`의 이름이 포함됩니다. 열거형을 에뮬레이트합니다.

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

`{<type1>, <type2>}`: 값은 유형 `type`이며, `<type1>` 또는 `<type2>` 중 하나여야 합니다. `<type1>` 및 `<type2>`는 `tf.DType`을 지원합니다. attr의 유형이 `type`임을 지정하지 않았습니다. `{...}`에 유형의 목록이 있을 때 암시됩니다. 예를 들어, 이 경우 attr `t`의 유형은 `int32`, `float` 또는 `bool`이어야 합니다.

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

다음은 일반적인 유형 제약 조건에 대한 바로 가기입니다.

- `numbertype`: 유형 `type`은 숫자(문자열도 부울도 아닌) 유형으로 제한됩니다.
- `realnumbertype`: 복잡한 유형이 없는 ` numbertype`과 유사합니다.
- `quantizedtype`: `numbertype`과 유사하지만, 양자화된 숫자 유형과 같습니다.

이들 제약 조건에서 허용되는 유형의 특정 목록은[`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h)에서 함수(예: `NumberTypes()`)로 정의됩니다. 이 예제에서 attr `t`는 숫자 유형 중 하나여야 합니다.

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

다음 op의 경우:

```python
tf.number_type(t=tf.int32)  # Valid
tf.number_type(t=tf.bool)   # Invalid
```

목록은 다른 목록 및 단일 유형과 결합될 수 있습니다. 다음 op에서는 attr `t`가 숫자 유형이거나 부울 유형일 수 있습니다.

```c++
REGISTER_OP("NumberOrBooleanType")
    .Attr("t: {numbertype, bool}");
```

다음 op의 경우:

```python
tf.number_or_boolean_type(t=tf.int32)  # Valid
tf.number_or_boolean_type(t=tf.bool)   # Valid
tf.number_or_boolean_type(t=tf.string) # Invalid
```

`int >= <n>`: 값은 `<n>`보다 크거나 같은 정수여야 합니다. `<n>`는 자연수입니다. 예를 들어, 다음 op 등록에서 attr `a`의 값은 `2` 이상이어야 함을 지정합니다.

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

`list(<type>) >= <n>`: 길이가 `<n>` 이상인 유형 `<type>`의 목록입니다. 예를 들어, 다음 op 등록에서 attr `a`은 유형 (`int32` 또는 `float`)의 목록이며, 적어도 3개 이상 있어야 함을 지정합니다.

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

attr의 기본값을 설정하려면(생성된 코드에서 선택 사항), 다음과 같이 끝에 `= <default>`를 추가합니다.

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

또한, 제약 조건과 기본값을 모두 지정할 수 있습니다.

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

지원되는 기본값 구문은 최종 GraphDef 정의의 프로토타입 표현에 사용되는 구문입니다.

다음은 모든 유형의 기본값을 지정하는 방법에 대한 예제입니다.

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

특히, 유형 `type`의 값은 `tf.DType`을 사용합니다.

#### 다형성

##### 유형 다형성

다른 유형을 입력으로 사용하거나 다른 출력 유형을 생성할 수 있는 op의 경우, op 등록에서 [입력 또는 출력 유형](#inputs-and-outputs)에 [attr](#attrs)을 지정할 수 있습니다. 일반적으로, 지원되는 각 유형에 대해 `OpKernel`을 등록합니다.

예를 들어, `int32` 이외에 `float`에 대해 `ZeroOut` op가 작동하게 하려면 op 등록은 다음과 같을 수 있습니다.

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

op 등록에서 이제 입력의 유형이 `float` 또는 `int32`여야 함을 지정합니다. 입력과 출력 유형이 모두 `T`이므로 출력의 유형도 같습니다.

###### 명명

입력, 출력 및 attrs에는 일반적으로 snake_case 이름이 지정되어야 합니다. 한 가지 예외는 입력의 유형 또는 출력의 유형으로 사용되는 attrs입니다. 이러한 attrs는 op가 그래프에 추가될 때 유추될 수 있으므로 op의 함수에는 나타나지 않습니다. 예를 들어, 이 ZeroOut의 최종 정의는 다음과 같은 Python 함수를 생성합니다.

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

`to_zero`에 `int32` 텐서가 전달되면, `T`는 자동으로 `int32`로 설정됩니다(실제로 `DT_INT32`). 유추된 attrs에는 대문자 또는 CamelCase 이름이 지정됩니다.

유추된 attrs를 출력 유형을 결정하는 유형 attr이 있는 op와 비교합니다.

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

이 경우, 사용자는 생성된 Python에서와 같이 출력 유형을 지정해야 합니다.

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

###### 유형 다형성 예제

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

[이전 버전과의 호환성](#backwards-compatibility)을 유지하려면, 기존 op에 attr을 추가할 때 [기본값](#default-values-and-constraints)을 지정해야 합니다.

```c++
REGISTER_OP("ZeroOut")
  .Attr("T: {float, int32} = DT_INT32")
  .Input("to_zero: T")
  .Output("zeroed: T")
```

더 많은 유형을 추가하고 싶다고 가정해 봅시다. 예: `double`

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

위와 같이 중복 코드로 또 다른 `OpKernel`을 작성하는 대신, 종종 C++ 템플릿을 사용할 수 있습니다. 오버로드당 여전히 하나의 커널 등록(`REGISTER_KERNEL_BUILDER` 호출)이 있습니다.

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

오버로드가 두 개 이상인 경우, 등록을 매크로에 넣을 수 있습니다.

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

커널을 등록하려는 유형의 목록에 따라 [`tensorflow/core/framework/register_types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h)에서 제공되는 매크로를 사용할 수 있습니다.

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

##### 입력 및 출력 목록

다양한 유형을 허용하거나 생성할 수 있을 뿐만 아니라 ops는 다양한 개수의 텐서를 소비하거나 생성할 수 있습니다.

다음 예제에서, attr `T`는 유형의 *list*를 보유하고, 상기 입력 `in`과 출력 `out`으로 사용됩니다. 입력 및 출력은 해당 유형의 텐서 목록입니다(출력의 텐서 수와 유형은 입력과 출력의 유형이 모두 `T`이므로 입력과 같습니다).

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

목록에서 지정할 수 있는 유형에 제한을 둘 수도 있습니다. 이 경우, 입력은 `float` 및 `double` 텐서의 목록입니다. op는 예를 들어, 입력 유형 `(float, double, float)`을 허용하며, 이 경우 출력 유형도 `(float, double, float)`입니다.

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

목록의 모든 텐서가 같은 유형이 되도록 하려면, 다음과 같이 할 수 있습니다.

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

`int32` 텐서의 목록을 허용하고 `int` attr `N`을 사용하여 목록의 길이를 지정합니다.

[다형 유형](#type-polymorphism)으로 만들 수도 있습니다. 다음 예제에서, 입력은 유형 (`"T"`)이 같은 (하지만 지정되지는 않은) 텐서(길이 `"N"`)의 목록이며, 출력은 일치하는 유형의 단일 텐서입니다.

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

기본적으로, 텐서 목록의 최소 길이는 1입니다. [해당 attr에 대한 `">="` 제약 조건](#default-values-and-constraints)을 사용하여 해당 기본값을 변경할 수 있습니다. 다음 예제에서 입력은 `int32` 텐서가 2개 이상인 목록입니다.

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

같은 구문이 `"list(type)"` attrs에서 작동합니다.

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### 입력 및 출력

위의 내용을 요약하면, op 등록에는 여러 개의 입력과 출력이 있을 수 있습니다.

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

각 입력 또는 출력 사양의 형식은 다음과 같습니다.

```
<name>: <io-type-expr>
```

`<name>`은 문자로 시작하며 영숫자와 밑줄로 구성될 수 있습니다. `<io-type-expr>`은 다음 유형 표현식 중의 하나입니다.

- `<type>`, `<type>`은 지원되는 입력 유형입니다(예: `float`, `int32`, `string`). 특정 유형의 단일 텐서를 지정합니다.

    `tf.DType`을 참조하세요.

    ```c++
    REGISTER_OP("BuiltInTypesExample")
        .Input("integers: int32")
        .Input("complex_numbers: complex64");
    ```

- `<attr-type>`, `<attr-type>`은 유형이 `type` 또는 `list(type)`(가능한 유형 제한이 있는)인 [Attr](#attrs)의 이름입니다. 이 구문은 [다형 ops](#polymorphism)를 허용합니다.

    ```c++
    REGISTER_OP("PolymorphicSingleInput")
        .Attr("T: type")
        .Input("in: T");

    REGISTER_OP("RestrictedPolymorphicSingleInput")
        .Attr("T: {int32, int64}")
        .Input("in: T");
    ```

    유형이 `list(type)`인 attr을 참조하면 텐서 시퀀스를 받아들일 수 있습니다.

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

    출력 `out`에서 텐서의 수 및 유형은 입력 `in`에서와 같은데, 입력과 출력의 유형이 모두 `T`이기 때문입니다.

- 유형이 같은 텐서 시퀀스의 경우: `<number>*<type>`에서 `<number>`는 유형이 `int`인 [Attr](#attrs)의 이름입니다. `<type>`은 `tf.DType`이거나 유형이 `type`인 attr의  이름입니다. 첫 번째의 예로, 이 op는 `int32` 텐서의 목록을 허용합니다.

    ```c++
    REGISTER_OP("Int32SequenceExample")
        .Attr("NumTensors: int")
        .Input("in: NumTensors * int32")
    ```

    이 op는 모든 유형의 텐서 목록을 허용하는데, 이때 텐서의 유형은 모두 같습니다.

    ```c++
    REGISTER_OP("SameTypeSequenceExample")
        .Attr("NumTensors: int")
        .Attr("T: type")
        .Input("in: NumTensors * T")
    ```

- 텐서에 대한 참조: `Ref(<type>)`, `<type>`은 이전 유형 중의 하나입니다.

입력의 유형에 사용된 모든 attr가 유추됩니다. 일반적으로, 유추된 attr은 (`T` 또는 `N`과 같은) 대문자 이름을 사용합니다. 그렇지 않으면, 입력, 출력 및 attr의 이름은 함수 매개변수(예: `num_outputs`)와 같습니다. 자세한 내용은 [명명에 관한 이전 섹션](#naming)을 참조하세요.

자세한 내용은 [`tensorflow/core/framework/op_def_builder.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h)를 참조하세요.

#### 이전 버전과의 호환성

멋진 사용자 지정 op를 작성하고 다른 사용자와 공유했다고 가정하여 연산을 사용하는 행복한 고객이 있습니다. 그러나 op를 변경하고 싶습니다.

일반적으로, 기존의 확인된(checked-in) 사양에 대한 변경 사항은 이전 버전과 호환되어야 합니다. op의 사양을 변경한 후 이전 사양에서 생성된 이전의 직렬화된 `GraphDef` 프로토콜 버퍼가 손상되면 안 됩니다. `GraphDef` 호환성에 대한 자세한 내용은 [여기에 설명](./versions.md#compatibility_of_graphs_and_checkpoints)되어 있습니다.

이전 버전과의 호환성을 유지하는 몇 가지 방법이 있습니다.

1. 연산에 추가된 새 attrs에는 기본값이 정의되어 있어야 하며, 해당 기본값을 가진 op는 원래 동작이 있어야 합니다. 다형이 아닌 연산에서 다형 연산으로 변경하려면, 기본적으로 원래 서명을 유지하기 위해 새 유형 attr에 기본값을 *지정해야* 합니다. 예를 들어, 연산이 다음과 같은 경우,

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: float")
        .Output("out: float");
    ```

    다음을 사용하여 이전 버전과 호환되는 다형 연산으로 만들 수 있습니다.

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: T")
        .Output("out: T")
        .Attr("T: numerictype = DT_FLOAT");
    ```

2. attr에 대한 제약 조건을 덜 제한적으로 안전하게 만들 수 있습니다. 예를 들어, `{int32, int64}`에서 `{int32, int64, float}` 또는 `type`로 변경할 수 있습니다. 또는 `{"apple", "orange"}`에서 `{"apple", "banana", "orange"}` 또는 `string`로 변경할 수 있습니다.

3. 목록 유형의 기본값이 이전 서명과 일치하는 한 단일 입력/출력을 목록 입력/출력으로 변경할 수 있습니다.

4. 기본값이 비어 있으면 새 목록 입력/출력을 추가할 수 있습니다.

5. op 이름 앞에 프로젝트 고유의 이름을 붙여서 생성하는 모든 새로운 ops에 네임스페이스를 추가합니다. 이렇게 하면 이후 버전의 TensorFlow에 포함될 수 있는 ops와 해당 op가 충돌하지 않습니다.

6. 미리 계획하세요! op의 향후 용도를 예상합니다. 서명을 일부 변경하는 것은 호환 가능한 방식으로 수행할 수 없습니다(예: 같은 유형의 목록을 다양한 유형의 목록으로 만들기).

안전하거나 안전하지 않은 변경 사항의 전체 목록은 [`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc) 에서 찾을 수 있습니다. 이전 버전과 호환되도록 연산을 변경할 수 없는 경우, 새 의미 체계를 사용하여 새 이름으로 새 연산을 만듭니다.

또한, 이러한 변경 사항은 `GraphDef` 호환성을 유지할 수 있지만, 생성된 Python 코드는 이전 호출자와 호환되지 않는 방식으로 변경될 수 있습니다. Python API는 새로운 선택적 인수를 끝에 추가하는 것을 제외하고 이전 서명을 유지함으로써 손으로 작성한 Python 래퍼를 신중하게 변경하여 호환성을 유지할 수 있습니다. 일반적으로, 호환되지 않는 변경 사항은 TensorFlow의 주요 버전이 변경될 때만 수행될 수 있으며 <a data-md-type="raw_html" href="./versions.md#compatibility_of_graphs_and_checkpoints">`GraphDef`버전 의미 체계</a>를 준수해야 합니다.

### GPU 지원

[서로 다른 유형의 커널을 등록](#polymorphism)하는 것처럼 서로 다른 OpKernel을 구현하고 CPU 및 GPU용 커널을 각각 등록할 수 있습니다. [ `tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/)에 GPU를 지원하는 커널의 몇 가지 예가 있습니다. 일부 커널에는 `.cc` 파일의 CPU 버전, `_gpu.cu.cc`로 끝나는 파일의 GPU 버전 및 `.h` 파일에서 공통으로 공유되는 코드가 있습니다.

예를 들어, `tf.pad`는 [`tensorflow/core/kernels/pad_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc)에 GPU 커널을 제외한 모든 것이 있습니다. GPU 커널은 [`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc)에 있으며, 공유 코드는 [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h)에 정의된 템플릿 형식의 클래스입니다. 코드를 이 방식으로 구성하는 데는 두 가지 이유가 있습니다. CPU와 GPU 구현 간에 공통 코드를 공유할 수 있으며 GPU 구현을 별도의 파일에 넣어 GPU 컴파일러로만 컴파일할 수 있습니다.

`pad`의 GPU 커널 버전을 사용하더라도 CPU 메모리에 여전히 `"paddings"` 입력이 필요합니다. 입력 또는 출력이 CPU에서 유지된다는 것을 표시하려면, 커널 등록에 `HostMemory()` 호출을 추가합니다. 예를 들면, 다음과 같습니다.

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### GPU 기기용 커널 컴파일하기

CUDA 커널을 사용하여 op를 구현하는 예는 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc)를 참조하세요. `tf_custom_op_library`은 CUDA 커널(`*.cu.cc` 파일)을 포함하는 소스 파일의 목록을 지정할 수있는 `gpu_srcs` 인수를 허용합니다. TensorFlow의 바이너리 설치에서 사용하려면, CUDA 커널을 NVIDIA의 `nvcc` 컴파일러로 컴파일해야 합니다. 다음은 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) 및 [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc)를 동적으로 로드 가능한 단일 라이브러리로 컴파일하는 데 사용할 수 있는 명령 시퀀스입니다.

```bash
nvcc -std=c++14 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

위에서 생성된 `cuda_op_kernel.so`는 `tf.load_op_library` 함수를 사용하여 Python에서 평소와 같이 로드할 수 있습니다.

CUDA 라이브러리가 `/usr/local/lib64`에 설치되지 않은 경우, 위의 두 번째(g++) 명령에서 경로를 명시적으로 지정해야 합니다. 예를 들어, CUDA가 `/usr/local/cuda-8.0`에 설치되어 있는 경우, `-L /usr/local/cuda-8.0/lib64/`를 추가합니다.

참고: 일부 Linux 설정에서는 `nvcc` 컴파일 단계에 대한 추가 옵션이 필요합니다. `-D_MWAITXINTRIN_H_INCLUDED`를 `nvcc` 명령줄에 추가하여 `mwaitxintrin.h`의 오류를 방지합니다.

### Python에서 그래디언트 구현하기

ops의 그래프에서 TensorFlow는 자동 미분(역전파)을 사용하여 기존 op에 대한 그래디언트를 나타내는 새 ops를 추가합니다. 새로운 ops에 대해 자동 미분을 수행하려면, ops의 출력에 대한 그래디언트가 지정된 ops의 입력에 대한 그래디언트를 계산하는 그래디언트 함수를 등록해야 합니다.

수학적으로, op가 (y = f(x))를 계산하는 경우, 등록된 그래디언트 op는 (y)에 대한 손실 (L)의 그래디언트 (\partial L/ \partial y)를 연쇄 규칙을 통해 (x)에 대한 그래디언트 (\partial L/ \ partial x)로 변환합니다.

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

`ZeroOut`의 경우, 입력의 한 항목만 출력에 영향을 미치므로 입력에 대한 그래디언트는 "원-핫" 희소 텐서입니다. 다음과 같이 표현됩니다.

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

`tf.RegisterGradient`로 그래디언트 함수를 등록하는 방법에 대한 세부 사항은 다음과 같습니다.

- 출력이 하나인 op의 경우, 그래디언트 함수는 `tf.Operation`, `op` 및 `tf.Tensor` `grad`를 사용하고 텐서 `op.inputs[i]`, `op.outputs[i]` 및 `grad`에서 새 ops를 빌드합니다. 모든 attrs에 대한 정보는 `tf.Operation.get_attr`을 통해 찾을 수 있습니다.

- 출력이 여러 개인 op인 경우, 그래디언트 함수는 `op` 및 `grads`를 사용하고, 이때 `grads`는 각 출력에 대한 그래디언트의 목록입니다. 그래디언트 함수의 결과는 각 입력에 대한 그래디언트를 나타내는 `Tensor` 객체의 목록이어야 합니다.

- 인덱스로 사용되는 정수 입력과 같이 일부 입력에 대해 잘 정의된 그래디언트가 없는 경우, 반환되는 해당 그래디언트는 `None`이어야 합니다. 예를 들어, 부동 소수점 텐서 `x` 및 정수 인덱스 `i`를 사용하는 op의 경우, 그래디언트 함수는 `[x_grad, None]를 반환`합니다.

- op에 의미 있는 그래디언트가 없는 경우, 그래디언트를 등록할 필요가 없으며, op의 그래디언트가 필요하지 않은 한 문제 없습니다. 경우에 따라 op에 잘 정의된 그래디언트가 없어도 그래디언트 계산에 관여할 수 있습니다. 이때 `ops.NotDifferentiable`을 사용하여 자동으로 0을 뒤로 전파할 수 있습니다.

그래디언트 함수가 호출될 때 텐서 데이터 자체가 아니라 ops의 데이터 흐름 그래프만 사용할 수 있습니다. 따라서, 모든 계산은 그래프 실행 시간에 실행되도록 다른 tensorflow ops를 사용하여 수행해야 합니다.

### C++의 형상 함수

TensorFlow API에 "도형 유추"라는 특성이 있어 그래프를 실행하지 않고도 텐서 도형에 대한 정보를 제공합니다. 도형 유추는 C++ `REGISTER_OP` 선언에서 각 op 유형에 등록된 "도형 함수"에 의해 지원되며 두 가지 역할을 수행합니다. 입력의 도형이 그래프 생성 중에 호환되는지 확인하고 출력의 도형을 지정합니다.

형상 함수는 `shape_inference::InferenceContext` 클래스에 대한 연산으로 정의됩니다. 예를 들어, ZeroOut의 형상 함수에서

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output (0, c->input (0));`은 첫 번째 출력의 형상이 첫 번째 입력의 형상으로 설정되어야 함을 선언합니다. 위의 예제에서와 같이 인덱스에 의해 출력이 선택된 경우, `set_output`의 두 번째 매개변수는 `ShapeHandle` 객체여야 합니다. 기본 생성자로 빈 `ShapeHandle` 객체를 만들 수 있습니다. 인덱스 `idx`를 가진 입력에 대한 `ShapeHandle` 객체는 `c->input(idx)`로 구할 수 있습니다.

`shape_inference::UnchangedShape`와 같이 많은 ops에 적용되는 공통 형상 함수가 여러 개 있으며, [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h)에서 찾을 수 있고, 다음과 같이 사용됩니다.

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

형상 함수는 입력의 형상을 제한할 수도 있습니다. 벡터 형상 제약 조건이있는 [`ZeroOut`](#conditional-checks-and-validation)의 경우, 형상 함수는 다음과 같습니다.

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

`WithRank` 호출은 입력 형상 `c->input(0)` 이 정확히 1차원의 형상인지 확인합니다(또는 입력 형상을 알 수 없는 경우, 출력 형상은 알 수 없는 1차원의 벡터가 됨).

[입력이 여러 개인 다형](#polymorphism) op인 경우, `InferenceContext`의 멤버를 사용하여 검사할 형상의 수를 결정하고 `Merge`의 멤버를 사용하여 형상이 모두 호환되는지 확인합니다(또는 길이를 나타내는 액세스 속성과 op의 속성에 대한 액세스를 제공하는 `InferenceContext::GetAttr`).

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

형상 유추는 선택적인 특성이며 텐서의 형상은 동적으로 변할 수 있으므로 형상 함수는 모든 입력의 불완전한 형상 정보에 대해 견고해야 합니다. [`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h)의 `Merge` 메서드를 사용하면 두 가지 형상 중 하나 또는 둘 다에 완전한 정보가 없는 경우에도 호출자가 두 형상이 같음을 확인할 수 있습니다. 형상 함수는 모든 핵심 TensorFlow ops에 대해 정의되며 다양한 사용 예를 제공합니다.

`InferenceContext` 클래스에는 형상 함수 조작을 정의하는 데 사용할 수 있는 많은 함수가 있습니다. 예를 들어, 특정 차원에 `InferenceContext::Dim` 및`InferenceContext::WithValue`를 사용하는 매우 특정한 값이 있는지 확인하고, 출력 차원이 `InferenceContext::Add` 및 `InferenceContext::Multiply`를 사용하는 두 입력 차원의 합/곱임을 지정할 수 있습니다. 지정할 수 있는 다양한 형상 조작에 대해서는 `InferenceContext` 클래스를 참조하세요. 다음 예제는 첫 번째 출력의 형상을 (n, 3)으로 설정합니다. 여기에서 첫 번째 입력의 형상은 (n, ...)입니다.

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

복잡한 형상 함수가 있는 경우, 다양한 입력 형상의 조합이 예상되는 출력형상의 조합을 생성하는지 확인하는 테스트를 추가하는 것이 좋습니다.  일부 [핵심 ops 테스트](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc)에서 테스트를 작성하는 방법에 대한 예제를 볼 수 있습니다. (`INFER_OK` 및 `INFER_ERROR`의 구문이 약간 까다롭지만, 테스트에서 입력 및 출력 형상 사양을 간결하게 표현하세요. 지금은 해당 테스트의 주변 주석을 참조하여 형상 문자열 사양을 이해하세요.)

## 사용자 정의 op용 pip 패키지 빌드하기

op에 대한 `pip` 패키지를 빌드하려면, [tensorflow/custom-op](https://github.com/tensorflow/custom-op) 예제를 참조하세요. 이 가이드는 소스에서 TensorFlow를 빌드하는 대신 TensorFlow pip 패키지에서 사용자 정의 op를 빌드하는 방법을 보여줍니다.

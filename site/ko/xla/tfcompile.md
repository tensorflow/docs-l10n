# AOT 컴파일 사용하기

## tfcompile이란?

`tfcompile`은 Ahead-of-Time(AOT)이 TensorFlow 그래프를 실행 가능한 코드로 컴파일하는 독립 실행형 도구입니다. 이를 통해 총 바이너리 크기를 줄이고 일부 런타임 오버헤드를 방지할 수 있습니다. `tfcompile`은 일반적으로, 추론 그래프를 모바일 기기용 실행 코드로 컴파일하는 데 사용됩니다.

TensorFlow 그래프는 일반적으로, TensorFlow 런타임에 의해 실행됩니다. 이로 인해 그래프의 각 노드를 실행하는 데 약간의 런타임 오버헤드가 발생합니다. 또한 그래프 자체 외에도 TensorFlow 런타임용 코드를 사용할 수 있어야 하므로 총 바이너리 크기가 더 커집니다. `tfcompile`에 의해 생성된 실행 코드는 TensorFlow 런타임을 사용하지 않으며, 실제로 계산에 사용되는 커널에 대한 종속성만 가지고 있습니다.

컴파일러는 XLA 프레임워크 위에 구축됩니다. TensorFlow를 XLA 프레임워크에 연결하는 코드는 [tensorflow/compiler](https://www.tensorflow.org/code/tensorflow/compiler/) 아래에 있습니다.

## tfcompile의 역할은?

`tfcompile`은 피드 및 가져오기의 TensorFlow 개념으로 식별되는 서브 그래프를 가져와서, 해당 서브 그래프를 구현하는 함수를 생성합니다. `feeds`는 함수에 대한 입력 인수이고 `fetches`는 함수에 대한 출력 인수입니다. 모든 입력은 피드에 의해 완전히 지정되어야 합니다. 이렇게 잘라낸 서브 그래프에는 자리 표시자 또는 변수 노드가 포함될 수 없습니다. 모든 자리 표시자와 변수를 피드로 지정하는 것이 일반적이며, 그러면 결과적인 서브 그래프에 더 이상 이러한 노드가 포함되지 않습니다. 생성된 함수는 함수 서명을 내보내는 헤더 파일 및 구현을 포함하는 객체 파일과 함께 `cc_library` 패키지로 만들어집니다. 사용자가 생성된 함수를 적절하게 호출하는 코드를 작성합니다.

## tfcompile 사용하기

이 섹션에서는 TensorFlow 서브 그래프에서 `tfcompile`을 사용하여 실행 가능한 바이너리를 생성하기 위한 단계를 개략적으로 설명합니다. 단계는 다음과 같습니다.

- 1단계: 컴파일할 서브 그래프 구성하기
- 2단계: `tf_library` 빌드 매크로를 사용하여 서브 그래프 컴파일하기
- 3단계: 서브 그래프를 호출하는 코드 작성하기
- 4단계: 최종 바이너리 생성하기

### 1단계: 컴파일할 서브 그래프 구성하기

생성된 함수의 입력 및 출력 인수에 해당하는 피드와 페치를 확인합니다. 그런 다음 [`tensorflow.tf2xla.Config`](https://www.tensorflow.org/code/tensorflow/compiler/tf2xla/tf2xla.proto) proto에서 `feeds`와 `fetches`를 구성합니다.

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

### 2단계: tf_library 빌드 매크로를 사용하여 서브 그래프 컴파일하기

이 단계에서는 `tf_library` 빌드 매크로를 사용하여 그래프를 `cc_library`로 변환합니다. `cc_library`는 생성된 코드에 대한 액세스를 제공하는 헤더 파일과 함께 그래프에서 생성된 코드를 포함하는 객체 파일로 구성됩니다. `tf_library`는 `tfcompile`을 사용하여 TensorFlow 그래프를 실행 가능한 코드로 컴파일합니다.

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

> 이 예의 경우에 GraphDef proto (test_graph_tfmatmul.pb)를 생성하려면 [make_test_graphs.py](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/make_test_graphs.py)를 실행하고 --out_dir 플래그로 출력 위치를 지정합니다.

일반적인 그래프에는 훈련을 통해 학습된 가중치를 나타내는 [`Variables`](https://www.tensorflow.org/guide/variables)가 포함되지만, `tfcompile`은 `Variables`가 포함된 서브 그래프를 컴파일할 수 없습니다. [freeze_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py) 도구는 체크포인트 파일에 저장된 값을 사용하여 변수를 상수로 변환합니다. 편의를 위해 `tf_library` 매크로는 `freeze_checkpoint` 인수를 지원하며 도구를 실행합니다. 더 많은 예를 보려면 [tensorflow/compiler/aot/tests/BUILD](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/BUILD)를 참조하세요.

> 컴파일된 서브 그래프에 표시되는 상수는 생성된 코드로 직접 컴파일됩니다. 상수를 컴파일하여 넣는 대신 생성된 함수로 전달하려면, 간단히 피드로 전달하면 됩니다.

`tf_library` 빌드 매크로에 대한 자세한 내용은 [tfcompile.bzl](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile.bzl)을 참조하세요.

기본 `tfcompile` 도구에 대한 자세한 내용은 [tfcompile_main.cc](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile_main.cc)를 참조하세요.

### 3단계: 서브 그래프를 호출하는 코드 작성하기

This step uses the header file (`test_graph_tfmatmul.h`) generated by the `tf_library` build macro in the previous step to invoke the generated code. The header file is located in the `bazel-bin` directory corresponding to the build package, and is named based on the name attribute set on the `tf_library` build macro. For example, the header generated for `test_graph_tfmatmul` would be `test_graph_tfmatmul.h`. Below is an abbreviated version of what is generated. The generated file, in `bazel-bin`, contains additional useful comments.

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

The generated C++ class is called `MatMulComp` in the `foo::bar` namespace, because that was the `cpp_class` specified in the `tf_library` macro. All generated classes have a similar API, with the only difference being the methods to handle arg and result buffers. Those methods differ based on the number and types of the buffers, which were specified by the `feed` and `fetch` arguments to the `tf_library` macro.

There are three types of buffers managed within the generated class: `args` representing the inputs, `results` representing the outputs, and `temps` representing temporary buffers used internally to perform the computation. By default, each instance of the generated class allocates and manages all of these buffers for you. The `AllocMode` constructor argument may be used to change this behavior. All buffers are aligned to 64-byte boundaries.

The generated C++ class is just a wrapper around the low-level code generated by XLA.

[`tfcompile_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/tfcompile_test.cc)를 기반으로 생성된 함수를 호출하는 예:

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

### 4단계: 최종 바이너리 생성하기

이 단계에서는 2단계에서 `tf_library`가 생성한 라이브러리와 3단계에서 작성된 코드를 결합하여 최종 바이너리를 만듭니다. 다음은 `bazel` BUILD 파일의 예입니다.

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

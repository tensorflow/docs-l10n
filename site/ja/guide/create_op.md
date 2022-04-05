# 演算子を作成する

注意: C++ のカスタム演算子が TensorFlow 公式 の pip パッケージと ABI 互換になることを保証できるように、[カスタム演算子リポジトリ](https://github.com/tensorflow/custom-op)にあるガイドに従ってください。カスタム演算子をビルドして配布するための Docker のイメージのほか、エンドツーエンドのコード例も含まれています。

既存の TensorFlow ライブラリに含まれていない演算子を作成する場合は、まず既存の Python 演算子または関数の複合として、Python で演算子を作成することをお勧めします。これを行えない場合は、カスタム C++ 演算子を作成することもできます。カスタム C++ 演算子を作成するのには、いくつかの理由があります。

- 既存の演算子を合成して演算子を表現するのが容易ではないか不可能である。
- 既存のプリミティブを合成して演算子を表現するのは非効率的である。
- 将来のコンパイラーでは融合が困難と思われるプリミティブの合成を手動で融合する。

例えとして、"MaxPool" 演算子に似ていても、最大値の代わりにスライディングウィンドウで中央値を計算する「median pooling」のようなものを実装するとしましょう。複合演算子を使ってこれを行うことは可能ですが（ExtractImagePatches と TopK を使用するなど）、単一の融合演算でより賢明に実行できるネイティブ演算子ほどのパフォーマンス効率またはメモリ効率は得られません。いつも通り、まずは表現しようとしているものを演算子を組み合わせて作成し、それが困難であるか非効率であることがわかった場合にのみ、新しい演算子を追加することをお勧めします。

カスタム演算子を導入するには、次を行う必要があります。

1. C++ ファイルに新しい演算子を登録します。演算子を登録すると、演算子の機能のインターフェース（仕様）が定義されます。これは、演算子の実装に依存していません。たとえば、演算子の登録によって、演算子の名前と入出力のほか、テンソルの形状推論に使用される形状の関数も定義されます。
2. C++ で演算子を実装します。演算子の実装はカーネルとして知られており、手順 1 で登録した仕様の具象実装です。さまざまな入力/出力の型またはアーキテクチャ（CPU、GPU など）に対し複数のカーネルが存在することがあります。
3. Python のラッパーを作成します（オプション）。このラッパーは Python で演算子を作成するために使用されるパブリック API です。デフォルトのラッパーは演算子の登録によって生成されるため、それを直接使用することも追加することもできます。
4. 演算子に使用する勾配を計算する関数を記述します（オプション）。
5. 演算子をテストします。通常は便宜上、Python でテストしますが、C++ でテストすることも可能です。勾配を定義した場合は、Python の `tf.test.compute_gradient_error` を使って検証することができます。Relu に似た演算子のフォワード関数と勾配をテストする例は、[`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py) をご覧ください。

### 前提条件

- C++ にある程度精通していること
- [TensorFlow バイナリ](../../install)がインストール済みであるか、[TensorFlow のソースコードがダウンロード済み](../../install/source.md)で、ビルドできること

## 演算子のインターフェースを定義する

TensorFlow システムで演算子を登録することで、演算子のインターフェースを定義します。登録では、演算子の名前、演算子の入力（型と名前）と出力（型と名前）、および演算子が必要とする docstrings や[属性](#%E3%82%A2%E3%83%88%E3%83%AA%E3%83%93%E3%83%A5%E3%83%BC%E3%83%88)を指定します。

この仕組みを確認するには、`int32` のテンソルを取って、最初の要素以外のすべての要素をゼロに設定したテンソルのコピーを出力する演算子を作成することをお勧めします。これを行うには、`zero_out.cc` というファイルを作成し、演算子のインターフェースを定義する `REGISTER_OP` マクロの呼び出しを追加します。

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

この `ZeroOut` 演算子は、入力として 32 ビット整数のテンソル `to_zero` を取り、32 ビット整数のテンソル `zeroed` を出力します。この演算子は形状関数を使用して、出力テンソルの形状が入力テンソルの形状と同じであることを保証します。たとえば、入力が形状 [10, 20] のテンソルである場合、この形状関数は出力形状も [10, 20] であることを示します。

注意: 演算子名はキャメルケースであり、バイナリに登録されているすべての演算子の中で一意である必要があります。

## 演算子のカーネルを実装する

インターフェースを定義したら、演算子の実装を 1 つ以上提供します。これらのカーネルの 1 つを作成するには、`OpKernel` を拡張して `Compute` メソッドをオーバーライドするクラスを作成します。`Compute` メソッドは、`OpKernelContext*` 型の `context` 引数を 1 つ提供します。これを介して入力テンソルや出力テンソルなどにアクセスすることができます。

上記で作成したファイルにカーネルを追加します。 カーネルは次のように記述されます。

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

カーネルを実装したら、それを TensorFlow システムに登録します。 登録時には、このカーネルが実行する際のさなざまな制約を指定します。たとえば、CPU 向けに作成したカーネルと GPU 向けの別のカーネルがある場合があります。

これを `ZeroOut` 演算子で行うには、次のコードを `zero_out.cc` に追加します。

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 重要: OpKernelのインスタンスは、同時にアクセスされることがあるため、`Compute` メソッドはをスレッドセーフにする必要があります。クラスメンバーへのアクセスは mutex で保護してください。または、クラスメンバー経由で状態を共有しないようにする方が推奨されます！演算子の状態を追跡するために、[`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h) を使用することを検討してください。

### マルチスレッドの CPU カーネル

マルチスレッド化された CPU カーネルを書くには、[`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) にあるシャード関数を利用できます。この関数は、内部演算子スレッドに使用されるために構成されたスレッド間で計算関数をシャーディングします（[`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto) の intra_op_parallelism_threads をご覧ください）。

### GPU カーネル

GPU カーネルは、OpKernel と CUDAカーネルおよびそのローンチコードの 2 部で実装されています。

入力の検査や出力の割り当てなど、CPU と GPU カーネルにおいて、OpKernel の実装はが共通している場合があります。その場合、次のように実装することが推奨されます。

1. デバイスとプリミティブ型のテンソルで OpKernel をテンプレート化して定義します。
2. 実際に出力を計算するために、Compute 関数はテンプレート化されたファンクタ構造体を呼び出します。
3. CPUDevice 用の特化したファンクタは同じファイルに定義されますが、GPUDevice 用の特化したファンクタは CUDA コンパイラによってコンパイルされるため、.cu.cc ファイルに定義されます。

実装例を示します。

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

## 演算子ライブラリをビルドする

### システムコンパイラーを使って演算子をコンパイルする（TensorFlow バイナリインストール）

システムで提供されている `g++` や `clang` のような `C++` コンパイラを使えば、`zero_out.cc` をコンパイルすることができます。バイナリ PIP パッケージは、コンパイルに必要なヘッダーファイルとライブラリをシステム固有の場所にインストールしますが、TensorFlow python ライブラリには、ヘッダーのディレクトリを取得する `get_include` 関数と、リンクされる共有オブジェクトがあるディレクトリを取得する `get_lib` 関数があります。Ubuntu マシン上でのこれらの関数の出力を次に示します。

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/tensorflow'
```

`g++` がインストールされていることを前提に、ここでは演算子を動的ライブラリにコンパイルするための一連のコマンドを示します。

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

macOS では、`.so` ファイルをビルドするときに、"-undefined dynamic_lookup" という追加フラグが必要です。

> `gcc` のバージョンが `>=5` のときの注意点: gccは、バージョン `5` から新しい C++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx) を利用しています。TensorFlow のウェブサイトで提供されているパッケージは、古い ABI を利用する `gcc4` でビルドされています。演算子ライブラリを `gcc>=5` でコンパイルする場合、コマンドラインに `-D_GLIBCXX_USE_CXX11_ABI=0` を追加し、ライブラリが古い ABI に対応できるようにしてください。

### bazel を使って演算子をコンパイルする（TensorFlow ソースインストール）

TensorFlow ソースがインストールされている場合は、TensorFlow のビルドシステムを利用して演算子をコンパイルすることができます。BUILD ファイルを次の Bazel ビルドルールに従って、[`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) ディレクトリに配置してください。

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

次のコマンドを使用して、`zero_out.so` をビルドします。

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

`Example` 演算のコンパイルでは、CUDA カーネルで、`tf_custom_op_library` の `gpu_srcs` パラメーターを使用する必要があります。BUILD ファイルを次の Bazel ビルドツールに従って、[`tensorflow/core/user_ops`](https://www.tensorflow.org/code/tensorflow/core/user_ops/) ディレクトリの新しいフォルダ（"example_gpu" など）に配置してください。

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)
```

次のコマンドを使用して、`kernel_example.so` をビルドします。

```bash
$ bazel build --config opt //tensorflow/core/user_ops/example_gpu:kernel_example.so
```

注意: 前述のとおり、gcc&gt;=5 でコンパイルする場合は、Bazel のコマンドライン引数に `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` を追加してください。

> 注意: 標準の `cc_library` ルールを利用して共有ライブラリ（`.so` ファイル）を作成できますが、`tf_custom_op_library` マクロを利用することを強く推奨します。このマクロは、必要となる依存関係を追加し、共有ライブラリが TensorFlow のプラグイン読み込みメカニズムに対応しているかを確認します。

## 演算子を Python で使用する

TensorFlow Python API には、動的ライブラリを読み込んで演算心を TensorFlow フレームワークに登録する `tf.load_op_library` 関数があります。`load_op_library` は、演算子とカーネルの Python ラッパーを含む Python モジュールを返します。そのため、演算子をビルドしたら、Python から次のようにして実行することができます。

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
print(zero_out_module.zero_out([[1, 2], [3, 4]]).numpy())

# Prints
array([[1, 0], [0, 0]], dtype=int32)
```

生成された関数は、スネークケースの名前が与えられることを覚えておいてください（[PEP8](https://www.python.org/dev/peps/pep-0008/) に準拠）。そのため、C++ ファイルで `ZeroOut` と名付けられた演算子は、Python 関数では `zero_out` となります。

Python モジュールから `import` 可能な、通常の関数として演算子を利用できるようにするには、次のように Python ソースファイルで `load_op_library` の呼び出しを使用すると役立つ可能性があります。

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## 演算子の動作を検証する

演算子が正しく実装されたことを検証するには、テストを書くことをお勧めします。次のコンテンツで `zero_out_op_test.py` を作成します。

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

次に、テストを実行します（TensorFlow がインストール済みであることが前提です）。

```sh
$ python zero_out_op_test.py
```

## 演算子に高度な機能を組み込む

基本的な（ある程度の制限が付いた）演算子のビルド方法と実装について理解したので、演算子に通常組み込む必要のある、より複雑な機能を確認しましょう。

- 条件チェックと検証
- 演算子の登録
    - 属性
    - 属性のタイプ
    - [ポリモーフィズム](#polymorphism)
    - [入力と出力](#inputs-and-outputs)
    - 下位互換性
- [GPU サポート](#gpu-support)
    - GPU デバイス向けのカーネルのコンパイル
- Python での勾配の実装
- C++ での形状関数

### 条件チェックと検証

ここまでの例では、あらゆる形状のテンソルに適用される演算子が想定されていましたが、ベクトルにのみ適用する場合はそうでしょうか。つまり、上記の OpKernel 実装にチェックを追加するということです。

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

これは入力がベクトルであることを表明し、ベクトルでない場合は `InvalidArgument` ステータスを設定して戻します。`OP_REQUIRES` マクロは、次の 3 つの引数を取ります。

- `context`。`OpKernelContext` または `OpKernelConstruction` のポインタで（[`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) を参照）、`SetStatus()` メソッドに使用します。
- 条件文。例として [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h) には、テンソルの形状を検証するための関数があります。
- エラー。`Status` オブジェクトで表現されます。[`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h) をご覧ください。`Status` には、型（通常は `InvalidArgument` ですが、型のリストをご覧ください）とメッセージの両方があります。エラーを構築する関数は、[`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h) にある場合があります。

または、ある関数から返された `Status` オブジェクトがエラーであるかをテストし、エラーである場合はそれを返して [`OP_REQUIRES_OK`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h) を使用することもできます。これらのマクロは、エラー時に関数から返されます。

### 演算子の登録

#### 属性

演算子には属性があり、その値は演算子がグラフに追加された時に設定されます。これらは演算子を構成するために使用され、その値にはカーネル実装内から演算子登録の入力と出力の型でアクセスすることができます。入力の方が柔軟であるため、できる限り属性の代わりに入力を使用するようにしてください。一方で入力はテンソルであり、値は動的に変わります。すなわち、入力はステップごとに変化したり、フィードを使って設定されたりします。属性は、入力で行えない、シグネチャ（入力または出力の数または型）に影響する構成やステップごとに変更できない構成に使用されます。

属性は演算子を登録する際に定義します。`Attr` メソッドを使って名前と型を指定しますが、次の形式を使用する必要があります。

```
<name>: <attr-type-expr>
```

上記の `<name>` は、文字で始まり、英数字とアンダースコアを使用できます。`<attr-type-expr>` は、[以下で説明](#%E3%82%A2%E3%83%88%E3%83%AA%E3%83%93%E3%83%A5%E3%83%BC%E3%83%88%E5%9E%8B)される形式の型表現です。

たとえば、0 番目の要素だけでなくユーザー指定のインデックスを保持するために演算子を `ZeroOut` する場合は、次のように演算子を登録できます。

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

（一連の[属性の型](#%E3%82%A2%E3%83%88%E3%83%AA%E3%83%93%E3%83%A5%E3%83%BC%E3%83%88%E5%9E%8B)は、入力と出力に使用される `tf.DType` と異なることに注意してください。）

カーネルでは、`context` パラメータを通じてコンストラクタ内でこの属性にアクセスできます。

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

そして、これを `Compute` メソッドで使用することができます。

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

#### 属性の型

属性には、次の型がサポートされています。

- `string`: バイトシーケンス（UTF8 である必要はありません）
- `int`: 符号付き整数
- `float`: 浮動小数点数
- `bool`: True または False
- `type`: [`DataType`](https://www.tensorflow.org/code/tensorflow/core/framework/types.cc) のいずれかの（ref型ではない）値
- `tensor`: [`TensorProto`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor.proto)
- `list(<type>)`: `<type>` のリスト。`<type>` は上記のいずれかの型です。`list(list(<type>))` は無効であることに注意してください。

完全なリストについては、[`op_def_builder.cc:FinalizeAttr`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc) もご覧ください。

##### デフォルト値と制約

属性にはデフォルト値がある場合があり、一部の型に制約を付けることができます。制約付きの属性を定義するには、次の `<attr-type-expr>` を利用できます。

`{'<string1>', '<string2>'}`: 値は `<string1>` または `<string2>` を持つ文字列である必要があります。この構文を利用する場合、型名  `string` は暗喩されます。これは enum をエミュレーションします。

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

`{<type1>, <type2>}`: 型 `type` の値であり、`<type1>` または `<type2>` のいずれかである必要があります。`<type1>` と `<type2>` は、サポートされている `tf.DType` です。属性の型が `type` であることは指定しません。これは、`{...}` に型のリストがある場合に暗喩されます。この場合には属性 `t` は、`int32`、 `float`、または `bool` のいずれかです。

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

一般的な型制約には、次のようなショートカットがあります。

- `numbertype`: 型 `type` は、数値型（文字列型、非ブール型ではない型）に制限されます。
- `realnumbertype`: 複素数型を除いた `numbertype` に似ています。
- `quantizedtype`: `numbertype` に似ていますが、量子化された数値型のみです。

これらで許可された具体的な型のリストは関数（`NumberTypes()` など）によって [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h) に定義されています。この例では、属性 `t` は数値型である必要があります。

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

この演算子の場合:

```python
tf.number_type(t=tf.int32)  # Valid
tf.number_type(t=tf.bool)   # Invalid
```

リストは他のリストや single 型と組み合わせることができます。次の演算子では、属性 `t` を任意の数値型またはブール型にすることができます。

```c++
REGISTER_OP("NumberOrBooleanType")
    .Attr("t: {numbertype, bool}");
```

この演算子の場合:

```python
tf.number_or_boolean_type(t=tf.int32)  # Valid
tf.number_or_boolean_type(t=tf.bool)   # Valid
tf.number_or_boolean_type(t=tf.string) # Invalid
```

`int >= <n>`: 値は、`<n>` 以上の整数型である必要があります。`<n>` は自然数です。たとえば、次の演算子の登録には、属性 `a` に `2` つ以上の値がある必要があることが示されています。

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

`list(<type>) >= <n>`: 長さが `<n>` 以上の型 `<type>` のリストです。たとえば、次の演算子の登録には、属性 `a` は型のリスト（`int32` または `float`）のリストであり、少なくとも 3 つ以上の値が必要であることが示されています。

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

属性のデフォルトの値を設定するには（生成されるコードのオプション）、次のように最後に `= <default>` を追加します。

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

さらに、制約とデフォルト値を同時に指定することもできます。

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

サポートされているデフォルト値のシンタックスは、GraphDefの定義の結果として表現される proto で利用できるものになります。

次に、すべての型にデフォルトを指定する例を示します。

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

型 `type` の値は `tf.DType` を使用することに特に注意してください。

#### ポリモーフィズム

##### 型ポリモーフィズム

異なる型を入力として取るか異なる型を出力する演算子については、オペレーションの登録において、[1 つの属性](#%E5%85%A5%E5%8A%9B%E3%81%A8%E5%87%BA%E5%8A%9B) を [入力または出力の型](#%E3%82%A2%E3%83%88%E3%83%AA%E3%83%93%E3%83%A5%E3%83%BC%E3%83%88) に指定できます。一般的にはその後で、サポートされたそれぞれの型について `OpKernel` を登録します。

たとえば、`ZeroOut` 演算子を `float` と `int32` に使用する場合、演算子の登録は次のようになります。

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

これで、演算子の登録は、入力の型が `float` または `int32` であり、出力にも `T` が使用されているためも同じ型で出力されるように指定されました。

###### 命名

入力、出力、および属性は通常、スネークケースで命名される必要があります。ただし、入力の型として使用されている属または出力の型に使用されている属性は例外です。これらの属性は、演算子がグラフに追加されるときに推論されるため、演算子の関数には出現しません。たとえば、この最後の ZeroOut の定義では、次のような Python 関数が生成されます。

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

`int32` のテンソルが `to_zero` に渡されてきた場合、`T` は自動的に `int32` （実際は、`DT_INT32`）に設定されます。これらの推論される属性は、大文字もしくはキャメルケースで命名されます。

これを、出力型を決定する型属性のある演算子と比較して見ましょう。

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

この場合、ユーザーは生成される Python で出力型を指定する必要があります。

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

###### 型ポリモーフィズムの例

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

[下位互換性](#%E5%BE%8C%E6%96%B9%E4%BA%92%E6%8F%9B%E6%80%A7) を保つには、既存の演算子に属性を追加するときに[デフォルト値](#%E3%83%87%E3%83%95%E3%82%A9%E3%83%AB%E3%83%88%E5%80%A4%E3%81%A8%E5%88%B6%E7%B4%84)を指定する必要があります。

```c++
REGISTER_OP("ZeroOut")
  .Attr("T: {float, int32} = DT_INT32")
  .Input("to_zero: T")
  .Output("zeroed: T")
```

たとえば、`double` 型など、ほかの型を追加するとしましょう。

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

ほとんどの場合、前述のような冗長なコードでほかの `OpKernel` を書く代わりに、C++ テンプレートを使うことができます。オーバーロードごとに 1つのカーネル登録（`REGISTER_KERNEL_BUILDER` 呼び出し）が必要になります。

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

オーバーロード数が 3 つ以上ある場合は、登録をマクロに入れ込むことができます。

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

カーネルを登録する型のリストによっては、[`tensorflow/core/framework/register_types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h) が提供するマクロを使用することも可能です。

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

##### 入力と出力のリスト

異なる型を受け入れたり生成したりできるほか、演算子は、テンソルの可変数を消費または生成することができます。

次の例では、属性 `T` は型の*リスト*を保持し、入力  `in` と出力 `out` の両方の型として使用されます。入力と出力はその方のテンソルのリストです（出力のテンソルの数と型にも `T` が使用されているため、入力と同じになります）。

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

また、どの型をリストに指定できるの制約を付けることもできます。次のケースでは、入力は `float` 型と `double` 型のテンソルのリストです。演算子は、たとえば、入力型 `(float, double, float)` を受け入れ、その場合の出力型も `(float, double, float)` になります。

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

リスト内のすべてのテンソルを同じ型にする場合は、次のようにします。

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

これは、`int32` テンソルのリストを受け入れ、`int` 属性 `N` を使用して、リストの長さを指定します。

これを [型ポリモーフィック](#type-polymorphism)にすることもできます。次の例では、入力は同じ（未指定）型（`"T"`）のテンソルのリスト（長さ `"N"`）で、出力は同じ型の 1 つのテンソルです。

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

デフォルトのテンソルの長さは 1 以上です。このデフォルトは、[対応する属性に `">="` 制約](#default-values-and-constraints)を使って変更することができます。次の例では、入力は少なくとも 2 つの `int32` 型テンソルのリストです。

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

同じ構文は、`"list(type)"` 属性でも動作します。

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### 入力と出力

上記をまとめると、演算子の登録は複数の入力と出力を持つことができます。

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

それぞれの入力または出力の仕様は、次の形式です。

```
<name>: <io-type-expr>
```

上記の `<name>` は、文字で始まり、英数字とアンダースコアを使用できます。`<io-type-expr>` は、次の型表現のいずれかです。

- `<type>`: `<type>` は、サポートされる入力型（`float`、`int32`、`string` など）です。これは特定の型の単一のテンソルを示します。

    `tf.DType` をご覧ください。

    ```c++
    REGISTER_OP("BuiltInTypesExample")
        .Input("integers: int32")
        .Input("complex_numbers: complex64");
    ```

- `<attr-type>`: `<attr-type>` は、[属性](#attrs)の名前で、型 `type` または `list(type)` を持ち、型制限の可能性があります。この構文では[ポリモーフィズムな演算子](#%E3%83%9D%E3%83%AA%E3%83%A2%E3%83%BC%E3%83%95%E3%82%A3%E3%82%BA%E3%83%A0)が可能です。

    ```c++
    REGISTER_OP("PolymorphicSingleInput")
        .Attr("T: type")
        .Input("in: T");

    REGISTER_OP("RestrictedPolymorphicSingleInput")
        .Attr("T: {int32, int64}")
        .Input("in: T");
    ```

    型 `list(type)` の属性を参照することで、一連のテンソルを受け入れることができます。

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

    出力と入力の型は `T` であるため、出力 `out` のテンソルの数と型は入力 `in` と同じです。

- おなじ型をもつテンソルのシーケンス: `<number> * <type>`。`<number>` は[属性](#attrs)の数で、`int` 型です。`<type>` は `tf.DType` または `type` 型の属性の数です。前者の例として、この演算子は `int32` テンソルのリストを受け入れます。

    ```c++
    REGISTER_OP("Int32SequenceExample")
        .Attr("NumTensors: int")
        .Input("in: NumTensors * int32")
    ```

    この演算子はすべての型が同じである場合に限り、任意の型のリストを受け入れます。

    ```c++
    REGISTER_OP("SameTypeSequenceExample")
        .Attr("NumTensors: int")
        .Attr("T: type")
        .Input("in: NumTensors * T")
    ```

- テンソルの参照: `Ref(<type>)`。`<type>` は前述した型のいずれかです。

入力の型に使用される属性は推論されます。推論された属性には大文字の名前（`T` または `N` など）が使用されます。そうでない場合、入力、出力、および属性には、関数パラメーター（`num_outputs` など）のような名前が付けられます。詳細については、[命名規則に関する前方のセクション](#naming)をご覧ください。

詳細については、[`tensorflow/core/framework/op_def_builder.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h) をご覧ください。

#### 下位互換性

カスタム演算子をうまく書けたので、みんなに利用してもらえるように共有したとします。ところが、その演算子に変更を適用することになりました。

一般的に、既存のチェックイン済みの仕様への変更は、下位互換性である必要があります。ある演算子の仕様を変更することによって、以前に古い仕様を使ってシリアル化した `GraphDef` プロトコルバッファが動作しなくなっては大変です。`GraphDef` 対応の詳細については、[こちらに説明](./versions.md#compatibility_of_graphs_and_checkpoints)されています。

下位互換性を維持するにはいくつかの方法があります。

1. 演算に追加された新しい属性にはデフォルト値が定義されている必要があり、演算子の元の動作はそのデフォルト値に基づく必要があります。演算を非ポリモーフィックからポリモーフィックに変更するには、新しい型属性にデフォルトの値を提供して、元のシグネチャをデフォルトで維持できるようにする*必要があります*。たとえば、次の演算があったとします。

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: float")
        .Output("out: float");
    ```

    次のようにして、下位互換性の方法でポリモーフィックにすることができます。

    ```c++
    REGISTER_OP("MyGeneralUnaryOp")
        .Input("in: T")
        .Output("out: T")
        .Attr("T: numerictype = DT_FLOAT");
    ```

2. 属性の制約の制限を安全に緩和することができます。たちえば、アトリビュートの制約を緩めるのは安全に行えます。たとえば、`{int32, int64}` から `{int32, int64, float}` または `type` に変更できます。または、`{"apple", "orange"}` を `{"apple", "banana", "orange"}` または `string` に変更することができます。

3. リスト型のデフォルトが以前のシグネチャに一致する場合に限り、単一の入力/出力をリストの入力出力に変更できます。

4. デフォルトが空であれば、新たなリストの入力/出力を追加できます。

5. 演算子の名前に、プロジェクトに固有のプレフィックスを付けることで、作成した演算子の名前空間を作れます。これにより、TensorFlow の将来のバージョンで含まれる可能性のあるすべての演算子との競合を回避できます。

6. 前もって計画しましょう！演算子の将来の使われ方を予測します。シグネチャの中には互換性のある方法で実行できないものがあります（同じ型のリストを型の異なるリストに変更するなど）。

安全な変更と安全でない変更の全リストは、[`tensorflow/core/framework/op_compatibility_test.cc`](#%E3%83%9D%E3%83%AA%E3%83%A2%E3%83%BC%E3%83%95%E3%82%A3%E3%82%BA%E3%83%A0) をご覧ください。演算子への変更を下位互換性にできない場合は、新しいセマンティクスと新しい名前で新しい演算を作成してください。

また、これらの変更によって `GraphDef` 互換性を維持できるかもしれませんが、生成される Python コードが以前のコーラーと互換性のない方法に変更される可能性があります。Python API は、手書きの Python ラッパーを注意深く変更することで互換性を維持できます。ただし、最後に新しいオプションの引数を追加する場合を除いて、以前のシグネチャを保持することもできます。一般的に互換性のない変更は、TensorFlow がメジャーバージョンを変更する場合にのみ行うことができ、<a data-md-type="raw_html" href="./versions.md#compatibility_of_graphs_and_checkpoints">`GraphDef` バージョンのセマンティクス</a>に準拠する必要があります。

### GPU のサポート

[異なる型のカーネルを登録](#polymorphism)できるのと同様に、CPU と GPU で別々の OpKernel を実装して登録することができます。[`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/) には GPU をサポートしたカーネルの例がいくつかあります。一部のカーネルには `.cc` ファイルの CPU バージョン、`_gpu.cu.cc` ファイルの GPU バージョン、および `.h` ファイルの共通コードがあります。

たとえば `tf.pad` は、[`tensorflow/core/kernels/pad_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc) に GPU のカーネル以外のすべてが存在します。GPU のカーネルは、[`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc) にあり、共通のコードは [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h) に定義されたテンプレートクラスです。コードをこのように編成しているは、CPU と GPU の実装間で共通のコードを共有できるようにし、GPU の実装を別のファイルに置くことで GPU コンパイラだけがコンパイルできるようにしているためです。

`pad` の GPU カーネルバージョンが使用されている場合であっても、CPU メモリに `"paddings"` が必要であるということに注意してください。その入力または出力が CPU に維持されているとマークするには、次のように、カーネル登録に `HostMemory()` 呼び出しを追加します。

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### GPU デバイス向けのカーネルのコンパイル

CUDA カーネルを使用して演算子を実装している例については [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) をご覧ください。`tf_custom_op_library` は、CUDA カーネル（`*.cu.cc` ファイル）を含むソースファイルのリストが指定されている `gpu_srcs` 引数を受け入れます。TensorFlow のバイナリインストールで使用する場合、CUDA カーネルは NVIDIA の `nvcc` コンパイラを使用してコンパイルされる必要があります。次は、[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) と [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc) を単一の動的に読み込まれるライブラリにコンパイルするために使用できるコマンドのシーケンスです。

```bash
nvcc -std=c++14 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++14 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

上記で生成された `cuda_op_kernel.so` は Python の `tf.load_op_library` 関数を使用して通常通り読み込むことができます。

CUDA ライブラリが `/usr/local/lib64` にインストールされていない場合は、上記の 2 つ目のコマンド（g++）に明示的にパスを指定する必要があります。たとえば、CUDA が `/usr/local/cuda-8.0` にインストールされている場合は `-L /usr/local/cuda-8.0/lib64/` を追加します。

注意: 特定の Linux の設定では、`nvcc` によるコンパイルのステップに追加のオプションが必要になることに注意してください。`mwaitxintrin.h` からのエラーを回避するには、`nvcc` コマンドラインに `-D_MWAITXINTRIN_H_INCLUDED` を追加してください。

### Python での勾配の実装

特定の演算子のグラフにおいて、TesorFlow は自動微分（バックプロパゲーション）を使用して、既存の演算子に対する勾配を表現する新しい演算子を追加します。自動微分が新しい演算子でも動作するようにするには、演算子の入力指定勾配に対する勾配を計算する勾配関数を演算子の出力に対して登録する必要があります。

数学的には、演算子が (y = f(x))  を計算する場合、登録されている勾配演算子は、(y) に関する損失 (L) の勾配 (\partial L/ \partial y) を連鎖規則を介して (x) に関する勾配 (\partial L/ \partial x) に変換します。

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

`ZeroOut` の場合、入力の 1 つのエントリのみが出力に影響するため、入力に関する勾配はスパース「ワンショット」テンソルになります。これは次のように表現されます。

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

`tf.RegisterGradient` で勾配関数を登録する方法:

- 出力が 1 つの演算子では、勾配関数は `tf.Operation`、`op`、および `tf.Tensor` `grad` を取って、テンソル `op.inputs[i]`、`op.outputs[i]`、および `grad` から新しい演算子をビルドします。属性に関する情報は、`tf.Operation.get_attr` にあります。

- 演算子に複数の出力がある場合、勾配関数は `op` と `grads` を取ります。`grads` は各出力に関する勾配のリストです。勾配関数の結果は、それぞれの入力に関する勾配を表現する `Tensor` オブジェクトのリストである必要があります。

- 整数の入力がインデックスとして使われている場合など、一部の入力の勾配が十分に定義されていない場合は、対応する結果の勾配は `None` になります。たとえば、浮動小数点数のテンソル `x` と整数インデックス `i` を取る演算子では、勾配関数は `[x_grad, None]` を返します。

- 演算子に意味のない勾配である場合は、ほとんどの場合、勾配を登録する必要はありません。また、演算の勾配がまったく必要でない限り、問題でもありません。ただし、一部のケースでは、十分に定義された勾配がない演算子が勾配の計算に関わっている場合があります。この場合は、`ops.NotDifferentiable` を使用して自動的にゼロ逆伝搬を行うことができます。

勾配関数が呼び出されるとき、演算子のデータフローグラフのみが利用でき、テンソルデータ自体は利用できない場合があることに注意してください。したがって、グラフ実行時に実行するには、すべての計算をほかの TensorFlow 演算子を使用して実行する必要があります。

### C++ での形状関数

TensorFlow API には、グラフを実行せずにテンソルの形状に関する情報を提供する「形状推論」と呼ばれる機能があります。形状推論は、C++ の `REGISTER_OP` 宣言の各演算子の型に登録されている「形状関数」によってサポートされており、グラフ構築中に入力の形状が互換していることをアサートすることと、出力の形状を指定することという 2 つの役割があります。

形状関数は、`shape_inference::InferenceContext`  に演算として定義されています。たとえば、ZeroOut の形状関数では、次のようになります。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` は、最初の出力の形状が最初の入力の形状に設定される必要があることを宣言しています。出力が上記の例のようにインデックスによって選択されている場合、`set_output` の 2 つ目のパラメーターは `ShapeHandle` である必要があります。空の `ShapeHandle` オブジェクトはデフォルトのコンストラクタで作成できます。インデックス `idx` の入力の `ShapeHandle` オブジェクトとは、`c->input(idx)` で取得できます。

多数の演算子に適用する `shape_inference::UnchangedShape` などの多数の共通する形状関数があり、これらは [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) にあり、次のように使用されます。

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

形状関数は、入力の形状も制約できます。[ベクトル形状制約のある `ZeroOut`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) のバージョンについては、形状関数は次のようになります。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

`WithRank` 呼び出しは、入力形状 `c->input(0)` にきっかり 1 次元の形状があることを検証します（または入力形状が不明である場合、出力形状は 1 つの不明な次元を持つベクトルがあることを検証します）。

演算子が[複数の入力を持つポリモーフィック](#polymorphism)である場合、`InferenceContext` のメンバーを使用して、チェックする形状の数を判定し、`Merge` を使用してすべての形状に互換性があることを検証します（または、`InferenceContext::GetAttr` で長さを示す属性にアクセスし、演算子の属性にアクセスできるようになります）。

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

形状推論はオプションの機能であり、テンソルの形状は非常に動的になる可能性があるため、形状関数はいずれかの入力に関する形状情報が不完全であることに堅牢である必要があります。`Merge` メソッド（[`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h)）を使うと、2 つの形状のいずれかまたは両方に完全な情報がない場合でも、コーラーはそれらが同じであることをアサートすることができます。形状関数は、すべてのコア TensorFlow 演算子に定義されており、多数のさまざまな使用例を提供しています。

`InferenceContext` クラスには、形状関数の操作を定義するために使用できる関数が多数あります。たとえば、`InferenceContext::Dim` と `InferenceContext::WithValue` を使用して、特定の次元に非常に具体的な値があることを検証することができます。また、`InferenceContext::Add` と `InferenceContext::Multiply` を使用して、出力の次元が 2 つの入力の和また積であることを指定することもできます。指定できる形状操作については、`InferenceContext` クラスをご覧ください。次の例は、最初の出力の形状を (n, 3) に設定しています。この最初の入力の 形状は (n, ...) です。

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

複雑な形状関数がある場合は、多様な入力形状の組み合わせによって、期待される出力形状の組み合わせが生成されることを検証するテストを追加することを検討してください。これらのテストの作成方法の例は、[core ops tests](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc) をご覧ください（`INFER_OK` と `INFER_ERROR` の構文は多少不可解ではありますが、テストで入力と出力の形状仕様を表現する場合は、コンパクトに収められるようにしてください。現時点では、これらのテストに含まれるコメントを見て、形状文字列の仕様を理解してください）。

## カスタム演算子の pip パッケージをビルドする

演算子の `pip` パッケージをビルドするには、[tensorflow/custom-op](https://github.com/tensorflow/custom-op) の例をご覧ください。このガイドでは、ソースから TensorFlow をビルドするのではなく、TensorFlow pip パッケージからカスタム演算子をビルドする方法が説明されています。

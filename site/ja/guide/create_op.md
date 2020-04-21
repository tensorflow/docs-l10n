# オペレーションを作成する

Note: C++のカスタムオペレーションが、TensorFlow公式のpipパッケージとABI互換になることを保証するため、[Custom opリポジトリ](https://github.com/tensorflow/custom-op) のガイドにしたがってください。
カスタムオペレーションをビルドして配布するためのDockerのイメージはもちろんのこと、初めから終わりまでのコード例が示されています。

既存のTensorFlowのライブラリに存在しないオペレーションを作りたい場合、既存のPythonのオペレーションや関数を組み合わせて、Pythonでオペレーションを書くことを推奨します。
もしそれが不可能なら、C++のカスタムオペレーションを作ってもよいです。
C++のカスタムオペレーションを作りたいと考える理由は、いくつかあります。

* 既存のオペレーションの組み合わせでオペレーションを表現するのが、不可能または簡単ではない
* 既存のプリミティブの組み合わせでオペレーションを表現するのが、効率的ではない
* 将来コンパイラが融合することが難しいプリミティブの組み合わせを、自前で融合したい

たとえば、"MaxPool" オペレーションと似ているが、最大値のかわりにウィンドウをスライドさせて中央値を計算する、"median pooling" のようなものを実装したいとしましょう。
これは、オペレーションの組み合わせ（たとえば、ExtractImagePatchesとTopKを使う）でも可能ですが、1つの融合したオペレーションとしてより賢明に実装したネイティブなオペレーションと比較して、性能とメモリ効率の面で劣るかもしれません。
いつも通り、オペレーションの組み合わせで、やりたいことを表現する試みには価値があります。
もしそれが難しいまたは非効率であることが証明されたときのみ、新しいオペレーションを追加することを検討しましょう。

カスタムオペレーションを組み込むために必要なことを、次に示します。

1. C++ファイル内で新しいオペレーションを登録します。オペレーションの登録では、オペレーションの実装とは独立であるオペレーションの機能のためのインターフェース（仕様）を定義します。たとえば、オペレーションの登録では、オペレーション名やオペレーションの入出力を定義します。また、テンソルのシェイプ推論に使用されるシェイプ関数を定義します。
2. C++でオペレーションを実装します。オペレーションの実装はカーネルとして知られ、Step 1で登録した仕様の実装を具体化します。異なる入出力型、アーキテクチャ（たとえば、CPUやGPU）のために複数のカーネルが存在することもあり得ます。
3. Pythonのラッパーを作成する（任意）。このラッパーは、Pythonでオペレーションを作るときに使われるパブリックなAPIです。デフォルトのラッパーは、オペレーションの登録から生成され、直接利用することもできますし、追加することもできます。
4. オペレーションの勾配を計算するための関数を書きます。（任意）
5. オペレーションをテストします。便宜上、たいていはPythonで行いますが、C++でオペレーションをテストすることも可能です。勾配を定義した場合、Python からは `tf.test.compute_gradient_error` を使って確認できます。Reluのような順伝搬の関数とその勾配をテストするための例については、[`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py) を見てください。


## 前提条件

* C++になじみがあること
* [TensorFlowのバイナリ](../../install) がインストールされていること。もしくは、[ダウンロードされたTensorFlowのソースコード](../../install/source.md) があり、ビルドできること


## オペレーションのインターフェース定義

TensorFlowのシステムを使って、オペレーションのインターフェースを登録して定義します。
登録にあたり、オペレーションの名前と入出力（型と名前）、オペレーションが必要とする場合があるdocstringsと [アトリビュート](#アトリビュート) を指定します。

どのように取り組むのかを見るために、`int32` のテンソルを受け取り、最初以外のすべての要素が0であるコピーされたテンソルを出力するオペレーションを作ることを考えます。
これを行うために、`zero_out.cc` と命名されたファイルを作成します。
続いて、オペレーションのインターフェースを定義するための `REGISTER_OP` マクロ呼び出しを追加します。

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

この `ZeroOut` オペレーションは、入力として32bit整数のテンソル `to_zero` を受け取り、32bit整数のテンソル `zeroed` を出力します。
このオペレーションは、出力テンソルが入力テンソルとおなじシェイプであることを保証するために、シェイプ関数を使っています。
たとえば、入力テンソルのシェイプが [10, 20] であるならば、このシェイプ関数は出力のシェイプも [10, 20] であることを明示します。

Note: オペレーションの名前はCamelCaseで、かつバイナリに登録されているすべてのオペレーションの中で唯一のものである必要があります。


## オペレーションのカーネル実装

インターフェースを定義したあとは、1つ以上のオペレーションの実装を提供する必要があります。
これらのカーネルを作成するためには、`OpKernel` を継承したクラスを作成し、`Compute` メソッドをオーバーライドします。
`Compute` メソッドは、`OpKernelContext*` 型である1つの `context` 引数を提供し、ここから入力や出力テンソルのような便利なものにアクセスできます。

上記で作成したファイルにカーネルを追加します。
カーネルはたとえば次のようなものになるかもしれません。

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 入力テンソルを取得する。
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 出力テンソルを作成する。
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // 最初以外のすべての要素を0にする。
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // 可能なら、最初の入力値は維持する。
    if (N > 0) output_flat(0) = input(0);
  }
};
```

カーネルを実装したあと、TensorFlowのシステムに登録します。
登録時には、このカーネルが動作するいろいろな制約を指定します。
たとえば、CPU向けに作成した1つのカーネルと、GPU向けの別のカーネルがあるとしましょう。

これを `ZeroOut` オペレーションで実現するためには、次を `zero_out.cc` に追加します。

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 重要: OpKernelのインスタンスは、同時にアクセスされることがあります。`Compute` メソッドは、スレッドセーフにしなければなりません。クラスメンバへのアクセスはmutexでガードしてください。いっそのこと、クラスメンバ経由で状態を共有しないようにしてください！オペレーションの状態を追跡するためには、[`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h) を使用することを検討してください。


### マルチスレッド化されたCPUカーネル

マルチスレッド化されたCPUカーネルを書くためには、[`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) にあるShard関数を利用できます。
この関数は、オペレーション内でのスレッド実行のために使われるスレッド間で計算を分割します。（[`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto) のintra_op_parallelism_threadsを見てください）


### GPUカーネル

GPUのカーネルは、OpKernelとCUDAカーネルとカーネルを起動するコードの2つの部分から実装されています。

入力の検査や出力の割り当てなど、時にはOpKernelの実装はCPUとGPU間で共通です。
その場合において、推奨される実装を次に示します。

1. デバイスとテンソルのプリミティブ型をテンプレート化した、OpKernelを定義します。
2. 実際に出力の計算をするために、Compute関数はテンプレート化されたファンクタ構造体を呼び出します。
3. CPUDeviceのために特化したファンクタはおなじファイルに定義しますが、GPUDeviceのために特化したものは、CUDAコンパイラによってコンパイルされるために.cu.ccファイルに定義します。

実装例を示します。

```c++
// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// GpuDeivce向けに部分特化したファンクタ。
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_
```

```c++
// kernel_example.cc
#include "kernel_example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPUに特化された実際の計算。
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernelの定義。
// テンプレートパラメータ<T>は、テンソルのデータ型。
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 入力テンソルを取得する。
    const Tensor& input_tensor = context->input(0);

    // 出力テンソルを作成する。
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // 計算する。
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// CPUカーネルを登録する。
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// GPUカーネルを登録する。
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* 明示的なインスタンス化は、kernel_example.cu.ccに定義する。 */  \
  extern template ExampleFunctor<GPUDevice, T>;                  \
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

// CUDAカーネルを定義する。
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// CUDAカーネルを起動するGPU向けの実装を定義する。
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // CUDAカーネルを起動する。
  //
  // 計算のためのblock数とthread_per_block数の例は、
  // core/util/gpu_kernel_helper.hを見てください。
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// 登録されたOpKernelの型のために、明示的にファンクタをインスタンス化します。
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```


## オペレーションライブラリの構築

### システムのコンパイラを使ってコンパイル（TensorFlowのバイナリインストール）

システム上で利用できる `g++` や `clang` のような `C++` コンパイラを使って、`zero_out.cc` をコンパイルするはずです。
バイナリのPIPパッケージは、オペレーションをコンパイルするために必要な、ヘッダファイルとライブラリをシステム固有の場所にインストールします。
しかし、TensorFlowのpythonライブラリは、ヘッダのディレクトリを取得する `get_include` 関数と、リンクされる共有オブジェクトがあるディレクトリを取得する `get_lib` 関数を提供しています。
Ubuntuマシン上におけるこれらの関数の出力を、次に示します。

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python2.7/site-packages/tensorflow'
```

`g++` がインストールされていることを想定し、ここではオペレーションを動的ライブラリにコンパイルするための一連のコマンドを示します。

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

Mac OS X上では、`.so` ファイルをビルドするときに、追加フラグ "-undefined dynamic_lookup" が必要です。

> `gcc` のバージョンが `>=5` のときの注意点: gccは、バージョン `5` から新しいC++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx) を利用します。TensorFlowのウェブサイトで利用可能なパッケージは、古いABIを利用する `gcc4` でビルドされています。もしオペレーションを `gcc>=5` でコンパイルする場合、コマンドラインに `-D_GLIBCXX_USE_CXX11_ABI=0` を追加し、古いABIと互換をもたせるようにしてください。


### bazelを使ってオペレーションをコンパイル（TensorFlowのソースコードインストール）

もしTensorFlowのソースコードがインストールされているなら、オペレーションをコンパイルするためにTensorFlowのビルドシステムを使用できます。
Bazelのビルドルールに従ったBUILDファイルを [`tensorflow/core/user_ops`][user_ops] ディレクトリに配置してください。

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

`zero_out.so` をビルドするために、次のコマンドを実行します。

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

Note: 前述したように、もしgcc>=5でコンパイルする場合は、bazelのコマンドラインに `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` を追加してください。

> Note: 標準の `cc_library` ルールを利用して共有ライブラリ（`.so` ファイル）を作成できますが、`tf_custom_op_library` マクロを利用することを強く推奨します。このマクロは、必要となるいくつかの依存関係を追加し、共有ライブラリがTensorFlowのプラグイン読み込み機構と適合しているかを確認します。


## オペレーションをPythonで使用する

TensorFlowのPython APIは、動的ライブラリをロードしてTensorFlowのフレームワークにオペレーションを登録する、`tf.load_op_library` 関数を提供しています。
`load_op_library` は、オペレーションとカーネルのPythonラッパーを含んだ、Pythonモジュールを返します。
たとえば、オペレーションを1度ビルドしたら、Pythonから次のように実行できます。

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# 出力表示
array([[1, 0], [0, 0]], dtype=int32)
```

([PEP8](https://www.python.org/dev/peps/pep-0008/) に従うために、)生成された関数は、スネークケースの名前が与えられることを覚えておいてください。
つまり、C++のファイル内で `ZeroOut` と名付けられたオペレーションは、Pythonの関数では `zero_out` で呼ぶことになるでしょう。

Pythonモジュールから `import` 可能な、通常の関数としてオペレーションを利用可能にするには、次のようにしてPythonのソースコードのファイルに、`load_op_library` の呼び出しをもつのが便利かもしれません。

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```


## オペレーションの動作を検証する

正しくオペレーションが実装できたことを確かめるよい方法は、テストを書くことです。
次の内容を持った `zero_out_op_test.py` を作ってください。

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

そして、テストを実行してください（TensorFlowがインストール済みであると想定しています）。

```sh
$ python zero_out_op_test.py
```


## より進んだ機能をオペレーションに作り込む

基本的な（そして、やや制限された）オペレーションと実装の構築方法を学んだため、オペレーションに対して作り込む必要が出てくるであろう、一般的でより複雑なことを見ていきましょう。

* [条件のチェックと検証](#条件チェックと検証)
* [オペレーションの登録](#オペレーションの登録)
  * [アトリビュート](#アトリビュート)
  * [アトリビュート型](#アトリビュート型)
  * [ポリモーフィズム](#ポリモーフィズム)
  * [入力と出力](#入力と出力)
  * [後方互換性](#後方互換性)
* [GPUサポート](#GPUサポート)
  * [GPUデバイス向けのカーネルコンパイル](#GPUデバイス向けのカーネルコンパイル)
* [Pythonにおける勾配の実装](#Pythonにおける勾配の実装)
* [C++でのシェイプ関数](#C++でのシェイプ関数)


### 条件チェックと検証

これまで想定してきた例では、いかなるシェイプのテンソルに対しても適用できるオペレーションを想定していました。
仮にベクトルに対してのみ適用したい場合、どうなるでしょうか？
これまでのOpKernelの実装に、チェックを追加することになります。

```c++
  void Compute(OpKernelContext* context) override {
    // 入力テンソルを取得する
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

これは入力がベクトルであることを表明し、もしそうでないなら `InvalidArgument` ステータスを設定して戻ります。
[`OP_REQUIRES` マクロ][validation-macro] は、3つの引数を受け取ります。

*  `context`、`SetStatus()` メソッドのための、`OpKernelContext` または `OpKernelConstruction` のポインタです ([`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) を確認してください)。
* 条件文。たとえば、[`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h) にあるテンソルのシェイプを検証する関数
* `Status` オブジェクトで表現されるエラーそのもの。[`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h) を見てください。`Status` は、型(`InvalidArgument` が頻出ですが、型のリストを見てください)とメッセージ。エラーを構築する関数は、[`tensorflow/core/lib/core/errors.h`][validation-macros] にあります。

あるいは、もしエラーとなった関数から `Status` オブジェクトを返したかどうかを判定し、エラーならそのまま返す場合は、[`OP_REQUIRES_OK`][validation-macros] が利用できます。
これらのマクロは、両方ともエラーとなった関数から戻ります。


### オペレーションの登録

#### アトリビュート

オペレーションは、オペレーションがグラフに追加されるときに設定される値である、アトリビュートをもつことができます。
これらはオペレーションを設定するときに使用され、カーネル実装やオペレーションの登録における入出力のデータ型の中で値にアクセスできます。
入力のほうがより柔軟であるため、できればアトリビュートの代わりに入力を使ってください。
これは、アトリビュートが定数値であり、グラフ構築時に定義する必要があるためです。
一方で入力はテンソルであり、値は動的に変わります。
すなわち、フィードを設定するように、入力はステップ毎に変わります。
アトリビュートは、入力ではできないことに対して使われます。
たとえば、シグネチャに影響するいかなる設定（入出力の型や数）や、ステップごとには変わらない設定に使われます。

オペレーションを登録するときに、次の仕様で `Attr` メソッドを使って名前と型を指定し、アトリビュートを定義します。

```
<name>: <attr-type-expr>
```

`<name>` は、アルファベットとアンダースコアで構成される文字で、`<attr-type-expr>` は、[以下で説明する](#アトリビュート型) 型表現です。

たとえば、0番目のエレメントのみの代わりに、ユーザが指定したインデックスを保存する `ZeroOut` オペレーションにしたい場合は、次のようにオペレーションを登録します。

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

（[アトリビュート型](#アトリビュート型) のセットが、入出力に使われていた `tf.DType` と異なることに注意してください。）

カーネルでは、コンストラクタ内で `context` パラメータを通してアクセスできます。

```c++
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // 値を保存するインデックスを取得する
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));
    // preserve_indexが正であるか確認する
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

そして、これを `Compute` メソッドで使用します。

```c++
  void Compute(OpKernelContext* context) override {
    // ...

    // 潜在的に動的な入力を検証するために、保存したアトリビュートを使います。
    // つまり、preserve_indexが範囲内であるかを確認する
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));

    // すべての出力テンソルの要素を0に設定する
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // リクエストされた入力の値を保存する
    output_flat(preserve_index_) = input(preserve_index_);
  }
```


#### アトリビュート型

次に示す型がアトリビュートでサポートされています。

* `string`: バイトシーケンス（UTF8である必要はない）
* `int`: 符号付き整数
* `float`: 浮動小数点数
* `bool`: TrueまたはFalse
* `type`: [`DataType`][DataTypeString] の（ref型ではない）値のいずれか
* `shape`: [`TensorShapeProto`][TensorShapeProto]
* `tensor`: [`TensorProto`][TensorProto]
* `list(<type>)`: `<type>` のリスト。`<type>` は、上記の型のいずれか。`list(list(<type>))` は無効であることに注意。

信頼のおけるリストである [`op_def_builder.cc:FinalizeAttr`][FinalizeAttr] も参照のこと。


##### デフォルト値と制約

アトリビュートはデフォルト値をもつことができ、いくつかのアトリビュート型には制約をもたせることができます。
制約を持ったアトリビュートを定義するために、次のような `<attr-type-expr>` を利用できます。

* `{'<string1>', '<string2>'}`: 値が `<string1>` もしくは `<string2>` である文字列でなければならない。このシンタックスを利用する場合、型名は `string` であることを暗に意味している。enumを真似る場合は、次のようにする。

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

* `{<type1>, <type2>}`: 型 `type` の値であり、`<type1>` もしくは `<type2>` のいずれかの値でなければならない。`<type1>` と `<type2>` は、サポートされた `tf.DType` である。アトリビュート型 `type` は指定しない。これは、`{...}` に型のリストを持っていることを暗に意味している。たとえば次のような場合、アトリビュート `t` は、`int32`、 `float`、 `bool` のいずれかになる。

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

* 一般的な型制約のために、手っ取り早い方法があります
  * `numbertype`: 型 `type` は、数値型（文字列でもなくBool型でもない）に制限されます。
  * `realnumbertype`: 複素数型を除いた `numbertype`
  * `quantizedtype`: 量子化された数値型に限定した `numbertype`

これらによって許されている特定の型リストは、[`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h) に定義された（`NumberTypes()` のような）関数によって定義されています。
この例では、アトリビュート `t` は数値型の1つでなければなりません。

```c++
REGISTER_OP("NumberType")
    .Attr("t: numbertype");
```

このオペレーションでは、次のようになります。

```python
tf.number_type(t=tf.int32)  # 有効
tf.number_type(t=tf.bool)   # 無効
```

リストは、ほかのリストや単一の型と組み合わせることができます。
次のオペレーションは、アトリビュート `t` が、数値型、Bool型のいずれについても許可しています。

```c++
REGISTER_OP("NumberOrBooleanType")
    .Attr("t: {numbertype, bool}");
```

このオペレーションでは、次のようになります。

```python
tf.number_or_boolean_type(t=tf.int32)  # 有効
tf.number_or_boolean_type(t=tf.bool)   # 有効
tf.number_or_boolean_type(t=tf.string) # 無効
```

* `int >= <n>`: 値は、`<n>` 以上の整数型でなければならない。`<n>` は自然数である。

たとえば、次に示すオペレーションの登録では、アトリビュート `a` が `2` 以上の値であることが必要であると示しています。

```c++
REGISTER_OP("MinIntExample")
    .Attr("a: int >= 2");
```

* `list(<type>) >= <n>`: 長さが `<n>` 以上の、型 `<type>` のリストである。

たとえば、次に示すオペレーションの登録では、アトリビュート `a` が、3つ以上の型（`int32` か `float`）のリストであることを示しています。

```c++
REGISTER_OP("TypeListExample")
    .Attr("a: list({int32, float}) >= 3");
```

（生成されたコードでは任意である）アトリビュートのデフォルト値を設定するためには、次のように最後に `= <default>` を追加します。

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

加えて、制約とデフォルト値を同時に指定することもできます。

```c++
REGISTER_OP("AttrConstraintAndDefaultExample")
    .Attr("i: int >= 1 = 1");
```

サポートされているデフォルト値のシンタックスは、GraphDefの定義の結果として表現されるprotoで利用できるものになります。

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

特に型 `type` の値については、`tf.DType` になります。


#### ポリモーフィズム

##### 型ポリモーフィズム

異なる型を入力として受け取るか、異なる型を出力するオペレーションについては、オペレーションの登録において、[入力または出力の型](#入力と出力) に [アトリビュート](#アトリビュート) を指定できます。
一般的に、サポートされたそれぞれの型について `OpKernel` を登録します。

たとえば、もし `ZeroOut` オペレーションについて、`int32` 型に加えて `float` 型をサポートするのであれば、オペレーションの登録は次のようになります。

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

オペレーションの登録は、どちらも型 `T` であることから、入力の型が `float` もしくは `int32` で、出力がおなじ型でなければならないと明示しています。

###### 命名

入力、出力、そしてアトリビュートは、スネークケースの名前にすべきです。入力の型や出力の型として使用されるアトリビュートは、例外です。これらのアトリビュートは、オペレーションがグラフに追加されるときに推論され、オペレーションの関数には現れてきません。たとえば、ZeroOutの最後の定義は、次のようなPythonの関数を生成します。

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

 もし `int32` のテンソルが `to_zero` に渡されてきた場合、`T` は自動的に `int32` （実際は、`DT_INT32`）が設定されます。これらの推論されたアトリビュートは、大文字もしくはキャメルケースで与えられます。

出力の型を決めるアトリビュート型を持つオペレーションと比較します。

```c++
REGISTER_OP("StringToNumber")
    .Input("string_tensor: string")
    .Output("output: out_type")
    .Attr("out_type: {float, int32} = DT_FLOAT");
    .Doc(R"doc(
Converts each string in the input Tensor to the specified numeric type.
)doc");
```

この場合、生成されたPythonのように、ユーザは出力の型を指定しなければなりません。

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
  // 前と同様
};

class ZeroOutFloatOp : public OpKernel {
 public:
  explicit ZeroOutFloatOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 入力テンソルを取得する。
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // 出力テンソルを作成する。
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<float>();

    // 出力テンソルのすべての要素を0にする。
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }

    // 最初の入力値は維持する。
    if (N > 0) output_flat(0) = input(0);
  }
};

// TypeConstraint<int32>("T") は、このテンプレートを具現化するときに使用するアトリビュート "T" が "int32" でなければならないことを意味していることに注意。
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

> [後方互換性](#後方互換性) を保つためには、既存のオペレーションに対してアトリビュートを追加するときに [デフォルト値](#デフォルト値と制約) を指定すべきです。
>
> ```c++
> REGISTER_OP("ZeroOut")
>   .Attr("T: {float, int32} = DT_INT32")
>   .Input("to_zero: T")
>   .Output("zeroed: T")
> ```

もしより多くの型、たとえば `double` を追加したいとしましょう。

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

前述のような冗長なコードでほかの `OpKernel` を書く代わりに、C++のテンプレートを使うことができます。
オーバーロード毎に、1つのカーネル登録（`REGISTER_KERNEL_BUILDER` 呼び出し）が必要になります。

```c++
template <typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // 入力テンソルを取得する。
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    
    // 出力テンソルを作成する。
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<T>();
    
    // 出力テンソルのすべての要素を0にする。
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
    
    // 最初の入力値は維持する。
    if (N > 0) output_flat(0) = input(0);
  }
};

// TypeConstraint<int32>("T") は、このテンプレートを具現化するときに使用するアトリビュート "T" が "int32" でなければならないことを意味していることに注意。
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

もし2つ以上のオーバーロードが必要な場合は、登録をマクロで行うことができます。

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

カーネルを登録する型のリストに依存しますが、[`tensorflow/core/framework/register_types.h`][register_types] で提供されているマクロを利用できます。

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

異なる型を受け取れる、もしくは出力できるようにすることに加えて、オペレーションは数多くのテンソルを消費または生成します。

次の例では、アトリビュート `T` は型の *リスト* を持ち、入力 `in` と出力 `out` の両方の型として使用されます。
入力と出力はその型のテンソルのリストです（そして、両方とも型 `T` をもつため、出力テンソルの型と数は入力とおなじです）。

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

リスト内で型を指定し、制限を加えることもできます。
次の場合は、入力が `float` と `double` のテンソルのリストです。
オペレーションは、たとえば入力型として `(float, double, float)` を受け取りますが、この場合出力型も `(float, double, float)` となります。

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

もしリスト内のすべてのテンソルがおなじ型であるならば、次のようにしてもよいです。

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

これは、`int32` のテンソルのリストを受け取り、リストの長さを指定するために、`int` のアトリビュート `N` を利用しています。

これは、[型ポリモーフィズム](#型ポリモーフィズム) でも同様です。
次の例では、入力がおなじ（ただし指定されていない）型（`"T"`）の（長さが `"N"` である）テンソルのリストで、出力がおなじ型の単一のテンソルです。

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

デフォルトでは、テンソルのリストは最小で1の長さを持ちます。
[対象とするアトリビュートの制約として `">="` ](#デフォルト値と制約) を利用することで、デフォルト値を変更できます。
次の例では、入力は少なくとも2つの `int32` のテンソルのリストです。

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

同様のシンタックスが、アトリビュート `"list(type)"` に対しても適用できます。

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```


#### 入力と出力

これまでの内容を要約すると、オペレーションの登録では、複数の入力と出力をもつことができます。

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

それぞれの入力と出力の指定は、次の形で行います。

```
<name>: <io-type-expr>
```

`<name>` は文字で始まり、英数字の文字とアンダースコアから構成できます。
`<io-type-expr>` は、次の型表現のいずれかになります。

* `<type>`: `<type>` は、サポートされる入力型（たとえば、`float`, `int32`, `string`）である。与えられた型の単一のテンソルであることを示す。`tf.DType` を参照のこと。

```c++
REGISTER_OP("BuiltInTypesExample")
    .Input("integers: int32")
    .Input("complex_numbers: complex64");
```

* `<attr-type>`: `<attr-type>` は、（型の制限がありうる）`type` もしくは `list(type)` の型をもつ [アトリビュート](#アトリビュート) の名前です。このシンタックスは、[ポリモーフィズムなオペレーション](#ポリモーフィズム) を許しています。

```c++
REGISTER_OP("PolymorphicSingleInput")
    .Attr("T: type")
    .Input("in: T");

REGISTER_OP("RestrictedPolymorphicSingleInput")
    .Attr("T: {int32, int64}")
    .Input("in: T");
```

`list(type)` 型のアトリビュートを参照することは、連続したテンソルを受け入れることになります。

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

両方ともおなじ型 `T` であるため、出力 `out` のテンソルの数と型は、入力 `in` とおなじであることに注意してください。

* おなじ型をもつテンソル列について: `<number> * <type>` となる。`<number>` は、`int` 型である [アトリビュート](#アトリビュート) の名前である。`<type>` は、`tf.DType` または `type` 型のアトリビュートの名前である。最初の例では、オペレーションは `int32` のテンソルのリストを受け取る。

```c++
REGISTER_OP("Int32SequenceExample")
    .Attr("NumTensors: int")
    .Input("in: NumTensors * int32")
```

一方このオペレーションは、すべておなじである限り、いかなる型のテンソルのリストを受け取ります。

```c++
REGISTER_OP("SameTypeSequenceExample")
    .Attr("NumTensors: int")
    .Attr("T: type")
    .Input("in: NumTensors * T")
```

* テンソルのリファレンスについて: `Ref(<type>)` となる。`<type>` は前述した型のいずれかになる。

入力の型として利用されている、いかなるアトリビュートは推論されます。慣例的に、推論されたアトリビュートは、（`T` や `N` のように）大文字の名前を使用します。そうでなければ、入力や出力、アトリビュートは、関数のパラメータ（たとえば `num_outputs`）のような名前を持ちます。詳細は、[前述した命名](#命名) を参照してください。

詳細は、[`tensorflow/core/framework/op_def_builder.h`][op_def_builder] を参照してください。


#### 後方互換性

すばらしいカスタムオペレーションを作り、ほかの人と共有し、顧客がそのオペレーションを使って喜んでいることを想定しましょう。
しかし、何らかの方法でオペレーションを変更したいとします。

一般的に、既存のチェックインされた仕様の変更は、後方互換でなければなりません。
すなわち、オペレーションの仕様変更は、前に古い仕様から構築してシリアライズ化された `GraphDef` プロトコルバッファを壊してはいけません。
`GraphDef` の互換性の詳細は、[ここで説明されています](https://github.com/tensorflow/docs/blob/master/site/en/guide/versions.md#compatibility-of-savedmodels-graphs-and-checkpoints)。

後方互換性を保つための方法は、いくつかあります。

1. オペレーションに新しく追加されたアトリビュートは、デフォルト値が定義されている必要があり、そのデフォルト値の場合には、オペレーションはもともとの動作にならなければなりません。非ポリモーフィズムからポリモーフィズムに変更するためには、新しい型アトリビュートにデフォルト値を与え、デフォルトでもとのシグネチャを保つ必要があります。たとえば、オペレーションが、次のようなものであった場合、

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: float")
           .Output("out: float");

   次のようにして、後方互換を維持しながらポリモーフィズムにできます。

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: T")
           .Output("out: T")
           .Attr("T: numerictype = DT_FLOAT");

2. アトリビュートの制約を緩めるのは安全に行えます。たとえば、`{int32, int64}` から `{int32, int64, float}` もしくは `type` に変更できます。また、`{"apple", "orange"}` を `{"apple", "banana", "orange"}` または `string` に変更できます。

3. リストの型のデフォルトが古いシグネチャと一致する限り、単一の入力/出力をリストの入力/出力に変更できます。

4. デフォルトが空であれば、新たなリストの入力/出力を追加できます。

5. オペレーションの名前に、プロジェクトに固有の名前をプレフィックスすることで、作成したオペレーションの名前空間を作れます。これにより、TensorFlowの将来のバージョンで含まれるかもしれないオペレーションとの衝突を回避できます。

6. 前もって計画してください！オペレーションの将来の使われ方を予想しましょう。いくつかのシグネチャの変更は、互換を保つ方法ではできません（たとえば、おなじ型のリストを異なる型のリストに変更するなど）。

変更が安全か安全でないかの一覧は、[`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc) にあります。
もし、オペレーションの変更を後方互換にできないならば、新しい名前かつ新しいセマンティクスで、新しいオペレーションを作ります。

これらの変更は、`GraphDef` の互換性を維持できますが、生成されたPythonコードは、呼び出し側にとって互換にならない変更をもたらすかもしれません。
Python APIは、新しい任意の引数を最後に追加するなどして古いシグネチャを保つなど、手書きのPythonラッパー内で注意深く変更することで、互換性を保てるかもしれません。
一般的に、互換性のない変更は、TensorFlowのメジャーバージョンが変更されたときのみ許されており、[`GraphDef` のバージョンのセマンティクス](./versions.md#compatibility_of_graphs_and_checkpoints) に従う必要があります。


### GPUサポート

[異なる型へカーネル登録](#ポリモーフィズム) できるのと同様に、CPUとGPU向けに異なるOpKernelを実装し、登録できます。
[`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/) には、GPUをサポートするカーネルの例がいくつかあります。
いくつかのカーネルは、`.cc` ファイルにCPUバージョン、`_gpu.cu.cc` で終わるファイルにはGPUバージョン、`.h` ファイルに共通で使用されるコードが存在していることに注意してください。

たとえば `tf.pad` は、[`tensorflow/core/kernels/pad_op.cc`][pad_op] にGPUのカーネル以外のすべてが存在します。
GPUのカーネルは、[`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc) にあり、共通のコードは [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h) に定義された、テンプレートクラスです。
CPUとGPUの実装間で共通のコードを共有できるようにし、GPUの実装を別のファイルに置くことでGPUのコンパイラだけがコンパイルできるようにする、という2つの理由から、このようなコードの管理になっています。

1つ注意することとしては、`pad` のGPUのカーネル版が使われたとしても、入力 `"paddings"` はCPUメモリに存在することです。
入力や出力をCPUに配置するためには、カーネル登録に `HostMemory()` 呼び出しを追加します。
たとえば、次のようにします。

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```


#### GPUデバイス向けのカーネルコンパイル

オペレーションを実装するために、CUDAカーネルを使用する例 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) を見てください。
`tf_custom_op_library` は、CUDAカーネル（`*.cu.cc` ファイル）を含むソースファイルのリストを指定できる `gpu_srcs` 引数を受け取ります。
バイナリインストールによるTensorFlowを使う場合、CUDAカーネルはNVIDIAの `nvcc` コンパイラによってコンパイルされます。
ここでは、[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) と [cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc) を1つの動的ロード可能なライブラリにコンパイルするために利用できる一連のコマンドを示します。

```bash
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

上記によって生成された `cuda_op_kernel.so` は、`tf.load_op_library` 関数を使って、いつものようにPythonからロードできます。

もしCUDAライブラリが `/usr/local/lib64` にインストールされていない場合、上記の2番目(g++)のコマンドにパスを指定する必要があることに注意してください。
たとえば、`/usr/local/cuda-8.0` にCUDAがインストールされている場合は、`-L /usr/local/cuda-8.0/lib64/` を追加します。

> 特定のLinuxの設定では、`nvcc` によるコンパイルの手順にいくつかのオプションが必要になることに注意してください。`mwaitxintrin.h` からのエラーを回避するためには、`nvcc` コマンドラインに `-D_MWAITXINTRIN_H_INCLUDED` を追加してください。


### Pythonにおける勾配の実装

オペレーションのグラフが与えられると、TensorFlowは自動微分（逆伝搬）を使って、存在するオペレーションに関して、勾配を表現するための新しいオペレーションを追加します。
新しいオペレーションに対して自動微分を動作させるためには、勾配を計算するための勾配関数を登録する必要があります。
これは、オペレーションの出力に関して与えられた勾配を入力とする、オペレーションに関する勾配を計算するものです。

数学的には、もしオペレーションが \\(y = f(x)\\) を計算するなら、登録された勾配のオペレーションは、\\(y\\) に関するロス \\(L\\) として、勾配 \\(\partial L/ \partial y\\) をチェインルールによって、\\(x\\) に関する勾配 \\(\partial L/ \partial x\\) に変換します。

$$\frac{\partial L}{\partial x}
    = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
    = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

`ZeroOut` の場合は、出力に影響を与える入力は1つのエントリだけであるため、入力に関する勾配は、スパースな "ワンホット" テンソルになります。
これは、次のように表現できます。

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """`zero_out` の勾配。

  Args:
    op: オリジナルのオペレーションの入力と出力を見つけるために使用する、
        微分対象の `zero_out` `Operation`。
    grad: `zero_out` オペレーションの出力に関する勾配。

  Returns:
    `zero_out` の入力に関する勾配。
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # 1つの入力しか持たないため、1つのテンソルのリスト
```

`tf.RegisterGradient` による勾配関数の登録の詳細を次に示します。

* 1つの出力をもつオペレーションでは、勾配関数は `tf.Operation` である `op`、`tf.Tensor` である `grad` を受け取り、`op.inputs[i]`、`op.outputs[i]`、`grad` を出力する新しいオペレーションをビルドします。アトリビュートに関する情報は、`tf.Operation.get_attr` 経由で確認できます。

* オペレーションが複数の出力をもつ場合、勾配関数は `op` と `grads` を受け取ります。`grads` は、それぞれの出力に関する勾配のリストです。勾配関数の結果は、それぞれの入力の勾配を表現するために、`Tensor` オブジェクトのリストでなければいけません。

* 整数の入力がインデックスとして使われている場合など、もし入力についてきちんと定義された勾配がない場合は、対応する結果の勾配の値は `None` にすべきです。たとえば、浮動小数点数のテンソル `x` とインデックス `i` を受け取るオペレーションについて、勾配関数は `[x_grad, None]` を返します。

* もしオペレーションの勾配がまったく意味がないものである場合、勾配を登録しなくてもよいでしょう。そしてこれは、オペレーションの勾配が必要でない限り、問題がありません。オペレーションがきちんと定義された勾配を持たない場合でも、勾配の演算に含めることはできます。ここで、自動的にゼロ逆伝搬を行うために、`ops.NotDifferentiable` を利用できます。

勾配関数が呼ばれたとき、オペレーションのデータフローグラフのみが利用可能であり、テンソルのデータ自体は利用できないことに注意してください。
このようにすべての計算は、グラフの実行時に実行されるほかのTensorFlowのオペレーションを使って行われなければなりません。


### C++でのシェイプ推論

TensorFlow APIは、グラフを実行することなくテンソルのシェイプの情報を提供するための"シェイプ推論"機能を持ちます。
シェイプ推論は、C++の `REGISTER_OP` 宣言における各オペレーションの型のために登録する"シェイプ関数"によってサポートされ、2つの役割を行います。
グラフ構築時に入力のシェイプに矛盾がないことを表明することと、出力のシェイプを決めることです。

シェイプ関数は、`shape_inference::InferenceContext` クラスのオペレーションとして定義されています。
たとえば、ZeroOutのシェイプ関数を次に示します。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` は、1番目の出力のシェイプに1番目の入力のシェイプが設定されるべきであると、宣言しています。
上記の例について、もし出力がインデックスによって選択される場合、`set_output` の2番目のパラメータは、`ShapeHandle` オブジェクトであるべきです。
デフォルトコンストラクタによって、空の `ShapeHandle` オブジェクトを作ることができます。
インデックス `idx` の入力のための `ShapeHandle` オブジェクトは、`c->input(idx)` によって得られます。 

[common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) にある `shape_inference::UnchangedShape` のように、数多くのオペレーションに適用する共通のシェイプ関数があり、次のように利用します。

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

シェイプ関数は、入力のシェイプに制約を与えることもできます。
[ベクトルのシェイプの制約がある `ZeroOut`](#条件のチェックと検証) について、シェイプ関数は次のようになります。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

`WithRank` は、入力のシェイプ `c->input(0)` がちょうど1次元のシェイプである（もしくは、もし入力のシェイプが不明なら、出力のシェイプは1つの不明な次元をもつベクトルになる）ことを検証します。

もしオペレーションが、[複数入力をもつポリモーフィズム](#ポリモーフィズム) なら、チェックするためのシェイプ数を決定したり、シェイプがすべて矛盾しないことを検証したりするために、`InferenceContext` のメンバ変数を利用できます（かわりに、オペレーションのアトリビュートへのアクセス手段を提供する `InferenceContext::GetAttr` を使って長さを示すアトリビュートにアクセスしてもよいです）。

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

シェイプ関数が任意の機能であり、テンソルのシェイプが動的に変更される場合があることから、シェイプ関数は入力の不完全なシェイプ情報に対して強固なものにしなければなりません。
[`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) にある `Merge` メソッドは、たとえどちらかまたは両方が完全な情報を持っていなくても、呼び出し元で2つのシェイプがおなじであることを、強く表明できます。
シェイプ関数は、すべてのTensorFlowのオペレーションに定義され、数多くの異なる使い方の例が提供されています。

`InferenceContext` クラスは、シェイプ関数の処理を定義するために使われる、多くの関数を持っています。
たとえば、`InferenceContext::Dim` と `InferenceContext::WithValue` を利用することで、特定の次元が特定の値をもつことを検査できます。
また、`InferenceContext::Add` と `InferenceContext::Multiply` を利用することで、出力の次元が2つの入力の次元の和または積であることを指定できます。
指定可能な数多くのシェイプの操作については、`InferenceContext` を参照してください。
次の例は、1番目の出力のシェイプを(n, 3)に設定し、1番目の入力がシェイプ(n, ...)をもつものです。

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

もし、複雑なシェイプ関数になる場合、さまざまなシェイプを持った入力を組み合わせ、期待した出力のシェイプの組み合わせが出力されることを検査するためのテストを追加することを考えてください。

このようなテストを書くための例が、[core ops tests](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc) にあります。（シンタックス `INFER_OK` と `INFER_ERROR` は少々不可解なものですが、テスト内で入力と出力のシェイプの仕様を表現するときに、簡潔になるようにしましょう。今のところは、シェイプの文字列指定の意味を理解するために、これらのテストにあるコメントを見てください。）


## カスタムオペレーションのpipパッケージをビルドする

オペレーションのpipパッケージをビルドするために、[tensorflow/custom-op](https://github.com/tensorflow/custom-op) の例を見てください。
このガイドでは、TensorFlowをソースコードからビルドする代わりに、TensorFlowのpipパッケージからカスタムオペレーションをビルドする方法が示されています。

[core-array_ops]:https://www.tensorflow.org/code/tensorflow/core/ops/array_ops.cc
[python-user_ops]:https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py
[tf-kernels]:https://www.tensorflow.org/code/tensorflow/core/kernels/
[user_ops]:https://www.tensorflow.org/code/tensorflow/core/user_ops/
[pad_op]:https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc
[standard_ops-py]:https://www.tensorflow.org/code/tensorflow/python/ops/standard_ops.py
[standard_ops-cc]:https://www.tensorflow.org/code/tensorflow/cc/ops/standard_ops.h
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[validation-macros]:https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h
[op_def_builder]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h
[register_types]:https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h
[FinalizeAttr]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc
[DataTypeString]:https://www.tensorflow.org/code/tensorflow/core/framework/types.cc
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[types-proto]:https://www.tensorflow.org/code/tensorflow/core/framework/types.proto
[TensorShapeProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto
[TensorProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor.proto

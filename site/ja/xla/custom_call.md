# XLA のカスタムコール

このドキュメントでは、XLA「カスタムコール」の記述と使い方について説明します。カスタムコールを使用すると、C++ や CUDA などのプログラミング言語で記述されたコードを、XLA プログラムから呼び出すことができます。

警告: カスタムコールは、パワーユーザ用の低レベル機能です。カスタムコールを使うとプログラムが壊れやすくなりデバッグしにくくなります（そして問題に気づきにくくなります）。問題が発生した場合、XLA をご自分でデバッグする準備ができていない場合は、カスタムコールを使うことは推薦できません。トラブルが発生した場合、XLA 開発者からのサポートはあまり期待できません。

警告: カスタムコールの API/ABI は、現時点では安定していません。きまぐれに変更するつもりはありませんが、変更する可能性はあります。今後の変更については以下で説明します。

## CPU でのカスタムコール

XLA クライアント API 経由で、カスタムコールを表す HLO 命令を作ることができます。これは、執筆時点では TensorFlow 経由では公開されていません。

たとえば、次のコードはカスタムコールを使用して、CPU で `A[i] = B[i % 128]+ C[i]` を計算します。（もちろん、通常の HLO でも実行でき、実行すべきです。）

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

関数 `do_custom_call` は、処理を実行するバッファの次元情報を知っている必要があります。この例では、サイズ 128 と 2048 をハードコーディングしています。ハードコーディングしない場合には、パラメータとして次元情報を関数に渡すことができます。

## GPU でのカスタムコール

GPU のカスタムコールのフレームワークは、CPU のフレームワークと多少異なります。ここでは、上記の CPU コードと同じ `A[i] = B[i % 128] + C[i]` の計算を行う CUDA の例をあげます。

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

まず、GPU カスタムコール関数が、*CPU 上で実行される関数である*ことに注意してください。CPU 用の `do_custom_call` 関数は、GPU 上での作業をキューに入れる役割を果たします。ここでは CUDA カーネルを起動していますが、cublas を呼び出すようなこともできます。

`buffers` はホスト上にあるポインタの配列で、各要素はデバイス（つまり GPU）メモリを指しています。パラメータが最初に来て、その後に出力の値が来ます。これは、CPU の呼び出し規約とは大きく異なり、２ つのパラメータ、`ins` と `out` があります。違う実装をした主な理由は、タプル型の入出力を効率的に処理するためです。以下のセクションを参照してください。

CPU の例のように、入出力バッファのサイズをカスタムコールにハードコーディングしました。しかし、CPU の場合とは異なり、オペランドとしてバッファの次元情報を渡してもうまく動作しません。通常、CPU 上でバッファのサイズが分かっている必要があります。例えば、カーネルを起動するとき、block/grid の次元情報が必要です。しかし、カスタムコールにオペランドとしてバッファサイズが渡されると、この値は GPU メモリ上にあります。処理の開始時に、この値を読むためのだけに処理が重い同期的なデバイスからホストへのメモリコピーを実行する必要があります。

これを回避するために `opaque` パラメータを用意しています。カスタムコールを作成するときに、任意のバイト文字列を設定できます。

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

`xla::Shape` はプロトコルバッファ表現を持つので、 `opaque` の内部にこのシリアライズされた表現を保存して GPU カスタムコールの内部でデシリアライズできます。ただし、`xla::ShapeProto` は頻繁には変更されませんが、*変更されることもあります*。git ログをチェックして、過去にどのような変更が行われたか確認してください。

## エラーの通知

カスタムコールでエラーが発生した場合は、CPU 上の関数に次のシグネチャを使用することで、エラーを XLA ランタイムに通知できます（クラッシュしたり、出力バッファーで意味のないものを返したりする代わりに）。

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

... GPU では次のようになります。

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

`XlaCustomCallStatusSetFailure` を使用して、エラーを通知できます。以下に例を示します。

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

`XlaCustomCallStatusSetSuccess` を使用して成功を示すこともできますが、`XlaCustomCallStatus` はデフォルトで成功状態であるため、完全に無視する場合は成功を示します。

このシグネチャでカスタムコール関数を使用する場合は、適切な API バージョンセットを使用して対応する `custom-call` 演算を作成する必要があります。以下に例を示します。

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

注意: 将来的には、すべてのクライアントがカスタムコール関数を新しい API バージョンに移行する必要があり、古いバージョンは推奨されなくなります。失敗しないカスタムコールの場合は、新しい `XlaCustomCallStatus*` パラメータを追加して無視するだけです。

失敗すると、カスタムコールの出力は使用されず、XLA ランタイムは計算を終了します。HLO 計算は、エラーから回復することはできません（例えば、エラーを見つけて処理することによって）。

## カスタムコールにタプルを渡す

以下のカスタムコールを考察してみましょう。

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
xla::CustomCall(&b, "do_custom_call", /*operands=*/<p>, out_shape);
```

CPU と GPU の両方で、タプルはポインタの配列としてメモリ内で表現されます。C++ 擬似コードでは、上記のパラメータ 0 は以下のように配置されます。

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

CPU と GPU でメモリ内表現は同じですが、CPU とGPU のカスタムコール呼び出し規約では処理方法が異なります。

### 一時バッファとしてのタプル出力

カスタムコールへのタプル入力は便利ですが、厳密には必須ではありません。カスタムコールへのタプル入力がサポートされていないなら、カスタムコールにタプルを渡す前に get-tuple-element を使ってタプルを分解できます。

一方、タプル*出力*は、他の方法ではできないことができます。

タプル出力を持つ明確な理由は、それがカスタムコール（または、他の XLA 命令）が複数の独立な配列を返す方法だからです。

明確な理由ではありませんが、タプル出力はカスタムコールに一時メモリを提供する方法でもあります。*出力*は一時バッファを表現できます。出力バッファは演算により書き込めるという性質を持っていて、書き込まれた後に読み出すことができます。これは、まさに一時バッファに必要とされるものです。

上の例で、`F32[1024]` を一時バッファとして使うとします。上記のように HLO を記述して、単にカスタムコールのタプルインデックス 1 を決して読まないようにします。

### CPU カスタムコールでのタプル

CPU コードには、`do_custom_call(const void** ins, void* out)` 関数があります。`ins` は `param0` を指す要素が 1 つだけの配列です。`param0` のサブバッファは、そのポインタをデリファレンスしてアクセスできます。`output_tuple` のサブバッファは、`out` をデリファレンスしてアクセスできます。

### GPU カスタムコールでのタプル

GPU コードには、`do_custom_call(..., void** buffers, ...)` 関数があります。この場合 `buffers` は、入出力の各末端のバッファが一要素に対応する、*６ 台*のデバイスポインタを持つホストの配列です。フラットリストを生成するために、パラメータと出力に対して反復処理を行い、それぞれについてその形状を行きがけ順に走査します。具体的な例は以下を参照してください。

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

# TensorFlow Lite デリゲート

注意: デリゲート API はまだ試験運用版であり、仕様変更の可能性があります。

## TensorFlow Lite デリゲートとは

A TensorFlow Lite delegate is a way to delegate part or all of graph execution to another executor.

## デリゲートを使用する理由

Running inference on compute-heavy machine learning models on mobile devices is resource demanding due to the devices' limited processing and power.

CPU に頼るのではなく、GPU や DSP などのハードウェアアクセラレータを搭載して、より高い性能とエネルギー効率を実現しているデバイスもあります。

## ビルトインのデリゲートを使用する

TensorFlow Lite は、ハードウェアアクセラレーション向けに以下のデリゲートを提供しています。

- **クロスプラットフォームアクセラレーション用 GPU デリゲート** - Android と iOS の両方で使用できます。GPU が利用可能な場合に 32 ビットおよび 16 ビットの浮動小数点数ベースのモデルを実行するように最適化されています。また、8 ビットの量子化されたモデルもサポートし、浮動小数点数モデルと同等の GPU パフォーマンスを提供します。GPU デリゲートに関する詳細については、[GPU の TensorFlow Lite](gpu_advanced.md) をご覧ください。Android および iOS の GPU デリゲートの使用に関する段階的なチュートリアルについては、[TensorFlow Lite GPU デリゲートチュートリアル](gpu.md)をご覧ください。
- **新しい Android デバイス用 NNAPI デリゲート** - NNAPI デリゲートは、GPU、DSP および/または NPU が利用可能な Android デバイス上でモデルの高速化に使用することができます。Android 8.1 (API 27+) 以上で利用可能です。NNAPI デリゲートの概要、段階的な手順、ベストプラクティスについては、[TensorFlow Lite NNAPI デリゲート](nnapi.md)をご覧ください。
- **古い Android デバイス用 Hexagon デリゲート** - Hexagon デリゲートは、Qualcomm Hexagon DSP を搭載した Android デバイスでモデルの高速化に使用することができます。NNAPI を完全にサポートしていない古いバージョンの Android OS デバイスで使用することができます。詳細については、[TensorFlow Lite Hexagon デリゲート](hexagon_delegate.md) をご覧ください。
- **新しい iPhone と iPad 用 Core ML デリゲート** - Neural Engine（ニューラルエンジン）が利用可能な最近の iPhone や iPad では、Core ML デリゲートを使用して 32 ビット浮動小数点数ベースのモデルの推論を高速化することができます。Neural Engine は、A12 SoC 以上の Apple モバイルデバイスで利用できます。Core ML デリゲートの概要と段階的な手順については、[TensorFlow Lite Core ML デリゲート](coreml_delegate.md)をご覧ください。

## デリゲートの仕組み

以下のような簡単なモデルグラフがあるとします。

![Original graph](../images/performance/tflite_delegate_graph_1.png "Original Graph")

特定の演算子用にデリゲートが提供されている場合、TensorFlow Lite はグラフを複数のサブグラフに分割し、各サブグラフをデリゲートで処理します。

ここでは、Conv2D と Mean の演算子について、デリゲート`MyDelegate`の方が実装が速いと仮定してみましょう。結果のメイングラフは以下のように更新されます。

![Graph with delegate](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/images/performance/tflite_delegate_graph_2.png?raw=true)

Each subgraph that is handled by a delegate will be replaced with a node that evaluates the subgraph on its invoked call.

モデルによっては、最終的なグラフが 1 つのノードで終わることがありますが、これはすべてのグラフがデリゲートされたか、複数のノードがサブグラフを処理したことを意味します。通常は、デリゲートからメイングラフに切り替えるたびにサブグラフからメイングラフに結果を渡すオーバーヘッドが発生するため、デリゲートによる複数のサブグラフの処理は避けるべきです。メモリの共有は必ずしも安全とは限りません。

## デリゲートの追加方法

*以下に使用する API は試験運用版であり、仕様変更の可能性があります。*

デリゲートを追加するには、前の項目に基づいて次のようにする必要があります。

1. デリゲートサブグラフの評価を担当するカーネルノードを定義します。
2. カーネルノードの登録とデリゲートが実行できるノードの取得を担当する [TfLiteDelegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L611) のインスタンスを作成します。

コードで表示するには、デリゲートを定義してそれを`MyDelegate`と呼びます。これは Conv2D と Mean 演算子の実行を高速化するものです。

```c++
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"

// This is where the execution of the operations or whole graph happens.
// The class below has an empty implementation just as a guideline
// on the structure.
class MyDelegate {
 public:
  // Returns true if my delegate can handle this type of op.
  static bool SupportedOp(const TfLiteRegistration* registration) {
    switch (registration->builtin_code) {
      case kTfLiteBuiltinConv2d:
      case kTfLiteBuiltinMean:
        return true;
      default:
        return false;
    }
  }

  // Any initialization code needed
  bool Init() {}
  // Any preparation work needed (e.g. allocate buffers)
  bool Prepare(TfLiteContext* context, TfLiteNode* node) {}
  // Actual running of the delegate subgraph.
  bool Invoke(TfLiteContext* context, TfLiteNode* node) {}
  // ... Add any other methods needed.
};

// Create the TfLiteRegistration for the Kernel node which will replace
// the subgraph in the main TfLite graph.
TfLiteRegistration GetMyDelegateNodeRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subgraph it replaces.
  // It is treated as an OP node. But in our case
  // Init will initialize the delegate.
  // Invoke will run the delegate graph.
  // Prepare for preparing the delegate.
  // Free for any cleaning needed by the delegate.
  TfLiteRegistration kernel_registration;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "MyDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<MyDelegate*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                   size_t) -> void* {
    // In the node init phase, initialize MyDelegate instance
    const TfLiteDelegateParams* delegate_params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    MyDelegate* my_delegate = new MyDelegate;
    if (!my_delegate->Init(context, params)) {
      return nullptr;
    }
    return my_delegate;
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                    TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

// TfLiteDelegate methods

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Claim all nodes that can be evaluated by the delegate and ask the
  // framework to update the graph with delegate kernel instead.
  std::vector<int> supported_nodes;
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
  TfLiteNode* node;
  TfLiteRegistration* registration;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (MyDelegate::SupportedOp(registration)) {
      supported_nodes.push_back(node_index);
    }
  }
  TfLiteRegistration my_delegate_kernel_registration =
      GetMyDelegateNodeRegistration();

  // This call split the graphs into subgraphs, for subgraphs that can be
  // handled by the delegate, it will replace it with a
  // 'my_delegate_kernel_registration'
  TfLiteIntArray* supported_nodes_int_array =
      ::tflite::ConvertVectorToTfLiteIntArray(supported_nodes);
  auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, my_delegate_kernel_registration,
      supported_nodes_int_array, delegate);
  TfLiteIntArrayFree(supported_nodes_int_array);
  return status
}

void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // Do any cleanups.
}

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                TfLiteDelegate* delegate,
                                TfLiteBufferHandle buffer_handle,
                                TfLiteTensor* tensor) {
  // Copies data from tensor to delegate buffer if needed.
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // Copies the data from delegate buffer into the tensor raw memory.
  return kTfLiteOk;
}

// Caller takes ownership of the returned pointer.
TfLiteDelegate* CreateMyDelegate() {
  TfLiteDelegate* delegate = new TfLiteDelegate;

  delegate->data_ = nullptr;
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &DelegatePrepare;
  // This cannot be null.
  delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
  // This can be null.
  delegate->CopyToBufferHandle = &CopyToBufferHandle;
  // This can be null.
  delegate->FreeBufferHandle = &FreeBufferHandle;

  return delegate;
}


// To add the delegate you need to call

auto* my_delegate = CreateMyDelegate();
if (interpreter->ModifyGraphWithDelegate(my_delegate) !=
        kTfLiteOk) {
  // Handle error
} else {
  interpreter->Invoke();
}
...
// Don't forget to delete your delegate
delete my_delegate;
```

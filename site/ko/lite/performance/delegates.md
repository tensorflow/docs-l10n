# TensorFlow Lite 대리자(delegate)

참고: 대리자 API는 아직 실험 단계이며 추후 변경될 수 있습니다.

## TensorFlow Lite 대리자란 무엇입니까?

TensorFlow Lite 대리자는 그래프 실행의 일부 또는 전체를 다른 executor에 위임하는 방법입니다.

## 대리자를 사용해야 하는 이유는 무엇입니까?

모바일 기기에서 컴퓨팅이 많은 머신러닝 모델에 대한 추론을 실행하는 것은 기기의 제한된 처리 및 전력으로 인해 리소스가 많이 필요합니다.

CPU에 의존하는 대신 일부 기기에는 GPU 또는 DSP와 같은 하드웨어 가속기가 있어 성능과 에너지 효율성을 높일 수 있습니다.

## 내장 대리자 사용하기

TensorFlow Lite는 하드웨어 가속을 위해 다음 대리자를 제공합니다.

- **크로스 플랫폼 가속을 위한 GPU 대리자** - GPU 대리자는 Android와 iOS 모두에서 사용할 수 있습니다. GPU를 사용할 수 있는 32bit 및 16bit 부동 기반 모델을 실행하도록 최적화되어 있습니다. GPU 대리자에 대한 개요는 [GPU의 TensorFlow Lite](gpu_advanced.md)를 참조하세요. Android 및 iOS에서 GPU 대리자를 사용하는 방법에 대한 단계별 튜토리얼은 [TensorFlow Lite GPU 대리자 튜토리얼](gpu.md)을 참조하세요.
- **최신 Android 기기용 NNAPI 대리자** - NNAPI 대리자를 사용하여 GPU, DSP 및/또는 NPU를 사용할 수 있는 Android 기기에서 모델을 가속화할 수 있습니다. Android 8.1(API 27+) 이상에서 사용할 수 있습니다. NNAPI 대리자 개요, 단계별 지침 및 모범 사례는 [TensorFlow Lite NNAPI 대리자](nnapi.md)를 참조하세요.
- **구형 Android 기기용 Hexagon 대리자** - Qualcomm Hexagon DSP를 사용하는 Android 기기에서 Hexagon 대리자를 사용하여 모델을 가속화할 수 있습니다. NNAPI를 완전히 지원하지 않는 이전 버전의 Android OS 기기에서 사용할 수 있습니다. 자세한 내용은 [TensorFlow Lite Hexagon 대리자](hexagon_delegate.md)를 참조하세요.
- **최신 iPhone 및 iPad용 Core ML 대리자** - Neural Engine을 사용할 수 있는 최신 iPhone 및 iPad의 경우 Core ML 대리자를 사용하여 32bit 부동 기반 모델에 대한 추론을 가속화할 수 있습니다. Neural Engine은 A12 SoC 이상의 Apple 모바일 기기를 사용할 수 있습니다. Core ML 대리자에 대한 개요 및 단계별 지침은 [TensorFlow Lite Core ML 대리자](coreml_delegate.md)를 참조하세요.

## 대리자는 어떻게 동작합니까?

다음과 같은 간단한 모델 그래프가 있다고 가정해 보겠습니다.

![Original graph](../images/performance/tflite_delegate_graph_1.png "원본 그래프")

특정 연산에 대리자가 제공된 경우 TensorFlow Lite는 그래프를 여러 하위 그래프로 분할하고 각 하위 그래프는 대리자가 처리합니다.

대리자 `MyDelegate`가 Conv2D 및 Mean 연산을 더 빠르게 구현한다고 가정해 보겠습니다. 그 결과 기본 그래프는 아래와 같이 보이도록 업데이트됩니다.

![Graph with delegate](../images/performance/tflite_delegate_graph_2.png "대리자가있는 그래프")

대리자가 처리하는 각 하위 그래프는 호출된 호출에서 하위 그래프를 평가하는 노드로 대체됩니다.

모델에 따라 최종 그래프는 하나의 노드로 끝날 수 있습니다. 즉, 모든 그래프가 위임되었거나 여러 노드가 하위 그래프를 처리했음을 의미합니다. 일반적으로 대리자에서 기본 그래프로 전환할 때마다 결과를 하위 그래프에서 기본 그래프로 전달하는 오버헤드가 있으므로 대리자가 여러 개의 하위 그래프를 처리하지 않도록 해야 합니다. 메모리 공유가 항상 안전한 것은 아닙니다.

## 대리자를 추가하는 방법

*아래 사용된 API는 실험적이며 추후 변경될 수 있습니다.*

이전 섹션에 따라 대리자를 추가하려면 다음을 수행해야 합니다.

1. 대리자 하위 그래프를 평가하는 커널 노드를 정의합니다.
2. 커널 노드를 등록하고 대리자가 실행할 수 있는 노드를 요청하는 [TfLiteDelegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L611) 인스턴스를 만듭니다.

코드에서 확인하기 위해 대리자를 정의하고 `MyDelegate`라고 부르면 Conv2D 및 Mean 연산을 더 빠르게 실행할 수 있습니다.

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

# 사용자 지정 대리자 구현

[TOC]

## TensorFlow Lite 대리자란?

TensorFlow Lite [Delegate](https://www.tensorflow.org/lite/performance/delegates)를 사용하면 다른 실행기에서 모델(일부 또는 전체)을 실행할 수 있습니다. 이 메커니즘은 추론을 위해 GPU 또는 Edge TPU(Tensor Processing Unit)와 같은 다양한 온디바이스 가속기를 활용할 수 있습니다. 이를 통해 개발자는 기본 TFLite에서 유연하고 분리된 방식으로 추론 속도를 높일 수 있습니다.

아래 다이어그램에 대리자를 요약했으며 자세한 내용은 아래 섹션을 참조하세요.

![TFLite 대리자](images/tflite_delegate.png "TFLite Delegates")

## 사용자 지정 대리자는 어떤 경우에 만들어야 합니까?

TensorFlow Lite에는 GPU, DSP, EdgeTPU 및 Android NNAPI와 같은 프레임워크 등 대상 가속기를 위한 다양한 대리자가 있습니다.

다음과 같은 경우에 고유한 대리자를 만들면 유용합니다.

- 기존 대리자가 지원하지 않는 새 ML 추론 엔진을 통합하려고 합니다.
- 알려진 시나리오에 대한 런타임을 개선하는 사용자 지정 하드웨어 가속기가 있습니다.
- 특정 모델의 속도를 높일 수 있는 CPU 최적화(예: 연산자 융합)를 개발 중입니다.

## 대리자는 어떻게 동작합니까?

다음과 같은 간단한 모델 그래프와 Conv2D 및 Mean 작업을 더 빠르게 구현하는 대리자 "MyDelegate"를 고려합니다.

![원본 그래프](../images/performance/tflite_delegate_graph_1.png "원본 그래프")

이 "MyDelegate"를 적용하면 원본 TensorFlow Lite 그래프가 다음과 같이 업데이트됩니다.

![대리자가 있는 그래프](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

위의 그래프는 TensorFlow Lite가 두 가지 규칙에 따라 원본 그래프를 분할할 때 얻어집니다.

- 대리자가 처리할 수 있는 특정 작업은 연산 사이의 원래 컴퓨팅 워크플로 종속성을 충족하면서 파티션에 배치됩니다.
- 각 대리할 파티션에는 대리자가 처리하지 않는 입력 및 출력 노드만 있습니다.

대리자가 처리하는 각 파티션은 호출 시 파티션을 평가하는 원본 그래프의 대리자 노드(대리자 커널이라고도 함)로 대체됩니다.

모델에 따라 최종 그래프는 하나 이상의 노드로 끝날 수 있으며 후자는 대리자가 일부 op를 지원하지 않음을 의미합니다. 일반적으로, 대리자가 여러 파티션을 처리하는 것은 바람직하지 않은데, 대리자에서 기본 그래프로 전환할 때마다 메모리 복사본(예: GPU에서 CPU로)으로 인해 위임된 하위 그래프의 결과를 기본 그래프로 전달하기 위한 오버헤드가 발생하기 때문입니다. 이러한 오버헤드는 특히 많은 양의 메모리 복사본이 있는 경우 성능 향상을 상쇄시킬 수 있습니다.

## 사용자 지정 대리자 구현하기

대리자를 추가하는 기본 방법은 [SimpleDelegate API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h)를 사용하는 것입니다.

새 대리자를 만들려면 2개의 인터페이스를 구현하고 인터페이스 메서드에 대한 고유한 구현을 제공해야 합니다.

### 1 - `SimpleDelegateInterface`

이 클래스는 지원되는 연산과 위임된 그래프를 캡슐화하는 커널을 생성하기 위한 팩토리 클래스의 기능을 나타냅니다. 자세한 내용은 이 [C++ 헤더 파일](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L71)에 정의된 인터페이스를 참조하세요. 코드의 주석은 각 API에 대해 자세히 설명합니다.

### 2 - `SimpleDelegateKernelInterface`

이 클래스는 위임된 파티션을 초기화/준비/실행하기 위한 논리를 캡슐화합니다.

다음이 있습니다([정의](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L43) 참조).

- Init(...): 일회성 초기화를 수행하기 위해 한 번 호출됩니다.
- Prepare(...): 이 노드의 서로 다른 인스턴스 각각에 대해 호출됩니다. 이는 위임된 파티션이 여러 개인 경우에 발생합니다. 일반적으로 여기에서 메모리 할당을 수행해야 하는데, 이는 텐서의 크기가 조정될 때마다 호출되기 때문입니다.
- Invoke(...): 추론을 위해 호출됩니다.

### 예제

이 예제에서는 float32 텐서만 있는 두 가지 유형의 연산 (ADD) 및 (SUB)만 지원할 수 있는 매우 간단한 대리자를 만듭니다.

```
// MyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class MyDelegate : public SimpleDelegateInterface {
 public:
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    if (kTfLiteBuiltinAdd != registration->builtin_code &&
        kTfLiteBuiltinSub != registration->builtin_code)
      return false;
    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32) return false;
    }
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "MyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<MyDelegateKernel>();
  }
};
```

다음으로, `SimpleDelegateKernelInterface`에서 상속하여 자신만의 대리자 커널을 만듭니다.

```
// My delegate kernel.
class MyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
    }
    return kTfLiteOk;
  }

 private:
  // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
  // and store the result in 'output_tensor'.
  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
        NumElements(input_tensor_1) != NumElements(output_tensor)) {
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);
    for (int i = 0; i < NumElements(input_tensor_1); ++i) {
      if (builtin_code == kTfLiteBuiltinAdd)
        output[i] = input_1[i] + input_2[i];
      else
        output[i] = input_1[i] - input_2[i];
    }
    return kTfLiteOk;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;
};


```

## 새 대리자 벤치마킹 및 평가

TFLite에는 TFLite 모델에 대해 빠르게 테스트할 수 있는 도구 세트가 있습니다.

- [모델 벤치마크 도구](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark): 이 도구는 TFLite 모델을 사용하여 임의 입력을 생성한 다음 지정된 실행 횟수만큼 모델을 반복적으로 실행합니다. 마지막에 집계된 대기 시간 통계를 인쇄합니다.
- [추론 Diff 도구](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff): 주어진 모델에 대해 이 도구는 임의의 가우시안 데이터를 생성하고 두 개의 다른 TFLite 인터프리터를 통해 전달합니다. 하나는 단일 스레드 CPU 커널을 실행하고 다른 하나는 사용자 정의 사양을 사용합니다. 요소별로 각 인터프리터의 출력 텐서 간의 절대 차이를 측정합니다. 이 도구는 정확도 문제를 디버깅하는 데도 유용할 수 있습니다.
- 이미지 분류 및 물체 감지를 위한 작업별 평가 도구도 있습니다. 이러한 도구는 [여기](https://www.tensorflow.org/lite/performance/delegates#tools_for_evaluation)에서 찾을 수 있습니다.

또한 TFLite에는 더 많은 범위의 새 대리자를 테스트하고 일반 TFLite 실행 경로가 단절되지 않았는지 확인하는 데 재사용할 수 있는 대규모 커널 및 op 단위 테스트가 있습니다.

새 대리자에 대해 TFLite 테스트 및 도구를 재사용하려면 다음 두 가지 중 하나를 사용할 수 있습니다.

- [대리자 등록계](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates) 메커니즘을 활용합니다.
- [외부 대리자](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external) 메커니즘을 활용합니다.

### 최상의 접근 방식 선택

두 접근 방식 모두 아래에 설명된 대로 몇 가지 변경 사항이 필요합니다. 그러나 첫 번째 접근 방식은 대리자를 정적으로 연결하고 테스트, 벤치마킹 및 평가 도구를 다시 빌드해야 합니다. 대조적으로, 두 번째 접근 방식은 대리자를 공유 라이브러리로 만들고 공유 라이브러리에서 생성/삭제 메서드를 노출하도록 요구합니다.

결과적으로, 외부 대리자 메커니즘은 TFLite의 [사전 구축된 Tensorflow Lite 도구 바이너리](#download-links-for-nightly-pre-built-tflite-tooling-binaries)와 함께 작동합니다. 그러나 덜 명시적이며 자동화된 통합 테스트에서 설정하기가 더 복잡할 수 있습니다. 명확성을 위해서는 대리자 등록계 접근 방식을 사용하세요.

### 옵션 1: 대리자 등록계 활용

[대리자 등록계](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)는 대리자 공급자 목록을 유지하며, 그 각각은 명령줄 플래그를 기반으로 TFLite 대리자를 만드는 쉬운 방법을 제공하므로 툴링에 편리합니다. 위에서 언급한 모든 Tensorflow Lite 도구에 새 대리자를 연결하려면 먼저 [이것](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/hexagon_delegate_provider.cc)과 같은 새 대리자 공급자를 만든 다음 BUILD 규칙을 약간만 변경합니다. 이 통합 프로세스의 전체 예가 아래에 나와 있습니다(코드는 [여기](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/dummy_delegate)에서 찾을 수 있음).

SimpleDelegate API를 구현하는 대리자와 아래와 같이 이 '더미' 대리자를 생성/삭제하는 외부 "C" API가 있다고 가정합니다.

```
// Returns default options for DummyDelegate.
DummyDelegateOptions TfLiteDummyDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options);

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate);
```

"DummyDelegate"를 벤치마크 도구 및 추론 도구와 통합하려면 아래와 같이 DelegateProvider를 정의합니다.

```
class DummyDelegateProvider : public DelegateProvider {
 public:
  DummyDelegateProvider() {
    default_params_.AddParam("use_dummy_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "DummyDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DummyDelegateProvider);

std::vector<Flag> DummyDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_dummy_delegate", params,
                                              "use the dummy delegate.")};
  return flags;
}

void DummyDelegateProvider::LogParams(const ToolParams& params) const {
  TFLITE_LOG(INFO) << "Use dummy test delegate : ["
                   << params.Get<bool>("use_dummy_delegate") << "]";
}

TfLiteDelegatePtr DummyDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_dummy_delegate")) {
    auto default_options = TfLiteDummyDelegateOptionsDefault();
    return TfLiteDummyDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

```

BUILD 규칙 정의는 라이브러리가 항상 링크되어 있고 최적화 프로그램에 의해 삭제되지 않도록 해야 하므로 중요합니다.

```
#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "dummy_delegate_provider",
    srcs = ["dummy_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1, # This is required so the optimizer doesn't optimize the library away.
)
```

이제 BUILD 파일에 이 두 래퍼 규칙을 추가하여 벤치마크 도구 및 추론 도구, 그리고 자체 대리자와 함께 실행할 수 있는 기타 평가 도구 버전을 만듭니다.

```
cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)
```

[여기](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#kernel-tests)에 설명된 대로 이 대리자 공급자를 TFLite 커널 테스트에 연결할 수도 있습니다.

### 옵션 2: 외부 대리자 활용

이 대안에서는 먼저 아래와 같이 외부 대리자 어댑터 [external_delegate_adaptor.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc)를 만듭니다. 이 접근 방식은 [앞서 언급한](#comparison-between-the-two-options) 옵션 1에 비해 선호도가 약간 떨어집니다.

```
TfLiteDelegate* CreateDummyDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();

  // Parse key-values options to DummyDelegateOptions.
  // You can achieve this by mimicking them as command-line flags.
  std::unique_ptr<const char*> argv =
      std::unique_ptr<const char*>(new const char*[num_options + 1]);
  constexpr char kDummyDelegateParsing[] = "dummy_delegate_parsing";
  argv.get()[0] = kDummyDelegateParsing;

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.get()[i + 1] = option_args.rbegin()->c_str();
  }

  // Define command-line flags.
  // ...
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(...),
      ...,
      tflite::Flag::CreateFlag(...),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.get(), flag_list)) {
    return nullptr;
  }

  return TfLiteDummyDelegateCreate(&options);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateDummyDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteDummyDelegateDelete(delegate);
}

#ifdef __cplusplus
}
#endif  // __cplusplus
```

이제 해당하는 BUILD 대상을 생성하여 아래와 같이 동적 라이브러리를 빌드합니다.

```
cc_binary(
    name = "dummy_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)
```

이 외부 대리자 .so 파일이 생성된 후, 바이너리가 [여기](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159)에 설명된 대로 명령줄 플래그를 지원하는 [external_delegate_provider](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider) 라이브러리와 연결되어 있는 한 바이너리를 빌드하거나 미리 빌드된 파일을 사용하여 새 대리자로 실행할 수 있습니다. 참고: 이 외부 대리자 공급자는 이미 기존 테스트 및 도구 바이너리에 연결되어 있습니다.

이 외부 대리자 접근 방식을 통해 더미 대리자를 벤치마킹하는 방법에 대한 설명은 [여기](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#option-2-utilize-tensorflow-lite-external-delegate)의 설명을 참조하세요. 앞서 언급한 테스트 및 평가 도구에 대해 유사한 명령을 사용할 수 있습니다.

*외부 대리자*는 <a>여기</a>에 표시된 것처럼 Tensorflow Lite Python 바인딩에서 해당 <em>대리자</em>의 C++ 구현이라는 점에 주목할 필요가 있습니다. 따라서 여기에서 만든 동적 외부 대리자 어댑터 라이브러리를 Tensorflow Lite Python API와 함께 직접 사용할 수 있습니다.

## 리소스

### 야간 사전 구축 TFLite 도구 바이너리에 대한 링크 다운로드

<table>
  <tr>
   <td>OS</td>
   <td>ARCH</td>
   <td>BINARY_NAME</td>
  </tr>
  <tr>
   <td rowspan="3">Linux</td>
   <td>x86_64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>arm</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td rowspan="2">Android</td>
   <td>arm</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
</table>

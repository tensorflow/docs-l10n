# カスタムデリゲートの実装

[TOC]

## TensorFlow Lite デリゲートとは

TensorFlow Lite [デリゲート](https://www.tensorflow.org/lite/performance/delegates)では、モデルの一部またはモデル全体を別の Executor で実行できます。このメカニズムでは、GPU や Edge TPU (Tensor Processing Unit) などのさまざまなオンデバイスアクセラレータを活用して、推論を実行できます。これにより、開発者は、既定の TFLite の柔軟な分離された方法を使用して、推論を高速化できます。

次の図は、デリゲートの概要です。詳細については、その後のセクションで説明します。

![TFLite デリゲート](images/tflite_delegate.png "TFLite Delegates")

## カスタムデリゲートを作成するとき

TensorFlow Lite には、GPU、DSP、EdgeTPU などのターゲットアクセラレータ、および Android NNAPI などのフレームワーク用の、さまざまなデリゲートがあります。

次のシナリオでは、独自のデリゲートの作成が効果的です。

- 既存のデリゲートではサポートされていない新しい ML 推論エンジンを統合したい。
- 既知のシナリオのランタイムを改善するカスタムハードウェアアクセラレータがある。
- 特定のモデルを高速化できる CPU 最適化 (オペレータ融合など) を開発している。

## デリゲートの仕組み

次のようなシンプルなモデルグラフと、Conv2D および Mean 演算の実装が高速なデリゲート「MyDelegate」について考えます。

![元のグラフ](../images/performance/tflite_delegate_graph_1.png "Original Graph")

この「MyDelegate」を適用した後、元の TensorFlow Lite グラフは次のように更新されます。

![デリゲートのグラフ](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

上記のグラフは、TensorFlow Lite が 次の 2 つのルールに従って元のグラフを分割するときに、取得されます。

- デリゲートで処理できる特定の演算が、演算間の元のコンピューティングワークフロー依存関係を満たしながら、パーティションに分割される。
- 各デリゲート対象パーティションには、デリゲートで処理されない入出力ノードのみが存在する。

デリゲートで処理される各パーティションは、呼び出し時にパーティションを評価する元のグラフで、デリゲートノード (別称: デリゲートカーネル) に置換されます。

モデルによっては、最終的なグラフのノードが 1 つになったり、複数になったりすることがあります。ノードが複数になる場合は、一部の演算がデリゲートによってサポートされません。一般的には、デリゲートで複数のパーティションを処理することは避けた方が良いでしょう。デリゲートをメイングラフに切り替えるたびに、メモリコピー (GPU から CPU へのコピーなど) が原因で、デリゲートされたサブグラフからメイングラフに結果を渡すときに負荷が発生するためです。特に、大量のメモリコピーが発生するときには、このような負荷によって、パフォーマンスの改善が帳消しになってしまう場合があります。

## カスタムデリゲートの実装

デリゲートを追加するには、[SimpleDelegate API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h) を使用することをお勧めします。

新しいデリゲートを作成するには、2 つのインターフェイスを実装し、インターフェイスメソッドを独自に実装する必要があります。

### 1 - `SimpleDelegateInterface`

このクラスは、演算がサポートされているデリゲートと、デリゲートされたグラフをカプセル化するカーネルを作成するためのファクトリクラスの機能を表します。詳細については、[C++ ヘッダーファイル](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L71)で定義されているインターフェイスを参照してください。コード内のコメントは、各 API の詳細説明です。

### 2 - `SimpleDelegateKernelInterface`

このクラスは、デリゲートされたパーティションを初期化、準備、実行するロジックをカプセル化します。

次のロジックがあります ([定義](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L43)を参照)。

- Init(...): 初期化を 1 回だけ実行するために呼び出されます。
- Prepare(...): このノードの各インスタンスで呼び出されます。この呼び出しは、複数のデリゲートされたパーティションがある場合に実行されます。テンソルのサイズが変更されるたびに呼び出されるため、通常は、ここでメモリ割り当てを実行するのが望ましいでしょう。
- Invoke(...): 推論で呼び出されます。

### 例

この例では、float32 テンソルで AAD と SUB の 2 種類の演算のみをサポートする、非常にシンプルなデリゲートを作成します。

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

次に、`SimpleDelegateKernelInterface` から継承し、独自のデリゲートカーネルを作成します。

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

## 新しいデリゲートのベンチマークと評価

TFLite には、TFLite モデルに対してすばやくテストができるさまざまなツールが用意されています。

- [モデルベンチマークツール](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark): TFLite モデルを取得し、ランダム入力を生成して、指定した実行回数の間、モデルを繰り返し実行します。最後に、遅延統計情報の集計が出力されます。
- [推論 Diff ツール](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff): 特定のモデルに対して、ランダム Gaussian データを生成し、そのデータを 2 つの異なる TFLite インタープリタ（シングルスレッド CPU カーネルを実行するインタープリタと、ユーザー定義仕様を使用いているインタープリタ) に渡します。各インタープリタからの出力テンソル間の絶対差異を、要素ごとに測定します。このツールは、精度の問題をデバッグするときに役立ちます。
- 画像分類および物体検出では、タスク固有の評価ツールを使用することもできます。これらのツールは、[こちら](https://www.tensorflow.org/lite/performance/delegates#tools_for_evaluation)をご覧ください。

また、TFLite には、多数のカーネルと演算単体テストのセットがあります。これらを再利用すると、対象範囲を広げて、新しいデリゲートをテストし、標準の TFLite 実行パスが破損していないことを確認できます。

新しいデリゲートで、TFLite テストとツールを再利用するには、次の 2 つのオプションのいずれかを使用します。

- [デリゲートレジストラ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)
- [外部デリゲート](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external)メカニズム

### 最適なアプローチの選定

以下で詳述するとおり、いずれのアプローチにもいくつかの変更が必要です。ただし、最初のアプローチでは、デリゲートを統計的にリンクし、テスト、ベンチマーク、評価ツールの再構築が必要です。それに対して、2 番目のアプローチでは、デリゲートを共有ライブラリにし、共有ライブラリから作成/削除メソッドを公開する必要があります。

このため、外部デリゲートメカニズムは、[ビルト済みの Tensorflow Lite ツールバイナリ](#download-links-for-nightly-pre-built-tflite-tooling-binaries)で動作します。ただし、自動統合テストで設定するのは、明示的ではなく、複雑化する可能性があります。明確にするには、デリゲートレジストラアプローチを使用してください。

### オプション 1: デリゲートレジストラの利用

[デリゲートレジストラ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates)は、デリゲートプロバイダのリストを保持します。各デリゲートプロバイダは、TFLite デリゲートを作成するためのコマンドラインフラグに基づく使いやすい方法と便利なツールを提供します。新しいデリゲートを上記のすべての Tensorflow Lite ツールに接続するには、まず、[このデリゲートプロバイダ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/hexagon_delegate_provider.cc)のような新しいデリゲートプロバイダを作成してから、ビルドルールにいくつかの変更を加えます。この統合プロセスの詳細な例については、以下を参照してください (コードは[こちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/dummy_delegate))。

次のように、SimpleDelegate API を実装するデリゲートと、この「ダミー」デリゲートを作成/削除する extern C API があるとします。

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

DummyDelegate をベンチマークツールと推論ツールに統合するには、次のように DelegateProvider を定義します。

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

ライブラリが常にリンクし、オプティマイザによって破棄されないことを保証する必要があるため、ビルドルール定義は重要です。

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

ビルドファイルにこれらの 2 つのラッパールールを追加して、ベンチマークツールと推論ツールのバージョン、および独自のデリゲートで実行できる他の評価ツールを作成します。

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

[こちら](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#kernel-tests)で説明しているように、このデリゲートプロバイダを TFLite カーネルテストに接続することもできます。

### オプション 2: 外部デリゲートの利用

この代替策では、まず、次のように、外部デリゲートアダプタ [external_delegate_adaptor.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc) を作成します。[上記](#comparison-between-the-two-options)のとおり、このアプローチよりも、オプション 1 が推奨されます。

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

次のように、対応するビルドターゲットを作成し、動的ライブラリを構築します。

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

この外部デリゲート .so ファイルが作成された後は、バイナリをビルドするか、ビルド済みのバイナリを使用して、バイナリが [external_delegate_provider](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159) ライブラリに関連付けられているかぎり、新しいデリゲートで実行できます。[こちら](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider)で説明しているとおり、このライブラリはコマンドラインフラグをサポートしています。注意: この外部デリゲートプロバイダはすでに既存のテストおよびツールバイナリに関連付けられています。

この外部デリゲートアプローチで、ダミーデリゲートをベンチマーク評価する方法の例については、[こちら](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#option-2-utilize-tensorflow-lite-external-delegate)の説明を参照してください。上記のテストおよび評価ツールでも、同様のコマンドを使用できます。

[こちら](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42)で示すように、*外部デリゲート*は、Tensorflow Lite Python バインディングにおける*デリゲート*の対応する C++ 実装です。このため、ここで作成された動的外部デリゲートアダプタライブラリは、直接 Tensorflow Lite Python API で使用できます。

## 参考資料

### 構築済み TFLite ツールバイナリのダウンロードリンク

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

# C/C++ API による GPU アクセラレーションのデリゲート

GPU を使って機械学習（ML）モデルを実行すると、ML 駆動型アプリケーションのパフォーマンスとユーザーエクスペリエンスを大幅に改善できます。Android デバイスでは、[*デリゲート*](../../performance/delegates)と以下のいずれかの API を使用して、モデルの GPU 高速実行が可能になります。

- Interpreter API - [ガイド](./gpu)
- Task library API - [ガイド](./gpu_task)
- Native（C/C++）API - このガイド

このページでは、C API と C++ API の GPU デリゲートの高度な使用方法と量子化モデルの使用について説明します。ベストプラクティスや高度な手法など、TensorFlow Lite で GPU アクセラレーションを使用する方法についての詳細は、[GPU デリゲート](../../performance/gpu)のページをご覧ください。

## GPU アクセラレーションの有効化

次のコード例に示すように、`TfLiteGpuDelegateV2Create()` でデリゲートを作成し、`TfLiteGpuDelegateV2Delete()` で破棄することにより、C または C++ で Android 用の TensorFlow Lite GPU デリゲートを使用します。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

`TfLiteGpuDelegateOptionsV2` オブジェクトコードを確認して、カスタムオプションを使用してデリゲートインスタンスをビルドします。`TfLiteGpuDelegateOptionsV2Default()` でデフォルトオプションを初期化し、必要に応じて変更できます。

C または C++ の Android 用の TensorFlow Lite GPU デリゲートは、[Bazel](https://bazel.io) ビルドシステムを使用します。次のコマンドを使用してデリゲートをビルドできます。

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

`Interpreter::ModifyGraphWithDelegate()` または `Interpreter::Invoke()` を呼び出す場合、呼び出し元はその時点のスレッドに `EGLContext` を持ち、`Interpreter::Invoke()` は同じ `EGLContext` から呼び出す必要があります。`EGLContext` が存在しない場合、デリゲートは内部で EGLContext を作成しますが、`Interpreter::ModifyGraphWithDelegate()` が呼び出されたスレッドと同じスレッドから `Interpreter::Invoke()` が常に呼び出されるようにする必要があります。

## 量子化モデル {:#quantized-models}

Android GPU デリゲートライブラリは、デフォルトで量子化モデルをサポートします。 GPU デリゲートで量子化モデルを使用するためにコードを変更する必要はありません。次のセクションでは、テストまたは実験目的で量子化サポートを無効にする方法について説明します。

#### 量子化モデルのサポートの無効化

次のコードは、量子化されたモデルのサポートを***無効***にする方法を示しています。

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

GPU アクセラレーションを使用した量子化モデルの実行の詳細については、[GPU デリゲート](../../performance/gpu#quantized-models)の概要を参照してください。

# カスタム演算子

TensorFlow Lite のビルトイン演算子ライブラリがサポートする TensorFlow 演算子は制限されているため、すべてのモデルは変換可能ではありません。詳細は、[演算子の互換性](ops_compatibility.md)をご覧ください。

変換を可能にするためには、カスタム演算子を使用します。カスタム演算子とは、TensorFlow Lite でサポートされていない TensorFlow 演算子のカスタム実装です。*また、サポートされていない（またはサポートされている）一連の TensorFlow 演算子を、融合して最適化された単一のカスタム演算子として使用できます。詳細は[演算子の融合](https://www.tensorflow.org/lite/convert/operation_fusion)を参照してください。*

カスタム演算子を使用するには、次の 4 つのステップに従います。

- [TensorFlow モデルを作成する。](#create-a-tensorflow-model)Saved Model（または Graph Def）が正しく名付けられた TensorFlow Lite 演算子を参照するようにします。

- [TensorFlow Lite モデルに変換する。](#convert-to-a-tensorflow-lite-model)モデルを正しく変換するには、適切な TensorFlow Lite コンバータ属性を指定します。

- [演算子を作成して登録する。](#create-and-register-the-operator)TensorFlow Lite ランタイムに、グラフの演算子とパラメータを実行可能な C/C++ コードにマッピングする方法を認識させるために行います。

- [演算子をテストしプロファイリングする。](#test-and-profile-your-operator)カスタム演算子のみをテストする場合は、カスタム演算子のみでモデルを作成し、[benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc) プログラムを使用することをお勧めします。

Let’s walk through an end-to-end example of running a model with a custom operator `tf.atan` (named as `Atan`, refer to #create-a-tensorflow-model) which is supported in TensorFlow, but unsupported in TensorFlow Lite.

Note: The `tf.atan` function is **not** a custom operator. It is a regular operator which is supported by both TensorFlow and TensorFlow Lite. But we **assume** that it is a custom operator in the following example in order to demonstrate a simple workflow.

The TensorFlow Text operator is an example of a custom operator. See the <a href="https://tensorflow.org/text/guide/text_tf_lite" class="external"> Convert TF Text to TF Lite</a> tutorial for a code example.

## Example: Custom `Atan` operator

Let’s walk through an example of supporting a TensorFlow operator that TensorFlow Lite does not have. Assume we are using the `Atan` operator and that we are building a very simple model for a function `y = atan(x + offset)`, where `offset` is trainable.

### TensorFlow モデルを作成する

The following code snippet trains a simple TensorFlow model. This model just contains a custom operator named `Atan`, which is a function `y = atan(x + offset)`, where `offset` is trainable.

```python
import tensorflow as tf

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-1.4288993, 0.98279375, 1.2490457, 1.2679114, 1.5658458]
offset = tf.Variable(0.0)

# Define a simple model which just contains a custom operator named `Atan`
@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def atan(x):
  return tf.atan(x + offset, name="Atan")

# Train model
optimizer = tf.optimizers.Adam(0.01)
def train(x, y):
    with tf.GradientTape() as t:
      predicted_y = atan(x)
      loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))

for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())
```

```python
The actual offset is: 1.0
The predicted offset is: 0.99999905
```

この時点では、デフォルトのコンバータフラグを使って TensorFlow Lite モデルを生成しようとすると、次のようなエラーメッセージが発生します。

```none
Error:
error: 'tf.Atan' op is neither a custom op nor a flex op.
```

### TensorFlow Lite モデルに変換する

カスタム演算子を使った TensorFlow Lite モデルを作成しましょう。次に示すように、コンバータの属性を`allow_custom_ops`に設定してください。

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([atan.get_concrete_function()], atan)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

At this point, if you run it with the default interpreter using commands such as follows:

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

You will still get the error:

```none
Encountered unresolved custom op: Atan.
```

### 演算を作成して登録する

すべての TensorFlow Lite 演算子（カスタムとビルトインの両方）は、次の 4 つの関数で構成される単純な Pure-C インターフェイスを使って定義されています。

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

`TfLiteContext`と<code>TfLiteNode</code>の詳細については、<a><code>common.h</code></a>をご覧ください。前者はエラー報告ファシリティと、すべてのテンソルを含むグローバルオブジェクトへのアクセスを提供し、後者は実装が入力と出力にアクセスできるようにします。

インタプリタがモデルを読み込む際、グラフの各ノード当たり`init()`が一度呼び出されます。つまりグラフで演算が 2 回以上使用される場合、`init()`は 2 回以上呼び出されることになります。カスタム演算子の場合、パラメータ名を値にマッピングする flexbuffer が含まれる構成バッファが提供されます。このバッファは、インタプリタが演算子のパラメータをパースするため、組み込み演算子の場合は空です。ステートが必要なカーネル実装はここで初期化して、オーナーシップを呼び出し元に移譲します。各 `init()`呼び出しには`free()`への対応する呼び出しがあるため、実装は、`init()`に割り当てられていた可能性のあるバッファを利用できるようになります。

入力テンソルのサイズが変更されるたび、インタプリタはグラフを通過して実装に変更を通知します。このため、実装は、内部バッファのサイズを変更し、入力形状と型の妥当性を検証し、出力形状を再計算するチャンスを得ることができます。このプロセスはすべて`prepare()`を介して行われるため、実装は`node->user_data`を使用してステートにアクセスすることができます。

最後に、推論が実行されるたび、インタプリタは、`invoke()`を呼び出してグラフをトラバースし、ここでもステートを`node->user_data`として使用することができます。

カスタム演算子は、定義済みの 4 つの関数と、通常は次のようなグローバル登録関数によって、組み込み演算子とまったく同じように実装できます。

```c++
namespace tflite {
namespace ops {
namespace custom {
  TfLiteRegistration* Register_MY_CUSTOM_OP() {
    static TfLiteRegistration r = {my_custom_op::Init,
                                   my_custom_op::Free,
                                   my_custom_op::Prepare,
                                   my_custom_op::Eval};
    return &r;
  }
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

登録は自動ではなく、明示的な`Register_MY_CUSTOM_OP`呼び出しを行う必要があることに注意してください。標準の`BuiltinOpResolver`（`:builtin_ops`ターゲットから利用可能）が組み込み関数を処理する間に、別のカスタムライブラリでカスタム演算子を収集する必要があります。

### TensorFlow Lite ランタイムでカーネルを定義する

TensorFlow Lite でこの演算子を使用できるようにするには、2 つの関数（`Prepare`および`Eval`）を定義し、`TfLiteRegistration`を構築する必要があります。

```cpp
TfLiteStatus AtanPrepare(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AtanEval(TfLiteContext* context, TfLiteNode* node) {
  using namespace tflite;
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i=0; i<count; ++i) {
    output_data[i] = atan(input_data[i]);
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_ATAN() {
  static TfLiteRegistration r = {nullptr, nullptr, AtanPrepare, AtanEval};
  return &r;
}
```

When initializing the `OpResolver`, add the custom op into the resolver (see below for an example). This will register the operator with Tensorflow Lite so that TensorFlow Lite can use the new implementation. Note that the last two arguments in `TfLiteRegistration` correspond to the `AtanPrepare` and `AtanEval` functions you defined for the custom op. If you used `AtanInit` and `AtanFree` functions to initialize variables used in the op and to free up space, respectively, then they would be added to the first two arguments of `TfLiteRegistration`; those arguments are set to `nullptr` in this example.

### カーネルライブラリで演算子を登録する

では、カーネルライブラリを使って演算子を登録する必要があります。これは、`OpResolver`を使って行います。舞台裏では、インタプリタは、モデルの各演算子を実行するために割り当てられるカーネルのライブラリを読み込みます。デフォルトのライブラリには組み込みカーネルのみが含まれますが、これをカスタムライブラリに置き換えたり拡張することが可能です。

演算子のコードと名前を実際のコードに変換する`OpResolver`クラスは、次のように定義されます。

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

通常の使用では、`BuiltinOpResolver`を使用して次のように書く必要があります。

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

上記で作成したカスタム演算子を追加するには、（レゾルバを`InterpreterBuilder`に渡す前に）`AddOp`を呼び出します。

```c++
resolver.AddCustom("Atan", Register_ATAN());
```

組み込み演算子のセットが大きすぎる場合、演算子の特定のサブセットに基づいて、新しい`OpResolver`をコード生成することができます。おそらく特定のモデルに含まれる演算子のみが含まれます。これが、TensorFlow の選択的登録に相当するものです（また、この単純なバージョンは、`tools`ディレクトリに提供されています）。

If you want to define your custom operators in Java, you would currently need to build your own custom JNI layer and compile your own AAR [in this jni code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc). Similarly, if you wish to define these operators available in Python you can place your registrations in the [Python wrapper code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

上記に類似するプロセスは、単一の演算子ではなく一連の演算子をサポートするために使用することができます。必要な数の`AddCustom`演算子を追加してください。また、`BuiltinOpResolver`を使用することで、`AddBuiltin`を使用して、組み込みの実装をオーバーライドできます。

### 演算子をテストしてプロファイリングする

To profile your op with the TensorFlow Lite benchmark tool, you can use the [benchmark model tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool) for TensorFlow Lite. For testing purposes, you can make your local build of TensorFlow Lite aware of your custom op by adding the appropriate `AddCustom` call (as show above) to [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/core/kernels/register.cc)

## ベストプラクティス

1. メモリの割り当てと解除の最適化には注意してください。`Prepare`にメモリを割り当てる方が、`Invoke`に割り当てるよりも効果的であり、ループの前に割り当てる方が、各イテレーションで割り当てるよりも有効です。自分で割り当てるよりも一時テンソルデータを使用してください（項目 2 参照）。できるだけコピーよりも、ポインタ/参照を使用してください。

2. 演算を通じてデータ構造が永続する場合、一時テンソルを使用してメモリを事前に割り当てておくことをお勧めします。ほかの関数でテンソルインデックスを参照するには、OpData 構造を使用する必要がある場合があります。[畳み込み用のカーネル](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc)の例をご覧ください。サンプルのコードスニペットを次に示します。

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3. メモリの浪費がかさばらないようであれば、実行のイテレーションごとに動的に割り当てられる`std::vector`よりも、静的な固定サイズの配列（または`Resize`の事前割り当て済みの `std::vector`）を使用することをお勧めします。

4. 存在しない標準ライブラリコンテナテンプレートをインスタンス化しないようにしてください。バイナリサイズが大きくなります。たとえば、ほかのカーネルに存在しない`std::map`が演算に必要な場合、直接インデックスマッピングで`std::vector`を使用すれば、バイナリサイズを小さくしたまま機能させることが可能です。ほかのカーネルが何を使用して洞察を得ているのかご覧ください。

5. `malloc`が返すメモリへのポインタを確認してください。このポインタが`nullptr`である場合、そのポイントを使って演算を実行してはいけません。関数で`malloc`を行い、エラーが存在する場合は、終了する前にメモリを解放してください。

6. 特定の条件を確認するには、`TF_LITE_ENSURE(context, condition)`を使います。`TF_LITE_ENSURE`が使用されるときには、コードはメモリを残してはいけません。これらのマクロは、リークするリソースが割り当てられる前に使用される必要があります。

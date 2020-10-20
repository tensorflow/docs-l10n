# TensorFlow Lite 演算子のバージョン

このドキュメントでは、TensorFlow Lite 演算のバージョン管理スキーマを説明します。演算のバージョン管理を行うことによって、既存の演算子に新しい機能やパラメータを追加することができます。さらに、次の項目が保証されます。

- 下位互換性: 新しい TensorFlow Lite 実装で以前のモデルファイルも処理します。
- 上位互換性: 以前の TensorFlow Lite 実装で、新しい機能が使用されない限り、新しいバージョンの TOCO が生成する新しいモデルファイルを処理できます。
- 上位非互換性検出: 以前の TensorFlow Lite 実装がサポートされていない新しいバージョンの演算を含む新しいモデルを読み取る場合に、エラーを報告します。

## 例: 畳み込みに膨張度を追加する

このドキュメントの残りの部分では、畳み込み演算に膨張パラメータを追加する方法を紹介しながら、TFLite における演算のバージョン管理を説明します。

このドキュメントを理解する上で、膨張に関する知識は必要ありません。次のことに注意してください。

- `dilation_width_factor` と `dilation_height_factor` という、2 つの新しい整数パラメータが追加されます。
- 膨張をサポートしない古い畳み込みカーネルは、膨張要因を 1 に設定するのと同等です。

### FlatBuffer スキーマを変更する

演算に新しいパラメータを追加するには、`lite/schema/schema.fbs` のオプションテーブルを変更します。

たとえば、畳み込みのオプションテーブルは、次のようになっています。

```
table Conv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;
}
```

新しいパラメータを追加する場合:

- どのパラメータがどのバージョンでサポートされているかを示すコメントを追加します。
- 新しい実装が新たに追加されたパラメータのデフォルト値を取得する場合、以前の実装とまったく同じように機能します。

新しいパラメータを追加した後のテーブルは次のようになります。

```
table Conv2DOptions {
  // Parameters supported by version 1:
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;

  // Parameters supported by version 2:
  dilation_width_factor:int = 1;
  dilation_height_factor:int = 1;
}
```

ファイル `lite/schema/schema_generated.h` は新しいスキーマに合わせて生成し直されます。

### C 構造とカーネル実装を変更する

TensorFlow Lite では、カーネル実装は FlatBuffer 定義から分離されており、カーネルは `lite/c/builtin_op_data.h` で定義された C 構造からパラメータを読み取ります。

元の畳み込みパラメータは次のとおりです。

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
} TfLiteConvParams;
```

FlatBuffer スキーマと同様に、どのパラメータが度のバージョン以降でサポートされているかを示すコメントを、次のように追加します。

```
typedef struct {
  // Parameters supported by version 1:
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;

  // Parameters supported by version 2:
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteConvParams;
```

新たに追加されたパラメータを C 構造から読み取るように、カーネル実装も変更してください。その詳細は、ここでは省略します。

### FlatBuffer 読み取りコードを変更する

FlatBuffer を読み取って C 構造を生成するロジックは、`lite/core/api/flatbuffer_conversions.cc` 内にあります。

次のように、新しいパラメータを処理するようにファイルを更新してください。

```
case BuiltinOperator_CONV_2D: {
  TfLiteConvParams* params = MallocPOD<TfLiteConvParams>();
  if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
    params->padding = parse_padding(conv_params->padding());
    params->stride_width = conv_params->stride_w();
    params->stride_height = conv_params->stride_h();
    params->activation =
        parse_activation(conv_params->fused_activation_function());
    params->dilation_width_factor = conv_params->dilation_width_factor();
    params->dilation_height_factor = conv_params->dilation_height_factor();
  }
  *builtin_data = reinterpret_cast<void*>(params);
  break;
}
```

ここでは、演算バージョンを確認する必要はありません。新しい実装が膨張係数のない古いモデルファイルを読み取る場合は、デフォルト値として 1 を使用するため、新しいカーネルは古いカーネルと一貫性を維持して機能します。

### カーネル登録を変更する

MutableOpResolver（`lite/op_resolver.h` で定義）は、演算カーネルを登録する関するをいくつか提供しています。最低バージョンと最大バージョンはデフォルトの 1 です。

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

組み込み演算は `lite/kernels/register.cc` に登録されています。この例では、`Conv2D` のバージョン 1 と 2 を処理できる新しい演算カーネルを実装しているため、次の行を変更する必要があります。

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
```

上記の行を次のように変更します。

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 2);
```

### TOCO TFLite エクスポータを変更する

次のステップでは、TOCO に実行に必要な最低バージョンを入力させます。この例では、次のことを行います。

- 膨張係数がすべて 1 である場合は、version=1 を入力する
- そうでない場合は version=2 とする

これを行うには、`lite/tools/versioning/op_version.cc` で演算クラスの `GetBuiltinOperatorVersion` 関数をオーバーライドする必要があります。

バージョンが 1 つしかない演算の場合は、`GetVersion` 関数は次のように定義されます。

```
int GetVersion(const Operator& op) const override { return 1; }
```

複数のバージョンをサポートする場合は、パラメータを確認し、次の例に示すようにして演算のバージョンを判定します。

```
int GetVersion(const Operator& op) const override {
  const auto& conv_op = static_cast<const ConvOperator&>(op);
  if (conv_op.dilation_width_factor != 1 ||
      conv_op.dilation_height_factor != 1) {
    return 2;
  }
  return 1;
}
```

### 演算子のバージョンマップを更新する

最後に、新しいバージョン情報を演算子バージョンマップに追加します。このバージョンマップに応じて、モデルで最小限必要となるランタイムバージョンを生成する必要があるため、これは必要なステップです。

これを行うには、`lite/toco/tflite/op_version.cc` に新しいマップエントリを追加する必要があります。

この例では、次のエントリを `op_version_map` に追加してください。

```
{{OperatorType::kConv, 3}, "kPendingReleaseOpVersion"}
```

（次の安定リリースでは、`kPendingReleaseOpVersion` は適切なリリースバージョンに置き換えられます。）

### 実装をデリゲートする

TensorFlow Lite には、演算をハードウェアのバックエンドにデリゲートすることのできるデリゲート API があります。デリゲートの `Prepare` 関数で、バージョンがデリゲーションコードのすべてのノードに対応しているかどうかを確認します。

```
const int kMinVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMinVersion) {
  // Reject the node if the version isn't supported.
}
```

これは、デリゲートがバージョン 1 の演算のみをサポートする場合でも必要な作業で、これにより、デリゲートがより高いバージョンの演算を得る場合に非互換性を検出できるようになります。

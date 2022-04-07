# TensorFlow Lite 연산자 버전

이 문서에서는 TensorFlow Lite의 op 버전 관리 스키마를 설명합니다. Op 버전 관리를 통해 개발자는 기존 ops에 새로운 기능과 매개변수를 추가할 수 있습니다. 또한 다음을 보장합니다.

- 이전 버전과의 호환성: 새로운 TensorFlow Lite 구현은 이전 모델 파일을 처리합니다.
- 이후 버전과의 호환성: 새 기능이 사용되지 않는다면 이전 TensorFlow Lite 구현은 새 버전의 TOCO에서 생성된 새 모델 파일을 처리합니다.
- 이후 버전과의 비호환성 감지: 이전 TensorFlow Lite 구현이 지원되지 않는 새 버전의 op를 포함한 새 모델을 읽는 경우, 오류를 보고합니다.

## 예: 컨볼루션에 dilation 추가

이 문서의 나머지 부분에서는 컨볼루션 연산에 dilation 매개변수를 추가하는 방법을 보여줌으로써 TFLite의 op 버전 관리를 설명합니다.

이 문서를 이해하기 위해 dilation에 대한 지식이 필요하지는 않습니다. 다음 사항을 참조하세요.

- `dilation_width_factor` 및 `dilation_height_factor`의 새로운 두 정수 매개변수가 추가됩니다.
- Dilation을 지원하지 않는 이전 컨볼루션 커널은 dilation 인자를 1로 설정하는 것과 같습니다.

### FlatBuffer 스키마 변경하기

Op에 새 매개변수를 추가하려면 `lite/schema/schema.fbs`의 옵션 테이블을 변경합니다.

예를 들어, 컨볼루션의 옵션 테이블은 다음과 같습니다.

```
table Conv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;
}
```

새 매개변수를 추가하는 경우:

- 어떤 매개변수가 어떤 버전에서 지원되는지 나타내는 주석을 추가합니다.
- 새 구현이 새로 추가된 매개변수의 기본값을 가져오면 이전 구현과 정확히 동일하게 동작합니다.

새 매개변수가 추가된 후 테이블은 다음과 같습니다.

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

새 스키마에 대해 `lite/schema/schema_generated.h` 파일을 다시 생성해야 합니다.

### C 구조 및 커널 구현 변경하기

TensorFlow Lite에서 커널 구현은 FlatBuffer 정의에서 분리됩니다. 커널은 `lite/c/builtin_op_data.h`에 정의된 C 구조에서 매개변수를 읽습니다.

원래 컨볼루션 매개변수는 다음과 같습니다.

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
} TfLiteConvParams;
```

FlatBuffer 스키마와 마찬가지로 어떤 버전부터 어떤 매개변수가 지원되는지를 나타내는 주석을 추가합니다. 결과는 다음과 같습니다.

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

C 구조에서 새로 추가된 매개변수를 읽으려면 커널 구현도 변경하세요. 여기서는 자세한 내용을 다루지 않습니다.

### FlatBuffer 읽기 코드 변경하기

FlatBuffer를 읽고 C 구조를 생성하는 로직은 `lite/core/api/flatbuffer_conversions.cc`에 들어 있습니다.

아래와 같이 새 매개변수를 처리하도록 파일을 업데이트합니다.

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

여기서 연산자 버전을 확인할 필요는 없습니다. 새로운 구현은 dilation 매개변수가 누락된 이전 모델 파일을 읽을 때 기본값으로 1을 사용하고, 새 커널은 이전 커널과 일관되게 동작합니다.

### 커널 등록 변경하기

MutableOpResolver(`lite/op_resolver.h`에 정의됨)는 op 커널을 등록하는 몇 가지 함수를 제공합니다. 최소 및 최대 버전은 기본적으로 1입니다.

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

내장 ops는 `lite/kernels/register.cc`에 등록됩니다. 이 예에서는 `Conv2D` 버전 1과 2를 처리할 수 있는 새로운 op 커널을 구현했으므로 다음 줄을 변경해야 합니다.

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
```

위의 줄을 아래 줄로 바꿉니다.

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 2);
```

### TOCO TFLite exporter 변경하기

다음 단계는 op를 실행하는 데 필요한 최소 버전을 채우도록 TOCO를 처리하는 것입니다. 이 예에서 의미하는 바는 다음과 같습니다.

- Dilation 인자가 모두 1인 경우, 버전=1을 채웁니다.
- 그렇지 않으면 버전=2를 채웁니다.

`DepthwiseConv2D` 케이스에 새 버전을 추가하여 `lite/tools/versioning/op_version.cc`의 연산자에 적합하게 `GetBuiltinOperatorVersion` 함수를 수정합니다.

```
int GetVersion(const Operator& op) const override { return 1; }
```

### 연산자 버전 맵 업데이트하기

마지막 단계는 새 버전 정보를 연산자 버전 맵에 추가하는 것입니다. 이 버전 맵을 기반으로 모델의 최소 필수 런타임 버전을 생성해야 하므로 이 단계가 필요합니다.

이를 위해 `lite/tools/versioning/runtime_version.cc`에 새 맵 항목을 추가해야 합니다.

이 예에서는 `op_version_map`에 다음 항목을 추가해야 합니다.

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

여기서 `%CURRENT_RUNTIME_VERSION%`는 [tensorflow/core/public/version.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)에 정의된 현재 런타임 버전에 해당합니다.

### 위임 구현

TensorFlow Lite는 하드웨어 백엔드에 ops를 위임할 수 있는 Delegation API를 제공합니다. 대리자의 `Prepare` 함수에서 위임 코드의 모든 노드에 대해 버전이 지원되는지 확인합니다.

```
{{OperatorType::kConv, 3}, "kPendingReleaseOpVersion"}
```

This is required even if the delegation only supports version 1 ops, so the delegation can detect incompatibility when getting a higher version op.

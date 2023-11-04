# Versões dos operadores do TensorFlow Lite

Este documento descreve o esquema de versionamento dos operadores do TensorFlow Lite. Com o versionamento de operações, os desenvolvedores podem adicionar novas funcionalidades e parâmetros aos operadores existentes. Além disso, garante o seguinte:

- Compatibilidade com versões anteriores: novas implementações do TensorFlow Lite devem funcionar com arquivos de modelos antigos.
- Compatibilidade com versões posteriores: implementações antigas do TensorFlow Lite devem funcionar com novos arquivos de modelos gerados pela nova versão do conversor, desde que nenhum novo recurso tenha sido usado.
- Detecção de compatibilidade com versões posteriores: se uma implementação antiga do TensorFlow Lite ler um novo modelo que contenha uma nova versão de uma operação sem suporte, deve comunicar o erro.

## Exemplo: como adicionar limitação a uma convolução com reconhecimento de profundidade

O restante deste documento explica o versionamento de operações no TF Lite ao mostrar como adicionar parâmetros de dilatação a uma operação de convolução com reconhecimento de profundidade.

Não é necessário conhecer conceitos de dilatação para entender este documento. Observe que:

- Dois novos parâmetros inteiros serão adicionados: `dilation_width_factor` e `dilation_height_factor`.
- Kernels antigos de convolução com reconhecimento de profundidade que não tenham suporte a dilatação são equivalentes a definir os fatores de dilatação como 1.

### Altere o esquema do FlatBuffer

Para adicionar novos parâmetros a uma operação, altere a tabela de opções em `lite/schema/schema.fbs`.

Por exemplo: a tabela de opções de convolução com reconhecimento de profundidade deve ser assim:

```
table DepthwiseConv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
}
```

Ao adicionar novos parâmetros:

- Adicione comentários indicando quais parâmetros têm suporte em quais versões.
- Quando a nova implementação obtiver os valores padrão dos parâmetros recém-adicionados, deve funcionar exatamente da mesma forma que a implementação antiga.

Após adicionar os parâmetros, a tabela deverá ser:

```
table DepthwiseConv2DOptions {
  // Parameters for DepthwiseConv version 1 or above.
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
  // Parameters for DepthwiseConv version 2 or above.
  dilation_w_factor:int = 1;
  dilation_h_factor:int = 1;
}
```

O arquivo `lite/schema/schema_generated.h` deve ser gerado novamente para o novo esquema.

### Altere as estruturas do C e a implementação do kernel

No TensorFlow Lite, a implementação do kernel está desacoplada da definição do FlatBuffer. Os kernels leem o parâmetro das estruturas do C definidas em `lite/c/builtin_op_data.h`.

O parâmetro original de convolução com reconhecimento de profundidade é o seguinte:

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
} TfLiteDepthwiseConvParams;
```

Tal como no esquema do FlatBuffer, adicione comentários indicando quais parâmetros têm suporte a partir de qual versão. Confira o resultado abaixo:

```
typedef struct {
  // Parameters for DepthwiseConv version 1 or above.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
  // Parameters for DepthwiseConv version 2 or above.
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteDepthwiseConvParams;
```

Altere também a implementação do kernel para ler os parâmetros recém-adicionados das estruturas do C. Os detalhes foram omitidos aqui.

### Altere o código de leitura do FlatBuffer

A lógica para ler o FlatBuffer e gerar a estrutura do C está em `lite/core/api/flatbuffer_conversions.cc`.

Atualize o arquivo para tratar os novos parâmetros, conforme mostrado abaixo:

```
TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const DepthwiseConv2DOptions* schema_params =
      op->builtin_options_as_DepthwiseConv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->depth_multiplier = schema_params->depth_multiplier();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}
```

Não é obrigatório verificar a versão da operação aqui. Quando a nova implementação ler um arquivo de modelo antigo, em que os fatores de dilatação estão ausentes, ela usará 1 como o valor padrão, e o novo kernel funcionará de forma consistente com o kernel antigo.

### Altere o registro do kernel

O MutableOpResolver (definido em `lite/mutable_op_resolver.h`) fornece algumas funções para registrar kernels de operações. A versão mínima e a versão máxima são 1 por padrão.

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

As operações integradas são registradas em `lite/kernels/register.cc`. Neste exemplo, implementamos um novo kernel de operação que pode tratar as versões 1 e 2 de `DepthwiseConv2D` e, portanto, precisamos alterar esta linha:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
```

para:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
             /* min_version = */ 1,
             /* max_version = */ 2);
```

### Altere a versão da operação do TF Lite

A próxima etapa é fazer o TF Lite preencher a versão mínima necessária para executar a operação. Neste exemplo, isso significa:

- Preencher version=1 quando todos os fatores de dilatação forem iguais a 1.
- Caso contrário, preencher version=2.

Modifique a função `GetBuiltinOperatorVersion` para o operador em `lite/tools/versioning/op_version.cc` adicionando a nova versão ao caso `DepthwiseConv2D`:

```
case BuiltinOperator_DEPTHWISE_CONV_2D:
  auto depthwise_conv_params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(op_sig.builtin_data);
  TFLITE_DCHECK(depthwise_conv_params != nullptr);
  if (depthwise_conv_params->dilation_width_factor != 1 ||
       depthwise_conv_params->dilation_height_factor != 1) {
    return 2;
  }
  return 1;
```

### Atualize o mapeamento de versões do operador

A última etapa é adicionar as informações da nova versão ao mapeamento de versões do operador. Essa etapa é necessária porque precisamos gerar a versão mínima do runtime exigida pelo modelo com base nesse mapeamento de versões.

Para fazer isso, você precisa adicionar uma nova entrada ao mapeamento em `lite/tools/versioning/runtime_version.cc`.

Neste exemplo, você precisa adicionar a entrada abaixo a `op_version_map`:

```
{{BuiltinOperator_DEPTHWISE_CONV_2D, 2}, %CURRENT_RUNTIME_VERSION%}
```

em que `%CURRENT_RUNTIME_VERSION%` corresponde à versão atual do runtime definida em [tensorflow/core/public/version.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h).

### Implementação de delegação

O TensorFlow Lite conta com uma API de delegação que permite delegar operações a back-ends de hardware. Na função `Prepare` do delegado, verifique se a versão tem suporte para cada nó do código de delegação.

```
const int kMaxVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMaxVersion) {
  // Reject the node if the version isn't supported.
}
```

Isso é necessário mesmo se a delegação tiver suporte somente a operações da versão 1 para que a delegação possa detectar incompatibilidade quando receber uma operação de versão superior.

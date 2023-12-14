# Versiones de operadores de TensorFlow Lite

Este documento describe el esquema de versionado de ops de TensorFlow Lite. El versionado de ops permite a los desarrolladores añadir nuevas funcionalidades y parámetros a las ops existentes. Además, garantiza lo siguiente:

- Retrocompatibilidad: La nueva implementación de TensorFlow Lite debe soportar un archivo de modelos anteriores.
- Compatibilidad hacia adelante: Una implementación previa de TensorFlow Lite debería soportar un nuevo archivo de modelo producido por una nueva versión del convertidor, siempre que no se usen nuevas características.
- Detección de incompatibilidad hacia adelante: Si una implementación antigua de TensorFlow Lite lee un nuevo modelo que contiene una nueva versión no admitida de una op, debería informar del error.

## Ejemplo: Añadir la dilatación a la convolución en profundidad

El resto de este documento explica el versionado de op en TFLite mostrando cómo añadir parámetros de demora a la operación de convolución en profundidad.

No es necesario saber sobre dilatación para entender este documento. Tenga en cuenta que:

- Se añadirán 2 nuevos parámetros enteros: `dilation_width_factor` y `dilation_height_factor`.
- Los kernels de convolución en profundidad anteriores que no admiten la dilatación equivalen a configurar los factores de dilatación en 1.

### Cambiar el esquema FlatBuffer

Para añadir nuevos parámetros en una op, modifique la tabla de opciones en `lite/schema/schema.fbs`.

Por ejemplo, la tabla de opciones de la convolución en profundidad tiene el siguiente aspecto:

```
table DepthwiseConv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
}
```

Al añadir nuevos parámetros:

- Añada comentarios que indiquen qué parámetros son compatibles con cada versión.
- Cuando la nueva implementación reciba los valores predeterminados para los nuevos parámetros añadidos, debería funcionar exactamente igual que la antigua implementación.

La tabla quedará así después de añadir los nuevos parámetros:

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

El archivo `lite/schema/schema_generated.h` debe volver a generarse para el nuevo esquema.

### Cambiar las estructuras en C e implementar el kernel

En TensorFlow Lite, la implementación del kernel está desacoplada de la definición de FlatBuffer. Los kernel leen el parámetro a partir de estructuras C definidas en `lite/c/builtin_op_data.h`.

El parámetro original de la convolución en profundidad es el siguiente:

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
} TfLiteDepthwiseConvParams;
```

Al igual que con el esquema FlatBuffer, añada comentarios que indiquen qué parámetros son compatibles a partir de qué versión. El resultado se ve a continuación:

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

Cambie también la implementación del kernel para leer los parámetros recién añadidos desde las estructuras en C. Los detalles se omiten aquí.

### Cambie el código de lectura de FlatBuffer

La lógica para leer FlatBuffer y producir la estructura en C se encuentra en `lite/core/api/flatbuffer_conversions.cc`.

Actualice el archivo para manejar los nuevos parámetros, como se muestra a continuación:

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

No es necesario verificar la versión op aquí. Cuando la nueva implementación lea un archivo de modelo antiguo en el que falten factores de dilatación, usará 1 como valor predeterminado, y el nuevo kernel funcionará de forma consistente con el kernel antiguo.

### Cambie el registro del kernel

El MutableOpResolver (definido en `lite/mutable_op_resolver.h`) aporta unas cuantas funciones para registrar kernels de op. La versión mínima y máxima son 1 por predeterminado:

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

Las ops integradas se registran en `lite/kernels/register.cc`. En este ejemplo, implementamos un nuevo kernel de ops que puede manejar `DepthwiseConv2D` versión 1 y 2, por lo que necesitamos cambiar esta línea:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
```

a:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
             /* min_version = */ 1,
             /* max_version = */ 2);
```

### Cambie la versión de op de TFLite

El siguiente paso es hacer que TFLite rellene la versión mínima necesaria para ejecutar la op. En este ejemplo, significa:

- Llenar versión=1 cuando los factores de dilatación sean todos 1.
- En caso contrario, llenar versión=2.

Modifique la función `GetBuiltinOperatorVersion` para el operador en `lite/tools/versioning/op_version.cc` añadiendo la nueva versión al caso de `DepthwiseConv2D`:

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

### Actualizar el mapa de versiones del operador

El último paso es añadir la información de la nueva versión en el mapa de versiones del operador. Este paso es necesario porque necesitamos generar la versión de runtime mínima requerida del modelo basándonos en este mapa de versiones.

Para ello, debe añadir una nueva entrada de mapeo en `lite/tools/versioning/runtime_version.cc`.

En este ejemplo, debe añadir la siguiente entrada en `op_version_map`:

```
{{BuiltinOperator_DEPTHWISE_CONV_2D, 2}, %CURRENT_RUNTIME_VERSION%}
```

donde `%CURRENT_RUNTIME_VERSION%` corresponde a la versión actual de runtime definida en [tensorflow/core/public/version.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h).

### Implementación de delegación

TensorFlow Lite ofrece una API de delegación que permite delegar ops a backends de hardware. En la función `Prepare` del delegado, compruebe si la versión es compatible para cada nodo en código de delegación.

```
const int kMaxVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMaxVersion) {
  // Reject the node if the version isn't supported.
}
```

Esto es necesario aunque la delegación sólo admita ops de la versión 1, para que pueda detectar la incompatibilidad al obtener una op de una versión más alta.

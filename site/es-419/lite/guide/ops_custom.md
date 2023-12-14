# Operadores personalizados

Dado que la librería de operadores incorporada en TensorFlow Lite sólo soporta un número limitado de operadores TensorFlow, no todos los modelos son convertibles. Si desea más detalles, consulte [compatibilidad de operadores](ops_compatibility.md).

Para permitir la conversión, los usuarios pueden realizar su propia implementación personalizada de un operador TensorFlow no soportado en TensorFlow Lite, lo que se conoce como operador personalizado. *Si por el contrario, desea combinar una serie de operadores TensorFlow no admitidos (o sí admitidos) en un único operador personalizado optimizado fusionado, consulte [fusión de operadores](https://www.tensorflow.org/lite/models/convert/operation_fusion).*

El uso de operadores personalizados consta de cuatro pasos.

- [Crear un Modelo TensorFlow.](#create-a-tensorflow-model)[ Asegúrese de que el Modelo Guardado (o Def. de grafo) hace referencia al operador TensorFlow Lite correctamente nombrado.](#create-a-tensorflow-model)

- [Convertir a un modelo TensorFlow Lite.](#convert-to-a-tensorflow-lite-model)[ Asegúrese de configurar correctamente el atributo del convertidor TensorFlow Lite para convertir correctamente el modelo.](#convert-to-a-tensorflow-lite-model)

- [Crear y registrar el operador.](#create-and-register-the-operator)[ Esto es para que el runtime de TensorFlow Lite sepa cómo mapear su operador y parámetros en su grafo a código C/C++ ejecutable.](#create-and-register-the-operator)

- [Probar y perfilar su operador.](#test-and-profile-your-operator) Si desea probar sólo su operador personalizado, lo mejor es crear un modelo sólo con su operador personalizado y usar el programa [benchmark_model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc).

Recorramos un ejemplo de extremo a extremo de funcionamiento de un modelo con un operador personalizado `tf.atan` (llamado `Atan`, consulte #create-a-tensorflow-model) que está admitido en TensorFlow, pero no en TensorFlow Lite.

Nota: La función `tf.atan` **no** es un operador personalizado. Es un operador regular admitido tanto por TensorFlow como por TensorFlow Lite. Pero **suponemos** que es un operador personalizado en el siguiente ejemplo para demostrar un flujo de trabajo sencillo.

El operador Text de TensorFlow es un ejemplo de operador personalizado. Vea el tutorial <a href="https://tensorflow.org/text/guide/text_tf_lite" class="external"> Convertir Text de TF a TF Lite</a> para ver un ejemplo de código.

## Ejemplo: Operador personalizado `Atan`

Repasemos un ejemplo de compatibilidad con un operador de TensorFlow que TensorFlow Lite no tiene. Supongamos que estamos usando el operador `Atan` y que estamos construyendo un modelo muy simple para una función `y = atan(x + offset)`, donde `offset` es entrenable.

### Cree un modelo TensorFlow

El siguiente fragmento de código entrena un modelo TensorFlow sencillo. Este modelo sólo contiene un operador personalizado llamado `Atan`, que es una función `y = atan(x + offset)`, donde `offset` es entrenable.

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

En este punto, si intenta generar un modelo TensorFlow Lite con los indicadores predeterminados del convertidor, obtendrá el siguiente mensaje de error:

```none
Error:
error: 'tf.Atan' op is neither a custom op nor a flex op.
```

### Convierta a un modelo TensorFlow Lite

Cree un modelo TensorFlow Lite con operadores personalizados, ajustando el atributo del convertidor `allow_custom_ops` como se muestra a continuación:

<pre>converter = tf.lite.TFLiteConverter.from_concrete_functions([atan.get_concrete_function()], atan)
&lt;b&gt;converter.allow_custom_ops = True&lt;/b&gt;
tflite_model = converter.convert()
</pre>

En este punto, si lo ejecuta con el intérprete predeterminado usando comandos como los siguientes:

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

Seguirá recibiendo el error:

```none
Encountered unresolved custom op: Atan.
```

### Cree y registre el operador.

Todos los operadores de TensorFlow Lite (tanto los personalizados como los incorporados) se definen utilizando una sencilla interfaz en C puro que consta de cuatro funciones:

```c++
typedef struct {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
} TfLiteRegistration;
```

Consulte [`common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h) para más información sobre `TfLiteContext` y `TfLiteNode`. El primero aporta facilidades de notificación de errores y acceso a objetos globales, incluidos todos los tensores. El segundo permite implementar el acceso a sus entradas y salidas.

Cuando el intérprete carga un modelo, llama a `init()` una vez por cada nodo del grafo. Un `init()` dado se llamará más de una vez si la op se usa varias veces en el grafo. Para las ops personalizadas se incluirá un búfer de configuración que contiene un búfer flexible que mapea los nombres de los parámetros con sus valores. El búfer está vacío para las ops integradas porque el intérprete ya ha parseado los parámetros de la op. Las implementaciones del kernel que requieran estado deben inicializarlo aquí y transferir la propiedad a la persona que llama. Para cada llamada a `init()`, habrá una llamada correspondiente a `free()`, lo que permitirá a las implementaciones deshacerse del búfer que puedan haber asignado en `init()`.

Siempre que se redimensionen los tensores de entrada, el intérprete recorrerá el grafo notificando el cambio a los implementadores. Esto les da la oportunidad de redimensionar su búfer interno, comprobar la validez de las formas y tipos de entrada y recalcular las formas de salida. Todo esto se hace a través de `prepare()`, y las implementaciones pueden acceder a su estado usando `node->user_data`.

Por último, cada vez que se ejecuta la inferencia, el intérprete recorre el grafo llamando a `invoke()`, y aquí también el estado está disponible como `node->user_data`.

Las ops personalizadas pueden implementarse exactamente igual que las integradas, definiendo esas cuatro funciones y una función de registro global que suele tener el siguiente aspecto:

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

Tenga en cuenta que el registro no es automático y que debe hacerse una llamada explícita a `Register_MY_CUSTOM_OP`. Si bien el `BuiltinOpResolver` estándar (disponible en el destino `:builtin_ops`) se encarga del registro de los builtins, las ops personalizadas deberán recopilarse en librerías personalizadas independientes.

### Defina el kernel en el runtime de TensorFlow Lite

Todo lo que tenemos que hacer para usar la op en TensorFlow Lite es definir dos funciones (`Prepare` y `Eval`), y construir un `TfLiteRegistration`:

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

Al inicializar el `OpResolver`, añada el op personalizado en el resolver (véase más abajo un ejemplo). Esto registrará el operador con Tensorflow Lite para que TensorFlow Lite pueda usar la nueva implementación. Tenga en cuenta que los dos últimos argumentos en `TfLiteRegistration` corresponden a las funciones `AtanPrepare` y `AtanEval` que definió para la op personalizada. Si usó las funciones `AtanInit` y `AtanFree` para inicializar las variables usadas en la op y para liberar espacio, respectivamente, entonces éstas se agregarían a los dos primeros argumentos de `TfLiteRegistration`; esos argumentos se configuran como `nullptr` en este ejemplo.

### Registre el operador con la librería del kernel

Ahora necesitamos registrar el operador con la librería del kernel. Esto se hace con un `OpResolver`. Entre bastidores, el intérprete cargará una librería de kernels que se asignarán para ejecutar cada uno de los operadores del modelo. Aunque la librería predeterminada sólo contiene kernels incorporados, es posible reemplazarla/ampliarla con una librería de operadores op personalizada.

La clase `OpResolver`, que traduce los códigos y nombres de los operadores en código real, se define así:

```c++
class OpResolver {
  virtual TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const = 0;
  virtual TfLiteRegistration* FindOp(const char* op) const = 0;
  virtual void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration) = 0;
  virtual void AddCustom(const char* op, TfLiteRegistration* registration) = 0;
};
```

Su uso normal requiere que use el `BuiltinOpResolver` y escriba:

```c++
tflite::ops::builtin::BuiltinOpResolver resolver;
```

Para añadir la op personalizada creada anteriormente, se llama a `AddOp` (antes de pasar el resolver al `InterpreterBuilder`):

```c++
resolver.AddCustom("Atan", Register_ATAN());
```

Si se considera que el conjunto de ops incorporadas es demasiado grande, podría generarse código para una nueva `OpResolver` basada en un subconjunto dado de ops, posiblemente sólo las contenidas en un modelo determinado. Esto equivale al registro selectivo de TensorFlow (y una versión simple del mismo está disponible en el directorio `tools`).

Si desea definir sus operadores personalizados en Java, actualmente tendría que construir su propia capa JNI personalizada y compilar su propio AAR [en este código jni](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/native/nativeinterpreterwrapper_jni.cc). Igualmente, si desea definir estos operadores disponibles en Python, puede colocar sus registros en el [código contenedor de Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc).

Recuerde que puede seguir un proceso similar al anterior para admitir un conjunto de operaciones en lugar de un único operador. Sólo tiene que añadir tantos operadores `AddCustom` como necesite. Además, `BuiltinOpResolver` también le permite anular implementaciones de builtins usando el `AddBuiltin`.

### Pruebe y perfile su operador

Para perfilar su op con la herramienta de benchmark de TensorFlow Lite, puede usar la [herramienta de modelo de benchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#tflite-model-benchmark-tool) para TensorFlow Lite. Para realizar pruebas, puede hacer que su versión local de TensorFlow Lite esté al tanto de su op personalizada añadiendo la llamada apropiada `AddCustom` (como se muestra arriba) a [register.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/core/kernels/register.cc).

## Prácticas recomendadas

1. Optimice las asignaciones y desasignaciones de memoria con precaución. Asignar memoria en `Prepare` es más eficiente que en `Invoke`, y asignar memoria antes de un bucle es mejor que en cada iteración. Use datos tensores temporales en lugar de mallocarlos usted mismo (véase el punto 2). Use punteros/referencias en lugar de copiar tanto como sea posible.

2. Si una estructura de datos va a persistir durante toda la operación, aconsejamos preasignar la memoria usando tensores temporales. Puede que necesite usar la estructura OpData para consultar los índices de los tensores en otras funciones. Véase el ejemplo en el [kernel de convolución](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/conv.cc). Un fragmento de código de ejemplo es el siguiente:

    ```
    auto* op_data = reinterpret_cast<OpData*>(node->user_data);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = op_data->temp_tensor_index;
    TfLiteTensor* temp_tensor = &context->tensors[op_data->temp_tensor_index];
    temp_tensor->type =  kTfLiteFloat32;
    temp_tensor->allocation_type = kTfLiteArenaRw;
    ```

3. Si no supone un gran coste de memoria desperdiciada, prefiera usar un arreglo estático de tamaño fijo (o un `std::vector` preasignado en `Resize`) en lugar de usar un `std::vector` asignado dinámicamente en cada iteración de la ejecución.

4. Evite instanciar plantillas de contenedores de bibliotecas estándares que no existan ya, porque afectan al tamaño de los binarios. Por ejemplo, si necesita un `std::map` en su operación que no existe en otros kernels, podría usar un `std::vector` con mapeado de indexación directa manteniendo el tamaño binario pequeño. Mire lo que usan otros kernels para hacerse una idea (o pregunte).

5. Revise el puntero a la memoria devuelto por `malloc`. Si este puntero es `nullptr`, no debe realizarse ninguna operación usando ese puntero. Si `malloc` en una función y tiene una salida de error, desasigne la memoria antes de salir.

6. Use `TF_LITE_ENSURE(context, condition)` para comprobar una condición específica. Su código no debe dejar memoria inactiva cuando se use `TF_LITE_ENSURE`, es decir, estas macros deben usarse antes de que se asigne ningún recurso que pueda tener fugas.

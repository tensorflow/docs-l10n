# Inferencia de TensorFlow Lite

El término *inferencia* se refiere al proceso de ejecución de un modelo de TensorFlow Lite en el dispositivo con el fin de realizar predicciones basadas en los datos de entrada. Para realizar una inferencia con un modelo de TensorFlow Lite, debe ejecutarlo a través de un *interprete*. El intérprete de TensorFlow Lite está diseñado para ser ágil y rápido. El intérprete usa un ordenamiento estático de grafos y un asignador de memoria personalizado (menos dinámico) para garantizar una carga, inicialización y latencia de ejecución mínimas.

Esta página describe cómo acceder al intérprete de TensorFlow Lite y realizar una inferencia utilizando C++, Java y Python, además de enlaces a otros recursos para cada plataforma [soportada](#supported-platforms).

[TOC]

## Conceptos importantes

La inferencia en TensorFlow Lite suele seguir los siguientes pasos:

1. **Cargar un modelo**

    Debe cargar en memoria el modelo `.tflite`, que contiene el grafo de ejecución del modelo.

2. **Transformar los datos**

    Los datos de entrada brutos para el modelo no suelen coincidir con el formato de datos de entrada esperado por el modelo. Por ejemplo, puede que necesite cambiar el tamaño de una imagen o cambiar su formato para que sea compatible con el modelo.

3. **Ejecutar la inferencia**

    Este paso consiste en usar la API de TensorFlow Lite para ejecutar el modelo. Implica algunos pasos como generar el intérprete, y asignar tensores, como se describe en las siguientes secciones.

4. **Interpretar la salida**

    Cuando reciba los resultados de la inferencia del modelo, deberá interpretar los tensores de una forma significativa que sea útil en su aplicación.

    Por ejemplo, un modelo podría devolver sólo una lista de probabilidades. Depende de usted mapear las probabilidades en categorías relevantes y presentarlo a su usuario final.

## Plataformas compatibles

Las API de inferencia de TensorFlow se ofrecen para las plataformas móviles/integradas más comunes, como [Android](#android-platform), [iOS](#ios-platform) y [Linux](#linux-platform), en múltiples lenguajes de programación.

En la mayoría de los casos, el diseño de las API refleja una preferencia por el rendimiento por encima de la facilidad de uso. TensorFlow Lite está diseñado para una inferencia rápida en dispositivos pequeños, por lo que no debería sorprender que las API traten de evitar copias innecesarias a expensas de la comodidad. Del mismo modo, la consistencia de las API de TensorFlow no era una meta explícita, por lo que cabe esperar cierta varianza entre los distintos lenguajes.

En todas las librerías, la API de TensorFlow Lite le permite cargar modelos, alimentar entradas y recuperar salidas de inferencia.

### Plataforma Android

En Android, la inferencia de TensorFlow Lite puede realizarse usando las API de Java o de C++. Las API de Java son prácticas y pueden usarse directamente dentro de las clases Activity de Android. Las API de C++ ofrecen más flexibilidad y velocidad, pero pueden requerir la escritura de contenedores JNI para mover datos entre las capas Java y C++.

Más abajo encontrará más detalles sobre cómo usar [C++](#load-and-run-a-model-in-c) y [Java](#load-and-run-a-model-in-java), o siga el [inicio rápido en Android](../android) para ver un tutorial y código de ejemplo.

#### Generador de código del contenedor TensorFlow Lite para Android

Nota: El generador de código contenedor TensorFlow Lite se encuentra en fase experimental (beta) y actualmente sólo es compatible con Android.

Para el modelo TensorFlow Lite mejorado con [metadatos](../inference_with_metadata/overview), los desarrolladores pueden usar el generador de código envolvente TensorFlow Lite para Android para crear código envolvente específico de la plataforma. El código contenedor elimina la necesidad de interactuar directamente con `ByteBuffer` en Android. En cambio, los desarrolladores pueden interactuar con el modelo de TensorFlow Lite con objetos tipados como `Bitmap` y `Rect`. Para más información, consulte el [Generador de código del contenedor de TensorFlow Lite para Android](../inference_with_metadata/codegen.md).

### Plataforma iOS

En iOS, TensorFlow Lite está disponible con librerías nativas de iOS escritas en [Swift](https://www.tensorflow.org/code/tensorflow/lite/swift) y [Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc). También puede usar la [API de C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) directamente en códigos Objective-C.

Consulte a continuación los detalles sobre cómo usar [Swift](#load-and-run-a-model-in-swift), [Objective-C](#load-and-run-a-model-in-objective-c) y la [API de C](#using-c-api-in-objective-c-code), o siga el [inicio rápido de iOS](ios.md) para ver un tutorial y código de ejemplo.

### Plataforma Linux

En plataformas Linux (incluyendo [Raspberry Pi](build_arm)), puede ejecutar inferencias usando las API de TensorFlow Lite disponibles en [C++](#load-and-run-a-model-in-c) y [Python](#load-and-run-a-model-in-python), como se muestra en las siguientes secciones.

## Ejecutar un modelo

Ejecutar un modelo TensorFlow Lite implica unos sencillos pasos:

1. Cargar el modelo en la memoria.
2. Generar un `Interpreter` basado en un modelo existente.
3. Ajustar los valores de los tensores de entrada (opcionalmente, cambiar el tamaño de los tensores de entrada si no se desean los tamaños predefinidos).
4. Invocar inferencia.
5. Leer los valores del tensor de salida.

En las secciones siguientes se describe cómo realizar estos pasos en cada lenguaje.

## Cargar y ejecutar un modelo en Java

*Plataforma: Android*

La API de Java para ejecutar una inferencia con TensorFlow Lite está diseñada principalmente para usarse con Android, por lo que está disponible como una dependencia de la librería de Android: `org.tensorflow:tensorflow-lite`.

En Java, usará la clase `Interpreter` para cargar un modelo y conducir la inferencia del modelo. En muchos casos, ésta puede ser la única API que necesite.

Puede inicializar un `Interpreter` usando un archivo `.tflite`:

```java
public Interpreter(@NotNull File modelFile);
```

O con un `MappedByteBuffer`:

```java
public Interpreter(@NotNull MappedByteBuffer mappedByteBuffer);
```

En ambos casos, debe proporcionar un modelo válido de TensorFlow Lite o la API lanzará `IllegalArgumentException`. Si usa `MappedByteBuffer` para inicializar un `Interpreter`, aquel debe permanecer inalterado durante toda la vida del `Interpreter`.

La forma preferida de ejecutar la inferencia en un modelo es usar firmas. Disponible para modelos convertidos a partir de Tensorflow 2.5

```Java
try (Interpreter interpreter = new Interpreter(file_of_tensorflowlite_model)) {
  Map<String, Object> inputs = new HashMap<>();
  inputs.put("input_1", input1);
  inputs.put("input_2", input2);
  Map<String, Object> outputs = new HashMap<>();
  outputs.put("output_1", output1);
  interpreter.runSignature(inputs, outputs, "mySignature");
}
```

El método `runSignature` recibe tres argumentos:

- **Entradas**: mapea las entradas del nombre de la entrada en la firma a un objeto de entrada.

- **Salidas**: mapea la salida a partir del nombre de la salida en la firma a los datos de salida.

- **Nombre de la firma** [opcional]: Nombre de la firma (puede dejarse vacío si el modelo tiene una sola firma).

Otra forma de ejecutar una inferencia cuando el modelo no tiene definidas las firmas. Basta con llamar a `Interpreter.run()`. Por ejemplo:

```java
try (Interpreter interpreter = new Interpreter(file_of_a_tensorflowlite_model)) {
  interpreter.run(input, output);
}
```

El método `run()` toma sólo una entrada y devuelve sólo una salida. Así que si su modelo tiene múltiples entradas o múltiples salidas, en su lugar use:

```java
interpreter.runForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

En este caso, cada entrada de `inputs` corresponde a un tensor de entrada y `map_of_indices_to_outputs` mapea índices de tensores de salida a los datos de salida correspondientes.

En ambos casos, los índices de los tensores deben corresponder a los valores que dio al [Convertidor de TensorFlow Lite](../models/convert/) cuando creó el modelo. Tenga en cuenta que el orden de los tensores en `input` debe coincidir con el orden dado al convertidor de TensorFlow Lite.

La clase `Interpreter` también ofrece prácticas funciones para que pueda obtener el índice de cualquier entrada o salida del modelo utilizando un nombre de operación:

```java
public int getInputIndex(String opName);
public int getOutputIndex(String opName);
```

Si `opName` no es una operación válida en el modelo, lanza una `IllegalArgumentException`.

Tenga en cuenta también que `Interpreter` posee recursos. Para evitar fugas de memoria, los recursos deben ser liberados después de ser usados mediante:

```java
interpreter.close();
```

Para ver un proyecto de ejemplo con Java, consulte la  [muestra de clasificación de imágenes en Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

### Tipos de datos admitidos (en Java)

Para usar TensorFlow Lite, los tipos de datos de los tensores de entrada y salida deben ser uno de los siguientes tipos primitivos:

- `float`
- `int`
- `long`
- `byte`

También se admiten los tipos `String`, pero se codifican de forma diferente a los tipos primitivos. En particular, la forma de un Tensor de cadenas dicta el número y la colocación de las cadenas en el Tensor, siendo cada elemento en sí mismo una cadena de longitud variable. En este sentido, el tamaño (en bytes) del Tensor no puede calcularse sólo a partir de la forma y el tipo, y en consecuencia las cadenas no pueden ser dadas como un argumento simple y llano `ByteBuffer`. Puede ver algunos ejemplos en esta [página](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter).

Si se usan otros tipos de datos, incluidos los tipos estándar como `Integer` y `Float`, se lanzará una `IllegalArgumentException`.

#### Entradas

Cada entrada debe ser un arreglo o un arreglo multidimensional de los tipos primitivos soportados, o un `ByteBuffer` sin procesar del tamaño apropiado. Si la entrada es un arreglo o un arreglo multidimensional, el tensor de entrada asociado se redimensionará implícitamente a las dimensiones del arreglo en el momento de la inferencia. Si la entrada es un ByteBuffer, el invocador deberá primero redimensionar manualmente el tensor de entrada asociado (mediante `Interpreter.resizeInput()`) antes de ejecutar la inferencia.

Cuando utilice `ByteBuffer`, prefiera usar buffers de bytes directos, ya que esto permite al `Interpreter` evitar copias innecesarias. Si el `ByteBuffer` es un búfer de bytes directo, su orden debe ser `ByteOrder.nativeOrder()`. Después de usarlo para la inferencia de un modelo, debe permanecer inalterado hasta que finalice la inferencia del modelo.

#### Salidas

Cada salida debe ser un arreglo o matriz multidimensional de los tipos primitivos admitidos, o un ByteBuffer del tamaño apropiado. Tenga en cuenta que algunos modelos tienen salidas dinámicas, en las que la forma de los tensores de salida puede variar en función de la entrada. No hay una forma directa de manejar esto con la actual API de inferencia de Java, pero hay extensiones previstas que lo harán posible.

## Cargar y ejecutar un modelo en Swift

*Plataforma: iOS*

La API de [Swift](https://www.tensorflow.org/code/tensorflow/lite/swift) está disponible en el Pod `TensorFlowLiteSwift` de Cocoapods.

En primer lugar, debe importar el módulo `TensorFlowLite`.

```swift
import TensorFlowLite
```

```swift
// Getting model path
guard
  let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
else {
  // Error handling...
}

do {
  // Initialize an interpreter with the model.
  let interpreter = try Interpreter(modelPath: modelPath)

  // Allocate memory for the model's input `Tensor`s.
  try interpreter.allocateTensors()

  let inputData: Data  // Should be initialized

  // input data preparation...

  // Copy the input data to the input `Tensor`.
  try self.interpreter.copy(inputData, toInputAt: 0)

  // Run inference by invoking the `Interpreter`.
  try self.interpreter.invoke()

  // Get the output `Tensor`
  let outputTensor = try self.interpreter.output(at: 0)

  // Copy output to `Data` to process the inference results.
  let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
  let outputData =
        UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
  outputTensor.data.copyBytes(to: outputData)

  if (error != nil) { /* Error handling... */ }
} catch error {
  // Error handling...
}
```

## Cargar y ejecutar un modelo en Objective-C

*Plataforma: iOS*

La [API de Objective-C](https://www.tensorflow.org/code/tensorflow/lite/objc) está disponible en el Pod `TensorFlowLiteObjC` de Cocoapods.

En primer lugar, debe importar el módulo `TensorFlowLite`.

```objc
@import TensorFlowLite;
```

```objc
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model"
                                                      ofType:@"tflite"];
NSError *error;

// Initialize an interpreter with the model.
TFLInterpreter *interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                  error:&error];
if (error != nil) { /* Error handling... */ }

// Allocate memory for the model's input `TFLTensor`s.
[interpreter allocateTensorsWithError:&error];
if (error != nil) { /* Error handling... */ }

NSMutableData *inputData;  // Should be initialized
// input data preparation...

// Get the input `TFLTensor`
TFLTensor *inputTensor = [interpreter inputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy the input data to the input `TFLTensor`.
[inputTensor copyData:inputData error:&error];
if (error != nil) { /* Error handling... */ }

// Run inference by invoking the `TFLInterpreter`.
[interpreter invokeWithError:&error];
if (error != nil) { /* Error handling... */ }

// Get the output `TFLTensor`
TFLTensor *outputTensor = [interpreter outputTensorAtIndex:0 error:&error];
if (error != nil) { /* Error handling... */ }

// Copy output to `NSData` to process the inference results.
NSData *outputData = [outputTensor dataWithError:&error];
if (error != nil) { /* Error handling... */ }
```

### Usar la API de C en código Objective-C

Actualmente, la API de Objective-C no admite delegados. Para usar delegados con código Objective-C, necesita llamar directamente a la [API de C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) subyacente.

```c
#include "tensorflow/lite/c/c_api.h"
```

```c
TfLiteModel* model = TfLiteModelCreateFromFile([modelPath UTF8String]);
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

// Create the interpreter.
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

// Allocate tensors and populate the input tensor data.
TfLiteInterpreterAllocateTensors(interpreter);
TfLiteTensor* input_tensor =
    TfLiteInterpreterGetInputTensor(interpreter, 0);
TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                           input.size() * sizeof(float));

// Execute inference.
TfLiteInterpreterInvoke(interpreter);

// Extract the output tensor data.
const TfLiteTensor* output_tensor =
    TfLiteInterpreterGetOutputTensor(interpreter, 0);
TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                         output.size() * sizeof(float));

// Dispose of the model and interpreter objects.
TfLiteInterpreterDelete(interpreter);
TfLiteInterpreterOptionsDelete(options);
TfLiteModelDelete(model);
```

## Cargar y ejecutar un modelo en C++

*Plataformas: Android, iOS, y Linux*

Nota: La API de C++ en iOS sólo está disponible cuando se usa bazel.

En C++, el modelo se almacena en la clase [`FlatBufferModel`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/flat-buffer-model.html). Encapsula un modelo TensorFlow Lite y se puede generar de un par de maneras diferentes, dependiendo de donde se almacena el modelo:

```c++
class FlatBufferModel {
  // Build a model based on a file. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromFile(
      const char* filename,
      ErrorReporter* error_reporter);

  // Build a model based on a pre-loaded flatbuffer. The caller retains
  // ownership of the buffer and should keep it alive until the returned object
  // is destroyed. Return a nullptr in case of failure.
  static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
      const char* buffer,
      size_t buffer_size,
      ErrorReporter* error_reporter);
};
```

Nota: Si TensorFlow Lite detecta la presencia de la [NNAPI de Android](https://developer.android.com/ndk/guides/neuralnetworks), intentará usar automáticamente la memoria compartida para almacenar el `FlatBufferModel`.

Ahora que tiene el modelo como un objeto `FlatBufferModel`, puede ejecutarlo con un [`Interpreter`](https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter.html). Un único `FlatBufferModel` puede ser usado simultáneamente por más de un `Interpreter`.

Precaución: El objeto `FlatBufferModel` debe permanecer válido hasta que todas las instancias de `Interpreter` que lo usen hayan sido destruidas.

Las partes importantes de la API del `Interprete` se muestran en el fragmento de código siguiente. Debe tenerse en cuenta que:

- Los tensores se representan mediante números enteros, para evitar las comparaciones de cadenas (y cualquier dependencia fija de las librerías de cadenas).
- No se debe acceder a un intérprete desde hilos concurrentes.
- La asignación de memoria para los tensores de entrada y salida debe activarse llamando a `AllocateTensors()` justo después de redimensionar los tensores.

El uso más sencillo de TensorFlow Lite con C++ es el siguiente:

```c++
// Load the model
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);

// Build the interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
interpreter->AllocateTensors();

float* input = interpreter->typed_input_tensor<float>(0);
// Fill `input`.

interpreter->Invoke();

float* output = interpreter->typed_output_tensor<float>(0);
```

Para más ejemplos de código, consulte [`minimal.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc) y [`label_image.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc).

## Cargar y ejecutar un modelo en Python

*Plataforma: Linux*

La API de Python para ejecutar una inferencia se ofrece en el módulo `tf.lite`. A partir del cual, en la mayoría de los casos sólo necesita [`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) para cargar un modelo y ejecutar una inferencia.

El siguiente ejemplo muestra cómo usar el intérprete de Python para cargar un archivo `.tflite` y ejecutar la inferencia con datos de entrada aleatorios:

Este ejemplo se recomienda si está convirtiendo desde SavedModel con un SignatureDef definido. Disponible a partir de TensorFlow 2.5

```python
class TestModel(tf.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def add(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 4}


SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH,
    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
  f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])
```

Otro ejemplo si el modelo no tiene SignatureDefs definidos.

```python
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

Como alternativa a cargar el modelo como un archivo `.tflite` preconvertido, puede combinar su código con la [API convertidora de Python de TensorFlow Lite](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter) (`tf.lite.TFLiteConverter`), lo que le permitirá convertir su modelo Keras al formato TensorFlow Lite y, a continuación, ejecutar la inferencia:

```python
import numpy as np
import tensorflow as tf

img = tf.keras.Input(shape=(64, 64, 3), name="img")
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

# Convert to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.Model(inputs=[img], outputs=[out]))
tflite_model = converter.convert()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Continue to get tensors and so forth, as shown above...
```

Para obtener más código Python de ejemplo, consulte [`label_image.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py).

Consejo: Ejecute `help(tf.lite.Interpreter)` en el terminal de Python para acceder a documentación detallada sobre el intérprete.

## Ejecutar la inferencia con el modelo de forma dinámica

Si desea ejecutar un modelo con forma de entrada dinámica, *redimensione la forma de entrada* antes de ejecutar la inferencia. De lo contrario, la forma `None` en los modelos Tensorflow será reemplazada por un marcador de posición de `1` en los modelos TFLite.

Los siguientes ejemplos muestran cómo redimensionar la forma de entrada antes de ejecutar la inferencia en diferentes lenguajes. Todos los ejemplos suponen que la forma de entrada está definida como `[1/None, 10]`, y necesita ser redimensionada a `[3, 10]`.

Ejemplo en C++:

```c++
// Resize input tensors before allocate tensors
interpreter->ResizeInputTensor(/*tensor_index=*/0, std::vector<int>{3,10});
interpreter->AllocateTensors();
```

Ejemplo en Python:

```python
# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
  
# Resize input shape for dynamic shape model and allocate tensor
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [3, 10])
interpreter.allocate_tensors()
  
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```




## Operaciones admitidas

TensorFlow Lite admite un subconjunto de operaciones de TensorFlow con algunas limitaciones. Para ver la lista completa de operaciones y limitaciones, consulte la página [Operaciones de TensorFlow Lite](https://www.tensorflow.org/mlir/tfl_ops).

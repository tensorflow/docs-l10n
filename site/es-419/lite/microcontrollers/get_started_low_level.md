# Introducción a los microcontroladores

Este documento explica cómo entrenar un modelo y ejecutar la inferencia usando un microcontrolador.

## El ejemplo de Hello World

El ejemplo [Hello World](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world) está diseñado para demostrar los fundamentos absolutos del uso de TensorFlow Lite para microcontroladores. Entrenamos y ejecutamos un modelo que reproduce una función seno, es decir, toma un único número como entrada y emite el valor [seno](https://en.wikipedia.org/wiki/Sine) del número. Cuando se implementa en el microcontrolador, sus predicciones se usan para hacer parpadear los LED o controlar una animación.

El flujo de trabajo de punta a punta implica los siguientes pasos:

1. [Entrenar un modelo](#train_a_model) (en Python): Un archivo python a fin de entrenar, convertir y optimizar un modelo para usarlo en el dispositivo.
2. [Ejecutar inferencia](#run_inference) (en C++ 17): Una prueba de unidad de extremo a extremo que ejecuta la inferencia sobre el modelo usando la [librería C++](library.md).

## Conseguir un dispositivo compatible

La aplicación de ejemplo que vamos a usar ha sido probada en los siguientes dispositivos:

- [Arduino Nano 33 BLE Sense](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense-with-headers) (usando Arduino IDE)
- [SparkFun Edge](https://www.sparkfun.com/products/15170) (construyendo directamente desde la fuente)
- [Kit Discovery STM32F746](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) (usando Mbed)
- [Adafruit EdgeBadge](https://www.adafruit.com/product/4400) (usando Arduino IDE)
- [Adafruit TensorFlow Lite para Kit de Microcontroladores](https://www.adafruit.com/product/4317) (usando Arduino IDE)
- [Adafruit Circuit Playground Bluefruit](https://learn.adafruit.com/tensorflow-lite-for-circuit-playground-bluefruit-quickstart?view=all) (usando Arduino IDE)
- [Espressif ESP32-DevKitC](https://www.espressif.com/en/products/hardware/esp32-devkitc/overview) (usando ESP IDF)
- [Espressif ESP-EYE](https://www.espressif.com/en/products/hardware/esp-eye/overview) (usando ESP IDF)

Aprenda más sobre las plataformas compatibles en [TensorFlow Lite para microcontroladores](index.md).

## Entrenar un modelo

Nota: Puede saltarse esta sección y usar el modelo entrenado incluido en el código de ejemplo.

Usar [train.py](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train.py) para el entrenamiento del modelo hello world para el reconocimiento de ondas sinusoidales

Ejecutar: `bazel build tensorflow/lite/micro/examples/hello_world:train` `bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/`

## Ejecutar inferencia

Para ejecutar el modelo en su dispositivo, seguiremos las instrucciones del archivo `README.md`:

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/README.md">Hello World README.md</a>

Las siguientes secciones le guiarán a través del ejemplo [`evaluate_test.cc`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world/evaluate_test.cc), una prueba de unidad que demuestra cómo ejecutar la inferencia usando TensorFlow Lite para microcontroladores. Carga el modelo y ejecuta la inferencia varias veces.

### 1. Incluya las cabeceras de la librería

Para usar la librería TensorFlow Lite para microcontroladores, debemos incluir los siguientes archivos de cabecera:

```C++
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

- [`micro_mutable_op_resolver.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) aporta las operaciones que utiliza el intérprete para ejecutar el modelo.
- [`micro_error_reporter.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h) emite información de depuración.
- [`micro_interpreter.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_interpreter.h) contiene código para cargar y ejecutar modelos.
- [`schema_generated.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/schema/schema_generated.h) contiene el esquema para el formato de archivo del modelo TensorFlow Lite [`FlatBuffer`](https://google.github.io/flatbuffers/).
- [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h) facilita información sobre versiones para el esquema de TensorFlow Lite.

### 2. Incluya la cabecera del modelo

El intérprete de TensorFlow Lite para microcontroladores espera que el modelo se facilite como un arreglo de C++. El modelo se define en los archivos `model.h` y `model.cc`. La cabecera se incluye con la siguiente línea:

```C++
#include "tensorflow/lite/micro/examples/hello_world/model.h"
```

### 3. Incluya la cabecera del marco de pruebas de unidad

Para crear una prueba de unidad, incluimos el marco de pruebas de unidad de TensorFlow Lite para microcontroladores con la siguiente línea:

```C++
#include "tensorflow/lite/micro/testing/micro_test.h"
```

La prueba se define usando las siguientes macros:

```C++
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  . // add code here
  .
}

TF_LITE_MICRO_TESTS_END
```

A continuación analizamos el código incluido en la macro anterior.

### 4. Configure el registro

Para configurar el registro, se crea un puntero `tflite::ErrorReporter` utilizando un puntero a una instancia `tflite::MicroErrorReporter`:

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```

Esta variable se pasará al intérprete, lo que le permitirá escribir logs. Dado que los microcontroladores suelen tener una gran variedad de mecanismos para registrar logs, la implementación de `tflite::MicroErrorReporter` está diseñada para ser personalizada para su dispositivo en particular.

### 5. Cargue un modelo

En el código siguiente, el modelo se instancia usando datos de un arreglo `char`, `g_model`, que se declara en `model.h`. A continuación, revisamos el modelo para asegurarnos de que su versión del esquema es compatible con la versión que estamos usando:

```C++
const tflite::Model* model = ::tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
}
```

### 6. Instancie el resolver de operaciones

Se declara una instancia [`MicroMutableOpResolver`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/micro_mutable_op_resolver.h). El intérprete la usará para registrar y acceder a las operaciones que utiliza el modelo:

```C++
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  return kTfLiteOk;

```

El `MicroMutableOpResolver` requiere un parámetro de plantilla que indique el número de ops que se registrarán. La función `RegisterOps` registra las ops con el resolver.

```C++
HelloWorldOpResolver op_resolver;
TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

```

### 7. Asigne memoria

Necesitamos preasignar una cierta cantidad de memoria para los arreglos de entrada, salida e intermedios. Esto se facilita como un arreglo `uint8_t` de tamaño `tensor_arena_size`:

```C++
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
```

El tamaño necesario dependerá del modelo que esté usando y puede que tenga que determinarlo experimentando.

### 8. Instancie el intérprete

Creamos una instancia de `tflite::MicroInterpreter`, pasándole las variables creadas anteriormente:

```C++
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     tensor_arena_size, error_reporter);
```

### 9. Asigne tensores

Le decimos al intérprete que asigne memoria del `tensor_arena` para los tensores del modelo:

```C++
interpreter.AllocateTensors();
```

### 10. Valide la forma de entrada

La instancia `MicroInterpreter` puede darnos un puntero al tensor de entrada del modelo llamando a `.input(0)`, donde `0` representa el primer (y único) tensor de entrada:

```C++
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);
```

Luego inspeccionamos este tensor para confirmar que su forma y tipo son los que esperamos:

```C++
// Make sure the input has the properties we expect
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
// The property "dims" tells us the tensor's shape. It has one element for
// each dimension. Our input is a 2D tensor containing 1 element, so "dims"
// should have size 2.
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
// The value of each element gives the length of the corresponding tensor.
// We should expect two single element tensors (one is contained within the
// other).
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
// The input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
```

El valor enum `kTfLiteFloat32` es una referencia a uno de los tipos de datos de TensorFlow Lite, y está definido en [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h).

### 11. Dar un valor de entrada

Para dar entrada al modelo, fijamos el contenido del tensor de entrada, como sigue:

```C++
input->data.f[0] = 0.;
```

En este caso, se introduce un valor de punto flotante que representa `0`.

### 12. Ejecute el modelo

Para ejecutar el modelo, podemos llamar a `Invoke()` en nuestra instancia `tflite::MicroInterpreter`:

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
}
```

Podemos revisar el valor de retorno, un `TfLiteStatus`, para determinar si la ejecución se ha realizado correctamente. Los posibles valores de `TfLiteStatus`, definidos en [`common.h`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/c/common.h), son `kTfLiteOk` y `kTfLiteError`.

El código siguiente afirma que el valor es `kTfLiteOk`, lo que significa que la inferencia se ha ejecutado correctamente.

```C++
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
```

### 13. Obtenga la salida

El tensor de salida del modelo puede obtenerse llamando a `output(0)` en el `tflite::MicroInterpreter`, donde `0` representa el primer (y único) tensor de salida.

En el ejemplo, la salida del modelo es un único valor de punto flotante contenido en un tensor 2D:

```C++
TfLiteTensor* output = interpreter.output(0);
TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);
```

Podemos leer el valor directamente del tensor de salida y afirmar que es lo que esperamos:

```C++
// Obtain the output value from the tensor
float value = output->data.f[0];
// Check that the output value is within 0.05 of the expected value
TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);
```

### 14. Ejecutar la inferencia de nuevo

El resto del código ejecuta la inferencia varias veces más. En cada ocasión, asignamos un valor al tensor de entrada, invocamos al intérprete y leemos el resultado del tensor de salida:

```C++
input->data.f[0] = 1.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

input->data.f[0] = 3.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

input->data.f[0] = 5.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
```

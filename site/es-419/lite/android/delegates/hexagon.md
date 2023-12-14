# Delegado Hexagon de TensorFlow Lite

Este documento explica cómo usar el delegado Hexagon de TensorFlow Lite en su aplicación utilizando la API Java y/o C. El delegado aprovecha la librería Qualcomm Hexagon para ejecutar kernels cuantificados en el DSP. Tenga en cuenta que el delegado está pensado para *complementar* la funcionalidad NNAPI, particularmente para dispositivos en los que la aceleración DSP NNAPI no está disponible (por ejemplo, en dispositivos más antiguos, o dispositivos que aún no tienen un controlador DSP NNAPI).

Nota: Este delegado se encuentra en fase experimental (beta).

**Dispositivos compatibles:**

Actualmente son compatibles las siguientes arquitecturas Hexagon, entre otras:

- Hexagon 680
    - Ejemplos de SoC: Snapdragon 821, 820, 660
- Hexagon 682
    - Ejemplos de SoC: Snapdragon 835
- Hexagon 685
    - Ejemplos de SoC: Snapdragon 845, Snapdragon 710, QCS410, QCS610, QCS605, QCS603
- Hexagon 690
    - Ejemplos de SoC: Snapdragon 855, RB5

**Modelos compatibles:**

El delegado Hexagon admite todos los modelos que se ajustan a nuestra especificación de cuantización simétrica de [8 bits](https://www.tensorflow.org/lite/performance/quantization_spec), incluidos los generados usando [cuantización entera posterior al entrenamiento](https://www.tensorflow.org/lite/performance/post_training_integer_quant). También son compatibles los modelos UInt8 entrenados con la ruta heredada de [entrenamiento consciente de la cuantización](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize), por ejemplo, [estas versiones cuantizadas](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models) en nuestra página de modelos alojados.

## API Java del delegado Hexagon

```java
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

### Ejemplo de uso

#### Paso 1. Edite app/build.gradle para usar el AAR delegado nocturno de Hexagon

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### Paso 2. Añada las librerías Hexagon a su app Android

- Descargue y ejecute hexagon_nn_skel.run. Debería ofrecer 3 librerías compartidas diferentes "libhexagon_nn_skel.so", "libhexagon_nn_skel_v65.so", "libhexagon_nn_skel_v66.so"
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

Nota: Deberá aceptar el Acuerdo de licencia.

Nota: A partir del 23/02/2021 debe usarse la v1.20.0.1.

Nota: Debe usar las librerías hexagon_nn con la versión compatible de la librería de interfaz. La librería de interfaz es parte del AAR y es obtenida por bazel a través del [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl). La versión en el bazel config es la versión que debe usar.

- Incluya las 3 en su app junto con otras librerías compartidas. Consulte [Cómo añadir librerías compartidas a su app](#how-to-add-shared-library-to-your-app). El delegado elegirá automáticamente la de mejor rendimiento en función del dispositivo.

Nota: Si su app será generada tanto para dispositivos ARM de 32 como de 64 bits, entonces necesitará añadir las librerías compartidas de Hexagon a ambas carpetas de librerías de 32 y 64 bits.

#### Paso 3. Cree un delegado e inicialice un intérprete de TensorFlow Lite

```java
import org.tensorflow.lite.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## API C del delegado Hexagon

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

### Ejemplo de uso

#### Paso 1. Edite app/build.gradle para usar el AAR delegado nocturno de Hexagon

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### Paso 2. Añada las librerías Hexagon a su app Android

- Descargue y ejecute hexagon_nn_skel.run. Debería ofrecer 3 librerías compartidas diferentes "libhexagon_nn_skel.so", "libhexagon_nn_skel_v65.so", "libhexagon_nn_skel_v66.so"
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

Nota: Deberá aceptar el Acuerdo de licencia.

Nota: A partir del 23/02/2021 debe usarse la v1.20.0.1.

Nota: Debe usar las librerías hexagon_nn con la versión compatible de la librería de interfaz. La librería de interfaz es parte del AAR y es obtenida por bazel a través del [config](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl). La versión en el bazel config es la versión que debe usar.

- Incluya las 3 en su app junto con otras librerías compartidas. Consulte [Cómo añadir librerías compartidas a su app](#how-to-add-shared-library-to-your-app). El delegado elegirá automáticamente la de mejor rendimiento en función del dispositivo.

Nota: Si su app será generada tanto para dispositivos ARM de 32 como de 64 bits, entonces necesitará añadir las librerías compartidas de Hexagon a ambas carpetas de librerías de 32 y 64 bits.

#### Paso 3. Incluya la cabecera en C

- El archivo de cabecera "hexagon_delegate.h" puede descargarse de [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h) o extraerse del AAR del delegado Hexagon.

#### Paso 4. Cree un delegado e inicialice un intérprete de TensorFlow Lite

- En su código, asegúrese de que la librería nativa Hexagon está cargada. Esto puede hacerse llamando a `System.loadLibrary("tensorflowlite_hexagon_jni");` <br>en su Activity o punto de entrada de Java.

- Cree un delegado, por ejemplo:

```c
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Assuming shared libraries are under "/data/local/tmp/"
// If files are packaged with native lib in android App then it
// will typically be equivalent to the path provided by
// "getContext().getApplicationInfo().nativeLibraryDir"
const char[] library_directory_path = "/data/local/tmp/";
TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
::tflite::TfLiteHexagonDelegateOptions params = {0};
// 'delegate_ptr' Need to outlive the interpreter. For example,
// If use case will need to resize input or anything that can trigger
// re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
    ::tflite::TfLiteHexagonDelegateDelete(delegate);
  });
interpreter->ModifyGraphWithDelegate(delegate.get());
// After usage of delegate.
TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

## Añada la librería compartida a su app

- Cree la carpeta "app/src/main/jniLibs", y cree un directorio para cada arquitectura objetivo. Por ejemplo,
    - ARM de 64-bits: `app/src/main/jniLibs/arm64-v8a`
    - ARM de 32-bits: `app/src/main/jniLibs/armeabi-v7a`
- Ponga su .so en el directorio que corresponda a la arquitectura.

Nota: Si está usando App Bundle para publicar su aplicación, puede que quiera configurar android.bundle.enableUncompressedNativeLibs=false en el archivo gradle.properties.

## Comentarios

En caso de problemas, cree una incidencia [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) con todos los detalles de reproducción necesarios, incluido el modelo de teléfono y placa utilizados (`adb shell getprop ro.product.device` y `adb shell getprop ro.board.platform`).

## Preguntas frecuentes

- ¿Qué operaciones admite el delegado?
    - Ver la lista actual de [operaciones y restricciones soportadas](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md)
- ¿Cómo puedo saber que el modelo está usando el DSP cuando habilito el delegado?
    - Se mostrarán dos mensajes de registro cuando habilite el delegado: uno para indicar si se ha creado el delegado y otro para indicar cuántos nodos se están ejecutando usando el delegado. <br> `Created TensorFlow Lite delegate for Hexagon.` <br> `Hexagon delegate: X nodes delegated out of Y nodes.`
- ¿Necesito que todas las operaciones del modelo sean compatibles para ejecutar el delegado?
    - No, el Modelo se particionará en subgrafos en función de las operaciones soportadas. Cualquier operación no compatible se ejecutará en la CPU.
- ¿Cómo puedo generar el AAR delegado de Hexagon a partir del código fuente?
    - Use `bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/hexagon/java:tensorflow-lite-hexagon`.
- ¿Por qué no se inicializa el delegado Hexagon a pesar de que mi dispositivo Android tiene un SoC compatible?
    - Verifique si su dispositivo tiene efectivamente un SoC compatible. Ejecute `adb shell cat /proc/cpuinfo | grep Hardware` y compruebe si devuelve algo como "Hardware : Qualcomm Technologies, Inc MSMXXXX".
    - Algunos fabricantes de teléfonos usan diferentes SoC para el mismo modelo de teléfono. Por lo tanto, es posible que el delegado Hexagon sólo funcione en algunos dispositivos del mismo modelo de teléfono, pero no en todos.
    - Algunos fabricantes de teléfonos restringen deliberadamente el uso del DSP Hexagon de las apps Android que no son del sistema, lo que hace que el delegado Hexagon no pueda funcionar.
- Mi teléfono tiene bloqueado el acceso al DSP. He rooteado el teléfono y sigo sin poder ejecutar el delegado, ¿qué hacer?
    - Asegúrese de desactivar SELinux enforce ejecutando `adb shell setenforce 0`

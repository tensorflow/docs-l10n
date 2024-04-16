# Medición del desempeño

## Herramientas de evaluación comparativa

Las herramientas de evaluación comparativa de TensorFlow Lite miden y calculan actualmente las estadísticas de las siguientes métricas de desempeño importantes:

- Tiempo de inicialización
- Tiempo de inferencia del estado de calentamiento
- Tiempo de inferencia del estado estacionario
- Uso de memoria durante el tiempo de inicialización
- Uso global de la memoria

Las herramientas de evaluación comparativa están disponibles como apps de evaluación comparativa para Android e iOS y como binarios nativos de línea de comandos, y todas comparten la misma lógica central de medición del desempeño. Tenga en cuenta que las opciones disponibles y los formatos de salida son ligeramente diferentes debido a las diferencias en el entorno runtime.

### App de evaluación comparativa para Android

Existen dos opciones para usar la herramienta de evaluación comparativa con Android. Una es un [binario de evaluación comparativa nativo](#native-benchmark-binary) y otra es una app de evaluación comparativa de Android, un mejor indicador de cómo funcionaría el modelo en la app. De cualquier forma, los números de la herramienta de referencia seguirán siendo ligeramente diferentes de los que se obtengan al ejecutar la inferencia con el modelo en la app real.

Esta app de evaluación comparativa para Android no tiene interfaz de usuario. Instálela y ejecútela usando el comando `adb` y recupere los resultados usando el comando `adb logcat`.

#### Descargar o compilar la app

Descargue las apps precompiladas nocturnas de evaluación comparativa de Android utilizando los enlaces que aparecen a continuación:

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk)

En cuanto a las apps de referencia para Android que admiten [ops TF](https://www.tensorflow.org/lite/guide/ops_select) a través del [delegado Flex](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex), use los siguientes enlaces:

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex.apk)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex.apk)

También puede compilar la app desde el código fuente siguiendo estas [instrucciones](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android).

Nota: Es necesario compilar la app desde el código fuente si desea ejecutar el apk de referencia de Android en CPU x86 o delegado Hexagon o si su modelo contiene [operadores select TF](../guide/ops_select) u [operadores personalizados](../guide/ops_custom).

#### Preparar la evaluación comparativa

Antes de ejecutar la app de benchmark, instale la app y transmita el archivo del modelo al dispositivo como se indica a continuación:

```shell
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```

#### Ejecutar evaluación comparativa

```shell
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/your_model.tflite \
              --num_threads=4"'
```

`graph` es un parámetro obligatorio.

- `graph`: `string` <br> La ruta al archivo del modelo TFLite.

Puede especificar más parámetros opcionales para ejecutar la evaluación comparativa.

- `num_threads`: `int` (predeterminado=1) <br> El número de hilos a usar para ejecutar el intérprete TFLite.
- `use_gpu`: `bool` (predeterminado=false) <br> Usar el [delegado GPU](gpu).
- `use_nnapi`: `bool` (predeterminado=false) <br> Usar el [delegado NNAPI](nnapi).
- `use_xnnpack`: `bool` (predeterminado=`false`) <br> Usar el [delegado XNNPACK](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/xnnpack).
- `use_hexagon`: `bool` (predeterminado=`false`) <br> Usar el [delegado Hexagon](hexagon_delegate).

Según el dispositivo que esté usando, es posible que algunas de estas opciones no estén disponibles o no tengan ningún efecto. Consulte [parámetros](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters) para ver más parámetros de rendimiento que podría ejecutar con la app de evaluación comparativa.

Vea los resultados usando el comando `logcat`:

```shell
adb logcat | grep "Inference timings"
```

Los resultados de la evaluación comparativa se presentan como:

```
... tflite  : Inference timings in us: Init: 5685, First inference: 18535, Warmup (avg): 14462.3, Inference (avg): 14575.2
```

### Binario de evaluación comparativa nativa

La herramienta Benchmark también se encuentra disponible como binario nativo `benchmark_model`. Puede ejecutar esta herramienta desde una línea de comandos shell en Linux, Mac, dispositivos incorporados y dispositivos Android.

#### Descargar o compilar el binario

Descargue los binarios nativos de línea de comandos precompilados nocturnos siguiendo los siguientes enlaces:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model)

En cuanto a los binarios nocturnos precompilados que soportan [ops TF](https://www.tensorflow.org/lite/guide/ops_select) a través del [delegado Flex](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex), use los siguientes enlaces:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_plus_flex)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_plus_flex)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_plus_flex)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_plus_flex)

Para realizar evaluaciones comparativas con [el delegado Hexagon de TensorFlow Lite](https://www.tensorflow.org/lite/android/delegates/hexagon), también hemos precompilado los archivos `libhexagon_interface.so` necesarios (consulte [aquí](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md) si desea más información sobre este archivo). Tras descargar el archivo de la plataforma correspondiente de los enlaces que aparecen a continuación, cambie el nombre del archivo a `libhexagon_interface.so`.

- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_libhexagon_interface.so)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_libhexagon_interface.so)

También puede compilar el binario nativo de evaluación comparativa desde el [código fuente](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) en su computadora.

```shell
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

Para compilar con la cadena de herramientas Android NDK, primero debe configurar el entorno de compilación siguiendo esta [guía](../android/lite_build#set_up_build_environment_without_docker), o usar la imagen docker como se describe en esta [guía](../android/lite_build#set_up_build_environment_using_docker).

```shell
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

Nota: Es un enfoque válido para insertar y ejecutar binarios directamente en un dispositivo Android para la evaluación comparativa, pero puede resultar en diferencias sutiles (pero observables) en el rendimiento en relación con la ejecución dentro de una app Android real. En particular, el programador de Android adapta el comportamiento en función de las prioridades de los hilos y los procesos, que difieren entre una actividad/aplicación en primer plano y un binario normal en segundo plano ejecutado a través de `adb shell ...`. Este comportamiento adaptado es más evidente cuando se habilita la ejecución multihilo en la CPU con TensorFlow Lite. Por lo tanto, se prefiere la app Android benchmark para la medición del desempeño.

#### Ejecutar evaluación comparativa

Para ejecutar evaluaciones comparativas en su computadora, ejecute el binario desde el intérprete de comandos.

```shell
path/to/downloaded_or_built/benchmark_model \
  --graph=your_model.tflite \
  --num_threads=4
```

Puede usar el mismo conjunto de [parámetros](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters) mencionado anteriormente con el binario nativo de la línea de comandos.

#### Perfilar las ops del modelo

El binario del modelo de referencia también le permite perfilar las ops del modelo y obtener los tiempos de ejecución de cada operario. Para ello, pase el indicador `--enable_op_profiling=true` a `benchmark_model` durante la invocación. Los detalles se explican [aquí](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#profiling-model-operators).

### Evaluación comparativa binaria nativa para múltiples opciones de rendimiento en una sola ejecución

También se proporciona un práctico y sencillo binario C++ para [comparar múltiples opciones de rendimiento](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#benchmark-multiple-performance-options-in-a-single-run) en una sola ejecución. Este binario se basa en la herramienta de evaluación antes mencionada que sólo podía evaluar una única opción de rendimiento cada vez. Comparten el mismo proceso de creación/instalación/ejecución, pero el nombre del destino BUILD de este binario es `benchmark_model_performance_options` y toma algunos parámetros adicionales. Un parámetro importante para este binario es:

`perf_options_list`: `string` (predeterminado='all') <br> Una lista separada por comas de las opciones de rendimiento de TFLite que se van a comparar.

Puede obtener binarios nocturnos precompilados para esta herramienta como se indica a continuación:

- [linux_x86-64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model_performance_options)
- [linux_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model_performance_options)
- [linux_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model_performance_options)
- [android_aarch64](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_performance_options)
- [android_arm](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model_performance_options)

### App de evaluación comparativa para iOS

Para ejecutar evaluaciones comparativas en un dispositivo iOS, debe compilar la app desde el [código fuente](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios). Coloque el archivo del modelo TensorFlow Lite en el directorio [benchmark_data](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios/TFLiteBenchmark/TFLiteBenchmark/benchmark_data) del árbol de origen y modifique el archivo `benchmark_params.json`. Esos archivos se empaquetan en la app y ésta lee los datos del directorio. Visite la [app para evaluación comparativa en iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios) para obtener instrucciones detalladas.

## Evaluación comparativa del rendimiento de modelos conocidos

Esta sección enumera las evaluaciones comparativas de rendimiento de TensorFlow Lite al ejecutar modelos bien conocidos en algunos dispositivos Android e iOS.

### Evaluación comparativa de rendimiento de Android

Estos números de referencia de rendimiento se generaron con el [binario de referencia nativo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

Para los puntos de referencia de Android, la afinidad de la CPU está configurada para usar núcleos grandes en el dispositivo para reducir la varianza (véase [detalles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#reducing-variance-between-runs-on-android)).

Se asume que los modelos se descargaron y descomprimieron en el directorio `/data/local/tmp/tflite_models`. El binario de evaluación comparativa se compila usando [estas instrucciones](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#on-android) y se supone que se encuentra en el directorio `/data/local/tmp`.

Para ejecutar la evaluación comparativa:

```sh
adb shell /data/local/tmp/benchmark_model \
  --num_threads=4 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50
```

Para ejecutarlo con delegado nnapi, establezca `--use_nnapi=true`. Para ejecutarlo con delegado GPU, establezca `--use_gpu=true`.

Los valores de rendimiento que figuran a continuación se miden en Android 10.

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Dispositivo</th>
      <th>CPU, 4 hilos</th>
      <th>GPU</th>
      <th>NNAPI</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
</td>
    <td>Pixel 3</td>
    <td>23.9 ms</td>
    <td>6.45 ms</td>
    <td>13.8 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>14.0 ms</td>
    <td>9.0 ms</td>
    <td>14.8 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>Pixel 3</td>
    <td>13.4 ms</td>
    <td>---</td>
    <td>6.0 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>5.0 ms</td>
    <td>---</td>
    <td>3.2 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
</td>
    <td>Pixel 3</td>
    <td>56 ms</td>
    <td>---</td>
    <td>102 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>34.5 ms</td>
    <td>---</td>
    <td>99.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>Pixel 3</td>
    <td>35.8 ms</td>
    <td>9.5 ms</td>
    <td>18.5 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>23.9 ms</td>
    <td>11.1 ms</td>
    <td>19.0 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>Pixel 3</td>
    <td>422 ms</td>
    <td>99.8 ms</td>
    <td>201 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>272.6 ms</td>
    <td>87.2 ms</td>
    <td>171.1 ms</td>
  </tr>
  <tr>
    <td rowspan="2">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>Pixel 3</td>
    <td>486 ms</td>
    <td>93 ms</td>
    <td>292 ms</td>
  </tr>
   <tr>
     <td>Pixel 4</td>
    <td>324.1 ms</td>
    <td>97.6 ms</td>
    <td>186.9 ms</td>
  </tr>
 </table>

### Evaluación comparativa de rendimiento de iOS

Estos números de evaluación comparativa del rendimiento se generaron con la [app de evaluación comparativa iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios).

Para ejecutar las pruebas de rendimiento de iOS, se modificó la app de evaluación comparativa para incluir el modelo adecuado y se modificó `benchmark_params.json` para definir `num_threads` como 2. Para usar el delegado de GPU, también se añadieron las opciones `"use_gpu" : "1"` y `"gpu_wait_type" : "aggressive"` a `benchmark_params.json`.

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Dispositivo</th>
      <th>CPU, 2 hilos</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
</td>
    <td>iPhone XS</td>
    <td>14.8 ms</td>
    <td>3.4 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>
</td>
    <td>iPhone XS</td>
    <td>11 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
</td>
    <td>iPhone XS</td>
    <td>30.4 ms</td>
    <td>---</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
</td>
    <td>iPhone XS</td>
    <td>21.1 ms</td>
    <td>15.5 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
</td>
    <td>iPhone XS</td>
    <td>261.1 ms</td>
    <td>45.7 ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
</td>
    <td>iPhone XS</td>
    <td>309 ms</td>
    <td>54.4 ms</td>
  </tr>
 </table>

## Trazar las funciones internas de TensorFlow Lite

### Trazar las funciones internas de TensorFlow Lite en Android

Nota: Esta función está disponible a partir de Tensorflow Lite v2.4.

Los eventos internos del intérprete TensorFlow Lite de una app Android pueden ser capturados por [Herramientas de trazado de Android](https://developer.android.com/topic/performance/tracing). Son los mismos eventos con la API [Trace](https://developer.android.com/reference/android/os/Trace) de Android, por lo que los eventos capturados del código Java/Kotlin se ven junto con los eventos internos de TensorFlow Lite.

Algunos ejemplos de eventos son:

- Invocación del operario
- Modificación del grafo por el delegado
- Asignación de tensores

Entre las diferentes opciones para capturar los trazados, esta guía cubre el perfilador de CPU de Android Studio y la app System Tracing. Si desea conocer otras opciones, consulte [Herramienta de línea de comandos Perfetto](https://developer.android.com/studio/command-line/perfetto) o [Herramienta de línea de comandos System Tracing](https://developer.android.com/topic/performance/tracing/command-line).

#### Añadir eventos de trazado en código Java

Este es un fragmento de código de la app de ejemplo [Image Classification](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android). El intérprete de TensorFlow Lite se ejecuta en la sección `recognizeImage/runInference`. Este paso es opcional, pero es útil para ayudar a tener presente dónde se realiza la llamada a la inferencia.

```java
  Trace.beginSection("recognizeImage");
  ...
  // Runs the inference call.
  Trace.beginSection("runInference");
  tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
  Trace.endSection();
  ...
  Trace.endSection();

```

#### Habilitar el trazado de TensorFlow Lite

Para activar el trazado de TensorFlow Lite, configure la propiedad del sistema Android `debug.tflite.trace` a 1 antes de iniciar la app Android.

```shell
adb shell setprop debug.tflite.trace 1
```

Si se ha configurado esta propiedad al inicializar el intérprete TensorFlow Lite, se rastrearán los eventos clave (por ejemplo, la invocación de un operario) del intérprete.

Después de capturar todos los trazos, desactive el trazado poniendo el valor de la propiedad en 0.

```shell
adb shell setprop debug.tflite.trace 0
```

#### Perfilador de CPU de Android Studio

Capture los trazos con el [Perfilador de CPU de Android Studio](https://developer.android.com/studio/profile/cpu-profiler) siguiendo los pasos que se indican a continuación:

1. Seleccione **Ejecutar &gt; Perfilar 'app'** en los menús de arriba.

2. Haga clic en cualquier lugar de la línea de tiempo de la CPU cuando aparezca la ventana del perfilador.

3. Seleccione 'Trazar llamadas al sistema' entre los modos de perfilado de la CPU.

    ![Seleccione 'Trazar llamadas al sistema'](images/as_select_profiling_mode.png)

4. Pulse el botón "Grabar".

5. Pulse el botón 'Detener'.

6. Investigue el resultado del trazado.

    ![trazo de Android Studio](images/as_traces.png)

En este ejemplo, puede ver la jerarquía de eventos en un hilo y las estadísticas de cada operario en el tiempo y también ver el flujo de datos de toda la app entre los hilos.

#### App System Tracing

Capture los trazos sin Android Studio siguiendo los pasos detallados en la [app System Tracing](https://developer.android.com/topic/performance/tracing/on-device).

En este ejemplo, se capturaron los mismos eventos TFLite y se guardaron en el formato Perfetto o Systrace dependiendo de la versión del dispositivo Android. Los archivos de trazado capturados pueden abrirse en la [UI de Perfetto](https://ui.perfetto.dev/#!/).

![trazo de Perfetto](images/perfetto_traces.png)

### Trazar las funciones internas de TensorFlow Lite en iOS

Nota: Esta función está disponible a partir de Tensorflow Lite v2.5.

Los eventos internos del intérprete TensorFlow Lite de una app iOS pueden ser capturados por la herramienta [Instruments](https://developer.apple.com/library/archive/documentation/ToolsLanguages/Conceptual/Xcode_Overview/MeasuringPerformance.html#//apple_ref/doc/uid/TP40010215-CH60-SW1) incluida con Xcode. Son los eventos [signpost](https://developer.apple.com/documentation/os/logging/recording_performance_data) de iOS, por lo que los eventos capturados del código Swift/Objective-C se ven junto con los eventos internos de TensorFlow Lite.

Algunos ejemplos de eventos son:

- Invocación del operario
- Modificación del grafo por el delegado
- Asignación de tensores

#### Habilitar el trazado de TensorFlow Lite

Configure la variable de entorno `debug.tflite.trace` siguiendo los pasos que se indican a continuación:

1. Seleccione **Producto &gt; Esquema &gt; Editar esquema...** en los menús de arriba de Xcode.

2. Haga clic en 'Perfil' en el panel izquierdo.

3. Desmarque la casilla 'Usar los argumentos y variables de entorno de la acción Ejecutar'.

4. Añada `debug.tflite.trace` en la sección 'Variables de entorno'.

    ![Configurar variable de entorno](images/xcode_profile_environment.png)

Si desea excluir los eventos de TensorFlow Lite al crear el perfil de la app para iOS, desactive el trazado eliminando la variable de entorno.

#### XCode Instruments

Capture los trazos siguiendo los pasos que se indican a continuación:

1. Seleccione **Producto &gt; Perfil** en los menús de arriba de Xcode.

2. Haga clic en **Registro** entre las plantillas de perfilado cuando se inicie la herramienta Instrumentos.

3. Pulse el botón 'Iniciar'.

4. Pulse el botón 'Detener'.

5. Haga clic en "os_signpost" para ampliar los elementos del subsistema de registro del sistema operativo.

6. Haga clic en 'org.tensorflow.lite' Subsistema de registro del sistema operativo.

7. Investigue el resultado del trazado.

    ![Trazo de Xcode Instruments](images/xcode_traces.png)

En este ejemplo, puede ver la jerarquía de eventos y estadísticas para cada tiempo de operario.

### Usar los datos de trazado

Los datos de trazado le permiten identificar los cuellos de botella en el rendimiento.

Estos son algunos ejemplos de información que puede obtener del perfilador y posibles soluciones para mejorar el rendimiento:

- Si el número de núcleos de CPU disponibles es menor que el número de hilos de inferencia, la sobrecarga de programación de la CPU puede provocar un rendimiento inferior. Puede reprogramar otras tareas intensivas de CPU en su aplicación para evitar la superposición con la inferencia de su modelo o ajustar el número de hilos de interpretación.
- Si los operadores no están totalmente delegados, algunas partes del grafo del modelo se ejecutan en la CPU en lugar de en el acelerador de hardware previsto. Puede sustituir los operadores no admitidos por otros similares admitidos.

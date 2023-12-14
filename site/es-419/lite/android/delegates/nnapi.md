# Delegado NNAPI de TensorFlow Lite

La [API de redes neuronales de Android (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) está disponible en todos los dispositivos que ejecuten Android 8.1 (nivel de API 27) o superior. Ofrece aceleración para modelos TensorFlow Lite en dispositivos Android con aceleradores de hardware compatibles, incluidos:

- Unidad de procesamiento gráfico (GPU)
- Procesador digital de señales (DSP)
- Unidad de procesamiento neuronal (NPU)

El rendimiento variará en función del hardware específico disponible en el dispositivo.

Esta página describe cómo usar el delegado NNAPI con el intérprete TensorFlow Lite en Java y Kotlin. Para las API de Android C, consulte la [documentación de Android Native Developer Kit](https://developer.android.com/ndk/guides/neuralnetworks).

## Probar el delegado NNAPI en su propio modelo

### Importación de Gradle

El delegado NNAPI forma parte del intérprete TensorFlow Lite para Android, versión 1.14.0 o superior. Puede importarlo a su proyecto añadiendo lo siguiente al archivo gradle de su módulo:

```groovy
dependencies {
   implementation 'org.tensorflow:tensorflow-lite:2.0.0'
}
```

### Inicializar del delegado NNAPI

Agregue el código para inicializar el delegado NNAPI antes de inicializar el intérprete de TensorFlow Lite.

Nota: Aunque NNAPI es compatible desde el Nivel de API 27 (Android Oreo MR1), la compatibilidad con las operaciones mejoró significativamente para el Nivel de API 28 (Android Pie) en adelante. Como resultado, recomendamos a los desarrolladores usar el delegado NNAPI para Android Pie o superior para la mayoría de los escenarios.

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

Interpreter.Options options = (new Interpreter.Options());
NnApiDelegate nnApiDelegate = null;
// Initialize interpreter with NNAPI delegate for Android Pie or above
if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
    nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);
}

// Initialize TFLite interpreter
try {
    tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
} catch (Exception e) {
    throw new RuntimeException(e);
}

// Run inference
// ...

// Unload delegate
tfLite.close();
if(null != nnApiDelegate) {
    nnApiDelegate.close();
}
```

## Prácticas recomendadas

### Pruebe el rendimiento antes de la implementación

El rendimiento en runtime puede variar considerablemente en función de la arquitectura del modelo, el tamaño, las operaciones, la disponibilidad del hardware y la utilización de éste en runtime. Por ejemplo, si una app utiliza en gran medida la GPU para el renderizado, quizá la aceleración NNAPI no mejore el rendimiento debido a la contención de recursos. Se recomienda ejecutar una sencilla prueba de rendimiento usando el depurador para medir el tiempo de inferencia. Ejecute la prueba en varios teléfonos con diferentes chipsets (del fabricante o de modelos del mismo fabricante) que sean representativos de su base de usuarios antes de habilitar NNAPI en producción.

Para desarrolladores avanzados, TensorFlow Lite también ofrece [una herramienta de evaluación comparativa de modelos para Android](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark).

### Cree una lista de exclusión de dispositivos

En producción, puede haber casos en los que NNAPI no funcione como se espera. Se recomienda a los desarrolladores que mantengan una lista de dispositivos que no deben usar la aceleración NNAPI en combinación con determinados modelos. Puede crear esta lista basándose en el valor de `"ro.board.platform"`, que puede recuperar utilizando el siguiente fragmento de código:

```java
String boardPlatform = "";

try {
    Process sysProcess =
        new ProcessBuilder("/system/bin/getprop", "ro.board.platform").
        redirectErrorStream(true).start();

    BufferedReader reader = new BufferedReader
        (new InputStreamReader(sysProcess.getInputStream()));
    String currentLine = null;

    while ((currentLine=reader.readLine()) != null){
        boardPlatform = line;
    }
    sysProcess.destroy();
} catch (IOException e) {}

Log.d("Board Platform", boardPlatform);
```

Para los desarrolladores avanzados, considere la posibilidad de conservar esta lista mediante un sistema de configuración remota. El equipo de TensorFlow está trabajando activamente en formas de simplificar y automatizar el descubrimiento y la aplicación de la configuración óptima de la NNAPI.

### Cuantización

La cuantización reduce el tamaño del modelo al usar enteros de 8 bits o flotantes de 16 bits en lugar de flotantes de 32 bits para el cálculo. El tamaño de los modelos con enteros de 8 bits es una cuarta parte del de las versiones con flotantes de 32 bits; con flotantes de 16 bits es la mitad. La cuantización puede mejorar el rendimiento de forma significativa, aunque el proceso podría sacrificar parte de la precisión del modelo.

Existen múltiples tipos de técnicas de cuantificación posteriores al entrenamiento, pero, para obtener la máxima compatibilidad y aceleración en el hardware actual, recomendamos [la cuantización de enteros completa](post_training_quantization#full_integer_quantization_of_weights_and_activations). Este enfoque convierte tanto la ponderación como las operaciones en enteros. Este proceso de cuantización requiere un conjunto de datos representativo para funcionar.

### Use modelos y ops compatibles

Si el delegado NNAPI no admite algunas de las ops o combinaciones de parámetros de un modelo, el framework sólo ejecuta las partes del grafo admitidas en el acelerador. El resto se ejecuta en la CPU, lo que resulta en una ejecución dividida. Debido al alto costo de la sincronización CPU/acelerador, esto puede resultar en un rendimiento más lento que la ejecución de todo el grafo sólo en la CPU.

NNAPI funciona mejor cuando los modelos sólo usan [ops compatibles](https://developer.android.com/ndk/guides/neuralnetworks#model). Se sabe que los siguientes modelos son compatibles con NNAPI:

- [Clasificación de imágenes MobileNet v1 (224x224) (descarga del modelo flotante)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [(descargar modelo cuantizado)](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) <br> *(modelo de clasificación de imágenes diseñado para aplicaciones de visión basadas en móviles e integradas)*
- [Detección de objetos SSD MobileNet v2](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(descargar)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite) <br> *(modelo de clasificación de imágenes que detecta múltiples objetos con cajas delimitadoras)*
- [Detección de objetos con detector de toma única (SSD) MobileNet v1(300x300)](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [(descargar)] (https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
- [PoseNet para la estimación de la pose](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [(descarga)](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite) <br> *(modelo de visión que estima las poses de una(s) persona(s) en imagen o vídeo)*

La aceleración NNAPI tampoco es compatible cuando el modelo contiene salidas de tamaño dinámico. En este caso, recibirá una advertencia del tipo:

```none
ERROR: Attempting to use a delegate that only supports static-sized tensors \
with a graph that has dynamic-sized tensors.
```

### Habilite la implementación de la CPU NNAPI

Un grafo que no pueda ser procesado completamente por un acelerador puede recurrir a la implementación de la CPU NNAPI. Sin embargo, dado que suele tener un rendimiento inferior al del intérprete TensorFlow, esta opción está desactivada por default en el delegado NNAPI para Android 10 (Nivel de API 29) o superior. Para anular este comportamiento, configure `setUseNnapiCpu` como `true` en el objeto `NnApiDelegate.Options`.

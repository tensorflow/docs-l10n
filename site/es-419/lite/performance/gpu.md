# Delegados de GPU para TensorFlow Lite

Usar unidades de procesamiento gráfico (GPU) para ejecutar sus modelos de aprendizaje automático (ML) puede mejorar drásticamente el rendimiento de su modelo y la experiencia de usuario de sus aplicaciones habilitadas para ML. TensorFlow Lite permite usar GPUs y otros procesadores especializados a través de un controlador de hardware llamado [*delegados*](./delegates). Si habilita el uso de GPU con sus aplicaciones ML de TensorFlow Lite, podrá tener los siguientes beneficios:

- **Velocidad**: Las GPU están diseñadas para un alto rendimiento de cargas de trabajo masivamente paralelas. Este diseño las hace muy adecuadas para las redes neuronales profundas, que constan de un enorme número de operarios, cada uno de los cuales trabaja sobre tensores de entrada que pueden procesarse en paralelo, lo que suele resultar en una latencia menor. En el mejor de los casos, la ejecución del modelo en una GPU puede ser lo suficientemente rápida como para permitir aplicaciones en tiempo real que antes no eran posibles.
- **Eficiencia energética**: las GPU realizan los cálculos de ML de forma muy eficiente y optimizada, por lo que suelen consumir menos energía y generar menos calor que la misma tarea ejecutada en CPU.

Este documento da una visión general del soporte de GPUs en TensorFlow Lite, y algunos usos avanzados de los procesadores GPU. Para obtener información más específica sobre cómo implementar el soporte de GPU en plataformas específicas, vea las siguientes guías:

- [Soporte de GPU para Android](../android/delegates/gpu)
- [Soporte de GPU para iOS](../ios/delegates/gpu)

## Soporte a las operaciones de ML en GPU {:#supported_ops}

Existen algunas limitaciones en cuanto a las operaciones de ML de TensorFlow, u *ops*, que puede acelerar el delegado de GPU de TensorFlow Lite. El delegado admite las siguientes ops en precisión flotante de 16 y 32 bits:

- `ADD`
- `AVERAGE_POOL_2D`
- `CONCATENATION`
- `CONV_2D`
- `DEPTHWISE_CONV_2D v1-2`
- `EXP`
- `FULLY_CONNECTED`
- `LOGICAL_AND`
- `LOGISTIC`
- `LSTM v2 (Basic LSTM only)`
- `MAX_POOL_2D`
- `MAXIMUM`
- `MINIMUM`
- `MUL`
- `PAD`
- `PRELU`
- `RELU`
- `RELU6`
- `RESHAPE`
- `RESIZE_BILINEAR v1-3`
- `SOFTMAX`
- `STRIDED_SLICE`
- `SUB`
- `TRANSPOSE_CONV`

De forma predeterminada, todas las ops sólo son compatibles con la versión 1. Al activar el [soporte de cuantización](#quantized-models) se activan las versiones adecuadas, por ejemplo, ADD v2.

### Solución de problemas de soporte de GPU

Si algunas de las ops no son compatibles con el delegado de la GPU, el marco de trabajo sólo ejecutará una parte del grafo en la GPU y la parte restante en la CPU. Debido al alto costo de la sincronización entre la CPU y la GPU, un modo de ejecución dividido como éste suele dar como resultado un rendimiento más lento que cuando todo el grafo se ejecuta sólo en la CPU. En este caso, la aplicación genera advertencias como:

```none
WARNING: op code #42 cannot be handled by this delegate.
```

No existe ninguna retrollamada para fallos de este tipo, ya que no se trata de un fallo real en tiempo de ejecución. Cuando pruebe la ejecución de su modelo con el delegado de la GPU, debe estar atento a estas advertencias. Un alto número de estas advertencias puede indicar que su modelo no es el más adecuado para usar la aceleración por GPU, y puede requerir la refactorización del modelo.

## Modelos de ejemplo

Los siguientes modelos de ejemplo están construidos para aprovechar la aceleración de la GPU con TensorFlow Lite y se proporcionan como referencia y prueba:

- [Clasificación de imágenes MobileNet v1 (224x224)](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html): Un modelo de clasificación de imágenes diseñado para aplicaciones de visión basadas en dispositivos móviles e integrados. ([modelo](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5))
- [Segmentación DeepLab (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html): modelo de segmentación de imágenes que asigna etiquetas semánticas, como perro, gato, coche, a cada pixel de la imagen de entrada. ([modelo](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1))
- [Detección de objetos SSD de MobileNet](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html): Un modelo de clasificación de imágenes que detecta múltiples objetos con cajas delimitadoras. ([modelo](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite))
- [PoseNet para la estimación de poses](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection): Un modelo de visión que estima las poses de las personas en imagen o vídeo. ([modelo](https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1))

## Optimización para GPU

Las siguientes técnicas pueden ayudarle a obtener un mejor rendimiento al ejecutar modelos en hardware de GPU usando el delegado GPU de TensorFlow Lite:

- **Operaciones de cambio de forma**: Algunas operaciones que son rápidas en una CPU pueden tener un alto costo para la GPU en dispositivos móviles. Las operaciones de cambio de forma son especialmente costosas de ejecutar, como `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH`, etc. Deberá examinar detenidamente el uso de las operaciones de cambio de forma y considerar que quizá se aplicaron sólo para explorar los datos o para las primeras iteraciones de su modelo. Si las elimina, puede mejorar significativamente el rendimiento.

- **Canales de datos de imagen**: En la GPU, los datos del tensor se trocean en 4 canales, por lo que un cálculo sobre un tensor con la forma `[B,H,W,5]` rinde más o menos lo mismo sobre un tensor de forma `[B,H,W,8]`, pero significativamente peor que `[B,H,W,4]`. Si el hardware de la cámara que está usando admite cuadros de imagen en RGBA, alimentar esa entrada de 4 canales es significativamente más rápido, ya que evita una copia de memoria de RGB de 3 canales a RGBX de 4 canales.

- **Modelos optimizados para móviles**: Para obtener el mejor rendimiento, debería considerar reentrenar su clasificador con una arquitectura de red optimizada para móviles. La optimización para inferir en el dispositivo puede reducir drásticamente la latencia y el consumo de energía aprovechando las características del hardware móvil.

## Compatibilidad avanzada con GPU

Puede usar otras técnicas avanzadas con el procesamiento en la GPU para mejorar aún más el rendimiento de sus modelos, como la cuantización y la serialización. En las secciones siguientes se describen estas técnicas con más detalle.

### Usar modelos cuantizados {:#quantized-models}

En esta sección se explica cómo el delegado de la GPU acelera los modelos cuantizados de 8 bits, entre los que se incluyen los siguientes:

- Modelos entrenados con [Entrenamiento consciente de la cuantización](https://www.tensorflow.org/model_optimization/guide/quantization/training)
- [Cuantización del rango dinámico](https://www.tensorflow.org/lite/performance/post_training_quant) postentrenamiento
- [Cuantización de enteros](https://www.tensorflow.org/lite/performance/post_training_integer_quant) postentrenamiento

Para optimizar el rendimiento, use modelos que tengan tensores de entrada y salida de punto flotante.

#### ¿Cómo funciona?

Dado que el backend de la GPU sólo admite la ejecución en punto flotante, ejecutamos los modelos cuantizados dándole una "vista en punto flotante" del modelo original. A alto nivel, esto implica los siguientes pasos:

- Los *tensores constantes* (como las ponderaciones/sesgos) se descuantizan una vez en la memoria de la GPU. Esta operación tiene lugar cuando el delegado está habilitado para TensorFlow Lite.

- Las *entradas y salidas* del programa de la GPU, si están cuantizadas a 8 bits, se descuantizan y cuantizan (respectivamente) para cada inferencia. Esta operación se realiza en la CPU usando los kernels optimizados de TensorFlow Lite.

- Los *simuladores de cuantización* se insertan entre las operaciones para imitar el comportamiento cuantizado. Este enfoque es necesario para los modelos en los que las ops esperan que las activaciones sigan los límites aprendidos durante la cuantización.

Para más información sobre cómo activar esta función con el delegado de GPU, consulte lo siguiente:

- Usar [modelos cuantizados con GPU en Android](../android/delegates/gpu#quantized-models)
- Usar [modelos cuantizados con GPU en iOS](../ios/delegates/gpu#quantized-models)

### Reducir el tiempo de inicialización con la serialización {:#delegate_serialization}

La función de delegado GPU le permite cargar desde el kernel código precompilado y datos del modelo serializados y guardados en disco de ejecuciones anteriores. Este enfoque evita tener que volver a compilar y puede reducir el tiempo de arranque hasta en un 90%. Esta mejora se consigue intercambiando espacio en disco por ahorro de tiempo. Puede activar esta función con unas pocas opciones de configuración, como se muestra en los siguientes ejemplos de código:

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.serialization_dir = kTmpDir;
    options.model_token = kModelToken;

    auto* delegate = TfLiteGpuDelegateV2Create(options);
    if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    GpuDelegate delegate = new GpuDelegate(
      new GpuDelegate.Options().setSerializationParams(
        /* serializationDir= */ serializationDir,
        /* modelToken= */ modelToken));

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

Cuando use la función de serialización, asegúrese de que su código cumple estas reglas de implementación:

- Almacene los datos de serialización en un directorio al que no puedan acceder otras apps. En dispositivos Android, use [`getCodeCacheDir()`](https://developer.android.com/reference/android/content/Context#getCacheDir()) que apunta a una ubicación privada para la aplicación actual.
- El token del modelo debe ser exclusivo del dispositivo para el modelo específico. Puede calcular un token de modelo generando una huella digital a partir de los datos del modelo usando bibliotecas como [`farmhash::Fingerprint64`](https://github.com/google/farmhash).

Nota: Para usar esta función de serialización se requiere el [SDK OpenCL](https://github.com/KhronosGroup/OpenCL-SDK).

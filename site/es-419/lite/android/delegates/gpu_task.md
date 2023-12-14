# Delegado de aceleración de GPU con librería de tareas

Usar unidades de procesamiento gráfico (GPU) para ejecutar sus modelos de aprendizaje automático (ML) puede mejorar drásticamente el rendimiento y la experiencia de usuario de sus aplicaciones habilitadas para ML. En los dispositivos Android, puede habilitar la ejecución acelerada por GPU de sus modelos usando un [*delegado*](../../performance/delegates) y una de las siguientes API:

- API del Intérprete: [guía](./gpu)
- API de librería de tareas: esta guía.
- API nativa (C/C++): [guía](./gpu_native)

Esta página describe cómo habilitar la aceleración por GPU para los modelos de TensorFlow Lite en apps Android usando la librería de tareas. Para obtener más información sobre el delegado de GPU para TensorFlow Lite, incluidas las mejores prácticas y técnicas avanzadas, consulte la página [Delegados de GPU](../../performance/gpu).

## Use la GPU con TensorFlow Lite con los servicios de Google Play

Las [librerías de tareas](../../inference_with_metadata/task_library/overview) de TensorFlow Lite ofrecen un conjunto de API específicas de tareas para crear aplicaciones de aprendizaje automático. Esta sección describe cómo usar el delegado del acelerador de la GPU con estas API utilizando TensorFlow Lite con los servicios de Google Play.

[TensorFlow Lite con los servicios de Google Play](../play_services) es la ruta recomendada para usar TensorFlow Lite en Android. Si su aplicación está dirigida a dispositivos que no ejecutan Google Play, consulte la sección [GPU con librería de tareas y TensorFlow Lite independiente](#standalone).

### Añada las dependencias del proyecto

Para habilitar el acceso al delegado de la GPU con las bibliotecas de tareas de TensorFlow Lite usando los servicios de Google Play, añada `com.google.android.gms:play-services-tflite-gpu` a las dependencias del archivo `build.gradle` de su app:

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### Habilite la aceleración de la GPU

Luego, verifique de forma asíncrona que el delegado de la GPU esté disponible para el dispositivo mediante la clase [`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu) y active la opción de delegado de la GPU para su clase modelo de la API de tareas con la clase [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por ejemplo, puede configurar la GPU en `ObjectDetector` como se muestra en los siguientes ejemplos de código:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

        lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
          val baseOptionsBuilder = BaseOptions.builder()
          if (task.result) {
            baseOptionsBuilder.useGpu()
          }
        ObjectDetectorOptions.builder()
                  .setBaseOptions(baseOptionsBuilder.build())
                  .setMaxResults(1)
                  .build()
        }
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">      Task&lt;Boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

      Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
        BaseOptions baseOptionsBuilder = BaseOptions.builder();
        if (task.getResult()) {
          baseOptionsBuilder.useGpu();
        }
        return ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setMaxResults(1)
                .build()
      });
      </pre>
    </section>
  </devsite-selector>
</div>

## Use la GPU con TensorFlow Lite autónomo {:#standalone}

Si su aplicación está dirigida a dispositivos que no ejecutan Google Play, es posible vincular el delegado de GPU a su aplicación y usarlo con la versión independiente de TensorFlow Lite.

### Añada las dependencias del proyecto

Para habilitar el acceso al delegado de la GPU con las librerías de tareas de TensorFlow Lite usando la versión independiente de TensorFlow Lite, añada `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` a las dependencias del archivo `build.gradle` de su app:

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Habilite la aceleración de la GPU

Luego habilite la opción de delegado de la GPU para su clase modelo de la API de tareas con la clase [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por ejemplo, puede configurar la GPU en `ObjectDetector` como se muestra en los siguientes ejemplos de código:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    val baseOptions = BaseOptions.builder().useGpu().build()

    val options =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build()

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    import org.tensorflow.lite.task.core.BaseOptions
    import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

    BaseOptions baseOptions = BaseOptions.builder().useGpu().build();

    ObjectDetectorOptions options =
        ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setMaxResults(1)
            .build();

    val objectDetector = ObjectDetector.createFromFileAndOptions(
      context, model, options);
      </pre>
    </section>
  </devsite-selector>
</div>

# TensorFlow Lite en servicios de Google Play

TensorFlow Lite está disponible en el runtime de los servicios de Google Play para todos los dispositivos Android que ejecuten la versión actual de los Play Services. Este runtime le permite ejecutar modelos de aprendizaje automático (ML) sin necesidad de integrar estáticamente las librerías de TensorFlow Lite en su app.

Con la API de servicios de Google Play, puede reducir el tamaño de sus apps y obtener un rendimiento mejorado de la última versión estable de las librerías. TensorFlow Lite en los servicios de Google Play es la forma recomendada de usar TensorFlow Lite en Android.

Para empezar familiarizarse con el runtime de los servicios de Play, consulte la sección [Inicio rápido](../android/quickstart), que contiene una guía paso a paso para implementar una aplicación de ejemplo. Si ya está utilizando TensorFlow Lite autónomo en su app, consulte la sección [Migrar desde TensorFlow Lite autónomo](#migrating) para actualizar una app existente y usar el runtime de los servicios Play. Para obtener más información sobre los servicios de Google Play, consulte el sitio web de [servicios de Google Play](https://developers.google.com/android/guides/overview).

<aside class="note"> <b>Condiciones:</b> Al acceder o usar las API de TensorFlow Lite en los servicios de Google Play, usted acepta las <a href="#tos">Condiciones del servicio</a>. Lea y comprenda todos los términos y políticas aplicables antes de acceder a las API.</aside>

## Usar el runtime de servicios Play

TensorFlow Lite en los servicios de Google Play está disponible a través de la [API de tareas de TensorFlow Lite](../api_docs/java/org/tensorflow/lite/task/core/package-summary) y la  [API de intérptere de TensorFlow Lite](../api_docs/java/org/tensorflow/lite/InterpreterApi). La librería de tareas ofrece interfaces optimizadas de modelos listos para usar para tareas comunes de aprendizaje automático que utilizan datos visuales, de audio y de texto. La API de intérprete de TensorFlow Lite, incluida en las librerías de soporte y runtime de TensorFlow, constituye una interfaz de propósito más general para crear y ejecutar modelos de ML.

Las siguientes secciones dan instrucciones sobre cómo implementar las API del intérprete y de la librería de tareas en los servicios de Google Play. Aunque una app puede usar tanto la API del intérprete como la de la librería de tareas, la mayoría de las apps sólo deberían usar un conjunto de APIs.

### Usar las API de la librería de tareas

La API de tareas de TensorFlow Lite encapsula la API del intérprete y brinda una interfaz de programación de alto nivel para tareas comunes de aprendizaje automático que usan datos visuales, de audio y de texto. Debe usar la API de tareas si su aplicación requiere una de las [tareas admitidas](../inference_with_metadata/task_library/overview#supported_tasks).

#### 1. Añada las dependencias del proyecto

La dependencia de su proyecto depende de su caso de uso de aprendizaje automático. Las API de tareas contienen las siguientes librerías:

- Librería de visión: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- Librería de audio: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- Librería de texto: `org.tensorflow:tensorflow-lite-task-text-play-services`

Añada una de las dependencias al código del proyecto de su app para acceder a la API de servicios de Play para TensorFlow Lite. Por ejemplo, use lo siguiente para implementar una tarea de visión:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Precaución: La librería de tareas de audio TensorFlow Lite versión 0.4.2 del repositorio maven está incompleta. Use en su lugar la versión 0.4.2.1 de esta librería: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. Añada la inicialización de TensorFlow Lite

Inicialice el componente TensorFlow Lite de la API de servicios de Google Play *antes* de usar las API de TensorFlow Lite. El siguiente ejemplo inicializa la librería de visión:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">init {
  TfLiteVision.initialize(context)
    }
  }
</pre>
    </section>
  </devsite-selector>
</div>

Importante: Asegúrese de que la tarea `TfLite.initialize` se complete antes de ejecutar código que acceda a las APIs de TensorFlow Lite.

Consejo: Los módulos de TensorFlow Lite se instalan simultáneamente a la instalación o actualización de su aplicación desde Play Store. Puede comprobar la disponibilidad de los módulos usando `ModuleInstallClient` de las API de servicios de Google Play. Para más información sobre la comprobación de la disponibilidad de los módulos, consulte [Garantizar la disponibilidad de la API con ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Ejecutar inferencia

Después de inicializar el componente TensorFlow Lite, llame al método `detect()` para generar inferencias. El código exacto dentro del método `detect()` varía dependiendo de la librería y del caso de uso. El ejemplo siguiente corresponde a un caso de uso sencillo de detección de objetos con la librería `TfLiteVision`:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">fun detect(...) {
  if (!TfLiteVision.isInitialized()) {
    Log.e(TAG, "detect: TfLiteVision is not initialized yet")
    return
  }

  if (objectDetector == null) {
    setupObjectDetector()
  }

  ...

}
</pre>
    </section>
  </devsite-selector>
</div>

Dependiendo del formato de los datos, puede que también necesite preprocesar y convertir sus datos dentro del método `detect()` antes de generar inferencias. Por ejemplo, los datos de imagen para un detector de objetos requieren lo siguiente:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Usar las API del intérprete

Las APIs del Intérprete ofrecen más control y flexibilidad que las APIs de la librería de Tareas. Debe usar las API del intérprete si su tarea de aprendizaje automático no es admitida por la librería de tareas, o si necesita una interfaz de propósito más general para construir y ejecutar modelos ML.

#### 1. Añada las dependencias del proyecto

Añada las siguientes dependencias al código del proyecto de su app para acceder a la API de servicios de Play para TensorFlow Lite:

```
dependencies {
...
    // Tensorflow Lite dependencies for Google Play services
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    // Optional: include Tensorflow Lite Support Library
    implementation 'com.google.android.gms:play-services-tflite-support:16.0.1'
...
}
```

#### 2. Añada la inicialización de TensorFlow Lite

Inicialice el componente TensorFlow Lite de la API de servicios de Google Play *antes* de usar las API de TensorFlow Lite:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
val initializeTask: Task&lt;Void&gt; by lazy { TfLite.initialize(this) }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
Task&lt;Void&gt; initializeTask = TfLite.initialize(context);
</pre>
    </section>
  </devsite-selector>
</div>

Nota: Asegúrese de que la tarea `TfLite.initialize` se complete antes de ejecutar código que acceda a las API de TensorFlow Lite. Use el método `addOnSuccessListener()`, como se muestra en la siguiente sección.

#### 3. Cree un intérprete y configure la opción runtime {:#step_3_interpreter}

Cree un intérprete utilizando `InterpreterApi.create()` y configúrelo para usar el runtime de los servicios de Google Play, llamando a `InterpreterApi.Options.setRuntime()`, como se muestra en el siguiente código de ejemplo:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private lateinit var interpreter: InterpreterApi
...
initializeTask.addOnSuccessListener {
  val interpreterOption =
    InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  interpreter = InterpreterApi.create(
    modelBuffer,
    interpreterOption
  )}
  .addOnFailureListener { e -&gt;
    Log.e("Interpreter", "Cannot initialize interpreter", e)
  }
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
...
private InterpreterApi interpreter;
...
initializeTask.addOnSuccessListener(a -&gt; {
    interpreter = InterpreterApi.create(modelBuffer,
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY));
  })
  .addOnFailureListener(e -&gt; {
    Log.e("Interpreter", String.format("Cannot initialize interpreter: %s",
          e.getMessage()));
  });
</pre>
    </section>
  </devsite-selector>
</div>

Debería usar la implementación anterior porque evita el bloqueo del hilo de la interfaz de usuario de Android. Si necesita administrar más de cerca la ejecución del hilo, puede añadir una llamada `Tasks.await()` a la creación del intérprete:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
import androidx.lifecycle.lifecycleScope
...
lifecycleScope.launchWhenStarted { // uses coroutine
  initializeTask.await()
}
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
@BackgroundThread
InterpreterApi initializeInterpreter() {
    Tasks.await(initializeTask);
    return InterpreterApi.create(...);
}
</pre>
    </section>
  </devsite-selector>
</div>

Advertencia: No llame a `.await()` en el hilo de interfaz de usuario en primer plano porque interrumpe la visualización de los elementos de la interfaz de usuario y crea una mala experiencia para el usuario.

#### 4. Ejecute inferencias

Usando el objeto `interpreter` que ha creado, llame al método `run()` para generar una inferencia.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
<pre class="prettyprint">
interpreter.run(inputBuffer, outputBuffer)
</pre>
    </section>
    <section>
      <h3>Java</h3>
<pre class="prettyprint">
interpreter.run(inputBuffer, outputBuffer);
</pre>
    </section>
  </devsite-selector>
</div>

## Aceleración del hardware {:#hardware-acceleration}

TensorFlow Lite le permite acelerar el rendimiento de su modelo usando procesadores de hardware especializados, como las unidades de procesamiento gráfico (GPU). Puede aprovechar estos procesadores especializados usando controladores de hardware llamados [*delegados*](https://www.tensorflow.org/lite/performance/delegates). Puede usar los siguientes delegados de aceleración de hardware con TensorFlow Lite en los servicios de Google Play:

- *[Delegado de GPU](https://www.tensorflow.org/lite/performance/gpu) (recomendado)*: Este delegado se suministra a través de los servicios de Google Play y se carga dinámicamente, al igual que las versiones de los servicios de Play de la API de tareas y la API de intérprete.

- [*Delegado NNAPI*](https://www.tensorflow.org/lite/android/delegates/nnapi): Este delegado está disponible como dependencia de una librería incluida en su proyecto de desarrollo Android, y se integra en su app.

Para obtener más información sobre la aceleración por hardware con TensorFlow Lite, consulte la página [Delegados de TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Comprobación de la compatibilidad del dispositivo

No todos los dispositivos son compatibles con la aceleración por hardware de la GPU con TFLite. Para mitigar errores y posibles fallos, use el método `TfLiteGpu.isGpuDelegateAvailable` para comprobar si un dispositivo es compatible con el delegado de GPU.

Use este método para confirmar si un dispositivo es compatible con la GPU y usar la CPU o el delegado NNAPI como alternativa para cuando la GPU no sea compatible.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Una vez que tenga una variable como `useGpuTask`, puede utilizarla para determinar si los dispositivos usan el delegado de la GPU. A continuación se muestran ejemplos de cómo hacerlo tanto con la librería de tareas como con la API del intérprete.

**Con la Api de tareas**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">lateinit val optionsTask = useGpuTask.continueWith { task -&gt;
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
    <pre class="prettyprint">Task&lt;ObjectDetectorOptions&gt; optionsTask = useGpuTask.continueWith({ task -&gt;
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

**Con la Api de intérprete**

<div>
<devsite-selector>
<section>
  <h3>Kotlin</h3>
    <pre class="prettyprint">val interpreterTask = useGpuTask.continueWith { task -&gt;
  val interpreterOptions = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
  if (task.result) {
      interpreterOptions.addDelegateFactory(GpuDelegateFactory())
  }
  InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOptions)
}
    </pre>
</section>
<section>
  <h3>Java</h3>
    <pre class="prettyprint">Task&lt;InterpreterApi.Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
  InterpreterApi.Options options =
      new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
  if (task.getResult()) {
     options.addDelegateFactory(new GpuDelegateFactory());
  }
  return options;
});
    </pre>
</section>
</devsite-selector>
</div>

### GPU con las API de la librería de tareas

Para usar el delegado de la GPU con las API de tareas:

1. Actualice las dependencias del proyecto para usar el delegado de GPU de los servicios Play:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. Inicialice el delegado de la GPU con `setEnableGpuDelegateSupport`. Por ejemplo, puede inicializar el delegado de GPU para `TfLiteVision` con lo siguiente:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLiteVision.initialize(context, TfLiteInitializationOptions.builder().setEnableGpuDelegateSupport(true).build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Habilite la opción de delegado GPU con [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val baseOptions = BaseOptions.builder().useGpu().build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        BaseOptions baseOptions = BaseOptions.builder().useGpu().build();
            </pre>
    </section>
    </devsite-selector>
    </div>

4. Configure las opciones utilizando `.setBaseOptions`. Por ejemplo, puede configurar la GPU en `ObjectDetector` de la forma siguiente:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        val options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build()
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        ObjectDetectorOptions options =
                ObjectDetectorOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMaxResults(1)
                    .build();
            </pre>
    </section>
    </devsite-selector>
    </div>

### GPU con API de intérprete

Para usar el delegado de la GPU con las API del intérprete:

1. Actualice las dependencias del proyecto para usar el delegado de GPU de los servicios Play:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. Habilite la opción de delegado de GPU en la inicialización de TFlite:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">        TfLite.initialize(context,
              TfLiteInitializationOptions.builder()
               .setEnableGpuDelegateSupport(true)
               .build());
            </pre>
    </section>
    </devsite-selector>
    </div>

3. Configure el delegado de la GPU en las opciones del intérprete para usar `DelegateFactory` llamando a `addDelegateFactory()` dentro de `InterpreterApi.Options()`:

    <div>
    <devsite-selector>
    <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
              val interpreterOption = InterpreterApi.Options()
               .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
               .addDelegateFactory(GpuDelegateFactory())
            </pre>
    </section>
    <section>
      <h3>Java</h3>
        <pre class="prettyprint">
              Options interpreterOption = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                .addDelegateFactory(new GpuDelegateFactory());
            </pre>
    </section>
    </devsite-selector>
    </div>

## Migrar desde TensorFlow Lite autónomo {:#migrating}

Si está planeando migrar su app de TensorFlow Lite independiente a la API de servicios de Play, estudie la siguiente guía adicional para actualizar el código del proyecto de su app:

1. Revise la sección [Limitaciones](#limitations) de esta página para asegurarse de que su caso de uso es compatible.
2. Antes de actualizar su código, realice comprobaciones de rendimiento y precisión de sus modelos, en particular si está usando versiones de TensorFlow Lite anteriores a la versión 2.1, para tener una línea de referencia con la que comparar la nueva implementación.
3. Si ha migrado todo su código para usar la API de servicios Play para TensorFlow Lite, debería eliminar las dependencias existentes de la *librería runtime* de TensorFlow Lite (entradas con <code>org.tensorflow:**tensorflow-lite**:*</code>) de su archivo build.gradle para poder reducir el tamaño de su app.
4. Identifique todas las apariciones de la creación de objetos `new Interpreter` en su código, y modifíquelo para que use la llamada InterpreterApi.create(). Esta nueva API es asíncrona, o sea que en la mayoría de los casos no se sustituye directamente, y deberá registrar un gestor de eventos para cuando se complete la llamada. Consulte el fragmento que aparece en el código del [Paso 3](#step_3_interpreter).
5. Añada `import org.tensorflow.lite.InterpreterApi;` e `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` a cualquier archivo fuente que use las clases `org.tensorflow.lite.Interpreter` o `org.tensorflow.lite.InterpreterApi`.
6. Si alguna de las llamadas resultantes a `InterpreterApi.create()` tiene un único argumento, añada `new InterpreterApi.Options()` a la lista de argumentos.
7. Añada `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` al último argumento de cualquier llamada a `InterpreterApi.create()`.
8. Reemplace todas las demás apariciones de la clase `org.tensorflow.lite.Interpreter` por `org.tensorflow.lite.InterpreterApi`.

Si desea usar TensorFlow Lite independiente y la API de servicios Play uno al lado del otro, debe usar TensorFlow Lite 2.9 (o posterior). TensorFlow Lite 2.8 y versiones anteriores no son compatibles con la versión de la API de servicios Play.

## Limitaciones

TensorFlow Lite en los servicios de Google Play tiene las siguientes limitaciones:

- La compatibilidad con los delegados de aceleración por hardware se limita a los delegados enumerados en la sección [Aceleración por hardware](#hardware-acceleration). No hay otros delegados de aceleración soportados.
- No se admite el acceso a TensorFlow Lite a través de [las API nativas](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c). Solo las API Java de TensorFlow Lite están disponibles a través de los servicios de Google Play.
- No se admiten las API experimentales u obsoletas de TensorFlow Lite, incluidas las ops personalizadas.

## Soporte y retroalimentación {:#support}

Puede dar retroalimentación y obtener soporte a través del TensorFlow Issue Tracker. Le rogamos que notifique los problemas y las solicitudes de asistencia utilizando la [plantilla de problemas](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) para TensorFlow Lite en los servicios de Google Play.

## Términos de servicio {:#tos}

El uso de TensorFlow Lite en las API de los servicios de Google Play está sujeto a las [Condiciones de servicio de las API de Google](https://developers.google.com/terms/).

### Privacidad y recopilación de datos

Cuando usa TensorFlow Lite en las API de servicios de Google Play, el procesamiento de los datos de entrada, como imágenes, vídeo, texto, se produce completamente en el dispositivo, y TensorFlow Lite en las API de servicios de Google Play no envía esos datos a los servidores de Google. En consecuencia, puede usar nuestras API para procesar datos que no deban salir del dispositivo.

Las API de servicios de TensorFlow Lite en Google Play pueden ponerse en contacto con los servidores de Google ocasionalmente para recibir información sobre corrección de errores, modelos actualizados y compatibilidad con aceleradores de hardware. Las API de servicios de TensorFlow Lite en Google Play también envían a Google métricas sobre el rendimiento y la utilización de las API de su app. Google utiliza estos datos de métricas para medir el rendimiento, depurar, hacer el mantenimiento y mejorar las API, y detectar usos indebidos o abusos, tal y como se describe con más detalle en nuestra [Política de privacidad](https://policies.google.com/privacy).

**Usted es responsable de informar a los usuarios de su app sobre el procesamiento por parte de Google de los datos de métricas de TensorFlow Lite en las API de los servicios de Google Play, tal y como exige la legislación aplicable.**

Los datos que recopilamos son los siguientes:

- Información del dispositivo (como el fabricante, el modelo, la versión del sistema operativo y la compilación) y los aceleradores de hardware de ML disponibles (GPU y DSP). Usado para diagnósticos y análisis de uso.
- Identificador del dispositivo usado para diagnósticos y análisis de uso.
- Información sobre la app (nombre del paquete, versión de la app). Usada para diagnósticos y análisis de uso.
- Configuración de la API (como qué delegados se están usando). Usada para diagnósticos y análisis de uso.
- Tipo de evento (como creación de intérprete, inferencia). Usado para diagnósticos y análisis de uso.
- Códigos de error. Usados para diagnósticos.
- Métricas de rendimiento. Usadas para diagnósticos.

## Siguientes pasos

Para obtener más información sobre cómo implementar el aprendizaje automático en su aplicación Móvil con TensorFlow Lite, consulte la [Guía del desarrollador de TensorFlow Lite](https://www.tensorflow.org/lite/guide). Puede encontrar modelos adicionales de TensorFlow Lite para clasificación de imágenes, detección de objetos y otras aplicaciones en el [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite).

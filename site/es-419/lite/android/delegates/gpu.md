# Delegado de aceleración de GPU con API de intérprete

Usar unidades de procesamiento gráfico (GPU) para ejecutar sus modelos de aprendizaje automático (ML) puede mejorar drásticamente el rendimiento y la experiencia de usuario de sus aplicaciones habilitadas para ML. En los dispositivos Android, puede habilitar

[*delegate*](../../performance/delegates) y una de las siguientes API:

- API del intérprete: esta guía
- API de librería de tareas: [guía](./gpu_task)
- API nativa (C/C++): [guía](./gpu_native)

Esta página describe cómo habilitar la aceleración por GPU para modelos TensorFlow Lite en apps Android usando la API de intérprete. Para saber más sobre cómo usar el delegado de GPU para TensorFlow Lite, incluyendo las mejores prácticas y técnicas avanzadas, consulte la página [Delegados de GPU](../../performance/gpu).

## Use la GPU con TensorFlow Lite con los servicios de Google Play

La [API del intérprete](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) de TensorFlow Lite ofrece un grupo de APIs de propósito general para crear aplicaciones de aprendizaje automático. Esta sección describe cómo usar el delegado del acelerador de GPU con estas API con TensorFlow Lite con los servicios de Google Play.

[TensorFlow Lite con los servicios de Google Play](../play_services) es la ruta recomendada para usar TensorFlow Lite en Android. Si su aplicación está dirigida a dispositivos que no ejecutan Google Play, consulte la sección [GPU con API de intérprete y TensorFlow Lite independiente](#standalone).

### Añada las dependencias del proyecto

Para habilitar el acceso al delegado de la GPU, añada `com.google.android.gms:play-services-tflite-gpu` al archivo `build.gradle` de su app:

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### Habilite la aceleración de la GPU

Después inicialice TensorFlow Lite con los servicios de Google Play con el soporte GPU:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">    val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)

    val interpreterTask = useGpuTask.continueWith { useGpuTask -&gt;
      TfLite.initialize(context,
          TfLiteInitializationOptions.builder()
          .setEnableGpuDelegateSupport(useGpuTask.result)
          .build())
      }
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">    Task&lt;boolean&gt; useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context);

    Task&lt;Options&gt; interpreterOptionsTask = useGpuTask.continueWith({ task -&gt;
      TfLite.initialize(context,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build());
    });
      </pre>
    </section>
  </devsite-selector>
</div>

Finalmente puede inicializar el intérprete pasando un `GpuDelegateFactory` por `InterpreterApi.Options`:

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">
    val options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(GpuDelegateFactory())

    val interpreter = InterpreterApi(model, options)

    // Run inference
    writeToInput(input)
    interpreter.run(input, output)
    readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">
    Options options = InterpreterApi.Options()
      .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
      .addDelegateFactory(new GpuDelegateFactory());

    Interpreter interpreter = new InterpreterApi(model, options);

    // Run inference
    writeToInput(input);
    interpreter.run(input, output);
    readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

Nota: El delegado GPU debe crearse en el mismo hilo que lo ejecuta. De lo contrario, puede aparecer el siguiente error, `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized.`

El delegado de la GPU también se puede usar con la vinculación de modelos ML en Android Studio. Para más información, consulte [Generar interfaces de modelos usando metadatos](../../inference_with_metadata/codegen#acceleration).

## Use la GPU con TensorFlow Lite autónomo {:#standalone}

Si su aplicación está dirigida a dispositivos que no ejecutan Google Play, es posible vincular el delegado de GPU a su aplicación y usarlo con la versión independiente de TensorFlow Lite.

### Añada las dependencias del proyecto

Para habilitar el acceso al delegado de la GPU, añada `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` al archivo `build.gradle` de su app:

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Habilite la aceleración de la GPU

Luego ejecute TensorFlow Lite en la GPU con `TfLiteDelegate`. En Java, puede especificar el `GpuDelegate` a través de `Interpreter.Options`.

<div>
  <devsite-selector>
    <section>
      <h3>Kotlin</h3>
      <p></p>
<pre class="prettyprint lang-kotlin">      import org.tensorflow.lite.Interpreter
      import org.tensorflow.lite.gpu.CompatibilityList
      import org.tensorflow.lite.gpu.GpuDelegate

      val compatList = CompatibilityList()

      val options = Interpreter.Options().apply{
          if(compatList.isDelegateSupportedOnThisDevice){
              // if the device has a supported GPU, add the GPU delegate
              val delegateOptions = compatList.bestOptionsForThisDevice
              this.addDelegate(GpuDelegate(delegateOptions))
          } else {
              // if the GPU is not supported, run on 4 threads
              this.setNumThreads(4)
          }
      }

      val interpreter = Interpreter(model, options)

      // Run inference
      writeToInput(input)
      interpreter.run(input, output)
      readFromOutput(output)
      </pre>
    </section>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">      import org.tensorflow.lite.Interpreter;
      import org.tensorflow.lite.gpu.CompatibilityList;
      import org.tensorflow.lite.gpu.GpuDelegate;

      // Initialize interpreter with GPU delegate
      Interpreter.Options options = new Interpreter.Options();
      CompatibilityList compatList = CompatibilityList();

      if(compatList.isDelegateSupportedOnThisDevice()){
          // if the device has a supported GPU, add the GPU delegate
          GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
          GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
          options.addDelegate(gpuDelegate);
      } else {
          // if the GPU is not supported, run on 4 threads
          options.setNumThreads(4);
      }

      Interpreter interpreter = new Interpreter(model, options);

      // Run inference
      writeToInput(input);
      interpreter.run(input, output);
      readFromOutput(output);
      </pre>
    </section>
  </devsite-selector>
</div>

### Modelos cuantizados  {:#quantized-models}

Las librerías de delegado de GPU de Android admiten modelos cuantizados de forma predeterminada. No es necesario realizar ningún cambio en el código para usar modelos cuantizados con la GPU delegada. En la siguiente sección se explica cómo desactivar el soporte cuantizado para realizar pruebas o con fines experimentales.

#### Deshabilite el soporte de modelos cuantizados

El siguiente código muestra cómo ***deshabilitar*** el soporte para modelos cuantizados.

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <p></p>
<pre class="prettyprint lang-java">GpuDelegate delegate = new GpuDelegate(new GpuDelegate.Options().setQuantizedModelsAllowed(false));

Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
      </pre>
    </section>
  </devsite-selector>
</div>

Para obtener más información sobre la ejecución de modelos cuantizados con aceleración de GPU, consulte la descripción general de [Delegado de GPU](../../performance/gpu#quantized-models).

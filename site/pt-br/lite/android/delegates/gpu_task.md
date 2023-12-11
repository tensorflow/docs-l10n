# Delegado de aceleração de GPU com a Biblioteca Task

O uso de unidades de processamento gráfico (GPUs) para executar seus modelos de aprendizado de máquina (ML) pode melhorar drasticamente o desempenho e a experiência do usuário dos seus aplicativos com tecnologia de ML. Nos dispositivos Android, você pode ativar a execução dos seus modelos com a aceleração de GPU usando um [*delegado*](../../performance/delegates) e uma das seguintes APIs:

- API Interpreter - [guia](./gpu)
- API Biblioteca Task - este guia
- API nativa (C/C++) - [guia](./gpu_native)

Esta página descreve como ativar a aceleração de GPU para os modelos do TensorFlow Lite nos apps Android usando a Biblioteca Task. Para mais informações sobre como usar o delegado de GPU para o TensorFlow Lite, incluindo práticas recomendadas e técnicas avançadas, confira a página [delegados de GPU](../../performance/gpu).

## Use a GPU com o TensorFlow Lite e o Google Play Services

As [Task Libraries](../../inference_with_metadata/task_library/overview) do TensorFlow Lite fornecem um conjunto de APIs específicas a tarefas para criar aplicativos de aprendizado de máquina. Esta seção descreve como usar o delegado acelerador de GPU com essas APIs usando o TensorFlow Lite com o Google Play Services.

O [TensorFlow com o Google Play Services](../play_services) é o método recomendado para usar o TensorFlow Lite no Android. Se o seu aplicativo segmentar dispositivos que não executam o Google Play, confira a seção [GPU com a Biblioteca Task e o TensorFlow Lite standalone](#standalone).

### Adicione as dependências do projeto

Para habilitar o acesso ao delegado de GPU com as Task Libraries do TensorFlow usando o Google Play Services, adicione `com.google.android.gms:play-services-tflite-gpu` às dependências do arquivo `build.gradle` do seu aplicativo:

```
dependencies {
  ...
  implementation 'com.google.android.gms:play-services-tflite-gpu:16.0.0'
}
```

### Ative a aceleração de GPU

Em seguida, verifique de maneira assíncrona se o delegado de GPU está disponível para o dispositivo usando a classe [`TfLiteGpu`](https://developers.google.com/android/reference/com/google/android/gms/tflite/gpu/support/TfLiteGpu) e ative a opção de delegado de GPU para sua classe de modelo da API Task com a classe [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por exemplo, você pode configurar a GPU em `ObjectDetector` conforme mostrado nos códigos de exemplo abaixo:

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

## Use a GPU com o TensorFlow Lite standalone {:#standalone}

Se o seu aplicativo segmentar dispositivos que não executam o Google Play, é possível empacotar o delegado de GPU com seu aplicativo e usá-lo com a versão standalone do TensorFlow Lite.

### Adicione as dependências do projeto

Para habilitar o acesso ao delegado de GPU com as Task Libraries usando a versão standalone do TensorFlow Lite, adicione `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` às dependências do arquivo `build.gradle`:

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite'
  implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Ative a aceleração de GPU

Em seguida, ative a opção de delegado de GPU para sua classe de modelo da API Task com a classe [<code>BaseOptions</code>](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder). Por exemplo, você pode configurar a GPU em `ObjectDetector` conforme mostrado nos códigos de exemplo a seguir:

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

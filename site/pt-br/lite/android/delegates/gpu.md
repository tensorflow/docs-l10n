# Delegado de aceleração de GPU com a API Interpreter

O uso de unidades de processamento gráfico (GPUs) para executar seus modelos de aprendizado de máquina (ML) pode melhorar drasticamente o desempenho a experiência do usuário dos seus aplicativos com tecnologia de ML. Nos dispositivos Android, você pode ativar

o [*delegado*](../../performance/delegates) e uma das seguintes APIs:

- API Interpreter - este guia
- API Biblioteca Task - [guia](./gpu_task)
- API nativa (C/C++) - [guia](./gpu_native)

Esta página descreve como ativar a aceleração de GPU para os modelos do TensorFlow Lite nos apps Android usando a API Interpreter. Para mais informações sobre como usar o delegado de GPU para o TensorFlow Lite, incluindo práticas recomendadas e técnicas avançadas, confira a página [delegados de GPU](../../performance/gpu).

## Use a GPU com o TensorFlow Lite e o Google Play Services

A [API Interpreter](https://tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi) do TensorFlow Lite fornece um conjunto de APIs de uso geral para criar aplicativos de aprendizado de máquina. Esta seção descreve como usar o delegado acelerador de GPU com essas APIs usando o TensorFlow Lite com o Google Play Services.

O [TensorFlow com o Google Play Services](../play_services) é o método recomendado para usar o TensorFlow Lite no Android. Se o seu aplicativo segmentar dispositivos que não executam o Google Play, confira a seção [GPU com a API Interpreter e o TensorFlow Lite standalone](#standalone).

### Adicione as dependências do projeto

Para habilitar o acesso ao delegado de GPU, adicione `com.google.android.gms:play-services-tflite-gpu` ao arquivo `build.gradle` do seu aplicativo:

```
dependencies {
    ...
    implementation 'com.google.android.gms:play-services-tflite-java:16.0.1'
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
}
```

### Ative a aceleração de GPU

Em seguida, inicialize o TensorFlow Lite com o Google Play Services e o suporte à GPU:

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

Por fim, você pode inicializar o interpretador passando um `GpuDelegateFactory` pelo `InterpreterApi.Options`:

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

Observação: o delegado de GPU precisa ser criado no mesmo thread que o executa. Caso contrário, você pode ver o seguinte erro: `TfLiteGpuDelegate Invoke: GpuDelegate must run on the same thread where it was initialized` (GpuDelegate precisa ser executado no mesmo thread em que foi inicializado).

O delegado de GPU também pode ser usado com a vinculação de modelo de ML no Android Studio. Para mais informações, consulte [Gere interfaces de modelo usando metadados](../../inference_with_metadata/codegen#acceleration).

## Use a GPU com o TensorFlow Lite standalone {:#standalone}

Se o seu aplicativo segmentar dispositivos que não executam o Google Play, é possível empacotar o delegado de GPU com seu aplicativo e usá-lo com a versão standalone do TensorFlow Lite.

### Adicione as dependências do projeto

Para habilitar o acesso ao delegado de GPU, adicione `org.tensorflow:tensorflow-lite-gpu-delegate-plugin` ao arquivo `build.gradle` do seu aplicativo:

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

### Ative a aceleração de GPU

Em seguida, execute o TensorFlow Lite na GPU com `TfLiteDelegate`. No Java, você pode especificar o `GpuDelegate` pelo `Interpreter.Options`.

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

### Modelos quantizados {:#quantized-models}

As bibliotecas de delegados GPU do Android são compatíveis com os modelos quantizados por padrão. Você não precisa fazer nenhuma alteração no código para usar modelos quantizados com o delegado de GPU. A seção a seguir explica como desativar o suporte quantizado para testes ou fins experimentais.

#### Desative o suporte ao modelo quantizado

O código a seguir mostra como ***desativar*** o suporte a modelos quantizados.

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

Para mais informações sobre como executar modelos quantizados com a aceleração de GPU, confira a visão geral do [delegado de GPU](../../performance/gpu#quantized-models).

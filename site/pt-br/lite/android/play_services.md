# TensorFlow Lite no Google Play Services

O TensorFlow Lite está disponível no runtime do Google Play Services para todos os dispositivos Android com a versão atual do Play Services. Esse runtime permite que você execute modelos de aprendizado de máquina (ML) sem empacotar estaticamente bibliotecas do TensorFlow Lite com seu app.

Com a API Google Play Services, você pode reduzir o tamanho dos seus apps e melhorar a performance da última versão estável das bibliotecas. O TensorFlow Lite no Google Play Services é a maneira recomendada de usar o TensorFlow Lite no Android.

Você pode começar com o runtime do Play Services usando o [Guia rápido](../android/quickstart), que oferece um tutorial passo a passo para a implementação de um aplicativo de exemplo. Se você já usa o TensorFlow Lite standalone no seu aplicativo e quer utilizar o runtime do Play Services, consulte a seção [Migrando do TensorFlow Lite standalone](#migrating). Para mais informações sobre o Google Play Services, consulte o site [Google Play Services](https://developers.google.com/android/guides/overview).

<aside class="note"> <b>Termos:</b> ao acessar ou usar o TensorFlow Lite nas APIs do Google Play Services, você concorda com os <a href="#tos">Termos de Serviço</a>. Por favor, leia e entenda todos os termos e políticas aplicáveis antes de acessar as APIs.</aside>

## Usando o runtime do Play Services

O TensorFlow Lite no Google Play Services está disponível pela [API Task do TensorFlow Lite](../api_docs/java/org/tensorflow/lite/task/core/package-summary) e [API Interpreter do TensorFlow Lite](../api_docs/java/org/tensorflow/lite/InterpreterApi). A Biblioteca Task oferece interfaces de modelo prontas para uso e otimizadas para tarefas comuns de aprendizado de máquina usando dados visuais, de áudio e texto. A API Interpreter do TensorFlow Lite, fornecida pelo runtime do TensorFlow e pelas bibliotecas de suporte, oferece uma interface de uso mais geral para criar e executar modelos de ML.

As seguintes seções apresentam instruções sobre como implementar as APIs Biblioteca Task e Interpreter no Google Play Services. Embora seja possível para um aplicativo usar tanto as APIs Interpreter como as APIs Biblioteca Task, a maioria só usa um conjunto de APIs.

### Usando as APIs Biblioteca Task

A API Task do TensorFlow Lite envolve a API Interpreter e fornece uma interface de programação de alto nível para tarefas comuns de aprendizado de máquina que usam dados visuais, de áudio e texto. Utilize a API Task se o seu aplicativo exigir uma das [tarefas compatíveis](../inference_with_metadata/task_library/overview#supported_tasks).

#### Adicione as dependências do projeto

A dependência do seu projeto depende do seu caso de uso de aprendizado de máquina. As APIs Task contêm as seguintes bibliotecas:

- Biblioteca de visão: `org.tensorflow:tensorflow-lite-task-vision-play-services`
- Biblioteca de áudio: `org.tensorflow:tensorflow-lite-task-audio-play-services`
- Biblioteca de texto: `org.tensorflow:tensorflow-lite-task-text-play-services`

Adicione uma das dependências ao código do projeto do aplicativo para acessar a API Play Services para o TensorFlow Lite. Por exemplo, use o código a seguir para implementar uma tarefa de visão:

```
dependencies {
...
    implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
...
}
```

Atenção: o repositório maven da versão 0.4.2 da biblioteca de áudio das tarefas do TensorFlow Lite está incompleto. Em vez disso, use a versão 0.4.2.1 para essa biblioteca: `org.tensorflow:tensorflow-lite-task-audio-play-services:0.4.2.1`.

#### 2. Adicione a inicialização do TensorFlow Lite

Inicialize o componente do TensorFlow Lite da API Google Play Services *antes* de usar as APIs do TensorFlow Lite. O exemplo a seguir inicializa a biblioteca de visão:

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

Importante: verifique se a tarefa `TfLite.initialize` foi concluída antes de executar código que acesse as APIs do TensorFlow Lite.

Dica: os módulos do TensorFlow Lite são instalados ao mesmo tempo que seu aplicativo ou atualizados a partir da Play Store. Você pode conferir a disponibilidade dos módulos usando `ModuleInstallClient` das APIs do Google Play Services. Para mais informações sobre a verificação da disponibilidade dos módulos, consulte [Garantindo a disponibilidade da API com ModuleInstallClient](https://developers.google.com/android/guides/module-install-apis).

#### 3. Realize inferências

Depois de inicializar o componente do TensorFlow Lite, chame o método `detect()` para gerar inferências. O código exato no método `detect()` varia dependendo da biblioteca e do caso de uso. O código a seguir é para um caso de uso de detecção de objetos simples com a biblioteca `TfLiteVision`:

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

Dependendo do formato dos dados, talvez você também precise pré-processar e converter seus dados no método `detect()` antes de gerar inferências. Por exemplo, os dados de imagem para um detector de objetos exigem o seguinte:

```kotlin
val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
val results = objectDetector?.detect(tensorImage)
```

### Usando as APIs Interpreter

As APIs Interpreter oferecem mais controle e flexibilidade do que as APIs Biblioteca Task. Você deve usar as APIs Interpreter se a sua tarefa de aprendizado de máquina não for compatível com a Biblioteca Task ou exigir uma interface de uso mais geral para criar e executar modelos de ML.

#### Adicione as dependências do projeto

Adicione as seguintes dependências ao código do projeto do aplicativo para acessar a API Play Services para o TensorFlow Lite:

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

#### 2. Adicione a inicialização do TensorFlow Lite

Inicialize o componente do TensorFlow Lite da API Google Play Services *antes* de usar as APIs do TensorFlow Lite:

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

Observação: verifique se a tarefa `TfLite.initialize` foi concluída antes de executar código que acesse as APIs do TensorFlow Lite. Use o método `addOnSuccessListener()`, conforme mostrado na próxima seção.

#### 3. Crie um interpretador e defina a opção de runtime {:#step_3_interpreter}

Crie um interpretador usando `InterpreterApi.create()` e o configure para usar o runtime do Google Play Services, ao chamar `InterpreterApi.Options.setRuntime()`, conforme exibido no código de exemplo a seguir:

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

Você deve usar a implementação acima, porque evita bloquear o thread da interface do usuário Android. Se você precisar gerenciar a execução do thread mais detalhadamente, pode adicionar uma chamada `Tasks.await()` para a criação do interpretador:

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

Aviso: não chame `.await()` no thread da interface do usuário em primeiro plano, porque interrompe a exibição dos elementos da interface do usuário e cria uma má experiência do usuário.

#### 3. Realize inferências

Usando o objeto `interpreter` criado, chame o método `run()` para gerar uma inferência.

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

## Aceleração de hardware {:#hardware-acceleration}

O TensorFlow Lite permite acelerar o desempenho do seu modelo usando processadores de hardware especializados, como unidades de processamento gráfico (GPUs). Você pode aproveitar esses processadores especializados usando drivers de hardware chamados [*delegados*](https://www.tensorflow.org/lite/performance/delegates). É possível usar os seguintes delegados de aceleração de hardware com o TensorFlow Lite no Google Play Services:

- *[Delegado de GPU](https://www.tensorflow.org/lite/performance/gpu) (recomendado)*: esse delegado é fornecido pelo Google Play Services e é carregado dinamicamente, como as versões do Play Services da API Task e Interpreter.

- [*Delegado NNAPI*](https://www.tensorflow.org/lite/android/delegates/nnapi): esse delegado está disponível como uma dependência da biblioteca incluída no seu projeto de desenvolvimento Android e é empacotado com seu aplicativo.

Para mais informações sobre a aceleração de hardware com o TensorFlow Lite, confira a página [Delegados do TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Verificando a compatibilidade do dispositivo

Nem todos os dispositivos oferecem suporte à aceleração de hardware de GPU com o TFLite. Para mitigar erros e possíveis falhas, use o método `TfLiteGpu.isGpuDelegateAvailable` para verificar se um dispositivo é compatível com o delegado de GPU.

Use esse método para confirmar se um dispositivo oferece suporte à GPU e use a CPU ou o delegado NNAPI como substituto quando a GPU não for compatível.

```
useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
```

Depois de ter uma variável como `useGpuTask`, você pode usá-la para determinar se os dispositivos usam o delegado de GPU. Os exemplos a seguir mostram como isso pode ser feito com ambas as APIs Interpreter e Biblioteca Task.

**Com a API Task**

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

**Com a API Interpreter**

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

### GPU com APIs Biblioteca Task

Para usar o delegado de GPU com as APIs Task:

1. Atualize as dependências do projeto para usar o delegado de GPU a partir do Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. Inicialize o delegado de GPU com `setEnableGpuDelegateSupport`. Por exemplo, você pode inicializar o delegado de GPU para `TfLiteVision` com o seguinte:

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

3. Ative a opção de delegado de GPU com [`BaseOptions`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/core/BaseOptions.Builder):

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

4. Configure as opções usando `.setBaseOptions`. Por exemplo, você pode configurar a GPU em `ObjectDetector` com o código a seguir:

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

### GPU com APIs Interpreter

Para usar o delegado de GPU com as APIs Interpreter:

1. Atualize as dependências do projeto para usar o delegado de GPU a partir do Play Services:

    ```
    implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ```

2. Ative a opção de delegado de GPU na inicialização do TFlite:

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

3. Defina o delegado de GPU nas opções de interpretador para usar `DelegateFactory` ao chamar `addDelegateFactory()` em `InterpreterApi.Options()`:

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

## Migrando do TensorFlow Lite standalone {:#migrating}

Se você planeja migrar seu aplicativo do TensorFlow Lite standalone para a API Play Services, leia as orientações adicionais a seguir para atualizar o código do projeto do aplicativo:

1. Revise a seção [Limitações](#limitations) desta página para garantir a compatibilidade com o caso de uso.
2. Antes de atualizar seu código, faça verificações de desempenho e exatidão nos seus modelos, especialmente se estiver usando versões do TensorFlow Lite anteriores à versão 2.1. Assim, você tem uma linha de base para comparar com a nova implementação.
3. Se você migrou todo o seu código para usar a API Play Services para o TensorFlow Lite, remova as dependências *runtime library* existentes do TensorFlow Lite (entradas com <code>org.tensorflow:**tensorflow-lite**:*</code>) do seu arquivo build.gradle para reduzir o tamanho do aplicativo.
4. Identifique todas as ocorrências de criação de objetos `new Interpreter` no seu código e modifique isso para usar a chamada InterpreterApi.create(). Essa nova API é assíncrona, ou seja, na maioria dos casos não é uma substituição direta, e você precisa registrar um listener para quando a chamada é concluída. Consulte o fragmento de código na [Etapa 3](#step_3_interpreter).
5. Adicione `import org.tensorflow.lite.InterpreterApi;` e `import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;` a quaisquer arquivos de código-fonte usando as classes `org.tensorflow.lite.Interpreter` ou `org.tensorflow.lite.InterpreterApi`.
6. Se qualquer uma das chamadas resultantes a `InterpreterApi.create()` tiver só um argumento, anexe `new InterpreterApi.Options()` à lista de argumentos.
7. Anexe `.setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)` ao último argumento de qualquer chamada a `InterpreterApi.create()`.
8. Substitua todas as outras ocorrências da classe `org.tensorflow.lite.Interpreter` por `org.tensorflow.lite.InterpreterApi`.

Se você quiser usar o TensorFlow Lite standalone e a API Play Services juntos, precisa usar o TensorFlow Lite 2.9 (ou mais recente). O TensorFlow Lite 2.8 e as versões mais antigas não são compatíveis com a versão da API Play Services.

## Limitações

O TensorFlow Lite no Google Play Services tem as seguintes limitações:

- O suporte aos delegados de aceleração de hardware é limitado aos delegados listados na seção [Aceleração de hardware](#hardware-acceleration). Nenhum outro delegado de aceleração é compatível.
- O acesso ao TensorFlow Lite por [APIs nativas](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c) não é suportado. Somente as APIs Java do TensorFlow Lite estão disponíveis pelo Google Play Services.
- As APIs experimentais ou descontinuadas do TensorFlow Lite, incluindo operações personalizadas, não são compatíveis.

## Suporte e feedback {:#support}

Você pode fornecer feedback e receber suporte pelo Issue Tracker do TensorFlow. Informe problemas e solicitações de suporte usando o [modelo de issue](https://github.com/tensorflow/tensorflow/issues/new?title=TensorFlow+Lite+in+Play+Services+issue&template=tflite-in-play-services.md) para o TensorFlow Lite no Google Play Services.

## Termos de serviço {:#tos}

O uso do TensorFlow Lite nas APIs do Google Play Services está sujeito aos [Termos de Serviço das APIs do Google](https://developers.google.com/terms/).

### Privacidade e coleta de dados

Ao usar o TensorFlow Lite nas APIs do Google Play Services, o processamento dos dados de entrada, como imagens, vídeos e textos, ocorre totalmente no dispositivo, e o TensorFlow Lite nas APIs do Google Play Services não envia esses dados aos servidores do Google. Como resultado, você pode usar nossas APIs para processar dados que não devem sair do dispositivo.

O TensorFlow Lite nas APIs do Google Play Services pode entrar em contato com os servidores do Google eventualmente para receber, por exemplo, correções de bug, modelos atualizados e informações sobre a compatibilidade de aceleradores de hardware. O TensorFlow Lite nas APIs do Google Play Services também pode enviar métricas sobre o desempenho e a utilização de APIs no seu aplicativo para o Google. O Google usa esses dados de métricas para medir o desempenho, depurar, manter e melhorar as APIs e detectar uso indevido ou abuso, conforme detalhado na nossa [Política de Privacidade](https://policies.google.com/privacy).

**Você é responsável por informar aos usuários do seu aplicativo sobre o processamento que o Google faz dos dados de métricas do TensorFlow Lite nas APIs do Google Play Services conforme exigido pela legislação aplicável.**

Os dados que coletamos incluem os seguintes:

- Informações do dispositivo (como fabricante, modelo, versão de SO e build) e aceleradores de hardware de ML disponíveis (GPU e DSP). Usadas para diagnóstico e análise de uso.
- Identificador do dispositivo usado para diagnóstico e análise de uso.
- Informações do aplicativo (nome do pacote, versão do aplicativo). Usadas para diagnóstico e análise de uso.
- Configuração da API (por exemplo, os delegados em uso). Usada para diagnóstico e análise de uso.
- Tipo de evento (como criação de interpretadores, inferência). Usado para diagnóstico e análise de uso.
- Códigos de erro. Usados para diagnóstico.
- Métricas de desempenho. Usadas para diagnóstico.

## Próximos passos

Para mais informações sobre como implementar o aprendizado de máquina no seu aplicativo para dispositivos móveis com o TensorFlow Lite, confira o [Guia para desenvolvedores do TensorFlow Lite](https://www.tensorflow.org/lite/guide). Encontre modelos adicionais do TensorFlow Lite para classificação de imagens, detecção de objetos e outros aplicativos no [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite).

# Reconhecimento de sons e palavras para Android

Este tutorial mostra como usar o TensorFlow Lite com modelos de aprendizado de máquina pré-criados para reconhecer sons e palavras faladas em um aplicativo Android. Os modelos de classificação de áudio como os exibidos neste tutorial podem ser usados para detectar atividades, identificar ações ou reconhecer comandos de voz.

![Demonstração animada de reconhecimento de áudio](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/audio_classification.gif){: .attempt-right} Este tutorial mostra como baixar o código de exemplo e carregar o projeto no [Android Studio](https://developer.android.com/studio/), bem como explica partes essenciais do código de exemplo para você começar a adicionar essa funcionalidade ao seu próprio aplicativo. O aplicativo de exemplo usa a [Biblioteca Task para áudio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier), que lida com a maior parte da gravação e do pré-processamento dos dados de áudio. Para mais informações sobre como o áudio é pré-processado para o uso com modelos de aprendizado de máquina, consulte [Preparação e ampliação de dados de áudio](https://www.tensorflow.org/io/tutorials/audio).

## Classificação de áudio com o aprendizado de máquina

O modelo de aprendizado de máquina neste tutorial reconhece sons ou palavras de amostras de áudio gravadas com um microfone em um dispositivo Android. O aplicativo de exemplo neste tutorial permite que você alterne entre o [YAMNet/classificador](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1), um modelo que reconhece sons e um modelo que reconhece determinadas palavras faladas, que foi [treinado](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) usando a ferramenta [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) do TensorFlow Lite. Os modelos executam previsões em clipes de áudio que contêm 15.600 amostras individuais por clipe e têm 1 segundo de duração.

## Configure e execute o exemplo

Para a primeira parte deste tutorial, baixe o código de exemplo do GitHub e o execute usando o Android Studio. As seções a seguir deste documento exploram as seções relevantes do exemplo, para você aplicá-las aos seus próprios aplicativos Android.

### Requisitos do sistema

- [Android Studio](https://developer.android.com/studio/index.html), versão 2021.1.1 (Bumblebee) ou mais recente.
- SDK do Android, versão 31 ou mais recente.
- Dispositivo Android com uma versão mínima de SO do SDK 24 (Android 7.0 - Nougat) com o modo desenvolvedor ativado.

### Obtenha o código de exemplo

Crie uma cópia local do código de exemplo. Você usará esse código para criar um projeto no Android Studio e executar o aplicativo de exemplo.

Para clonar e configurar o código de exemplo:

1. Clone o repositório git
    <pre class="devsite-click-to-copy">
        git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure sua instância git para usar o sparse checkout e ter somente os arquivos para o aplicativo de exemplo:
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### Importe e execute o projeto

Crie  um projeto a partir do código de exemplo baixado, compile e depois execute esse projeto.

Para importar e compilar o projeto do código de exemplo:

1. Inicie o [Android Studio](https://developer.android.com/studio).
2. No Android Studio, selecione **File &gt; New &gt; Import Project** (Arquivo &gt; Novo &gt; Importar projeto).
3. Acesse o diretório do código de exemplo com o arquivo `build.gradle` (`.../examples/lite/examples/audio_classification/android/build.gradle`) e selecione esse diretório.

Se você selecionar o diretório correto, o Android Studio cria e compila um novo projeto. Esse processo pode levar alguns minutos, dependendo da velocidade do seu computador e se você usou o Android Studio para outros projetos. Quando o build for concluído, o Android Studio exibirá uma mensagem `BUILD SUCCESSFUL` no painel de status **Build Output**.

Para executar o projeto:

1. No Android Studio, execute o projeto ao selecionar **Run &gt; Run 'app'** (Executar &gt; Executar 'app').
2. Selecione um dispositivo Android conectado com um microfone para testar o aplicativo.

Observação: se você usar um emulador para executar o aplicativo, [ative a entrada de áudio](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio) na máquina host.

As próximas seções mostram as modificações necessárias no projeto existente para adicionar essa funcionalidade ao seu próprio aplicativo, usando esse aplicativo de exemplo como um ponto de referência.

## Adicione as dependências do projeto

No seu próprio aplicativo, você precisa adicionar as dependências do projeto para executar os modelos de aprendizado de máquina do TensorFlow e acessar funções utilitárias que convertem formatos de dados padrão, como áudio, em um formato de dados de tensor que pode ser processado pelo modelo que você está usando.

O aplicativo de exemplo usa as seguintes bibliotecas do TensorFlow Lite:

- [API Biblioteca Task do TensorFlow para áudio:](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/package-summary) fornece as classes de entrada dos dados de áudio necessárias, a execução do modelo de aprendizado de máquina e os resultados gerados com o processamento do modelo.

As instruções a seguir mostram como adicionar as dependências necessárias ao seu próprio projeto de aplicativo Android.

Para adicionar dependências de módulo:

1. No módulo que usa o TensorFlow Lite, atualize o arquivo `build.gradle` para que inclua as seguintes dependências. No código de exemplo, esse arquivo está localizado aqui: `.../examples/lite/examples/audio_classification/android/build.gradle`
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
2. No Android Studio, sincronize as dependências do projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

## Inicialize o modelo de ML

No seu aplicativo Android, você precisa inicializar o modelo de aprendizado de máquina do TensorFlow com parâmetros antes de realizar previsões com o modelo. Esses parâmetros de inicialização dependem do modelo e podem incluir configurações como limites de exatidão mínima padrão para previsões e rótulos para palavras ou sons que o modelo consegue reconhecer.

Um modelo do TensorFlow Lite inclui um arquivo `*.tflite` contendo o modelo. O arquivo do modelo contém a lógica de previsão e geralmente inclui [metadados](../../models/convert/metadata) sobre como interpretar resultados de previsão, como nomes de classes de previsão. Os arquivos do modelo devem ser armazenados no diretório `src/main/assets` do seu projeto de desenvolvimento, como no código de exemplo:

- `<project>/src/main/assets/yamnet.tflite`

Para conveniência e legibilidade do código, o exemplo declara um objeto complementar que define as configurações para o modelo.

Para inicializar o modelo no seu aplicativo:

1. Crie um objeto complementar para definir as configurações para o modelo:
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
2. Crie as configurações para o modelo ao construir um objeto `AudioClassifier.AudioClassifierOptions`:
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
3. Use as configurações desse objeto para construir um objeto [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) do TensorFlow Lite que contém o modelo:
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### Ative a aceleração de hardware

Ao inicializar um modelo do TensorFlow Lite no seu aplicativo, você deve considerar usar os recursos de aceleração de hardware para acelerar os cálculos de previsão do modelo. Os [delegados](https://www.tensorflow.org/lite/performance/delegates) do TensorFlow Lite são módulos de software que aceleram a execução dos modelos de aprendizado de máquina usando hardware de processamento especializado em um dispositivo móvel, como unidades de processamento gráfico (GPUs) ou unidades de processamento de tensor (TPUs). O código de exemplo usa o delegado NNAPI para lidar com a aceleração de hardware da execução do modelo:

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

Usar os delegados para executar modelos do TensorFlow é recomendável, mas não obrigatório. Para mais informações sobre como usar os delegados com o TensorFlow Lite, consulte [Delegados do TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

## Prepare os dados para o modelo

No seu aplicativo Android, seu código fornece dados ao modelo para interpretação ao transformar dados existentes, como clipes de áudio, em um formato de dados de [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pode ser processado pelo modelo. Os dados em um Tensor passados a um modelo precisam ter dimensões específicas, ou um formato, que correspondam ao formato dos dados usados para treinar o modelo.

O [modelo YAMNet/classificador](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) e os modelos de  [comandos de voz](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) personalizados usados nesse código de exemplo aceitam objetos de dados de Tensor que representam clipes de áudio de canal único, ou mono, gravados a 16 kHz em clipes de 0,975 segundo (15.600 amostras). Ao realizar previsões em novos dados de áudio, seu aplicativo precisa transformar esses dados de áudio em objetos de dados de Tensor desse tamanho e formato. A [API para áudio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) da Biblioteca Task do TensorFlow Lite faz a transformação de dados para você.

Na classe `AudioClassificationHelper` do código de exemplo, o aplicativo grava o áudio em tempo real dos microfones do dispositivo usando um objeto [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) Android. O código usa o [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) para criar e configurar esse objeto para gravar áudio em uma taxa de amostragem apropriada para o modelo. O código também usa o AudioClassifier para criar um objeto [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) que armazena os dados de áudio transformados. Em seguida, o objeto TensorAudio é passado a um modelo para análise.

Para fornecer dados de áudio ao modelo de ML:

- Use o objeto `AudioClassifier` para criar um objeto `TensorAudio` e um objeto `AudioRecord`:
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

Observação: seu aplicativo precisa solicitar permissão para gravar áudio usando o microfone de um dispositivo Android. Veja a classe `fragments/PermissionsFragment` no projeto, por exemplo. Para mais informações sobre como solicitar permissões, confira [Permissões no Android](https://developer.android.com/guide/topics/permissions/overview).

## Realize previsões

No seu aplicativo Android, depois de conectar um objeto [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) e [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) a um objeto AudioClassifier,  você pode executar o modelo com esses dados para gerar uma previsão ou *inferência*. O código de exemplo deste tutorial realiza previsões com clipes de um stream de entrada de áudio gravado em tempo real a uma taxa específica.

A execução do modelo consome recursos significativos, então é importante realizar previsões de modelo de ML em um thread em segundo plano separado. O aplicativo de exemplo usa um objeto `[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)` para isolar o processamento do modelo de outras funções do aplicativo.

Os modelos de classificação de áudio que reconhecem sons com um início e fim claro, como palavras, podem produzir previsões mais exatas em um stream de áudio recebido ao analisar clipes de áudio sobrepostos. Essa abordagem ajuda o modelo a evitar perder previsões para palavras cortadas ao final de um clipe. No aplicativo de exemplo, sempre que você realiza uma previsão, o código pega o último clipe de 0,975 segundo do buffer de gravação de áudio e o analisa. Você pode fazer o modelo analisar clipes de áudio sobrepostos ao definir o valor `interval` do pool de thread em execução da análise do modelo para um comprimento mais curto do que a duração dos clipes analisados. Por exemplo, se o seu modelo analisa clipes de 1 segundo e você define o intervalo como 500 milissegundos, o modelo sempre analisará a última metade do clipe anterior e 500 milissegundos dos novos dados de áudio, criando uma sobreposição de análise de clipes de 50%.

Para começar a realizar previsões nos dados de áudio:

1. Use o método `AudioClassificationHelper.startAudioClassification()` para iniciar a gravação de áudio para o modelo:
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
2. Defina com que frequência o modelo gera uma inferência a partir dos clipes de áudio ao configurar um `interval` de taxa fixa no objeto `ScheduledThreadPoolExecutor`:
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
3. O objeto `classifyRunnable` no código acima executa o método `AudioClassificationHelper.classifyAudio()`, que carrega os dados de áudio mais recentes do gravador e realiza uma previsão:
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

Cuidado: não realize as previsões do modelo de ML no thread de execução principal do aplicativo. Ao fazer isso, a interface do usuário do seu aplicativo pode ficar lenta ou sem resposta.

### Interrompa o processamento da previsão

Garanta que o código do seu aplicativo interrompa a classificação de áudio quando o Fragmento ou a Atividade de processamento de áudio perder o foco. A execução contínua de um modelo de aprendizado de máquina afeta significativamente a duração da bateria de um dispositivo Android. Use o método `onPause()` na atividade ou no fragmento Android associado à classificação de áudio para interromper o processamento da previsão e a gravação de áudio.

Para interromper a classificação e gravação de áudio:

- Use o método `AudioClassificationHelper.stopAudioClassification()` para interromper a execução do modelo e a gravação, conforme mostrado abaixo na classe `AudioFragment`:
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## Processe a saída do modelo

No seu aplicativo Android, depois de processar um clipe de áudio, o modelo produz uma lista de previsões com que o código do seu aplicativo precisa lidar ao executar lógica de negócios adicional, exibindo resultados ao usuário ou realizando outras ações. O resultado de qualquer modelo do TensorFlow Lite varia em termos de número de previsões produzidas (uma ou mais) e informações descritivas de cada previsão. No caso dos modelos no aplicativo de exemplo, as previsões são uma lista de sons ou palavras reconhecidas. O objeto de opções AudioClassifier usado no código de exemplo permite definir o número máximo de previsões com o método `setMaxResults()`, conforme mostrado na seção [Inicialize o modelo de ML](#Initialize_the_ML_model).

Para obter os resultados de previsão do modelo:

1. Obtenha os resultados do método `classify()` <br> do objeto [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) e os passe ao objeto listener (código de referência):
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
2. Use a função onResult() do listener para processar a saída ao executar a lógica de negócios ou exibir os resultados ao usuário:
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

O modelo usado nesse exemplo gera uma lista de previsões com um rótulo para o som ou a palavra classificada e uma pontuação entre 0 e 1 como um Float representando a confiança da previsão, sendo 1 a pontuação mais alta. Em geral, as previsões com uma pontuação abaixo de 50% (0,5) são consideradas inconclusivas. No entanto, cabe a você decidir como lida com os resultados de previsão de valores baixos e as necessidades do seu aplicativo.

Depois que o modelo retornar um conjunto de resultados de previsão, seu aplicativo pode agir em relação a essas previsões ao apresentar o resultado ao seu usuário ou executar lógica adicional. No caso do código de exemplo, o aplicativo lista os sons ou as palavras identificadas na interface do usuário do aplicativo.

## Próximos passos

Você pode encontrar mais modelos do TensorFlow Lite para processamento de áudio no [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt) e na página [Guia de modelos pré-treinados](https://www.tensorflow.org/lite/models/trained). Para mais informações sobre como implementar o aprendizado de máquina no seu aplicativo móvel com o TensorFlow Lite, confira o [Guia para desenvolvedores do TensorFlow Lite](https://www.tensorflow.org/lite/guide).

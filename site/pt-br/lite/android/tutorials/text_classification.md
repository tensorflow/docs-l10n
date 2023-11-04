# Classificação de texto com o Android

Este tutorial mostra como criar um aplicativo Android usando o TensorFlow Lite para classificar texto de linguagem natural. Esse aplicativo foi feito para um dispositivo Android físico, mas também pode ser executado em um emulador de dispositivo.

O [aplicativo de exemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) usa o TensorFlow Lite para classificar texto como positivo ou negativo, usando a [Biblioteca Task para linguagem natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks) para permitir a execução de modelos de aprendizado de máquina de classificação de texto.

Se você estiver atualizando um projeto existente, pode usar o aplicativo de exemplo como referência ou modelo. Para instruções sobre como adicionar a classificação de texto a um aplicativo existente, consulte [Atualizando e modificando seu aplicativo](#modify_applications).

## Visão geral da classificação de texto

A *classificação de texto* é a tarefa de aprendizado de máquina que atribui um conjunto de categorias predefinidas a um texto aberto. Um modelo de classificação de texto é treinado com um corpus de texto de linguagem natural, em que palavras ou frases são classificadas manualmente.

O modelo treinado recebe texto como entrada e tenta categorizar esse texto de acordo com o conjunto de classes conhecidas que foi treinado para classificar. Por exemplo, o modelo neste exemplo aceita um fragmento de texto e determina se o sentimento do texto é positivo ou negativo. Para cada fragmento, o modelo de classificação gera uma pontuação que indica a confiança do texto ser classificado corretamente como positivo ou negativo.

Para mais informações sobre como os modelos neste tutorial são gerados, consulte o tutorial [Classificação de texto com o Model Maker do TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

## Modelos e dataset

Este tutorial usa modelos que foram treinados usando o dataset [SST-2](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank, ou Treebank de Sentimentos da Stanford). O SST-2 contém 67.349 avaliações de filmes para treinamento e 872 para teste, sendo cada uma categorizada como positiva ou negativa. Os modelos usados neste aplicativo foram treinados usando a ferramenta [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) do TensorFlow Lite.

O aplicativo de exemplo usa os seguintes modelos pré-treinados:

- [Average Word Vector](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) (`NLClassifier`): o `NLClassifier` da Biblioteca Task classifica o texto de entrada em diferentes categorias e consegue lidar com a maioria dos modelos de classificação de texto.

- [MobileBERT](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier) (`BertNLClassifier`): o `BertNLClassifier` da Biblioteca Task é semelhante ao NLClassifier, mas adaptado a casos que exigem tokenizações por palavra e frase fora do grafo.

## Configure e execute o aplicativo de exemplo

Para configurar o aplicativo de classificação de texto, baixe o aplicativo de exemplo do [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) e o execute usando o [Android Studio](https://developer.android.com/studio/).

### Requisitos do sistema

- **[Android Studio](https://developer.android.com/studio/index.html)**, versão 2021.1.1 (Bumblebee) ou mais recente.
- SDK do Android, versão 31 ou mais recente.
- Dispositivo Android com uma versão mínima de SO do SDK 21 (Android 7.0 - Nougat) com o [modo desenvolvedor](https://developer.android.com/studio/debug/dev-options) ativado.

### Obtenha o código de exemplo

Crie uma cópia local do código de exemplo. Você usará esse código para criar um projeto no Android Studio e executar o aplicativo de exemplo.

Para clonar e configurar o código de exemplo:

1. Clone o repositório git
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure sua instância git para usar o sparse checkout e ter somente os arquivos para o aplicativo de exemplo de classificação de texto:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/text_classification/android
        </pre>

### Importe e execute o projeto

Crie um projeto a partir do código de exemplo baixado, compile e depois execute esse projeto.

Para importar e compilar o projeto do código de exemplo:

1. Inicie o [Android Studio](https://developer.android.com/studio).
2. No Android Studio, selecione **File &gt; New &gt; Import Project** (Arquivo &gt; Novo &gt; Importar projeto).
3. Acesse o diretório do código de exemplo com o arquivo build.gradle (`.../examples/lite/examples/text_classification/android/build.gradle`) e selecione esse diretório.
4. Se o Android Studio solicitar o Gradle Sync, selecione OK.
5. Garanta que o dispositivo Android esteja conectado ao seu computador e que o modo desenvolvedor esteja ativado. Clique na seta `Run` verde.

Se você selecionar o diretório correto, o Android Studio cria e compila um novo projeto. Esse processo pode levar alguns minutos, dependendo da velocidade do seu computador e se você usou o Android Studio para outros projetos. Quando o build for concluído, o Android Studio exibirá uma mensagem `BUILD SUCCESSFUL` no painel de status **Build Output**.

Para executar o projeto:

1. No Android Studio, execute o projeto ao selecionar **Run &gt; Run…**.
2. Selecione um dispositivo Android conectado (ou emulador) para testar o aplicativo.

### Usando o aplicativo

![Aplicativo de exemplo de classificação de texto no Android](../../../images/lite/android/text-classification-screenshot.png){: .attempt-right width="250px"}

Depois de executar o projeto no Android Studio, o aplicativo abre automaticamente no dispositivo conectado ou emulador de dispositivo.

Para usar o classificador de texto:

1. Insira um fragmento de texto na caixa de texto.
2. No menu suspenso **Delegate**, selecione `CPU` ou `NNAPI`.
3. Especifique um modelo ao escolher `AverageWordVec` ou `MobileBERT`.
4. Selecione **Classify** (Classificar).

O aplicativo gera uma pontuação *positiva* e uma *negativa*. Essas duas pontuações juntas somam 1 e medem a probabilidade de o sentimento do texto inserido ser positivo ou negativo. Um número mais alto indica um maior nível de confiança.

Agora você tem um aplicativo de classificação de texto. Use as seguintes seções para entender melhor como o aplicativo de exemplo funciona e como implementar os recursos de classificação de texto nos seus aplicativos em produção:

- [Como o aplicativo funciona](#how_it_works): um tutorial da estrutura e dos principais arquivos do aplicativo de exemplo.

- [Modifique seu aplicativo](#modify_applications): instruções sobre como adicionar a classificação de texto a um aplicativo existente.

## Como o aplicativo de exemplo funciona {:#how_it_works}

O aplicativo usa a [Biblioteca Task para linguagem natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks) para implementar modelos de classificação de texto. Os dois modelos, Average Word Vector e MobileBERT, foram treinados usando o [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) do TensorFlow Lite. O aplicativo é executado na CPU por padrão, com a opção da aceleração de hardware usando o delegado NNAPI.

Os seguintes arquivos e diretórios contêm o código fundamental para esse aplicativo de classificação de texto:

- [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt): inicializa o classificador de texto e lida com a seleção de delegados e modelo.
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/MainActivity.kt): implementa o aplicativo, inclusive chamando `TextClassificationHelper` e `ResultsAdapter`.
- [ResultsAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/ResultsAdapter.kt): processa e formata os resultados.

## Modifique seu aplicativo {:#modify_applications}

As seguintes seções explicam os principais passos para modificar seu próprio aplicativo Android e executar o modelo mostrado no aplicativo de exemplo. Essas instruções usam o aplicativo de exemplo anterior como ponto de referência. As mudanças específicas necessárias no seu próprio aplicativo podem diferir do aplicativo de exemplo.

### Abra ou crie um projeto Android

Você precisa de um projeto de desenvolvimento Android no Android Studio para acompanhar o resto destas instruções. Siga as instruções abaixo para abrir um projeto existente ou criar um novo.

Para abrir um projeto de desenvolvimento Android existente:

- No Android Studio, selecione *File &gt; Open* (Arquivo &gt; Abrir) e escolha um projeto existente.

Para criar um projeto de desenvolvimento Android básico:

- Siga as instruções no Android Studio para [Criar um projeto básico](https://developer.android.com/studio/projects/create-project).

Para mais informações sobre como usar o Android Studio, consulte a [documentação do Android Studio](https://developer.android.com/studio/intro).

### Adicione as dependências do projeto

No seu próprio aplicativo, você precisa adicionar as dependências do projeto para executar os modelos de aprendizado de máquina do TensorFlow e acessar funções utilitárias que convertem dados como strings em um formato de dados de tensor que pode ser processado pelo modelo que você está usando.

As instruções a seguir explicam como adicionar as dependências de projeto e módulo necessárias ao seu próprio projeto de aplicativo Android.

Para adicionar dependências de módulo:

1. No módulo que usa o TensorFlow Lite, atualize o arquivo `build.gradle` para que inclua as seguintes dependências.

    No aplicativo de exemplo, as dependências estão localizadas em [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/build.gradle):

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.0'
    }
    ```

    O projeto precisa incluir a Biblioteca Task para texto (`tensorflow-lite-task-text`).

    Se você quiser modificar esse aplicativo para que seja executado em uma unidade de processamento gráfico (GPU), a biblioteca da GPU (`tensorflow-lite-gpu-delegate-plugin`) fornece a infraestrutura para executar o aplicativo na GPU e o Delegado (`tensorflow-lite-gpu`) oferece a lista de compatibilidade. A execução desse aplicativo na GPU está fora do escopo deste tutorial.

2. No Android Studio, sincronize as dependências do projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

### Inicialize os modelos de ML {:#initialize_models}

No seu aplicativo Android, você precisa inicializar o modelo de aprendizado de máquina do TensorFlow Lite com parâmetros antes de realizar previsões com o modelo.

Um modelo do TensorFlow Lite é armazenado como um arquivo `*.tflite`. O arquivo do modelo contém a lógica de previsão e geralmente inclui [metadados](../../models/convert/metadata) sobre como interpretar resultados de previsão, como nomes de classes de previsão. Geralmente, os arquivos do modelo são armazenados no diretório `src/main/assets` do seu projeto de desenvolvimento, como no código de exemplo:

- `<project>/src/main/assets/mobilebert.tflite`
- `<project>/src/main/assets/wordvec.tflite`

Observação: esse aplicativo de exemplo usa um arquivo `[download_model.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/download_model.gradle)` para baixar os modelos [Average Word Vector](https://www.tensorflow.org/lite/inference_with_metadata/task_library/nl_classifier) e [MobileBERT](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier) no tempo de build. Essa abordagem não é necessária nem recomendada para um aplicativo em produção.

Para conveniência e legibilidade do código, o exemplo declara um objeto complementar que define as configurações para o modelo.

Para inicializar o modelo no seu aplicativo:

1. Crie um objeto complementar para definir as configurações para o modelo. No aplicativo de exemplo, esse objeto está localizado em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    companion object {
      const val DELEGATE_CPU = 0
      const val DELEGATE_NNAPI = 1
      const val WORD_VEC = "wordvec.tflite"
      const val MOBILEBERT = "mobilebert.tflite"
    }
    ```

2. Crie as configurações para o modelo ao criar um objeto classificador e construir um objeto do TensorFlow Lite com `BertNLClassifier` ou `NLClassifier`.

    No aplicativo de exemplo, isso está localizado na função `initClassifier` em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    fun initClassifier() {
      ...
      if( currentModel == MOBILEBERT ) {
        ...
        bertClassifier = BertNLClassifier.createFromFileAndOptions(
          context,
          MOBILEBERT,
          options)
      } else if (currentModel == WORD_VEC) {
          ...
          nlClassifier = NLClassifier.createFromFileAndOptions(
            context,
            WORD_VEC,
            options)
      }
    }
    ```

    Observação: a maioria dos aplicativos em produção que usam a classificação de texto utilizarão `BertNLClassifier` ou `NLClassifier`, e não ambos.

### Ative a aceleração de hardware (opcional) {:#hardware_acceleration}

Ao inicializar um modelo do TensorFlow Lite no seu aplicativo, você deve considerar usar os recursos de aceleração de hardware para acelerar os cálculos de previsão do modelo. Os [delegados](https://www.tensorflow.org/lite/performance/delegates) do TensorFlow Lite são módulos de software que aceleram a execução dos modelos de aprendizado de máquina usando hardware de processamento especializado em um dispositivo móvel, como unidades de processamento gráfico (GPUs) ou unidades de processamento de tensor (TPUs).

Para ativar a aceleração de hardware no seu aplicativo:

1. Crie uma variável para definir o delegado que o aplicativo usará. No aplicativo de exemplo, essa variável está localizada no início em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    var currentDelegate: Int = 0
    ```

2. Crie um seletor de delegado. No aplicativo de exemplo, esse seletor está localizado na função `initClassifier` em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    val baseOptionsBuilder = BaseOptions.builder()
    when (currentDelegate) {
       DELEGATE_CPU -> {
           // Default
       }
       DELEGATE_NNAPI -> {
           baseOptionsBuilder.useNnapi()
       }
    }
    ```

Observação: é possível modificar esse aplicativo para usar um delegado de GPU, mas isso exige que o classificador seja criado no mesmo thread que está usando esse classificador. Isso está fora do escopo deste tutorial.

Usar os delegados para executar modelos do TensorFlow é recomendável, mas não obrigatório. Para mais informações sobre como usar os delegados com o TensorFlow Lite, consulte [Delegados do TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Prepare os dados para o modelo

No seu aplicativo Android, seu código fornece dados ao modelo para interpretação ao transformar dados existentes, como texto bruto, em um formato de dados de [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pode ser processado pelo modelo. Os dados em um Tensor passados a um modelo precisam ter dimensões específicas, ou um formato, que correspondam ao formato dos dados usados para treinar o modelo.

Esse aplicativo de classificação de texto aceita uma [string](https://developer.android.com/reference/java/lang/String.html) como entrada, e os modelos são treinados exclusivamente com um corpus de língua inglesa. Caracteres especiais e palavras que não estão em inglês são ignoradas durante a inferência.

Para fornecer dados de texto ao modelo:

1. Verifique se a função `initClassifier` contém o código para o delegado e os modelos, conforme explicado nas seções [Inicialize os modelos de ML](#initialize_models) e [Ative a aceleração de hardware](#hardware_acceleration).

2. Use o bloco `init` para chamar a função `initClassifier`. No aplicativo de exemplo, o `init` está localizado em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    init {
      initClassifier()
    }
    ```

### Realize previsões

No seu aplicativo Android, depois de inicializar um objeto [BertNLClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier) ou [NLClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier), você pode começar a inserir o texto para o modelo categorizar como "positivo" ou "negativo".

Para realizar previsões:

1. Crie uma função `classify`, que usa o classificador selecionado (`currentModel`) e mede o tempo que leva para classificar o texto de entrada (`inferenceTime`). No aplicativo de exemplo, a função `classify` está localizada em [TextClassificationHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/TextClassificationHelper.kt):

    ```
    fun classify(text: String) {
      executor = ScheduledThreadPoolExecutor(1)

      executor.execute {
        val results: List<Category>
        // inferenceTime is the amount of time, in milliseconds, that it takes to
        // classify the input text.
        var inferenceTime = SystemClock.uptimeMillis()

        // Use the appropriate classifier based on the selected model
        if(currentModel == MOBILEBERT) {
          results = bertClassifier.classify(text)
        } else {
          results = nlClassifier.classify(text)
        }

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        listener.onResult(results, inferenceTime)
      }
    }
    ```

2. Passe os resultados de `classify` ao objeto listener.

    ```
    fun classify(text: String) {
      ...
      listener.onResult(results, inferenceTime)
    }
    ```

### Processe a saída do modelo

Depois de inserir uma linha de texto, o modelo produz uma pontuação de previsão, expressa como um Float, entre 0 e 1 para as categorias "positivo" e "negativo".

Para obter os resultados de previsão do modelo:

1. Crie uma função `onResult` para o objeto listener processar a saída. No aplicativo de exemplo, o objeto listener está localizado em [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/app/src/main/java/org/tensorflow/lite/examples/textclassification/MainActivity.kt)

    ```
    private val listener = object : TextClassificationHelper.TextResultsListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        runOnUiThread {
          activityMainBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)

          adapter.resultsList = results.sortedByDescending {
            it.score
          }

          adapter.notifyDataSetChanged()
        }
      }
      ...
    }
    ```

2. Adicione uma função `onError` ao objeto listener para lidar com os erros:

    ```
      private val listener = object : TextClassificationHelper.TextResultsListener {
        ...
        override fun onError(error: String) {
          Toast.makeText(this@MainActivity, error, Toast.LENGTH_SHORT).show()
        }
      }
    ```

Depois que o modelo retornar um conjunto de resultados de previsão, seu aplicativo pode agir em relação a essas previsões ao apresentar o resultado ao seu usuário ou executar lógica adicional. O aplicativo de exemplo lista as pontuações de previsão na interface do usuário.

## Próximos passos

- Treine e implemente os modelos do zero com o tutorial [Classificação de texto com o Model Maker do TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).
- Explore mais [ferramentas de processamento de texto para o TensorFlow](https://www.tensorflow.org/text).
- Baixe outros modelos BERT no [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1).
- Descubra os usos do TensorFlow Lite nos [exemplos](../../examples).
- Saiba mais sobre como usar modelos de aprendizado de máquina com o TensorFlow Lite na seção [Modelos](../../models).
- Saiba mais sobre como implementar o aprendizado de máquina no seu aplicativo para dispositivos móveis no [Guia para desenvolvedores do TensorFlow Lite](../../guide).

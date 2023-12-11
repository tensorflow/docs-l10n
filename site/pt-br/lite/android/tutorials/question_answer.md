# Respondendo a perguntas com o Android

![Aplicativo de exemplo de resposta a perguntas no Android](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

Este tutorial mostra como criar um aplicativo Android usando o TensorFlow Lite para fornecer respostas a perguntas estruturadas em texto de linguagem natural. O [aplicativo de exemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) usa a API *BERT de resposta a perguntas* ([`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer)), na [Biblioteca Task para linguagem natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks), para promover modelos de aprendizado de máquina de resposta a perguntas.

Se você estiver atualizando um projeto existente, pode usar o aplicativo de exemplo como referência ou modelo. Para instruções sobre como adicionar a resposta a perguntas a um aplicativo existente, consulte [Atualizando e modificando seu aplicativo](#modify_applications).

## Visão geral da resposta a perguntas

A *resposta a perguntas* é a tarefa de aprendizado de máquina que responde perguntas em linguagem natural. Um modelo desse tipo treinado recebe uma passagem de texto e uma pergunta como entrada e tenta responder com base na interpretação das informações nessa passagem.

Um modelo de resposta a perguntas é treinado com um dataset correspondente, que consiste em um dataset de compreensão de leitura, além de pares de pergunta-resposta baseados em diferentes segmentos de texto.

Para mais informações sobre como os modelos neste tutorial são gerados, consulte o tutorial [BERT de resposta a perguntas com o Model Maker do TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## Modelos e dataset

O aplicativo de exemplo usa o modelo Mobile BERT Q&amp;A ([`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1)), que é uma versão mais leve e rápida do [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers). Para mais informações sobre `mobilebert`, consulte a pesquisa [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) (MobileBERT: um BERT agnóstico de tarefa e compacto para dispositivos de recursos limitados).

O modelo `mobilebert` foi treinado usando o Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)), ou Dataset de respostas a perguntas da Stanford: um dataset de compreensão de leitura que consiste em artigos da Wikipedia e um conjunto de pares pergunta-resposta para cada artigo.

## Configure e execute o aplicativo de exemplo

Para configurar o aplicativo de resposta a perguntas, baixe o aplicativo de exemplo do [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android) e o execute usando o [Android Studio](https://developer.android.com/studio/).

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
2. Opcionalmente, configure sua instância git para usar o sparse checkout e ter somente os arquivos para o aplicativo de exemplo de resposta a perguntas:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/bert_qa/android
        </pre>

### Importe e execute o projeto

Crie um projeto a partir do código de exemplo baixado, compile e depois execute esse projeto.

Para importar e compilar o projeto do código de exemplo:

1. Inicie o [Android Studio](https://developer.android.com/studio).
2. No Android Studio, selecione **File &gt; New &gt; Import Project** (Arquivo &gt; Novo &gt; Importar projeto).
3. Acesse o diretório do código de exemplo com o arquivo build.gradle (`.../examples/lite/examples/bert_qa/android/build.gradle`) e selecione esse diretório.
4. Se o Android Studio solicitar o Gradle Sync, selecione OK.
5. Garanta que o dispositivo Android esteja conectado ao seu computador e que o modo desenvolvedor esteja ativado. Clique na seta `Run` verde.

Se você selecionar o diretório correto, o Android Studio cria e compila um novo projeto. Esse processo pode levar alguns minutos, dependendo da velocidade do seu computador e se você usou o Android Studio para outros projetos. Quando o build for concluído, o Android Studio exibirá uma mensagem `BUILD SUCCESSFUL` no painel de status **Build Output**.

Para executar o projeto:

1. No Android Studio, execute o projeto ao selecionar **Run &gt; Run…**.
2. Selecione um dispositivo Android conectado (ou emulador) para testar o aplicativo.

### Usando o aplicativo

Depois de executar o projeto no Android Studio, o aplicativo abre automaticamente no dispositivo conectado ou emulador de dispositivo.

Para usar o aplicativo de exemplo de resposta a perguntas:

1. Escolha um tópico na lista de assuntos.
2. Escolha uma pergunta sugerida ou insira a sua na caixa de texto.
3. Ative a seta laranja para executar o modelo.

O aplicativo tenta identificar a resposta à pergunta a partir da passagem de texto. Se o modelo detectar uma resposta na passagem, o aplicativo destaca o trecho de texto relevante para o usuário.

Agora você tem um aplicativo de resposta a perguntas operacional. Use as seguintes seções para entender melhor como o aplicativo de exemplo funciona e como implementar os recursos de resposta a perguntas nos seus aplicativos em produção:

- [Como o aplicativo funciona](#how_it_works): um tutorial da estrutura e dos principais arquivos do aplicativo de exemplo.

- [Modifique seu aplicativo](#modify_applications): instruções sobre como adicionar a resposta a perguntas a um aplicativo existente.

## Como o aplicativo de exemplo funciona {:#how_it_works}

O aplicativo usa a API `BertQuestionAnswerer` no pacote da [Biblioteca Task para linguagem natural (NL)](../../inference_with_metadata/task_library/overview#supported_tasks). O modelo MobileBERT foi treinado usando o [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) do TensorFlow Lite. O aplicativo é executado na CPU por padrão, com a opção de aceleração de hardware usando o delegado GPU ou NNAPI.

Os seguintes arquivos e diretórios contêm o código fundamental para esse aplicativo:

- [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt): inicializa a resposta a perguntas e lida com a seleção de delegados e modelo.
- [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt): processa e formata os resultados.
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt): fornece a lógica de organização do aplicativo.

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

No seu próprio aplicativo, adicione as dependências do projeto para executar os modelos de aprendizado de máquina do TensorFlow Lite e acessar funções utilitárias. Essas funções convertem dados como strings em um formato de dados de tensor que pode ser processado pelo modelo. As seguintes instruções explicam como adicionar as dependências de projeto e módulo necessárias ao seu próprio projeto de app Android.

Para adicionar dependências de módulo:

1. No módulo que usa o TensorFlow Lite, atualize o arquivo `build.gradle` para que inclua as seguintes dependências.

    No aplicativo de exemplo, as dependências estão localizadas em [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle):

    ```
    dependencies {
      ...
      // Import tensorflow library
      implementation 'org.tensorflow:tensorflow-lite-task-text:0.3.0'

      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    O projeto precisa incluir a Biblioteca Task para texto (`tensorflow-lite-task-text`).

    Se você quiser modificar esse aplicativo para que seja executado em uma unidade de processamento gráfico (GPU), a biblioteca da GPU (`tensorflow-lite-gpu-delegate-plugin`) fornece a infraestrutura para executar o aplicativo na GPU e o Delegado (`tensorflow-lite-gpu`) oferece a lista de compatibilidade.

2. No Android Studio, sincronize as dependências do projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

### Inicialize os modelos de ML {:#initialize_models}

No seu aplicativo Android, você precisa inicializar o modelo de aprendizado de máquina do TensorFlow Lite com parâmetros antes de realizar previsões com o modelo.

Um modelo do TensorFlow Lite é armazenado como um arquivo `*.tflite`. O arquivo do modelo contém a lógica de previsão e geralmente inclui [metadados](../../models/convert/metadata) sobre como interpretar resultados de previsão. Geralmente, os arquivos do modelo são armazenados no diretório `src/main/assets` do seu projeto de desenvolvimento, como no código de exemplo:

- `<project>/src/main/assets/mobilebert_qa.tflite`

Observação: o aplicativo de exemplo usa um arquivo [`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle) para baixar o modelo [mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) e a [passagem de texto](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json) no tempo de build. Essa abordagem não é necessária para um aplicativo em produção.

Para conveniência e legibilidade do código, o exemplo declara um objeto complementar que define as configurações para o modelo.

Para inicializar o modelo no seu aplicativo:

1. Crie um objeto complementar para definir as configurações para o modelo. No aplicativo de exemplo, esse objeto está localizado em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106):

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

2. Crie as configurações para o modelo ao criar um objeto `BertQaHelper` e construir um objeto do TensorFlow Lite com `bertQuestionAnswerer`.

    No aplicativo de exemplo, isso está localizado na função `setupBertQuestionAnswerer()` em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76):

    ```
    class BertQaHelper(
        ...
    ) {
        ...
        init {
            setupBertQuestionAnswerer()
        }

        fun clearBertQuestionAnswerer() {
            bertQuestionAnswerer = null
        }

        private fun setupBertQuestionAnswerer() {
            val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)
            ...
            val options = BertQuestionAnswererOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .build()

            try {
                bertQuestionAnswerer =
                    BertQuestionAnswerer.createFromFileAndOptions(context, BERT_QA_MODEL, options)
            } catch (e: IllegalStateException) {
                answererListener
                    ?.onError("Bert Question Answerer failed to initialize. See error logs for details")
                Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            }
        }
        ...
        }
    ```

### Ative a aceleração de hardware (opcional) {:#hardware_acceleration}

Ao inicializar um modelo do TensorFlow Lite no seu aplicativo, você deve considerar usar os recursos de aceleração de hardware para acelerar os cálculos de previsão do modelo. Os [delegados](https://www.tensorflow.org/lite/performance/delegates) do TensorFlow Lite são módulos de software que aceleram a execução dos modelos de aprendizado de máquina usando hardware de processamento especializado em um dispositivo móvel, como unidades de processamento gráfico (GPUs) ou unidades de processamento de tensor (TPUs).

Para ativar a aceleração de hardware no seu aplicativo:

1. Crie uma variável para definir o delegado que o aplicativo usará. No aplicativo de exemplo, essa variável está localizada no início em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31):

    ```
    var currentDelegate: Int = 0
    ```

2. Crie um seletor de delegado. No aplicativo de exemplo, esse seletor está localizado na função `setupBertQuestionAnswerer` em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62):

    ```
    when (currentDelegate) {
        DELEGATE_CPU -> {
            // Default
        }
        DELEGATE_GPU -> {
            if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                baseOptionsBuilder.useGpu()
            } else {
                answererListener?.onError("GPU is not supported on this device")
            }
        }
        DELEGATE_NNAPI -> {
            baseOptionsBuilder.useNnapi()
        }
    }
    ```

Usar os delegados para executar modelos do TensorFlow é recomendável, mas não obrigatório. Para mais informações sobre como usar os delegados com o TensorFlow Lite, consulte [Delegados do TensorFlow Lite](https://www.tensorflow.org/lite/performance/delegates).

### Prepare os dados para o modelo

No seu aplicativo Android, seu código fornece dados ao modelo para interpretação ao transformar dados existentes, como texto bruto, em um formato de dados de [Tensor](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) que pode ser processado pelo modelo. Os dados em um Tensor passados a um modelo precisam ter dimensões específicas, ou um formato, que correspondam ao formato dos dados usados para treinar o modelo. Esse aplicativo de resposta a perguntas aceita [strings](https://developer.android.com/reference/java/lang/String.html) como entradas para tanto a passagem de texto como a pergunta. O modelo não reconhece caracteres especiais e palavras que não estiverem em inglês.

Para fornecer dados de passagem de texto ao modelo:

1. Use o objeto `LoadDataSetClient` para carregar os dados de passagem de texto no aplicativo. No aplicativo de exemplo, isso está localizado em [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45)

    ```
    fun loadJson(): DataSet? {
        var dataSet: DataSet? = null
        try {
            val inputStream: InputStream = context.assets.open(JSON_DIR)
            val bufferReader = inputStream.bufferedReader()
            val stringJson: String = bufferReader.use { it.readText() }
            val datasetType = object : TypeToken<DataSet>() {}.type
            dataSet = Gson().fromJson(stringJson, datasetType)
        } catch (e: IOException) {
            Log.e(TAG, e.message.toString())
        }
        return dataSet
    }
    ```

2. Use o objeto `DatasetFragment` para listar os títulos de cada passagem de texto e iniciar a tela **Pergunta e Resposta TFL**. No aplicativo de exemplo, isso está localizado em [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt):

    ```
    class DatasetFragment : Fragment() {
        ...
        override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
            super.onViewCreated(view, savedInstanceState)
            val client = LoadDataSetClient(requireActivity())
            client.loadJson()?.let {
                titles = it.getTitles()
            }
            ...
        }
       ...
    }
    ```

3. Use a função `onCreateViewHolder` no objeto `DatasetAdapter` para apresentar os títulos de cada passagem de texto. No aplicativo de exemplo, isso está localizado em [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt):

    ```
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }
    ```

Para fornecer perguntas do usuário ao modelo:

1. Use o objeto `QaAdapter` para fornecer a pergunta ao modelo. No aplicativo de exemplo, esse objeto está localizado em [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt):

    ```
    class QaAdapter(private val question: List<String>, private val select: (Int) -> Unit) :
      RecyclerView.Adapter<QaAdapter.ViewHolder>() {

      inner class ViewHolder(private val binding: ItemQuestionBinding) :
          RecyclerView.ViewHolder(binding.root) {
          init {
              binding.tvQuestionSuggestion.setOnClickListener {
                  select.invoke(adapterPosition)
              }
          }

          fun bind(question: String) {
              binding.tvQuestionSuggestion.text = question
          }
      }
      ...
    }
    ```

### Realize previsões

No seu aplicativo Android, depois de inicializar um objeto [BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer), você pode começar a inserir perguntas em texto de linguagem natural no modelo. O modelo tenta identificar a resposta na passagem de texto.

Para realizar previsões:

1. Crie uma função `answer`, que executa o modelo e mede o tempo que leva para identificar a resposta (`inferenceTime`). No aplicativo de exemplo, a função `answer` está localizada em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98):

    ```
    fun answer(contextOfQuestion: String, question: String) {
        if (bertQuestionAnswerer == null) {
            setupBertQuestionAnswerer()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val answers = bertQuestionAnswerer?.answer(contextOfQuestion, question)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        answererListener?.onResults(answers, inferenceTime)
    }
    ```

2. Passe os resultados de `answer` ao objeto listener.

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### Processe a saída do modelo

Depois de inserir uma pergunta, o modelo fornece no máximo cinco respostas possíveis na passagem.

Para obter os resultados do modelo:

1. Crie uma função `onResult` para o objeto listener processar a saída. No aplicativo de exemplo, o objeto listener está localizado em [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98)

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

2. Destaque seções da passagem com base nos resultados. No aplicativo de exemplo, isso está localizado em [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208):

    ```
    override fun onResults(results: List<QaAnswer>?, inferenceTime: Long) {
        results?.first()?.let {
            highlightAnswer(it.text)
        }

        fragmentQaBinding.tvInferenceTime.text = String.format(
            requireActivity().getString(R.string.bottom_view_inference_time),
            inferenceTime
        )
    }
    ```

Depois que o modelo retornar um conjunto de resultados, seu aplicativo pode agir em relação a essas previsões ao apresentar o resultado ao seu usuário ou executar lógica adicional.

## Próximos passos

- Treine e implemente os modelos do zero com o tutorial [Resposta a perguntas com o Model Maker do TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).
- Explore mais [ferramentas de processamento de texto para o TensorFlow](https://www.tensorflow.org/text).
- Baixe outros modelos BERT no [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1).
- Descubra os usos do TensorFlow Lite nos [exemplos](../../examples).

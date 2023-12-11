# Crie sua própria API de tarefas

A <a href="overview.md">biblioteca Task do TensorFlow Lite</a> conta com APIs pré-compiladas nativas/para Android/para iOS usando a mesma infraestrutura de abstração do TensorFlow. Você pode estender a infraestrutura da API de tarefas para criar APIs personalizadas caso as bibliotecas de tarefas existentes não sejam compatíveis com seu modelo.

## Visão geral

A infraestrutura da API de tarefas tem uma estrutura de duas camadas: a camada C++ inferior, que encapsula o runtime nativo do TF Lite, e a camada Java/Obj-C superior, que se comunica com a camada C++ por meio da JNI ou do encapsulador nativo.

Implementar toda a lógica do TensorFlow apenas no C++ minimiza o custo, maximiza o desempenho de inferência e simplifica o workflow geral entre as plataformas.

Para criar uma classe de tarefas, estenda a [BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) para fornecer uma lógica de conversão entre a interface de modelo do TF Lite e a interface da API de tarefas. Depois, use utilitários Java/ObjC para criar as APIs correspondentes. Com todos os detalhes do TensorFlow ocultos, você pode implantar o modelo do TF Lite em seus aplicativos sem qualquer conhecimento sobre aprendizado de máquina.

O TensorFlow Lite conta com algumas APIs pré-compiladas para a maioria das <a href="overview.md#supported_tasks">tarefas comuns de visão e NLP</a>. Você pode compilar suas próprias APIs para outras tarefas usando a infraestrutura da API de tarefas.

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg)</div>
<div align="center">Figura 1 – APIs de tarefas pré-compiladas</div>
<div align="left"></div>

## Compile sua própria API com a infraestrutura da API de tarefas

### API do C++

Todos os detalhes do TF Lite são implementados na API nativa. Crie um objeto de API usando uma das funções de fábrica e obtenha os resultados do modelo chamando as funções definidas na interface.

#### Exemplo de uso

Veja um exemplo de uso de [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h) em C++ para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).

```cpp
  char kBertModelPath[] = "path/to/model.tflite";
  // Create the API from a model file
  std::unique_ptr<BertQuestionAnswerer> question_answerer =
      BertQuestionAnswerer::CreateFromFile(kBertModelPath);

  char kContext[] = ...; // context of a question to be answered
  char kQuestion[] = ...; // question to be answered
  // ask a question
  std::vector<QaAnswer> answers = question_answerer.Answer(kContext, kQuestion);
  // answers[0].text is the best answer
```

#### Compilação da API

<div align="center">![native_task_api](images/native_task_api.svg)</div>
<div align="center">Figura 2 – API de tarefas nativa</div>
<div align="left"></div>

Para compilar um objeto de API, você precisa estender [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) para fornecer as seguintes informações:

- **Determine o I/O da API** – Sua API deve expor entradas/saídas similares nas diferentes plataformas. Por exemplo: `BertQuestionAnswerer` recebe duas strings `(std::string& context, std::string& question)` como entrada e gera como saída um vetor de possíveis respostas e probabilidades como `std::vector<QaAnswer>`. Isso é feito por meio da especificação dos tipos correspondentes no [parâmetro template](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;") de `BaseTaskApi`. Com os parâmetros template especificados, a função [`BaseTaskApi::Infer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") terá os tipos corretos de entrada/saída. Essa função pode ser chamada diretamente pelos clientes da API, mas é uma boa prática encapsulá-la dentro de uma função específica do modelo; neste caso, é `BertQuestionAnswerer::Answer`.

    ```cpp
    class BertQuestionAnswerer : public BaseTaskApi<
                                  std::vector<QaAnswer>, // OutputType
                                  const std::string&, const std::string& // InputTypes
                                  > {
      // Model specific function delegating calls to BaseTaskApi::Infer
      std::vector<QaAnswer> Answer(const std::string& context, const std::string& question) {
        return Infer(context, question).value();
      }
    }
    ```

- **Forneça lógica de conversão entre o I/O da API e o tensor de entrada/saída do modelo** – Com os tipos de entrada e saída especificados, as subclasses também precisam implementar as funções tipadas [`BaseTaskApi::Preprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74) e [`BaseTaskApi::Postprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80). As duas funções fornecem [entradas](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc) e [saídas](https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008) do `FlatBuffer` do TF Lite. A subclasse é responsável por atribuir valores do I/O da API aos tensores de I/O. Confira um exemplo completo de implementação em [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc).

    ```cpp
    class BertQuestionAnswerer : public BaseTaskApi<
                                  std::vector<QaAnswer>, // OutputType
                                  const std::string&, const std::string& // InputTypes
                                  > {
      // Convert API input into tensors
      absl::Status BertQuestionAnswerer::Preprocess(
        const std::vector<TfLiteTensor*>& input_tensors, // input tensors of the model
        const std::string& context, const std::string& query // InputType of the API
      ) {
        // Perform tokenization on input strings
        ...
        // Populate IDs, Masks and SegmentIDs to corresponding input tensors
        PopulateTensor(input_ids, input_tensors[0]);
        PopulateTensor(input_mask, input_tensors[1]);
        PopulateTensor(segment_ids, input_tensors[2]);
        return absl::OkStatus();
      }

      // Convert output tensors into API output
      StatusOr<std::vector<QaAnswer>> // OutputType
      BertQuestionAnswerer::Postprocess(
        const std::vector<const TfLiteTensor*>& output_tensors, // output tensors of the model
      ) {
        // Get start/end logits of prediction result from output tensors
        std::vector<float> end_logits;
        std::vector<float> start_logits;
        // output_tensors[0]: end_logits FLOAT[1, 384]
        PopulateVector(output_tensors[0], &end_logits);
        // output_tensors[1]: start_logits FLOAT[1, 384]
        PopulateVector(output_tensors[1], &start_logits);
        ...
        std::vector<QaAnswer::Pos> orig_results;
        // Look up the indices from vocabulary file and build results
        ...
        return orig_results;
      }
    }
    ```

- **Crie funções de fábrica da API** – São necessários um arquivo de modelo e um [`OpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h) para inicializar o [`tflite::Interpreter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h). [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h) conta com funções utilitárias para criar instâncias da BaseTaskApi.

    Observação: por padrão, [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h) conta com um [`BuiltInOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h). Caso o seu modelo precise de operações personalizadas ou um subconjunto de operações integradas, você criar um [`MutableOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h) para registrá-las.

    Você também precisa fornecer qualquer arquivo associado ao modelo. Por exemplo: `BertQuestionAnswerer` também pode ter um arquivo adicional para o vocabulário do tokenizador.

    ```cpp
    class BertQuestionAnswerer : public BaseTaskApi<
                                  std::vector<QaAnswer>, // OutputType
                                  const std::string&, const std::string& // InputTypes
                                  > {
      // Factory function to create the API instance
      StatusOr<std::unique_ptr<QuestionAnswerer>>
      BertQuestionAnswerer::CreateBertQuestionAnswerer(
          const std::string& path_to_model, // model to passed to TaskApiFactory
          const std::string& path_to_vocab  // additional model specific files
      ) {
        // Creates an API object by calling one of the utils from TaskAPIFactory
        std::unique_ptr<BertQuestionAnswerer> api_to_init;
        ASSIGN_OR_RETURN(
            api_to_init,
            core::TaskAPIFactory::CreateFromFile<BertQuestionAnswerer>(
                path_to_model,
                absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>(),
                kNumLiteThreads));

        // Perform additional model specific initializations
        // In this case building a vocabulary vector from the vocab file.
        api_to_init->InitializeVocab(path_to_vocab);
        return api_to_init;
      }
    }
    ```

### API do Android

Crie APIs do Android definindo a interface Java/Kotlin e delegando a lógica para a camada C++ por meio da JNI. No caso da API do Android, é preciso compilar a API nativa primeiro.

#### Exemplo de uso

Veja um exemplo de uso de [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java) em Java para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).

```java
  String BERT_MODEL_FILE = "path/to/model.tflite";
  String VOCAB_FILE = "path/to/vocab.txt";
  // Create the API from a model file and vocabulary file
    BertQuestionAnswerer bertQuestionAnswerer =
        BertQuestionAnswerer.createBertQuestionAnswerer(
            ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE, VOCAB_FILE);

  String CONTEXT = ...; // context of a question to be answered
  String QUESTION = ...; // question to be answered
  // ask a question
  List<QaAnswer> answers = bertQuestionAnswerer.answer(CONTEXT, QUESTION);
  // answers.get(0).text is the best answer
```

#### Compilação da API

<div align="center">![android_task_api](images/android_task_api.svg)</div>
<div align="center">Figura 3 – API de tarefas para Android</div>
<div align="left"></div>

Similar às APIs nativas, para compilar um objeto de API, o cliente precisa estender a [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java) para fornecer as informações abaixo, o que fornece manipulações de JNI para todas as APIs de tarefas para Java.

- **Determine o I/O da API** – Geralmente, espelha as interfaces nativas. Por exemplo: `BertQuestionAnswerer` recebe `(String context, String question)` como entrada e gera como saída `List<QaAnswer>`. A implementação chama uma função nativa privada com assinatura similar, exceto pelo parâmetro adicional `long nativeHandle`, que é o ponteiro retornado pelo C++.

    ```java
    class BertQuestionAnswerer extends BaseTaskApi {
      public List<QaAnswer> answer(String context, String question) {
        return answerNative(getNativeHandle(), context, question);
      }

      private static native List<QaAnswer> answerNative(
                                            long nativeHandle, // C++ pointer
                                            String context, String question // API I/O
                                           );

    }
    ```

- **Crie funções de fábrica da API** – Também espelham as funções de fábrica nativas, exceto pelas funções de fábrica do Android, que também precisam receber o [`Context`](https://developer.android.com/reference/android/content/Context) para acesso aos arquivos. A implementação chama um dos utilitários em [`TaskJniUtils`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java) para compilar o objeto de API C++ correspondente e passar seu ponteiro ao construtor da `BaseTaskApi`.

    ```java
      class BertQuestionAnswerer extends BaseTaskApi {
        private static final String BERT_QUESTION_ANSWERER_NATIVE_LIBNAME =
                                                  "bert_question_answerer_jni";

        // Extending super constructor by providing the
        // native handle(pointer of corresponding C++ API object)
        private BertQuestionAnswerer(long nativeHandle) {
          super(nativeHandle);
        }

        public static BertQuestionAnswerer createBertQuestionAnswerer(
                                            Context context, // Accessing Android files
                                            String pathToModel, String pathToVocab) {
          return new BertQuestionAnswerer(
              // The util first try loads the JNI module with name
              // BERT_QUESTION_ANSWERER_NATIVE_LIBNAME, then opens two files,
              // converts them into ByteBuffer, finally ::initJniWithBertByteBuffers
              // is called with the buffer for a C++ API object pointer
              TaskJniUtils.createHandleWithMultipleAssetFilesFromLibrary(
                  context,
                  BertQuestionAnswerer::initJniWithBertByteBuffers,
                  BERT_QUESTION_ANSWERER_NATIVE_LIBNAME,
                  pathToModel,
                  pathToVocab));
        }

        // modelBuffers[0] is tflite model file buffer, and modelBuffers[1] is vocab file buffer.
        // returns C++ API object pointer casted to long
        private static native long initJniWithBertByteBuffers(ByteBuffer... modelBuffers);

      }
    ```

- **Implemente o módulo JNI para funções nativas** – Todos os métodos Java nativos são implementados chamando uma função nativa correspondente do módulo de JNI. As funções de fábrica criam um objeto da API nativa e retornam seu ponteiro como um tipo long para o Java. Em chamadas posteriores à API do Java, o ponteiro de tipo long é passado de volta à JNI e convertido de volta no objeto da API nativa. Os resultados da API nativa são então convertidos de volta para resultados Java.

    Por exemplo, [bert_question_answerer_jni](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc) é implementado assim.

    ```cpp
      // Implements BertQuestionAnswerer::initJniWithBertByteBuffers
      extern "C" JNIEXPORT jlong JNICALL
      Java_org_tensorflow_lite_task_text_qa_BertQuestionAnswerer_initJniWithBertByteBuffers(
          JNIEnv* env, jclass thiz, jobjectArray model_buffers) {
        // Convert Java ByteBuffer object into a buffer that can be read by native factory functions
        absl::string_view model =
            GetMappedFileBuffer(env, env->GetObjectArrayElement(model_buffers, 0));

        // Creates the native API object
        absl::StatusOr<std::unique_ptr<QuestionAnswerer>> status =
            BertQuestionAnswerer::CreateFromBuffer(
                model.data(), model.size());
        if (status.ok()) {
          // converts the object pointer to jlong and return to Java.
          return reinterpret_cast<jlong>(status->release());
        } else {
          return kInvalidPointer;
        }
      }

      // Implements BertQuestionAnswerer::answerNative
      extern "C" JNIEXPORT jobject JNICALL
      Java_org_tensorflow_lite_task_text_qa_BertQuestionAnswerer_answerNative(
      JNIEnv* env, jclass thiz, jlong native_handle, jstring context, jstring question) {
      // Convert long to native API object pointer
      QuestionAnswerer* question_answerer = reinterpret_cast<QuestionAnswerer*>(native_handle);

      // Calls the native API
      std::vector<QaAnswer> results = question_answerer->Answer(JStringToString(env, context),
                                             JStringToString(env, question));

      // Converts native result(std::vector<QaAnswer>) to Java result(List<QaAnswerer>)
      jclass qa_answer_class =
        env->FindClass("org/tensorflow/lite/task/text/qa/QaAnswer");
      jmethodID qa_answer_ctor =
        env->GetMethodID(qa_answer_class, "<init>", "(Ljava/lang/String;IIF)V");
      return ConvertVectorToArrayList<QaAnswer>(
        env, results,
        [env, qa_answer_class, qa_answer_ctor](const QaAnswer& ans) {
          jstring text = env->NewStringUTF(ans.text.data());
          jobject qa_answer =
              env->NewObject(qa_answer_class, qa_answer_ctor, text, ans.pos.start,
                             ans.pos.end, ans.pos.logit);
          env->DeleteLocalRef(text);
          return qa_answer;
        });
      }

      // Implements BaseTaskApi::deinitJni by delete the native object
      extern "C" JNIEXPORT void JNICALL Java_task_core_BaseTaskApi_deinitJni(
          JNIEnv* env, jobject thiz, jlong native_handle) {
        delete reinterpret_cast<QuestionAnswerer*>(native_handle);
      }
    ```

### API do iOS

Crie APIs do iOS encapsulando um objeto da API nativa em um objeto da API do ObjC. O objeto da API criado pode ser usado no ObjC ou no Swift. No caso da API do iOS, é preciso compilar a API nativa primeiro.

#### Exemplo de uso

Veja um exemplo de uso de [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h) em ObjC para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).

```swift
  static let mobileBertModelPath = "path/to/model.tflite";
  // Create the API from a model file and vocabulary file
  let mobileBertAnswerer = TFLBertQuestionAnswerer.mobilebertQuestionAnswerer(
      modelPath: mobileBertModelPath)

  static let context = ...; // context of a question to be answered
  static let question = ...; // question to be answered
  // ask a question
  let answers = mobileBertAnswerer.answer(
      context: TFLBertQuestionAnswererTest.context, question: TFLBertQuestionAnswererTest.question)
  // answers.[0].text is the best answer
```

#### Compilação da API

<div align="center">![ios_task_api](images/ios_task_api.svg)</div>
<div align="center">Figura 4 – API de tarefas para iOS</div>
<div align="left"></div>

A API do iOS é um encapsulador ObjC simples que usa a API nativa. Para compilar a API, siga as etapas abaixo:

- **Defina o encapsulador ObjC** – Defina uma classe ObjC e delegue as implementações para o objeto da API nativa correspondente. Atenção: as dependências nativas só podem aparecer em um arquivo .mm devido à incapacidade de o Swift fazer a interoperabilidade com o C++.

    - Arquivo .h

    ```objc
      @interface TFLBertQuestionAnswerer : NSObject

      // Delegate calls to the native BertQuestionAnswerer::CreateBertQuestionAnswerer
      + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString*)modelPath
                                                    vocabPath:(NSString*)vocabPath
          NS_SWIFT_NAME(mobilebertQuestionAnswerer(modelPath:vocabPath:));

      // Delegate calls to the native BertQuestionAnswerer::Answer
      - (NSArray<TFLQAAnswer*>*)answerWithContext:(NSString*)context
                                         question:(NSString*)question
          NS_SWIFT_NAME(answer(context:question:));
    }
    ```

    - Arquivo .mm

    ```objc
      using BertQuestionAnswererCPP = ::tflite::task::text::BertQuestionAnswerer;

      @implementation TFLBertQuestionAnswerer {
        // define an iVar for the native API object
        std::unique_ptr<QuestionAnswererCPP> _bertQuestionAnswerwer;
      }

      // Initialize the native API object
      + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString *)modelPath
                                              vocabPath:(NSString *)vocabPath {
        absl::StatusOr<std::unique_ptr<QuestionAnswererCPP>> cQuestionAnswerer =
            BertQuestionAnswererCPP::CreateBertQuestionAnswerer(MakeString(modelPath),
                                                                MakeString(vocabPath));
        _GTMDevAssert(cQuestionAnswerer.ok(), @"Failed to create BertQuestionAnswerer");
        return [[TFLBertQuestionAnswerer alloc]
            initWithQuestionAnswerer:std::move(cQuestionAnswerer.value())];
      }

      // Calls the native API and converts C++ results into ObjC results
      - (NSArray<TFLQAAnswer *> *)answerWithContext:(NSString *)context question:(NSString *)question {
        std::vector<QaAnswerCPP> results =
          _bertQuestionAnswerwer->Answer(MakeString(context), MakeString(question));
        return [self arrayFromVector:results];
      }
    }
    ```

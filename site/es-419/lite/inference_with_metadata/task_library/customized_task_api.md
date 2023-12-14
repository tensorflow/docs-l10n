# Genere su propia API de tareas

La <a href="overview.md">librería de tareas TensorFlow Lite</a> da API nativas/Android/iOS precompiladas sobre la misma infraestructura que abstrae TensorFlow. Puede ampliar la infraestructura de la API de tareas para crear API personalizadas si su modelo no es compatible con las librerías de tareas existentes.

## Visión general

La infraestructura de la API de tareas tiene una estructura de dos capas: la capa inferior C++ que encapsula el runtime nativo TFLite y la capa superior Java/ObjC que se comunica con la capa C++ a través de JNI o de un contenedor nativo.

Implementar toda la lógica de TensorFlow sólo en C++ minimiza el costo, maximiza el rendimiento de la inferencia y simplifica el flujo de trabajo global en todas las plataformas.

Para crear una clase Task, extienda la [BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) para aportar la lógica de conversión entre la interfaz del modelo TFLite y la interfaz de la tarea API, y luego use las utilidades Java/ObjC para crear las API correspondientes. Con todos los detalles de TensorFlow ocultos, puede implementar el modelo TFLite en sus apps sin ningún conocimiento de aprendizaje automático.

TensorFlow Lite ofrece algunas API precompiladas para las tareas más populares de <a href="overview.md#supported_tasks">Visión y PNL</a>. Puede generar sus propias API para otras tareas usando la infraestructura de API de tareas.

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg)</div>
<div align="center">Figure 1. API de tareas precompiladas</div>
<div align="left"></div>

## Genere su propia API con la infra API de tareas

### API de C/C++

Todos los detalles de TFLite se implementan en la API nativa. Cree un objeto API usando una de las funciones de fábrica y obtenga los resultados del modelo llamando a las funciones definidas en la interfaz.

#### Ejemplo de uso

A continuación se muestra un ejemplo usando el [`BertQuestionAnswerer` en C++](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h) para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).

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

#### Generar la API

<div align="center">![native_task_api](images/native_task_api.svg)</div>
<div align="center">Figure 2. API de tareas nativa</div>
<div align="left"></div>

Para generar un objeto API, debe aportar la siguiente información extendiendo [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h)

- **Determine la E/S de la API**: Su API debería presentar una entrada/salida similar en las distintas plataformas. Por ejemplo, `BertQuestionAnswerer` toma dos cadenas `(std::string& context, std::string& question)` como entrada y emite un vector de posibles respuestas y probabilidades como `std::vector<QaAnswer>`. Esto se hace especificando los tipos correspondientes en el [parámetro de plantilla] de `BaseTaskApi`(https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;"). Con los parámetros de plantilla especificados, la función [`BaseTaskApi::Infer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") tendrá los tipos de entrada/salida correctos. Esta función puede ser llamada directamente por los clientes de la API, pero es una práctica recomendable encapsularla dentro de una función específica del modelo, en este caso, `BertQuestionAnswerer::Answer`.

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

- **Provea la lógica de conversión entre la E/S de la API y el tensor de entrada/salida del modelo**: Con los tipos de entrada y salida especificados, las subclases también necesitan implementar las funciones tipadas [`BaseTaskApi::Preprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74) y [`BaseTaskApi::Postprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80). Las dos funciones proveen [entradas](https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1007) y [salidas](https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008) del TFLite `FlatBuffer`. La subclase es responsable de asignar valores desde la E/S de la API a los tensores de E/S. Véase el ejemplo de implementación completo en [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc).

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

- **Crear funciones de fábrica de la API**: Se necesita un archivo modelo y un [`OpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h) para inicializar el [`tflite::Interpreter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h). [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h) ofrece funciones de utilidad para crear instancias de BaseTaskApi.

    Nota: De forma predeterminada, [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h) ofrece un [`BuiltInOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h). Si su modelo necesita ops personalizadas o un subconjunto de ops incorporadas, puede registrarlas creando un [`MutableOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h).

    También debe aportar cualquier archivo asociado al modelo. Por ejemplo, `BertQuestionAnswerer` también puede tener un archivo adicional para el vocabulario de su tokenizador.

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

### API Android

Cree APIs Android definiendo la interfaz Java/Kotlin y delegando la lógica a la capa C++ a través de JNI. La API de Android requiere que se genere primero la API nativa.

#### Ejemplo de uso

Aquí tiene un ejemplo usando [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java) Java para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1).

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

#### Generar la API

<div align="center">![android_task_api](images/android_task_api.svg)</div>
<div align="center">Figura 3. API de tareas de Android</div>
<div align="left"></div>

De forma similar a las API nativas, para generar un objeto API, el cliente debe facilitar la siguiente información extendiendo [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java), que proporciona manejadores JNI para todas las API de tareas Java.

- **Determinar la E/S de la API**: Esto suele reflejar las interfaces nativas. Por ejemplo, `BertQuestionAnswerer` toma `(String context, String question)` como entrada y da como salida `List<QaAnswer>`. La implementación llama a una función nativa privada con firma similar, excepto que tiene un parámetro adicional `long nativeHandle`, que es el puntero devuelto desde C++.

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

- **Crear funciones de fábrica de la API**: Esto también refleja las funciones de fábrica nativas, excepto que las funciones de fábrica de Android también necesitan tomar [`Context`](https://developer.android.com/reference/android/content/Context) para el acceso al archivo. La implementación llama a una de las utilidades de [`TaskJniUtils`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java) para generar el objeto C++ de la API correspondiente y pasar su puntero al constructor `BaseTaskApi`.

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

- **Implemente el módulo JNI para funciones nativas**: Todos los métodos nativos de Java se implementan llamando a una función nativa correspondiente desde el módulo JNI. Las funciones de fábrica crearían un objeto API nativo y devolverían su puntero como un tipo long a Java. En posteriores llamadas a la API de Java, el puntero de tipo long se pasa de nuevo a JNI y se vuelve a convertir al objeto de la API nativa. Luego, los resultados de la API nativa se convierten de nuevo en resultados de Java.

    Por ejemplo, así es como se implementa [bert_question_answerer_jni](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc).

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

### API de iOS

Cree API de iOS encapsulando un objeto API nativo en un objeto API de ObjC. El objeto API creado puede usarse tanto en ObjC como en Swift. La API de iOS requiere que se cree primero la API nativa.

#### Ejemplo de uso

Este es un ejemplo usando [`TFLBertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h) en ObjC para [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1) en Swift.

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

#### Generar la API

<div align="center">![ios_task_api](images/ios_task_api.svg)</div>
<div align="center">Figura 4. API de tareas de iOS</div>
<div align="left"></div>

La API de iOS es un simple contenedor ObjC sobre la API nativa. Genere la API siguiendo los pasos que se indican a continuación:

- **Defina el contenedor ObjC**: Defina una clase ObjC y delegue las implementaciones al objeto API nativo correspondiente. Tenga en cuenta que las dependencias nativas sólo pueden aparecer en un archivo .mm debido a la incapacidad de Swift para interoperar con C++.

    - Archivo .h

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

    - Archivo .mm

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

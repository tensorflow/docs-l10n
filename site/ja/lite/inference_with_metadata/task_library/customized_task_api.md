# 独自の Task API を構築する

<a href="overview.md">TensorFlow Lite Task Library</a> は、TensorFlow を抽象化するインフラストラクチャと同じインフラストラクチャの上に、構築済みのネイティブ/Android/iOS API を提供します。モデルが既存の Task ライブラリでサポートされていない場合は、Task API インフラストラクチャを拡張して、カスタマイズされた API を構築できます。

## 概要

Task API インフラストラクチャは 2 レイヤー構造になっています。下部の C++ レイヤーはネイティブ TFLite ランタイムをカプセル化し、上部の Java/ObjC レイヤーは JNI またはネイティブラッパーを介して C++ レイヤーと通信します。

すべての TensorFlow ロジックを C ++ のみで実装すると、コストの最小化や推論パフォーマンスの最大化が可能になりプラットフォーム全体のワークフロー全体が簡素化されます。

Task クラスを作成するには、[BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) を拡張して TFLite モデルインターフェースと Task API インターフェース間の変換ロジックを提供し、Java/ObjC ユーティリティを使用して対応する API を作成します。TensorFlow の詳細をすべて非表示にすると、機械学習の知識がなくても TFLite モデルをアプリにデプロイできます。

TensorFlow Lite は、最も人気のある <a href="overview.md#supported_tasks">Vision と NLP タスク</a>用にいくつかの構築済み API を提供します。Task API インフラストラクチャを使用すると、他のタスク用に独自の API を構築できます。

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg)</div>
<div align="center">図 1. 事前構築済みの Task API</div>
<div align="left"></div>

## Task API infra を使用して独自の API を構築する

### C++ API

すべての TFLite の詳細は、ネイティブ API に実装されています。ファクトリ関数の 1 つを使用して API オブジェクトを作成し、インターフェースで定義された関数を呼び出してモデルの結果を取得します。

#### 使用例

以下は、[MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1) で C ++[ `BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h) を使用した例です。

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

#### API の構築

<div align="center">![native_task_api](images/native_task_api.svg)</div>
<div align="center">図 2. ネイティブ Task API</div>
<div align="left"></div>

API オブジェクトを作成するには、[`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) を拡張して次の情報を提供する必要があります

- **API I/Oを決定する** - API は、異なるプラットフォーム間で同様の入出力を公開する必要があります。 例: `BertQuestionAnswerer` は、2 つの文字列 `(std::string& context, std::string& question)` を入力として取り、可能な答えと確率のベクトルを `std::vector<QaAnswer>` として出力します。これは、`BaseTaskApi` の [テンプレートパラメータ] で対応するタイプを指定することで行われます (https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;")。テンプレートパラメータを指定すると、[`BaseTaskApi::Infer`] (https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") 関数は正しい入力/出力の型になります。この関数は API クライアントから直接呼び出すことができますが、モデル固有の関数 (この場合は`BertQuestionAnswerer::Answer`) 内にラップすることをお勧めします。

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

- **モデルの API I/O と入出力テンソル間の変換ロジックを提供する** - 入力と出力の型が指定されている場合、サブクラスは型付き関数 [`BaseTaskApi::Preprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74) と [`BaseTaskApi::Postprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80) も実装する必要があります。2 つの関数は、TFLite <code>FlatBuffer</code> からの[入力](https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008)と<a>出力</a>を提供します。サブクラスは、API I/O から I/O テンソルへ値を割り当てます。[`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc) で完全な実装例を参照してください。

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

- **API のファクトリ関数を作成する** - [`tflite::Interpreter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h) を初期化するには、モデルファイルと [`OpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h) が必要です。 [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h) は、BaseTaskApi インスタンスを作成するためのユーティリティ関数を提供します。

    注意: デフォルトでは、[`BuiltInOpResolver` ](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h)は [`TaskAPIFactory`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h) を提供します。モデルにカスタマイズされた演算または組み込み演算のサブセットが必要な場合は、[`MutableOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h) を作成して登録できます。

    また、モデルに関連付けられているファイルも提供する必要があります。たとえば、`BertQuestionAnswerer` には、トークナイザーの語彙用のファイルを追加することもできます。

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

### Android API

Java/Kotlin インターフェースを定義し、ロジックを JNI を介して C ++ レイヤーにデリゲートすることにより、Android API を作成します。Android API では、ネイティブ API を最初に作成する必要があります。

#### 使用例

以下は、[MobileBert](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java) で Java<a> <code>BertQuestionAnswerer</code></a> を使用した例です。

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

#### API の構築

<div align="center">![android_task_api](images/android_task_api.svg)</div>
<div align="center">図 3. Android Task API</div>
<div align="left"></div>

ネイティブ API と同様に、API オブジェクトを構築するには、クライアントはすべての Java Task API に JNI 処理を提供する [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java) を拡張して次の情報を提供する必要があります。

- **API I/O を決定する** - これは通常、ネイティブインターフェースを反映しています。たとえば、`BertQuestionAnswerer` は、`(String context, String question)` を入力として取り、`List<QaAnswer>` を出力します。実装は、同様のシグネチャを持つプライベートネイティブ関数を呼び出しますが、C++ から返されるポインタであるパラメータ `long nativeHandle` が追加されます。

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

- **API のファクトリ関数を作成する** - これはネイティブファクトリ関数もミラーリングしますが、Android ファクトリ関数もファイルアクセスのために [`Context`](https://developer.android.com/reference/android/content/Context) を取る必要があります。実装は、[`TaskJniUtils`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java) のユーティリティの 1 つを呼び出して、対応する C ++ API オブジェクトを構築し、そのポインタを `BaseTaskApi` コンストラクタに渡します。

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

- **ネイティブ関数用の JNI モジュールを実装する** - すべての Java ネイティブメソッドは、JNI モジュールから対応するネイティブ関数を呼び出すことによって実装されます。ファクトリ関数はネイティブ API オブジェクトを作成し、そのポインタを long 型として Java に返します。その後、Java API の呼び出しでは、long 型のポインタが JNI に渡され、ネイティブ API オブジェクトにキャストされます。ネイティブ API の結果は、Java の結果に変換されます。

    たとえば、これは [bert_question_answerer_jni](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc) の実装方法です。

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

### iOS API

ネイティブ API オブジェクトを ObjC API オブジェクトにラップして iOS API を作成します。作成された API オブジェクトは、ObjC または Swift で使用できます。iOS API では、ネイティブ API を最初に作成する必要があります。

#### 使用例

以下は、Swift で [MobileBert](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h) の ObjC <a><code>TFLBertQuestionAnswerer</code></a> を使用した例です。

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

#### API の構築

<div align="center">![ios_task_api](images/ios_task_api.svg)</div>
<div align="center">図 4. iOS Task API</div>
<div align="left"></div>

iOS API は、ネイティブ API の上にあるシンプルな ObjC ラッパーです。以下の手順に沿って API を構築します。

- **ObjC ラッパーを定義する** - ObjC クラスを定義し、対応するネイティブ API オブジェクトに実装をデリゲートします。Swift は C ++ と相互運用できないため、ネイティブの依存関係は .mm ファイルにのみ表示されることに注意してください。

    - .h ファイル

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

    - .mm ファイル

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

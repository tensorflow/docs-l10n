# 独自の Task API を構築する

<a href="overview.md">TensorFlow Lite Task Library</a> は、TensorFlow を抽象化するインフラストラクチャと同じインフラストラクチャの上に、構築済みのネイティブ/Android/iOS API を提供します。モデルが既存の Task ライブラリでサポートされていない場合は、Task API インフラストラクチャを拡張して、カスタマイズされた API を構築できます。

## 概要

Task API インフラストラクチャは 2 レイヤー構造になっています。下部の C++ レイヤーはネイティブ TFLite ランタイムをカプセル化し、上部の Java/ObjC レイヤーは JNI またはネイティブラッパーを介して C++ レイヤーと通信します。

すべての TensorFlow ロジックを C ++ のみで実装すると、コストの最小化や推論パフォーマンスの最大化が可能になりプラットフォーム全体のワークフロー全体が簡素化されます。

Task クラスを作成するには、[BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) を拡張して TFLite モデルインターフェイスと Task API インターフェイス間の変換ロジックを提供し、Java/Obj ユーティリティを使用して対応する API を作成します。TensorFlow の詳細をすべて非表示にすると、機械学習の知識がなくても TFLite モデルをアプリにデプロイできます。

TensorFlow Lite は、最も人気のある <a href="overview.md#supported_tasks">Vision と NLP タスク</a>用にいくつかの構築済み API を提供します。Task API インフラストラクチャを使用すると、他のタスク用に独自の API を構築できます。

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg) <div align="center">図 1. 構築済み Task API <div align="left"> </div> <h>Task API インフラストラクチャで独自の API を作成</h> <h>C++ API</h> <p data-md-type="paragraph">すべての TFLite 詳細は、ネイティブ API で実装されます。API オブジェクトを作成するには、ファクトリ関数のいずれかを使用し、インターフェイスで定義された関数を呼び出して、モデル結果を取得します。</p> <h>使用例</h> <p data-md-type="paragraph">C++ を使用した例<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> for <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a>.</p> <pre data-md-type="block_code" data-md-language="cpp">  char kBertModelPath[] = "path/to/model.tflite";
  // Create the API from a model file
  std::unique_ptr&lt;BertQuestionAnswerer&gt; question_answerer =
      BertQuestionAnswerer::CreateFromFile(kBertModelPath);

  char kContext[] = ...; // context of a question to be answered
  char kQuestion[] = ...; // question to be answered
  // ask a question
  std::vector&lt;QaAnswer&gt; answers = question_answerer.Answer(kContext, kQuestion);
  // answers[0].text is the best answer
</pre> <h>API の構築</h> <div data-md-type="block_html"><div align="center">![native_task_api](images/native_task_api.svg) <div align="center">図 2. ネイティブ Task API <div align="left"> </div> <p data-md-type="paragraph">API オブジェクトを構築するには、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a> を拡張して、次の情報を入力する必要があります。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">API I/O の決定</strong> - API はさまざまなプラットフォームで類似した入出力を公開します。例: <code data-md-type="codespan">BertQuestionAnswerer</code> は 2 つの文字列の <code data-md-type="codespan">(std::string&amp; context, std::string&amp; question)</code> を入力値として取り、考えられる答えと確率のベクトルを <code data-md-type="codespan">std::vector&lt;QaAnswer&gt;</code> として出力します。これを行うには、<code data-md-type="codespan">BaseTaskApi</code> の [template parameter](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;") で対応する型を指定します。テンプレートパラメータを指定すると、[<code data-md-type="codespan">BaseTaskApi::Infer</code>](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") 関数に正しい入出力型が定義されます。この関数は直接 API クライアントから呼び出せますが、モデル固有の関数内で隠蔽することをお勧めします。この例では、<code data-md-type="codespan">BertQuestionAnswerer::Answer</code> です。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
                              std::vector&lt;QaAnswer&gt;, // OutputType
                              const std::string&amp;, const std::string&amp; // InputTypes
                              &gt; {
  // Model specific function delegating calls to BaseTaskApi::Infer
  std::vector&lt;QaAnswer&gt; Answer(const std::string&amp; context, const std::string&amp; question) {
    return Infer(context, question).value();
  }
}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">モデルの API I/O と入出力テンソルの間の変換ロジックを指定する</strong> - 入出力型を指定すると、サブクラスでも入力された関数の <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Preprocess</code></a> と <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Postprocess</code></a> を実装する必要があります。この 2 つの関数は、TFLite <code data-md-type="codespan">FlatBuffer</code> からの<a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1007" data-md-type="link">入力</a>と<a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008" data-md-type="link">出力</a>を提供します。サブクラスは、API I/O から I/O テンソルに値を割り当てます。詳細な実装例については、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> を参照してください。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
                              std::vector&lt;QaAnswer&gt;, // OutputType
                              const std::string&amp;, const std::string&amp; // InputTypes
                              &gt; {
  // Convert API input into tensors
  absl::Status BertQuestionAnswerer::Preprocess(
    const std::vector&lt;TfLiteTensor*&gt;&amp; input_tensors, // input tensors of the model
    const std::string&amp; context, const std::string&amp; query // InputType of the API
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
  StatusOr&lt;std::vector&lt;QaAnswer&gt;&gt; // OutputType
  BertQuestionAnswerer::Postprocess(
    const std::vector&lt;const TfLiteTensor*&gt;&amp; output_tensors, // output tensors of the model
  ) {
    // Get start/end logits of prediction result from output tensors
    std::vector&lt;float&gt; end_logits;
    std::vector&lt;float&gt; start_logits;
    // output_tensors[0]: end_logits FLOAT[1, 384]
    PopulateVector(output_tensors[0], &amp;end_logits);
    // output_tensors[1]: start_logits FLOAT[1, 384]
    PopulateVector(output_tensors[1], &amp;start_logits);
    ...
    std::vector&lt;QaAnswer::Pos&gt; orig_results;
    // Look up the indices from vocabulary file and build results
    ...
    return orig_results;
  }
}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">API のファクトリ関数を作成する</strong> - <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h" data-md-type="link"><code data-md-type="codespan">tflite::Interpreter</code></a> を初期化するには、モデルファイルと <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h" data-md-type="link"><code data-md-type="codespan">OpResolver</code></a> が必要です。<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> は、BaseTaskApi インスタンスを作成するためのユーティリティ関数を提供します。</p> <p data-md-type="paragraph">注意: 既定では、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> は <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h" data-md-type="link"><code data-md-type="codespan">BuiltInOpResolver</code></a> を提供します。モデルでカスタマイズされた処理が必要な場合、またはビルトイン処理のサブセットが必要な場合は、<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h" data-md-type="link"><code data-md-type="codespan">MutableOpResolver</code></a> を作成して、登録することができます。</p> <p data-md-type="paragraph">モデルに関連付けられたファイルもすべて提供する必要があります。例: <code data-md-type="codespan">BertQuestionAnswerer</code> では、トークナイザの字句解析用のファイルを追加することもできます。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
                              std::vector&lt;QaAnswer&gt;, // OutputType
                              const std::string&amp;, const std::string&amp; // InputTypes
                              &gt; {
  // Factory function to create the API instance
  StatusOr&lt;std::unique_ptr&lt;QuestionAnswerer&gt;&gt;
  BertQuestionAnswerer::CreateBertQuestionAnswerer(
      const std::string&amp; path_to_model, // model to passed to TaskApiFactory
      const std::string&amp; path_to_vocab  // additional model specific files
  ) {
    // Creates an API object by calling one of the utils from TaskAPIFactory
    std::unique_ptr&lt;BertQuestionAnswerer&gt; api_to_init;
    ASSIGN_OR_RETURN(
        api_to_init,
        core::TaskAPIFactory::CreateFromFile&lt;BertQuestionAnswerer&gt;(
            path_to_model,
            absl::make_unique&lt;tflite::ops::builtin::BuiltinOpResolver&gt;(),
            kNumLiteThreads));

    // Perform additional model specific initializations
    // In this case building a vocabulary vector from the vocab file.
    api_to_init-&gt;InitializeVocab(path_to_vocab);
    return api_to_init;
  }
}
</pre> </li> </ul> <h>Android API</h> <p data-md-type="paragraph">Java/Kotlin インターフェイスを定義し、JNI 経由でロジックを C ++ レイヤーにデリゲートして、Android API を作成します。初めて Android API を構築するときには、ネイティブ API が必要です。</p> <h>使用例</h> <p data-md-type="paragraph">次に、<a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> で Java <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> を使用する例を示します。</p> <pre data-md-type="block_code" data-md-language="java">  String BERT_MODEL_FILE = "path/to/model.tflite";
  String VOCAB_FILE = "path/to/vocab.txt";
  // Create the API from a model file and vocabulary file
    BertQuestionAnswerer bertQuestionAnswerer =
        BertQuestionAnswerer.createBertQuestionAnswerer(
            ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE, VOCAB_FILE);

  String CONTEXT = ...; // context of a question to be answered
  String QUESTION = ...; // question to be answered
  // ask a question
  List&lt;QaAnswer&gt; answers = bertQuestionAnswerer.answer(CONTEXT, QUESTION);
  // answers.get(0).text is the best answer
</pre> <h>API の構築</h> <div data-md-type="block_html"><div align="center">![android_task_api](images/android_task_api.svg) <div align="center">図 3. Android Task API <div align="left"> </div> <p data-md-type="paragraph">ネイティブ API のように、API オブジェクトを構築するには、クライアントで、すべての Java Task API の JNI 処理を提供する、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a> を拡張して、次の情報を提供する必要があります。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">API I/O の決定</strong> - 通常、これはネイティブインターフェイスを反映します。例: <code data-md-type="codespan">BertQuestionAnswerer</code> は <code data-md-type="codespan">(String context, String question)</code> を入力して受け取り、<code data-md-type="codespan">List&lt;QaAnswer&gt;</code> を出力します。実装は、C ++ から返されたポインタである追加の <code data-md-type="codespan">long nativeHandle</code> パラメータがある点を除き、類似したシグネチャを使用してプライベートネイティブ関数を呼び出します。</p> <pre data-md-type="block_code" data-md-language="java">class BertQuestionAnswerer extends BaseTaskApi {
  public List&lt;QaAnswer&gt; answer(String context, String question) {
    return answerNative(getNativeHandle(), context, question);
  }

  private static native List&lt;QaAnswer&gt; answerNative(
                                        long nativeHandle, // C++ pointer
                                        String context, String question // API I/O
                                       );

}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">API のクラスファクトリ関数</strong> - これは、Android ファクトリ関数ではファイルアクセスの <a href="https://developer.android.com/reference/android/content/Context" data-md-type="link"><code data-md-type="codespan">Context</code></a> を取得する必要があるという点を除き、ネイティブファクトリ関数を反映します。この実装は、<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java" data-md-type="link"><code data-md-type="codespan">TaskJniUtils</code></a> のユーティリティのいずれかを呼び出し、対応する C ++ API オブジェクトを呼び出して、そのポインタを <code data-md-type="codespan">BaseTaskApi</code> コンストラクタに渡します。</p> <pre data-md-type="block_code" data-md-language="java">  class BertQuestionAnswerer extends BaseTaskApi {
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
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">ネイティブ関数の JNI モジュールの実装</strong> - すべての Java ネイティブメソッドは、JNI モジュールから対応するネイティブ関数を呼び出して実装されます。ファクトリ関数はネイティブ API オブジェクトを作成し、Long 型としてポインタを Java に返します。後から Java API を呼び出すときに、Long 型ポインタが JNI に戻され、ネイティブ API オブジェクトに渡されます。ネイティブ API 結果は Java 結果に変換されます。</p> <p data-md-type="paragraph">たとえば、これは <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc" data-md-type="link">bert_question_answerer_jni</a> の実装方法です。</p> <pre data-md-type="block_code" data-md-language="cpp">  // Implements BertQuestionAnswerer::initJniWithBertByteBuffers
  extern "C" JNIEXPORT jlong JNICALL
  Java_org_tensorflow_lite_task_text_qa_BertQuestionAnswerer_initJniWithBertByteBuffers(
      JNIEnv* env, jclass thiz, jobjectArray model_buffers) {
    // Convert Java ByteBuffer object into a buffer that can be read by native factory functions
    absl::string_view model =
        GetMappedFileBuffer(env, env-&gt;GetObjectArrayElement(model_buffers, 0));

    // Creates the native API object
    absl::StatusOr&lt;std::unique_ptr&lt;QuestionAnswerer&gt;&gt; status =
        BertQuestionAnswerer::CreateFromBuffer(
            model.data(), model.size());
    if (status.ok()) {
      // converts the object pointer to jlong and return to Java.
      return reinterpret_cast&lt;jlong&gt;(status-&gt;release());
    } else {
      return kInvalidPointer;
    }
  }

  // Implements BertQuestionAnswerer::answerNative
  extern "C" JNIEXPORT jobject JNICALL
  Java_org_tensorflow_lite_task_text_qa_BertQuestionAnswerer_answerNative(
  JNIEnv* env, jclass thiz, jlong native_handle, jstring context, jstring question) {
  // Convert long to native API object pointer
  QuestionAnswerer* question_answerer = reinterpret_cast&lt;QuestionAnswerer*&gt;(native_handle);

  // Calls the native API
  std::vector&lt;QaAnswer&gt; results = question_answerer-&gt;Answer(JStringToString(env, context),
                                         JStringToString(env, question));

  // Converts native result(std::vector&lt;QaAnswer&gt;) to Java result(List&lt;QaAnswerer&gt;)
  jclass qa_answer_class =
    env-&gt;FindClass("org/tensorflow/lite/task/text/qa/QaAnswer");
  jmethodID qa_answer_ctor =
    env-&gt;GetMethodID(qa_answer_class, "&lt;init&gt;", "(Ljava/lang/String;IIF)V");
  return ConvertVectorToArrayList&lt;QaAnswer&gt;(
    env, results,
    [env, qa_answer_class, qa_answer_ctor](const QaAnswer&amp; ans) {
      jstring text = env-&gt;NewStringUTF(ans.text.data());
      jobject qa_answer =
          env-&gt;NewObject(qa_answer_class, qa_answer_ctor, text, ans.pos.start,
                         ans.pos.end, ans.pos.logit);
      env-&gt;DeleteLocalRef(text);
      return qa_answer;
    });
  }

  // Implements BaseTaskApi::deinitJni by delete the native object
  extern "C" JNIEXPORT void JNICALL Java_task_core_BaseTaskApi_deinitJni(
      JNIEnv* env, jobject thiz, jlong native_handle) {
    delete reinterpret_cast&lt;QuestionAnswerer*&gt;(native_handle);
  }
</pre> </li> </ul> <h>iOS API</h> <p data-md-type="paragraph">ネイティブ API オブジェクトを ObjC API オブジェクトに隠蔽して、iOS API を作成します。作成された API オブジェクトは、ObjC または Swift で使用できます。iOS API では、最初にネイティブ API を構築する必要があります。</p> <h>サンプルの使用方法</h> <p data-md-type="paragraph">ObjC の使用例を示します。<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h" data-md-type="link">Swift の </a><a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> の <code data-md-type="codespan">TFLBertQuestionAnswerer</code>。</p> <pre data-md-type="block_code" data-md-language="swift">  static let mobileBertModelPath = "path/to/model.tflite";
  // Create the API from a model file and vocabulary file
  let mobileBertAnswerer = TFLBertQuestionAnswerer.mobilebertQuestionAnswerer(
      modelPath: mobileBertModelPath)

  static let context = ...; // context of a question to be answered
  static let question = ...; // question to be answered
  // ask a question
  let answers = mobileBertAnswerer.answer(
      context: TFLBertQuestionAnswererTest.context, question: TFLBertQuestionAnswererTest.question)
  // answers.[0].text is the best answer
</pre> <h>API の構築</h> <div data-md-type="block_html"><div align="center">![ios_task_api](images/ios_task_api.svg) <div align="center">図 4. iOS Task API <div align="left"> </div> <p data-md-type="paragraph">iOS API はネイティブ API の上の ObjC ラッパーです。次のステップに従って、API を構築します。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">ObjC ラッパーの定義</strong> - ObjC クラスを定義し、実装を対応するネイティブ API オブジェクトにデリゲートします。Swift は C ++との相互運用性がないため、ネイティブ依存関係は .mm ファイルでのみ表示できます。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.h ファイル</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  @interface TFLBertQuestionAnswerer : NSObject

  // Delegate calls to the native BertQuestionAnswerer::CreateBertQuestionAnswerer
  + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString*)modelPath
                                                vocabPath:(NSString*)vocabPath
      NS_SWIFT_NAME(mobilebertQuestionAnswerer(modelPath:vocabPath:));

  // Delegate calls to the native BertQuestionAnswerer::Answer
  - (NSArray&lt;TFLQAAnswer*&gt;*)answerWithContext:(NSString*)context
                                     question:(NSString*)question
      NS_SWIFT_NAME(answer(context:question:));
}
</pre> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.mm ファイル</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  using BertQuestionAnswererCPP = ::tflite::task::text::BertQuestionAnswerer;

  @implementation TFLBertQuestionAnswerer {
    // define an iVar for the native API object
    std::unique_ptr&lt;QuestionAnswererCPP&gt; _bertQuestionAnswerwer;
  }

  // Initialize the native API object
  + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString *)modelPath
                                          vocabPath:(NSString *)vocabPath {
    absl::StatusOr&lt;std::unique_ptr&lt;QuestionAnswererCPP&gt;&gt; cQuestionAnswerer =
        BertQuestionAnswererCPP::CreateBertQuestionAnswerer(MakeString(modelPath),
                                                            MakeString(vocabPath));
    _GTMDevAssert(cQuestionAnswerer.ok(), @"Failed to create BertQuestionAnswerer");
    return [[TFLBertQuestionAnswerer alloc]
        initWithQuestionAnswerer:std::move(cQuestionAnswerer.value())];
  }

  // Calls the native API and converts C++ results into ObjC results
  - (NSArray&lt;TFLQAAnswer *&gt; *)answerWithContext:(NSString *)context question:(NSString *)question {
    std::vector&lt;QaAnswerCPP&gt; results =
      _bertQuestionAnswerwer-&gt;Answer(MakeString(context), MakeString(question));
    return [self arrayFromVector:results];
  }
}
</pre> </li> </ul> </div>
</div></div>
</div>
</div></div>
</div>
</div></div>
</div>
</div>

# 构建您自己的任务 API

<a href="overview.md">TensorFlow Lite Task Library</a> 在抽象 TensorFlow 的相同基础架构上提供了预构建的原生/Android/iOS API。如果现有的 Task 库不支持您的模型，您可以扩展 Task API 基础架构来构建自定义 API。

## 概述

Task API 基础架构有两层结构：底部的 C++ 层封装了原生的 TFLite 运行时，顶部的 Java/ObjC 层通过 JNI 或原生封装容器与 C++ 层通信。

只用 C++ 实现所有 TensorFlow 逻辑，能够最大限度地降低成本，最大限度地提高推断性能，并简化跨平台的整体工作流。

要创建 Task 类，请扩展 [BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) 来提供 TFLite 模型接口和 Task API 接口之间的转换逻辑，然后使用 Java/ObjC 实用工具来创建相应的 API。由于隐藏了所有的 TensorFlow 细节，您可以在没有任何机器学习知识的情况下在应用中部署 TFLite 模型。

TensorFlow Lite 为最热门的<a href="overview.md#supported_tasks">视觉和 NLP 任务</a>提供了一些预构建的 API。您可以使用 Task API 基础架构为其他任务构建自己的 API。

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg) <div align="center">图 1. 预构建的 Task API<div align="left"> </div> <h>使用 Task API 基础架构构建您自己的 API</h> <h>C++ API</h> <p data-md-type="paragraph">所有 TFLite 详细信息都在原生 API 中实现。使用一个工厂函数创建 API 对象，并通过调用在接口中定义的函数获取模型结果。</p> <h>示例用法</h> <p data-md-type="paragraph">下面是一个将 C++ <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> 用于 <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> 的示例。</p> <pre data-md-type="block_code" data-md-language="cpp">  char kBertModelPath[] = "path/to/model.tflite";
  // Create the API from a model file
  std::unique_ptr&lt;BertQuestionAnswerer&gt; question_answerer =
      BertQuestionAnswerer::CreateFromFile(kBertModelPath);

  char kContext[] = ...; // context of a question to be answered
  char kQuestion[] = ...; // question to be answered
  // ask a question
  std::vector&lt;QaAnswer&gt; answers = question_answerer.Answer(kContext, kQuestion);
  // answers[0].text is the best answer
</pre> <h>构建 API</h> <div data-md-type="block_html"><div align="center">![native_task_api](images/native_task_api.svg) <div align="center">图 2. 原生 Task API <div align="left"> </div> <p data-md-type="paragraph">要构建 API 对象，您必须通过扩展 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a> 提供以下信息</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">确定 API I/O</strong> - 您的 API 应在不同的平台之间公开相似的输入/输出。例如，<code data-md-type="codespan">BertQuestionAnswerer</code> 将两个字符串 <code data-md-type="codespan">(std::string&amp; context, std::string&amp; question)</code> 作为输入，并以 <code data-md-type="codespan">std::vector&lt;QaAnswer&gt;</code> 形式输出可能的回答和概率的矢量。为此，您需要在 <code data-md-type="codespan">BaseTaskApi</code> 的[模板参数](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;")中指定对应类型。指定模板参数后，[<code data-md-type="codespan">BaseTaskApi::Infer</code>](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") 函数将具有正确的输入/输出类型。此函数可以由 API 客户端直接调用，但是最好将其封装在模型特定的函数中，在本例中为 <code data-md-type="codespan">BertQuestionAnswerer::Answer</code>。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
                              std::vector&lt;QaAnswer&gt;, // OutputType
                              const std::string&amp;, const std::string&amp; // InputTypes
                              &gt; {
  // Model specific function delegating calls to BaseTaskApi::Infer
  std::vector&lt;QaAnswer&gt; Answer(const std::string&amp; context, const std::string&amp; question) {
    return Infer(context, question).value();
  }
}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">提供 API I/O 与模型输入/输出张量之间的转换逻辑</strong> - 指定输入和输出类型后，子类还需要实现类型函数 <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Preprocess</code></a> 和 <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Postprocess</code></a>。这两个函数从 TFLite <code data-md-type="codespan">FlatBuffer</code> 提供<a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1007" data-md-type="link">输入</a>和<a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008" data-md-type="link">输出</a>。子类负责将值从 API I/O 分配给 I/O 张量。请参阅 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> 中的完整实现示例。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
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
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">创建 API 的工厂函数</strong> - 初始化 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h" data-md-type="link"><code data-md-type="codespan">tflite::Interpreter</code></a> 需要模型文件和 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h" data-md-type="link"><code data-md-type="codespan">OpResolver</code></a>。<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> 提供了创建 BaseTaskApi 实例的效用函数。</p> <p data-md-type="paragraph">注：默认情况下，<a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> 提供了 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h" data-md-type="link"><code data-md-type="codespan">BuiltInOpResolver</code></a>。如果您的模型需要自定义运算或一部分内置内置运算，您可以通过创建一个 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h" data-md-type="link"><code data-md-type="codespan">MutableOpResolver</code></a> 来注册它们。</p> <p data-md-type="paragraph">您还必须提供与模型关联的任何文件。例如，<code data-md-type="codespan">BertQuestionAnswerer</code> 也可以为其分词器的词汇额外使用一个文件。</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
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
</pre> </li> </ul> <h>Android API</h> <p data-md-type="paragraph">定义 Java/Kotlin 接口并通过 JNI 将逻辑委托给 C++ 层以创建 Android API。Android API 要求先构建原生 API。</p> <h>示例用法</h> <p data-md-type="paragraph">下面是一个将 Java <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> 用于 <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> 的示例。</p> <pre data-md-type="block_code" data-md-language="java">  String BERT_MODEL_FILE = "path/to/model.tflite";
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
</pre> <h>构建 API</h> <div data-md-type="block_html"><div align="center">![android_task_api](images/android_task_api.svg) <div align="center">图 3. Android Task API <div align="left"> </div> <p data-md-type="paragraph">与原生 API 类似，要构建 API 对象，客户端也需要通过扩展 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a> 提供以下信息，后者为所有 Java Task API 提供了 JNI 处理。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">确定 API I/O</strong> - 这通常可以镜像原生接口，例如 <code data-md-type="codespan">BertQuestionAnswerer</code> 将 <code data-md-type="codespan">(String context, String question)</code> 作为输入，并输出 <code data-md-type="codespan">List&lt;QaAnswer&gt;</code>。该实现调用了一个具有类似签名的私有原生函数，但该函数具有一个额外的参数 <code data-md-type="codespan">long nativeHandle</code>，这是从 C++ 返回的指针。</p> <pre data-md-type="block_code" data-md-language="java">class BertQuestionAnswerer extends BaseTaskApi {
  public List&lt;QaAnswer&gt; answer(String context, String question) {
    return answerNative(getNativeHandle(), context, question);
  }

  private static native List&lt;QaAnswer&gt; answerNative(
                                        long nativeHandle, // C++ pointer
                                        String context, String question // API I/O
                                       );

}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">创建 API 的工厂函数</strong> - 这也会镜像原生工厂函数，但获取文件访问的 <a href="https://developer.android.com/reference/android/content/Context" data-md-type="link"><code data-md-type="codespan">Context</code></a> 也需要 Android  工厂函数。该实现调用了 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java" data-md-type="link"><code data-md-type="codespan">TaskJniUtils</code></a> 中的一个实用工具来构建对应的 C++ API 对象并将其指针传递给 <code data-md-type="codespan">BaseTaskApi</code> 构造函数。</p> <pre data-md-type="block_code" data-md-language="java">  class BertQuestionAnswerer extends BaseTaskApi {
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
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">为原生函数实现 JNI 模块</strong> - 所有 Java 原生方法均通过从 JNI 模块调用对应的原生函数来实现。工厂函数将创建一个原生 API 对象并将其指针作为长类型返回给 Java。在对 Java API 的后续调用中，长类型指针将被传回 JNI 并转换回原生 API 对象。随后，原生 API 结果会被转换回 Java 结果。</p> <p data-md-type="paragraph">例如，以下是 <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc" data-md-type="link">bert_question_answerer_jni</a> 的实现方式。</p> <pre data-md-type="block_code" data-md-language="cpp">  // Implements BertQuestionAnswerer::initJniWithBertByteBuffers
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
</pre> </li> </ul> <h>iOS API</h> <p data-md-type="paragraph">通过将原生 API 对象封装到 ObjC API 对象中来创建  iOS API。创建的 API 对象可以用于 ObjC 或 Swift。iOS API 要求先构建原生 API。</p> <h>Sample usage</h> <p data-md-type="paragraph">下面是一个在 Swift 中将 ObjC <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h" data-md-type="link"><code data-md-type="codespan">TFLBertQuestionAnswerer</code></a> 用于 <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> 的示例。</p> <pre data-md-type="block_code" data-md-language="swift">  static let mobileBertModelPath = "path/to/model.tflite";
  // Create the API from a model file and vocabulary file
  let mobileBertAnswerer = TFLBertQuestionAnswerer.mobilebertQuestionAnswerer(
      modelPath: mobileBertModelPath)

  static let context = ...; // context of a question to be answered
  static let question = ...; // question to be answered
  // ask a question
  let answers = mobileBertAnswerer.answer(
      context: TFLBertQuestionAnswererTest.context, question: TFLBertQuestionAnswererTest.question)
  // answers.[0].text is the best answer
</pre> <h>构建 API</h> <div data-md-type="block_html"><div align="center">![ios_task_api](images/ios_task_api.svg) <div align="center">图 4. iOS Task API <div align="left"> </div> <p data-md-type="paragraph">iOS API 是一个基于原生 API 的简单 ObjC 封装容器。按照以下步骤操作来构建 API：</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">定义 ObjC 封装容器</strong> - 定义一个 ObjC 类并将实现委托给对应的原生 API 对象。请注意，由于 Swift 无法与 C++ 互操作，原生依赖项仅可以出现在 .mm 文件中。</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.h 文件</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  @interface TFLBertQuestionAnswerer : NSObject

  // Delegate calls to the native BertQuestionAnswerer::CreateBertQuestionAnswerer
  + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString*)modelPath
                                                vocabPath:(NSString*)vocabPath
      NS_SWIFT_NAME(mobilebertQuestionAnswerer(modelPath:vocabPath:));

  // Delegate calls to the native BertQuestionAnswerer::Answer
  - (NSArray&lt;TFLQAAnswer*&gt;*)answerWithContext:(NSString*)context
                                     question:(NSString*)question
      NS_SWIFT_NAME(answer(context:question:));
}
</pre> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.mm 文件</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  using BertQuestionAnswererCPP = ::tflite::task::text::qa::BertQuestionAnswerer;

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

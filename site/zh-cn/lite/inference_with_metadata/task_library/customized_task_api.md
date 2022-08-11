# 构建您自己的任务 API

<a href="overview.md">TensorFlow Lite Task Library</a> 在抽象 TensorFlow 的相同基础架构上提供了预构建的原生/Android/iOS API。如果现有的 Task 库不支持您的模型，您可以扩展 Task API 基础架构来构建自定义 API。

## 概述

Task API 基础架构有两层结构：底部的 C++ 层封装了原生的 TFLite 运行时，顶部的 Java/ObjC 层通过 JNI 或原生封装容器与 C++ 层通信。

只用 C++ 实现所有 TensorFlow 逻辑，能够最大限度地降低成本，最大限度地提高推断性能，并简化跨平台的整体工作流。

要创建 Task 类，请扩展 [BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h) 来提供 TFLite 模型接口和 Task API 接口之间的转换逻辑，然后使用 Java/ObjC 实用工具来创建相应的 API。由于隐藏了所有的 TensorFlow 细节，您可以在没有任何机器学习知识的情况下在应用中部署 TFLite 模型。

TensorFlow Lite 为最热门的<a href="overview.md#supported_tasks">视觉和 NLP 任务</a>提供了一些预构建的 API。您可以使用 Task API 基础架构为其他任务构建自己的 API。

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg) <div align="center">Figure 1. prebuilt Task APIs <div align="left"> </div> <h>Build your own API with Task API infra</h> <h>C++ API</h> <p data-md-type="paragraph">All TFLite details are implemented in the native API. Create an API object by using one of the factory functions and get model results by calling functions defined in the interface.</p> <h>Sample usage</h> <p data-md-type="paragraph">Here is an example using the C++ <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> for <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a>.</p> <pre data-md-type="block_code" data-md-language="cpp">  char kBertModelPath[] = "path/to/model.tflite";
  // Create the API from a model file
  std::unique_ptr&lt;BertQuestionAnswerer&gt; question_answerer =
      BertQuestionAnswerer::CreateFromFile(kBertModelPath);

  char kContext[] = ...; // context of a question to be answered
  char kQuestion[] = ...; // question to be answered
  // ask a question
  std::vector&lt;QaAnswer&gt; answers = question_answerer.Answer(kContext, kQuestion);
  // answers[0].text is the best answer
</pre> <h>Building the API</h> <div data-md-type="block_html"><div align="center">![native_task_api](images/native_task_api.svg) <div align="center">Figure 2. Native Task API <div align="left"> </div> <p data-md-type="paragraph">To build an API object,you must provide the following information by extending <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a></p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Determine the API I/O</strong> - Your API should expose similar input/output across different platforms. e.g. <code data-md-type="codespan">BertQuestionAnswerer</code> takes two strings <code data-md-type="codespan">(std::string&amp; context, std::string&amp; question)</code> as input and outputs a vector of possible answer and probabilities as <code data-md-type="codespan">std::vector&lt;QaAnswer&gt;</code>. This is done by specifying the corresponding types in <code data-md-type="codespan">BaseTaskApi</code>'s [template parameter](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;"). With the template parameters specified, the [<code data-md-type="codespan">BaseTaskApi::Infer</code>](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") function will have the correct input/output types. This function can be directly called by API clients, but it is a good practice to wrap it inside a model-specific function, in this case, <code data-md-type="codespan">BertQuestionAnswerer::Answer</code>.</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
                              std::vector&lt;QaAnswer&gt;, // OutputType
                              const std::string&amp;, const std::string&amp; // InputTypes
                              &gt; {
  // Model specific function delegating calls to BaseTaskApi::Infer
  std::vector&lt;QaAnswer&gt; Answer(const std::string&amp; context, const std::string&amp; question) {
    return Infer(context, question).value();
  }
}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Provide conversion logic between API I/O and input/output tensor of the model</strong> - With input and output types specified, the subclasses also need to implement the typed functions <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Preprocess</code></a> and <a href="https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80" data-md-type="link"><code data-md-type="codespan">BaseTaskApi::Postprocess</code></a>. The two functions provide <a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1007" data-md-type="link">inputs</a> and <a href="https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008" data-md-type="link">outputs</a> from the TFLite <code data-md-type="codespan">FlatBuffer</code>. The subclass is responsible for assigning values from the API I/O to I/O tensors. See the complete implementation example in <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a>.</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
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
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Create factory functions of the API</strong> - A model file and a <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h" data-md-type="link"><code data-md-type="codespan">OpResolver</code></a> are needed to initialize the <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h" data-md-type="link"><code data-md-type="codespan">tflite::Interpreter</code></a>. <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> provides utility functions to create BaseTaskApi instances.</p> <p data-md-type="paragraph">Note: By default <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h" data-md-type="link"><code data-md-type="codespan">TaskAPIFactory</code></a> provides a <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h" data-md-type="link"><code data-md-type="codespan">BuiltInOpResolver</code></a>. If your model needs customized ops or a subset of built-in ops, you can register them by creating a <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h" data-md-type="link"><code data-md-type="codespan">MutableOpResolver</code></a>.</p> <p data-md-type="paragraph">You must also provide any files associated with the model. e.g, <code data-md-type="codespan">BertQuestionAnswerer</code> can also have an additional file for its tokenizer's vocabulary.</p> <pre data-md-type="block_code" data-md-language="cpp">class BertQuestionAnswerer : public BaseTaskApi&lt;
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
</pre> </li> </ul> <h>Android API</h> <p data-md-type="paragraph">Create Android APIs by defining Java/Kotlin interface and delegating the logic to the C++ layer through JNI. Android API requires native API to be built first.</p> <h>Sample usage</h> <p data-md-type="paragraph">Here is an example using Java <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java" data-md-type="link"><code data-md-type="codespan">BertQuestionAnswerer</code></a> for <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a>.</p> <pre data-md-type="block_code" data-md-language="java">  String BERT_MODEL_FILE = "path/to/model.tflite";
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
</pre> <h>Building the API</h> <div data-md-type="block_html"><div align="center">![android_task_api](images/android_task_api.svg) <div align="center">Figure 3. Android Task API <div align="left"> </div> <p data-md-type="paragraph">Similar to Native APIs, to build an API object, the client needs to provide the following information by extending <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java" data-md-type="link"><code data-md-type="codespan">BaseTaskApi</code></a>, which provides JNI handlings for all Java Task APIs.</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Determine the API I/O</strong> - This usually mirrors the native interfaces. e.g <code data-md-type="codespan">BertQuestionAnswerer</code> takes <code data-md-type="codespan">(String context, String question)</code> as input and outputs <code data-md-type="codespan">List&lt;QaAnswer&gt;</code>. The implementation calls a private native function with similar signature, except it has an additional parameter <code data-md-type="codespan">long nativeHandle</code>, which is the pointer returned from C++.</p> <pre data-md-type="block_code" data-md-language="java">class BertQuestionAnswerer extends BaseTaskApi {
  public List&lt;QaAnswer&gt; answer(String context, String question) {
    return answerNative(getNativeHandle(), context, question);
  }

  private static native List&lt;QaAnswer&gt; answerNative(
                                        long nativeHandle, // C++ pointer
                                        String context, String question // API I/O
                                       );

}
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Create factory functions of the API</strong> - This also mirrors native factory functions, except Android factory functions also need to take <a href="https://developer.android.com/reference/android/content/Context" data-md-type="link"><code data-md-type="codespan">Context</code></a> for file access. The implementation calls one of the utilities in <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java" data-md-type="link"><code data-md-type="codespan">TaskJniUtils</code></a> to build the corresponding C++ API object and pass its pointer to the <code data-md-type="codespan">BaseTaskApi</code> constructor.</p> <pre data-md-type="block_code" data-md-language="java">  class BertQuestionAnswerer extends BaseTaskApi {
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
</pre> </li> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Implement the JNI module for native functions</strong> - All Java native methods are implemented by calling a corresponding native function from the JNI module. The factory functions would create a native API object and return its pointer as a long type to Java. In later calls to Java API, the long type pointer is passed back to JNI and cast back to the native API object. The native API results are then converted back to Java results.</p> <p data-md-type="paragraph">For example, this is how <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc" data-md-type="link">bert_question_answerer_jni</a> is implemented.</p> <pre data-md-type="block_code" data-md-language="cpp">  // Implements BertQuestionAnswerer::initJniWithBertByteBuffers
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
</pre> </li> </ul> <h>iOS API</h> <p data-md-type="paragraph">Create iOS APIs by wrapping a native API object into a ObjC API object. The created API object can be used in either ObjC or Swift. iOS API requires the native API to be built first.</p> <h>Sample usage</h> <p data-md-type="paragraph">Here is an example using ObjC <a href="https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h" data-md-type="link"><code data-md-type="codespan">TFLBertQuestionAnswerer</code></a> for <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1" data-md-type="link">MobileBert</a> in Swift.</p> <pre data-md-type="block_code" data-md-language="swift">  static let mobileBertModelPath = "path/to/model.tflite";
  // Create the API from a model file and vocabulary file
  let mobileBertAnswerer = TFLBertQuestionAnswerer.mobilebertQuestionAnswerer(
      modelPath: mobileBertModelPath)

  static let context = ...; // context of a question to be answered
  static let question = ...; // question to be answered
  // ask a question
  let answers = mobileBertAnswerer.answer(
      context: TFLBertQuestionAnswererTest.context, question: TFLBertQuestionAnswererTest.question)
  // answers.[0].text is the best answer
</pre> <h>Building the API</h> <div data-md-type="block_html"><div align="center">![ios_task_api](images/ios_task_api.svg) <div align="center">Figure 4. iOS Task API <div align="left"> </div> <p data-md-type="paragraph">iOS API is a simple ObjC wrapper on top of native API. Build the API by following the steps below:</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="false"> <li data-md-type="list_item" data-md-list-type="unordered"> <p data-md-type="paragraph"><strong data-md-type="double_emphasis">Define the ObjC wrapper</strong> - Define an ObjC class and delegate the implementations to the corresponding native API object. Note the native dependencies can only appear in a .mm file due to Swift's inability to interop with C++.</p> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.h file</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  @interface TFLBertQuestionAnswerer : NSObject

  // Delegate calls to the native BertQuestionAnswerer::CreateBertQuestionAnswerer
  + (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString*)modelPath
                                                vocabPath:(NSString*)vocabPath
      NS_SWIFT_NAME(mobilebertQuestionAnswerer(modelPath:vocabPath:));

  // Delegate calls to the native BertQuestionAnswerer::Answer
  - (NSArray&lt;TFLQAAnswer*&gt;*)answerWithContext:(NSString*)context
                                     question:(NSString*)question
      NS_SWIFT_NAME(answer(context:question:));
}
</pre> <ul data-md-type="list" data-md-list-type="unordered" data-md-list-tight="true"> <li data-md-type="list_item" data-md-list-type="unordered">.mm file</li> </ul> <pre data-md-type="block_code" data-md-language="objc">  using BertQuestionAnswererCPP = ::tflite::task::text::BertQuestionAnswerer;

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

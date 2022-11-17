# 나만의 Task API 구축하기

<a href="overview.md">TensorFlow Lite Task Library</a>는 TensorFlow를 추상화하는 동일한 인프라 위에 사전 빌드된 네이티브/Android/iOS API를 제공합니다. 해당 모델이 기존 작업 라이브러리에서 지원되지 않는 경우, Task API 인프라를 확장하여 사용자 정의 API를 빌드할 수 있습니다.

## 개요

Task API 인프라는 2개 레이어 구조로 되어 있습니다. 즉, 기본 TFLite 런타임을 캡슐화하는 하위 C++ 레이어와 JNI 또는 네이티브 래퍼를 통해 C++ 레이어와 정보를 소통하는 상위 Java/ObjC 레이어입니다.

모든 TensorFlow 로직을 C++로만 구현하면 비용이 최소화되고 추론 성능이 최대화되며 플랫폼 전반에서 전체 워크플로가 단순해집니다.

Task 클래스를 생성하려면 [BaseTaskApi](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h)를 확장하여 TFLite 모델 인터페이스와 Task API 인터페이스 간에 변환 논리를 제공한 다음, Java/ObjC 유틸리티를 사용하여 해당 API를 생성합니다. 모든 TensorFlow 세부 사항들이 숨겨진 상태이므로 머신러닝에 대한 지식이 없어도 앱에 TFLite 모델을 배포할 수 있습니다.

TensorFlow Lite는 주요 <a href="overview.md#supported_tasks">Vision 및 NLP 작업</a>을 위해 사전 빌드된 API를 제공합니다. Task API 인프라를 사용하여 다른 작업을 위한 고유한 API를 빌드할 수 있습니다.

<div align="center">![prebuilt_task_apis](images/prebuilt_task_apis.svg)</div>
<div align="center">그림 1. 사전 빌드된 Task API</div>
<div align="left"></div>

## Task API 인프라로 나만의 API 빌드하기

### C++ API

모든 TFLite 세부 내용은 네이티브 API에서 구현됩니다. 팩터리 함수 중 하나를 사용하여 API 객체를 만들고 인터페이스에 정의된 함수를 호출하여 모델 결과를 가져옵니다.

#### 샘플 사용법

다음은 [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)용 C++ [`BertQuestionAnswerer`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)를 사용하는 예입니다.

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

#### API 구축하기

<div align="center">![native_task_api](images/native_task_api.svg)</div>
<div align="center">그림 2. 네이티브 Task API</div>
<div align="left"></div>

API 객체를 빌드하려면 [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h)를 확장하여 다음 정보를 제공해야 합니다.

- **API I/O 결정** - 해당 API는 여러 플랫폼에 걸쳐 비슷한 입력/출력을 노출해야 합니다. 예를 들어, `BertQuestionAnswerer`는 두 개의 문자열 `(std::string& context, std::string& question)`을 입력으로 받아서 가능한 답변 및 확률 벡터를 `std::vector<QaAnswer>`로 출력합니다. 이를 위해 `BaseTaskApi`의 [템플릿 매개변수](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="template &lt;class OutputType, class... InputTypes&gt;")에서 해당 유형을 지정합니다. 템플릿 매개변수가 지정되면 [`BaseTaskApi::Infer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/base_task_api.h?q="Infer(InputTypes... args)") 함수가 올바른 입력/출력 유형을 갖게 됩니다. 이 함수는 API 클라이언트에서 직접 호출할 수 있지만, 모델에 특정한 함수, 이 경우에는 `BertQuestionAnswerer::Answer` 내에서 이를 래핑하는 것이 좋은 습관입니다.

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

- **API I/O와 모델의 입력/출력 텐서 간에 변환 논리 제공** - 입력 및 출력 유형이 지정되면, 서브 클래스에서 형식화된 함수 [`BaseTaskApi::Preprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L74) 및 [`BaseTaskApi::Postprocess`](https://github.com/tensorflow/tflite-support/blob/5cea306040c40b06d6e0ed4e5baf6c307db7bd00/tensorflow_lite_support/cc/task/core/base_task_api.h#L80)도 구현해야 합니다. 두 함수는 TFLite <code>FlatBuffer</code>에서 [입력](https://github.com/tensorflow/tensorflow/blob/1b84e5af78f85b8d3c4687b7dee65b78113f81cc/tensorflow/lite/schema/schema.fbs#L1008)과 <a>출력</a>을 제공합니다. 이 서브 클래스는 API I/O의 값을 I/O 텐서에 할당하는 역할을 합니다. [`BertQuestionAnswerer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.cc)의 전체 구현 예를 참조하세요.

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

- **API의 팩터리 함수 만들기** - [`tflite::Interpreter`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/api/op_resolver.h)를 초기화하려면 모델 파일과 [`OpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h)가 필요합니다. [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h)는 BaseTaskApi 인스턴스를 생성하는 유틸리티 함수를 제공합니다.

    참고: 기본적으로, [`TaskAPIFactory`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/core/task_api_factory.h)는 [`BuiltInOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/register.h)를 제공합니다. 해당 모델에 사용자 정의 ops 또는 내장 ops의 일부가 필요한 경우, [`MutableOpResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/mutable_op_resolver.h)를 만들어 이러한 ops를 등록할 수 있습니다.

    모델과 관련된 모든 파일도 제공해야 합니다. 예를 들어, `BertQuestionAnswerer`에는 토큰화된 어휘의 추가 파일이 있을 수도 있습니다.

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

Java/Kotlin 인터페이스를 정의하고 JNI를 통해 C++ 레이어에 논리를 위임하여 Android API를 만듭니다. Android API를 사용하려면 먼저 네이티브 API를 빌드해야 합니다.

#### 샘플 사용법

다음은 [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)용 Java [`BertQuestionAnswerer`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)를 사용하는 예입니다.

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

#### API 구축하기

<div align="center">![android_task_api](images/android_task_api.svg)</div>
<div align="center">그림 3. Android Task API</div>
<div align="left"></div>

Native API와 마찬가지로 API 객체를 빌드하려면 클라이언트가 모든 Java Task API에 대한 JNI 처리를 제공하는 [`BaseTaskApi`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/BaseTaskApi.java)를 확장하여 다음 정보를 제공해야 합니다.

- **API I/O 결정** - 일반적으로, 기본 인터페이스를 미러링합니다. 예를 들어, `BertQuestionAnswerer`는 `(String context, String question)`을 입력으로 받아 `List<QaAnswer>`를 출력합니다. 이 구현은 C++에서 반환된 포인터인 추가 매개변수 `long nativeHandle`이 있다는 점을 제외하고 유사한 서명을 사용하여 비공개 네이티브 함수를 호출합니다.

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

- **API의 팩터리 함수 만들기** - Android 팩터리 함수가 파일 액세스를 위해 [`Context`](https://developer.android.com/reference/android/content/Context)도 받아야 한다는 점을 제외하고, 네이티브 팩터리 함수를 미러링합니다. 구현을 위해 [`TaskJniUtils`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core/TaskJniUtils.java)의 유틸리티 중 하나를 호출하여 해당 C++ API 객체를 빌드하고 포인터를 `BaseTaskApi` 생성자로 전달합니다.

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

- **네이티브 함수에 대한 JNI 모듈 구현** -모든 Java 네이티브 메서드는 JNI 모듈에서 해당 네이티브 함수를 호출하여 구현됩니다. 팩터리 함수는 네이티브 API 객체를 생성하고 포인터를 long 형식으로 Java에 반환합니다. Java API에 대한 이후 호출에서 long 형식 포인터는 JNI로 다시 전달되고 네이티브 API 객체로 다시 캐스팅됩니다. 그런 다음, 네이티브 API 결과가 Java 결과로 다시 변환됩니다.

    예를 들어, 다음은 [bert_question_answerer_jni](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/native/task/text/qa/bert_question_answerer_jni.cc)가 어떻게 구현되는지를 보여줍니다.

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

네이티브 API 객체를 ObjC API 개체로 래핑하여 iOS API를 만듭니다. 생성된 API 객체는 ObjC 또는 Swift에서 사용할 수 있습니다. iOS API를 사용하려면 먼저 네이티브 API를 빌드해야 합니다.

#### 샘플 사용법

다음은 [MobileBert](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)용 ObjC [`TFLBertQuestionAnswerer`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1)를 Swift로 사용하는 예입니다.

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

#### API 구축하기

<div align="center">![ios_task_api](images/ios_task_api.svg)</div>
<div align="center">그림 4. iOS Task API</div>
<div align="left"></div>

iOS API는 네이티브 API 상위에 있는 간단한 ObjC 래퍼입니다. 아래 단계에 따라 API를 빌드하세요.

- **ObjC 래퍼 정의** - ObjC 클래스를 정의하고 구현을 해당 네이티브 API 객체에 위임합니다. Swift는 C++와 상호 운용할 수 없기 때문에 기본 종속성은 .mm 파일에만 나타날 수 있습니다.

    - .h 파일

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

    - .mm 파일

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

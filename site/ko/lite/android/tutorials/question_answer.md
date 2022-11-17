# Android로 질문에 답하기

![Android의 질의응답 예제 앱](../../examples/bert_qa/images/screenshot.gif){: .attempt-right width="250px"}

이 튜토리얼은 TensorFlow Lite를 사용하여 Android 애플리케이션을 빌드하는 방법을 제시하여 자연어 텍스트에서 구조화된 질문에 대한 답변을 제공합니다. [예제 애플리케이션](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)은 [자연어(NL)용 작업 라이브러리](../../inference_with_metadata/task_library/overview#supported_tasks)에서 *BERT 질의응답기* ([`BertQuestionAnswerer`](../../inference_with_metadata/task_library/bert_question_answerer)) API를 사용해 질의응답 머신러닝 모델을 활성화합니다. 이 애플리케이션은 실제 Android 기기용으로 설계되었지만 기기 에뮬레이터에서도 실행할 수 있습니다.

기존 프로젝트를 업데이트하는 중이라면 예제 애플리케이션을 참조 또는 템플릿으로 사용할 수 있습니다. 기존 애플리케이션에 질의응답을 추가하는 방법에 대한 지침은 [애플리케이션 업데이트 및 수정](#modify_applications)을 참조하세요.

## 질의응답 개요

*질의응답*은 자연어로 제기된 질의응답에 대한 머신러닝 작업입니다. 훈련된 질의응답 모델은 입력으로 텍스트 구절과 질문을 받고 구절 내의 정보에 대한 해석을 기반으로 질문에 답하려고 시도합니다.

질의응답 모델은 텍스트의 서로 다른 세그먼트를 바탕으로 한 질문-답변 쌍과 함께 독해 데이터세트를 읽는 것으로 구성된 질의응답 데이터세트로 훈련됩니다.

이 튜토리얼의 모델이 생성되는 방법에 대한 더 자세한 정보는 [TensorFlow Lite Model Maker를 통한 BERT Question Answer](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) 튜토리얼을 참조하세요.

## 모델 및 데이터세트

예제 앱은 Mobile BERT Q&amp;A ([`mobilebert`](https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1)) 모델을 사용하며 이는 [BERT](https://arxiv.org/abs/1810.04805)(Bidirectional Encoder Representations from Transformers)의 더 가볍고 빠른 버전입니다. `mobilebert`에 대한 자세한 내용은 [MobileBERT: 리소스가 제한된 기기용 Compact Task-Agnostic BERT](https://arxiv.org/abs/2004.02984) 연구 논문을 참조하세요.

`mobilebert` 모델은 Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)) 데이터세트, Wikipedia 및 각 아티클의 질문-답변 세트의 아티클로 구성된 독해 데이터세트를 사용하여 훈련되었습니다.

## 예제 앱 설치 및 실행

질의응답 애플리케이션을 설치하려면 [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android)에서 예제 앱을 다운로드하고 [Android Studio](https://developer.android.com/studio/)를 사용해 실행합니다.

### 시스템 요구 사항

- **[Android Studio](https://developer.android.com/studio/index.html)** 버전 2021.1.1(Bumblebee) 이상
- Android SDK 버전 31 이상
- [개발자 모드](https://developer.android.com/studio/debug/dev-options)가 활성화된 상태로 최소 OS 버전의 SDK 21(Android 7.0 - Nougat)가 설치된 Android 기기 또는 Android Emulator.

### 예제 코드 가져오기

예제 코드의 로컬 복사본을 만듭니다. 이 코드를 사용하여 Android Studio에서 프로젝트를 만들고 예제 애플리케이션을 실행합니다.

예제 코드를 복제하고 설정하려면:

1. git 리포지토리를 복제합니다.
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 선택적으로, git 인스턴스를 구성하여 sparse checkout을 사용하면 질의응답 예제 앱에 대한 파일만 남게 됩니다.
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/bert_qa/android
        </pre>

### 프로젝트 가져오기 및 실행하기

다운로드한 예제 코드에서 프로젝트를 생성하고 프로젝트를 빌드한 후 실행합니다.

예제 코드 프로젝트를 가져오고 빌드하려면:

1. [Android Studio](https://developer.android.com/studio)를 시작합니다.
2. Android 스튜디오에서 **File(파일) &gt; New(새로 만들기) &gt; Import Project(프로젝트 가져오기)**를 선택합니다.
3. build.gradle 파일(`.../examples/lite/examples/bert_qa/android/build.gradle`)이 포함된 예제 코드 디렉터리로 이동하여 해당 디렉터리를 선택합니다.
4. Android Studio에서 Gradle 동기화를 요청하면 OK(확인)를 선택합니다.
5. 기기가 컴퓨터에 연결되어 있고 개발자 모드가 활성화되어 있는지 확인합니다. 녹색 `Run`(실행) 화살표를 클릭합니다.

올바른 디렉터리를 선택하면 Android Studio에서 새 프로젝트를 만들고 빌드합니다. 이 프로세스는 컴퓨터 속도와 다른 프로젝트에 Android Studio를 사용했는지 여부에 따라 몇 분이 소요될 수 있습니다. 빌드가 완료되면 Android Studio가 **빌드 출력** 상태 패널에 `BUILD SUCCESSFUL` 메시지를 표시합니다.

프로젝트를 실행하려면:

1. Android Studio에서 **Run(실행) &gt; Run(실행)…**을 선택하여 프로젝트를 실행합니다.
2. 연결된 Android 기기(또는 에뮬레이터)를 선택하여 앱을 테스트합니다.

### 애플리케이션 사용하기

Android Studio에서 프로젝트를 실행한 후, 애플리케이션은 연결된 기기 또는 기기 에뮬레이터에서 자동으로 열립니다.

질의응답 앱을 사용하려면:

1. 주제 목록에서 화제를 선택합니다.
2. 제안된 질문을 선택하거나 직접 텍스트 박스에 입력합니다.
3. 주황색 화살표를 토글하여 모델을 실행합니다.

애플리케이션은 구절 텍스트에서 질문에 대한 답변을 식별하려고 시도합니다. 모델이 구절 내에서 답변을 감지하면 애플리케이션은 사용자를 위해 텍스트의 관련 범위를 강조합니다.

이제 작동하는 질의응답 애플리케이션이 있습니다. 예제 애플리케이션이 작동하는 법과 프로덕션 애플리케이션에서 질의응답 기능을 구현하는 법에 대해 더 잘 이해하기 위해 다음 섹션을 사용하세요.

- [애플리케이션 작동 방식](#how_it_works) - 예제 애플리케이션의 구조 및 키 파일 연습입니다.

- [애플리케이션 수정](#modify_applications) - 기존의 애플리케이션에 질의 응답을 추가하는 것에 대한 지침입니다.

## 예재 앱 작동 방법 {:#how_it_works}

애플리케이션은 [자연어(NL)용 작업 라이브러리](../../inference_with_metadata/task_library/overview#supported_tasks) 패키지 내의 `BertQuestionAnswerer` API를 사용합니다. MobileBERT 모델은 TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer)를 사용하여 훈련되었습니다. 이 애플리케이션은 GPU 또는 NNAPI 대리자를 사용하는 하드웨어 가속화 옵션과 함께 기본으로 CPU에서 실행됩니다.

다음 파일과 디렉터리는 이 애플리케이션에 대한 주요 코드를 포함합니다.

- [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt) - 질의응답기를 초기화하고 모델 및 대리자 선택을 처리합니다.
- [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt) - 결과를 처리하고 포맷합니다.
- [MainActivity.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/MainActivity.kt) - 앱의 구성 논리를 제공합니다.

## 애플리케이션 수정 {:#modify_applications}

다음 섹션은 예제 앱에 표시된 모델을 실행하도록 자신의 Android 앱을 수정하는 주요 단계를 설명합니다. 이러한 지침은 예제 앱을 참조 포인트로 사용합니다. 자신의 앱에 필요한 특정 변경 사항은 예제 앱과 다를 수 있습니다.

### Android 프로젝트 열기 또는 만들기

나머지 지침을 따르려면 Android Studio의 Android 개발 프로젝트가 필요합니다. 아래 지침을 따라 기존의 프로젝트를 열거나 새로운 프로젝트를 만드세요.

기존 Android 개발 프로젝트 열기

- Android Studio에서 *File(파일) &gt; Open(열기)*를 선택하고 기존 프로젝트를 선택합니다.

기본 Android 개발 프로젝트 만들기

- Android Studio의 지침을 따라 [기본 프로젝트를 만듭니다](https://developer.android.com/studio/projects/create-project).

Android Studio를 사용에 대한 더 자세한 정보는 [Android Studio 설명서](https://developer.android.com/studio/intro)를 참조하세요.

### 프로젝트 종속성 추가

자신의 애플리케이션에서 특정 프로젝트 종속성을 추가하여 TensorFlow Lite 머신러닝 모델을 실행하고 유틸리티 함수에 액세스합니다. 이러한 함수는 문자열과 같은 데이터를 모델이 프로세스 할 수 있는 텐서 데이터 형식으로 변환합니다. 다음 지침은 요구되는 프로젝트와 모듈 종속성을 자신의 Android 앱 프로젝트에 추가하는 방법을 설명합니다.

모듈 종속성을 추가하려면:

1. TensorFlow Lite를 사용하는 모듈에서 모듈의 `build.gradle` 파일을 다음 종속성을 포함하도록 업데이트하세요.

    예제 애플리케이션에서, 종속성은 [app/build.gradle](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/build.gradle)에 있습니다.

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

    프로젝트는 텍스트 작업 라이브러리(`tensorflow-lite-task-text`)를 포함해야 합니다.

    이 앱을 수정하여 그래픽 처리 장치(GPU)에서 실행하길 원한다면 GPU 라이브러리(`tensorflow-lite-gpu-delegate-plugin`)는 인프라를 제공하여 GPU에서 앱을 실행하고 대리자(`tensorflow-lite-gpu`)는 호환성 목록을 제공합니다.

2. Android Studio에서 **File(파일) &gt; Sync Project with Gradle Files(프로젝트를 Gradle 파일과 동기화)**를 선택하여 프로젝트 종속성을 동기화합니다.

### ML 모델 초기화하기 {:#initialize_models}

Android 앱에서, 모델로 예측을 실행하기 전에 매개변수를 사용해 TensorFlow Lite 머신러닝 모델을 초기화해야 합니다.

TensorFlow Lite 모델은 `*.tflite` 파일로 저장됩니다. 모델 파일은 예측 논리를 포함하며 일반적으로 예측 결과를 해거하는 방법에 대한 [메타데이터](../../models/convert/metadata)를 포함합니다. 일반적으로, 모델 파일은 코드 예제에서와 같이 개발 프로젝트의 `src/main/assets` 디렉터리에 저장됩니다.

- `<project>/src/main/assets/mobilebert_qa.tflite`

참고: 이 예제 앱은 [`download_model.gradle`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/download_models.gradle) 파일을 사용하여 빌드 시 [mobilebert_qa](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer) 모델과 [구절 텍스트](https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/contents_from_squad.json)를 다운로드받습니다. 이 접근 방식은 프로덕션 앱에는 필요하지 않습니다.

편의와 코드 가독성을 위해 이 예제에서는 모델에 대한 설정을 정의하는 컴패니언 객체를 선언합니다.

앱에서 모델을 초기화하려면:

1. 컴패니언 객체를 만들어 모델에 대한 설정을 정의하세요. 예제 애플리케이션에서 이 객체는 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L100-L106)에 있습니다.

    ```
    companion object {
        private const val BERT_QA_MODEL = "mobilebert.tflite"
        private const val TAG = "BertQaHelper"
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
    }
    ```

2. `BertQaHelper` 객체를 빌드하여 모델에 대한 설정을 만들고 `bertQuestionAnswerer`로 TensorFlow Lite 객체를 구성합니다.

    예제 애플리케이션에서, 이것은 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L41-L76)의 `setupBertQuestionAnswerer()` 함수에 있습니다.

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

### 하드웨어 가속화 사용(선택 사항) {:#hardware_acceleration}

앱에서 TensorFlow Lite 모델을 초기화할 때 하드웨어 가속화 기능을 사용하여 모델의 예측 계산 속도를 높이는 방식을 고려해야 합니다. TensorFlow Lite [대리자](https://www.tensorflow.org/lite/performance/delegates)는 GPU(그래픽 처리 장치) 또는 TPU(텐서 처리 장치)와 같은 모바일 기기의 특수 처리 하드웨어를 사용하여 머신러닝 모델의 실행을 가속하는 소프트웨어 모듈입니다.

앱에서 하드웨어 가속화 사용하기

1. 변수를 만들어 애플리케이션이 사용할 대리자를 정의합니다. 예제 애플리케이션에서, 이 변수는 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L31)의 초기에 있습니다.

    ```
    var currentDelegate: Int = 0
    ```

2. 대리자 선택자를 만듭니다. 예제 애플리케이션에서, 대리자 선택자는 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L48-L62)의 `setupBertQuestionAnswerer` 함수에 있습니다.

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

TensorFlow Lite 모델 실행에 대리자를 사용하는 것이 권장되지만 필수는 아닙니다. TensorFlow Lite에서 대리자를 사용하는 방법에 대한 자세한 내용은 [TensorFlow Lite 대리자](https://www.tensorflow.org/lite/performance/delegates)를 참조하세요.

### 모델에 대한 데이터 준비하기

Android 앱에서, 코드는 원시 데이터와 같은 기존 데이터를 모델이 처리할 수 있는 [텐서](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) 데이터 형식으로 변환하여 해석을 위해 데이터를 모델에 제공합니다. 모델에 전달하는 텐서는 반드시 모델을 훈련하는 데 사용된 데이터 형식과 일치하는 특정 차원 또는 형상이어야 합니다. 이 질의응답 앱은 텍스트 구절 및 질문 모두에 대한 입력으로 [문자열](https://developer.android.com/reference/java/lang/String.html)을 받아들입니다. 모델은 특정 글자 및 비영어 단어를 인식하지 않습니다.

모델에 구절 텍스트 데이터 제공하기

1. `LoadDataSetClient` 객체를 사용하여 앱에 구절 텍스트 데이터를 로드합니다. 예제 애플리케이션에서, 이것은 [LoadDataSetClient.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/dataset/LoadDataSetClient.kt#L25-L45)에 있습니다.

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

2. `DatasetFragment` 객체를 사용하여 텍스트 각 구절에 대한 제목을 나열하고 **TFL 질의응답** 화면을 시작합니다. 예제 애플리케이션에서, 이것은 [DatasetFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetFragment.kt)에 있습니다.

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

3. `DatasetAdapter` 객체의 `onCreateViewHolder` 함수를 사용하여 텍스트 각 구절에 대한 제목을 표시합니다. 예제 애플리케이션에서, 이것은 [DatasetAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/DatasetAdapter.kt)에 있습니다.

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

모델에 사용자 질문 제공하기

1. `QaAdapter` 객체를 사용하여 모델에 질문을 제공합니다. 예제 애플리케이션에서, 이것은 [QaAdapter.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaAdapter.kt)에 있습니다.

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

### 예측 실행하기

Android 앱에서, [BertQuestionAnswerer](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer) 객체를 초기화하면 모델에 자연어 텍스트 형식으로 질문을 입력하길 시작할 수 있습니다. 모델은 텍스트 구절 내에서 답을 식별하려고 시도합니다.

예측을 실행하려면:

1. 모델을 실행하고 답변(`inferenceTime`)을 식별하는 데 소요된 시간을 측정하는 `answer` 함수를 만듭니다. 예제 애플리케이션에서, `answer` 함수는 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L78-L98)에 있습니다.

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

2. 결과를 `answer`에서 리스너 객체로 전달합니다.

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

### 모델 출력 처리하기

질문을 입력한 후, 모델은 구절에서 최대 다섯 가지 가능한 답변을 제공합니다.

모델에서 결과 얻기

1. 리스너 객체에 대한 `onResult` 함수를 만들어 출력을 처리합니다. 예제 애플리케이션에서, 리스너 객체는 [BertQaHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/BertQaHelper.kt#L92-L98)에 있습니다.

    ```
    interface AnswererListener {
        fun onError(error: String)
        fun onResults(
            results: List<QaAnswer>?,
            inferenceTime: Long
        )
    }
    ```

2. 결과를 바탕으로 구절의 섹션을 강조합니다. 예제 애플리케이션에서, 이것은 [QaFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/fragments/QaFragment.kt#L199-L208)에 있습니다.

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

모델이 일련의 결과를 반환하면 애플리케이션은 사용자에게 결과를 제공하거나 추가 논리를 실행하여 이러한 예측에 따라 조치를 취할 수 있습니다.

## 다음 단계

- [TensorFlow Lite Model Maker를 이용한 질의응답](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer) 튜토리얼을 통해 처음부터 모델을 훈련하고 구현합니다.
- [TensorFlow를 위한 텍스트 처리 도구](https://www.tensorflow.org/text)를 더 살펴봅니다.
- [TensorFlow Hub](https://tfhub.dev/google/collections/bert/1)에서 기타 BERT 모델을 다운로드합니다.
- [예제](../../examples)에서 TensorFlow Lite의 다양한 용도를 살펴봅니다.

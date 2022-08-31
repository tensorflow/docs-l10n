# Android용 소리 및 단어 인식

이 튜토리얼에서는 미리 빌드된 머신 러닝 모델과 함께 TensorFlow Lite를 사용하여 Android 앱에서 소리와 말을 인식하는 방법을 보여줍니다. 이 튜토리얼에 표시된 것과 같은 오디오 분류 모델을 사용하여 활동을 감지하거나, 작업을 식별하거나, 음성 명령을 인식할 수 있습니다.

![오디오 인식 애니메이션 데모](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/audio_classification.gif){: .attempt-right} 이 튜토리얼에서는 예제 코드를 다운로드하고 프로젝트를 [Android Studio](https://developer.android.com/studio/)에 로드하는 방법을 보여주고, 이 기능을 자신의 앱에 추가할 수 있도록 코드 예제의 주요 부분을 설명합니다. 예제 앱 코드는 대부분의 오디오 데이터 녹음과 전처리를 처리하는 TensorFlow의 [Task Library for Audio](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)를 사용합니다. 머신러닝 모델과 함께 사용하기 위해 오디오를 전처리하는 방법에 대한 자세한 내용은 [오디오 데이터 준비 및 증강](https://www.tensorflow.org/io/tutorials/audio)을 참조하세요.

## 머신 러닝을 통한 오디오 분류

이 튜토리얼의 머신 러닝 모델은 Android 장치에서 마이크를 사용하여 녹음된 오디오 샘플의 소리나 단어를 인식합니다. 이 튜토리얼의 예제 앱을 사용하면 소리를 인식하는 모델인 [YAMNet/classifier](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)와 TensorFlow Lite [Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) 도구를 사용하여 [훈련된](https://www.tensorflow.org/lite/models/modify/model_maker) 특정 구어를 인식하는 모델 사이에서 전환할 수 있습니다. 모델은 클립당 15600개의 개별 샘플을 포함하고 길이가 약 1초인 오디오 클립에 대한 예측을 실행합니다.

## 예제 설정 및 실행

이 튜토리얼의 첫 번째 부분에서는 GitHub에서 샘플을 다운로드하고 Android Studio를 사용하여 이를 실행합니다. 이 튜토리얼의 다음 섹션에서는 이를 자신의 Android 앱에 적용할 수 있도록 예제의 관련 섹션을 탐색합니다.

### 시스템 요구 사항

- <strong><a>Android Studio</a></strong> 버전 2021.1.1(Bumblebee) 이상
- Android SDK 버전 31 이상
- 개발자 모드가 활성화된 상태로 최소 OS 버전의 SDK 24(Android 7.0 - Nougat)가 설치된 Android 기기

### 예제 코드 가져오기

예제 코드의 로컬 복사본을 만듭니다. 이 코드를 사용하여 Android Studio에서 프로젝트를 만들고 샘플 애플리케이션을 실행합니다.

예제 코드를 복제하고 설정하려면:

1. git 리포지토리를 복제합니다.
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. 선택적으로, 스파스 체크아웃을 사용하도록 git 인스턴스를 구성합니다. 그러면 예제 앱에 대한 파일만 남게 됩니다.
    ```
    cd examples
    git sparse-checkout init --cone
    git sparse-checkout set lite/examples/audio_classification/android
    ```

### 프로젝트 가져오기 및 실행하기

다운로드한 예제 코드에서 프로젝트를 생성하고 프로젝트를 빌드한 후 실행합니다.

예제 코드 프로젝트를 가져오고 빌드하려면:

1. [Android Studio](https://developer.android.com/studio)를 시작합니다.
2. Android Studio에서 **File(파일) &gt; New(새로 만들기) &gt; Import Project(프로젝트 가져오기)**를 선택합니다.
3. `build.gradle` 파일(`.../examples/lite/examples/audio_classification/android/build.gradle`)이 포함된 예제 코드 디렉터리로 이동하여 해당 디렉터리를 선택합니다.

올바른 디렉터리를 선택하면 Android Studio에서 새 프로젝트를 만들고 빌드합니다. 이 프로세스는 컴퓨터 속도와 다른 프로젝트에 Android Studio를 사용했는지 여부에 따라 몇 분이 소요될 수 있습니다. 빌드가 완료되면 Android Studio가 <strong>빌드 출력</strong> 상태 패널에 <code>BUILD SUCCESSFUL</code> 메시지를 표시합니다.

프로젝트를 실행하려면:

1. Android Studio에서 **Run(실행) &gt; Run 'app'('앱' 실행)**을 선택하여 프로젝트를 실행합니다.
2. 마이크가 있는 연결 Android 기기를 선택하여 앱을 테스트합니다.

참고: 에뮬레이터를 사용하여 앱을 실행하는 경우, 호스트 시스템에서 [오디오 입력을 활성화](https://developer.android.com/studio/releases/emulator#29.0.6-host-audio)해야 합니다.

다음 섹션에서는 이 예제 앱을 참조점으로 사용하여 이 기능을 자신의 앱에 추가하기 위해 기존 프로젝트에 수행해야 하는 변경 내용을 보여줍니다.

## 프로젝트 종속성 추가

자신의 애플리케이션에서 TensorFlow Lite 머신 러닝 모델을 실행하고 오디오와 같은 표준 데이터 형식을 사용 중인 모델에서 처리할 수 있는 텐서 데이터 형식으로 변환하는 유틸리티 기능에 액세스하려면 특정 프로젝트 종속성을 추가해야 합니다.

예제 앱은 다음 TensorFlow Lite 라이브러리를 사용합니다.

- [TensorFlow Lite Task library Audio API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/package-summary) - 필요한 오디오 데이터 입력 클래스, 머신 러닝 모델 실행, 모델 처리의 출력 결과를 제공합니다.

다음 지침은 필요한 프로젝트 종속성을 자신의 Android 앱 프로젝트에 추가하는 방법을 보여줍니다.

모듈 종속성을 추가하려면:

1. TensorFlow Lite를 사용하는 모듈에서 다음 종속성을 포함하도록 모듈의 `build.gradle` 파일을 업데이트합니다. 예제 코드에서 이 파일은 `.../examples/lite/examples/audio_classification/android/build.gradle`에 있습니다.
    ```
    dependencies {
    ...
        implementation 'org.tensorflow:tensorflow-lite-task-audio'
    }
    ```
2. Android Studio에서 **File(파일) &gt; Sync Project with Gradle Files(프로젝트를 Gradle 파일과 동기화)**를 선택하여 프로젝트 종속성을 동기화합니다.

## ML 모델 초기화

Android 앱에서 모델로 예측을 실행하기 전에 매개변수로 TensorFlow Lite 머신 러닝 모델을 초기화해야 합니다. 이러한 초기화 매개변수는 모델에 따라 다르며 모델이 인식할 수 있는 단어 또는 소리의 레이블 및 예측에 대한 기본 최소 정확도 임계값과 같은 설정을 포함할 수 있습니다.

TensorFlow Lite 모델에는 모델이 들어 있는 `*.tflite` 파일이 포함되어 있습니다. 모델 파일에는 예측 논리가 포함되며 일반적으로 예측 클래스 이름과 같은 예측 결과를 해석하는 방법에 대한 [메타데이터](../../models/convert/metadata)가 포함됩니다. 모델 파일은 코드 예제와 같이 개발 프로젝트의 `src/main/assets` 디렉터리에 저장해야 합니다.

- `<project>/src/main/assets/yamnet.tflite`

편의와 코드 가독성을 위해 이 예제에서는 모델에 대한 설정을 정의하는 컴패니언 객체를 선언합니다.

앱에서 모델을 초기화하려면:

1. 모델에 대한 설정을 정의하는 컴패니언 객체를 만듭니다.
    ```
    companion object {
      const val DISPLAY_THRESHOLD = 0.3f
      const val DEFAULT_NUM_OF_RESULTS = 2
      const val DEFAULT_OVERLAP_VALUE = 0.5f
      const val YAMNET_MODEL = "yamnet.tflite"
      const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
    ```
2. `AudioClassifier.AudioClassifierOptions` 객체를 빌드하여 모델에 대한 설정을 만듭니다.
    ```
    val options = AudioClassifier.AudioClassifierOptions.builder()
      .setScoreThreshold(classificationThreshold)
      .setMaxResults(numOfResults)
      .setBaseOptions(baseOptionsBuilder.build())
      .build()
    ```
3. 이 설정 객체를 사용하여 모델이 포함된 TensorFlow Lite [`AudioClassifier`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) 객체를 구성합니다.
    ```
    classifier = AudioClassifier.createFromFileAndOptions(context, "yamnet.tflite", options)
    ```

### 하드웨어 가속 사용

앱에서 TensorFlow Lite 모델을 초기화할 때 하드웨어 가속 기능을 사용하여 모델의 예측 계산 속도를 높이는 것을 고려해야 합니다. TensorFlow Lite [대리자](https://www.tensorflow.org/lite/performance/delegates)는 GPU(그래픽 처리 장치) 또는 TPU(텐서 처리 장치)와 같은 모바일 장치의 특수 처리 하드웨어를 사용하여 머신 러닝 모델의 실행을 가속화하는 소프트웨어 모듈입니다. 코드 예제에서는 NNAPI 대리자를 사용하여 모델 실행의 하드웨어 가속을 처리합니다.

```
val baseOptionsBuilder = BaseOptions.builder()
   .setNumThreads(numThreads)
...
when (currentDelegate) {
   DELEGATE_CPU -> {
       // Default
   }
   DELEGATE_NNAPI -> {
       baseOptionsBuilder.useNnapi()
   }
}
```

TensorFlow Lite 모델 실행에 대리자를 사용하는 것이 권장되지만 필수는 아닙니다. TensorFlow Lite에서 대리자를 사용하는 방법에 대한 자세한 내용은 [TensorFlow Lite 대리자](https://www.tensorflow.org/lite/performance/delegates)를 참조하세요.

## 모델에 대한 데이터 준비하기

Android 앱에서 코드는 오디오 클립과 같은 기존 데이터를 모델에서 처리할 수 있는 [텐서](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor) 데이터 형식으로 변환하여 해석을 수행할 수 있게 모델에 데이터를 제공합니다. 모델로 전달되는 텐서의 데이터에는 모델 훈련에 사용되는 데이터 형식과 일치하는 특정 차원 또는 형상이 있어야 합니다.

이 코드 예제에서 사용된 [YAMNet/분류기 모델](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) 및 사용자 지정 [음성 명령](https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition) 모델은 0.975초 클립(15600개 샘플)에서 16kHz로 기록된 단일 채널 또는 모노 오디오 클립을 나타내는 텐서 데이터 객체를 받아들입니다. 새로운 오디오 데이터에 대한 예측을 실행하는 앱은 해당 오디오 데이터를 해당 크기와 모양의 텐서 데이터 객체로 변환해야 합니다. TensorFlow Lite Task Library [Audio API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier)는 데이터 변환을 자동으로 처리합니다.

예제 코드 `AudioClassificationHelper` 클래스에서 앱은 Android [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) 객체를 사용하여 장치 마이크의 라이브 오디오를 녹음합니다. 이 코드는 [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier)를 사용하여 모델에 적절한 샘플링 속도로 오디오를 녹음하도록 해당 객체를 빌드하고 구성합니다. 이 코드는 또한 AudioClassifier를 통해 [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) 객체를 빌드하여 변환된 오디오 데이터를 저장합니다. 그런 다음 TensorAudio 객체가 분석을 위해 모델에 전달됩니다.

ML 모델에 오디오 데이터를 제공하려면:

- `AudioClassifier` 객체를 사용하여 `TensorAudio` 객체와 `AudioRecord` 객체를 생성합니다.
    ```
    fun initClassifier() {
    ...
      try {
        classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
        // create audio input objects
        tensorAudio = classifier.createInputTensorAudio()
        recorder = classifier.createAudioRecord()
      }
    ```

참고: 앱은 Android 기기 마이크를 사용하여 오디오를 녹음할 수 있는 권한을 요청해야 합니다. 예제는 프로젝트의 `fragments/PermissionsFragment` 클래스를 참조하세요. 권한 요청에 대한 자세한 내용은 [Android 권한](https://developer.android.com/guide/topics/permissions/overview)을 참조하세요.

## 예측 실행하기

Android 앱에서 [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) 객체와 [TensorAudio](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/audio/TensorAudio) 객체를 AudioClassifier 객체에 연결하면 해당 데이터에 대해 모델을 실행하여 예측 또는 *추론*을 생성할 수 있습니다. 이 튜토리얼의 예제 코드는 라이브 녹음된 오디오 입력 스트림의 클립에 대해 특정 속도로 예측을 실행합니다.

모델 실행에는 상당한 리소스가 소비되므로 별도의 백그라운드 스레드에서 ML 모델 예측을 실행하는 것이 중요합니다. 예제 앱은 `[ScheduledThreadPoolExecutor](https://developer.android.com/reference/java/util/concurrent/ScheduledThreadPoolExecutor)` 객체를 사용하여 모델 처리를 앱의 다른 기능과 분리합니다.

단어와 같이 시작과 끝이 명확한 소리를 인식하는 오디오 분류 모델은 겹치는 오디오 클립을 분석하여 들어오는 오디오 스트림에 대해 보다 정확한 예측을 생성할 수 있습니다. 이 접근 방식은 모델이 클립 끝에서 잘린 단어를 예측하지 못하는 것을 방지하는 데 도움이 됩니다. 예제 앱에서 예측을 실행할 때마다 코드는 오디오 녹음 버퍼에서 최신 0.975초 클립을 가져와 분석합니다. 모델 분석 스레드 실행 풀 `interval` 값을 분석 중인 클립의 길이보다 짧은 길이로 설정하여 모델이 겹치는 오디오 클립을 분석하도록 할 수 있습니다. 예를 들어, 모델이 1초 클립을 분석하고 간격을 500밀리초로 설정하면 모델은 매번 이전 클립의 마지막 절반과 새 오디오 데이터의 500밀리초를 분석하여 50%의 클립 분석 중첩을 생성합니다.

오디오 데이터에 대한 예측 실행을 시작하려면:

1. `AudioClassificationHelper.startAudioClassification()` 메서드를 사용하여 모델에 대한 오디오 녹음을 시작합니다.
    ```
    fun startAudioClassification() {
      if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        return
      }
      recorder.startRecording()
    }
    ```
2. `ScheduledThreadPoolExecutor` 객체에서 고정 속도 `interval`을 설정하여 모델이 오디오 클립에서 추론을 생성하는 빈도를 설정합니다.
    ```
    executor = ScheduledThreadPoolExecutor(1)
    executor.scheduleAtFixedRate(
      classifyRunnable,
      0,
      interval,
      TimeUnit.MILLISECONDS)
    ```
3. 위 코드의 `classifyRunnable` 객체는 `AudioClassificationHelper.classifyAudio()` 메서드를 실행합니다. 이 메서드는 레코더에서 사용 가능한 최신 오디오 데이터를 로드하고 예측을 수행합니다.
    ```
    private fun classifyAudio() {
      tensorAudio.load(recorder)
      val output = classifier.classify(tensorAudio)
      ...
    }
    ```

주의: 애플리케이션의 기본 실행 스레드에서 ML 모델 예측을 실행하지 마십시오. 그렇지 않으면 앱 사용자 인터페이스가 느려지거나 응답하지 않을 수 있습니다.

### 예측 처리 중지

앱의 오디오 처리 조각 또는 활동이 포커스를 잃으면 앱 코드가 오디오 분류를 중지하는지 확인하세요. 머신 러닝 모델을 지속적으로 실행하면 Android 기기의 배터리 수명에 상당한 영향을 미칩니다. 오디오 분류와 연결된 Android 활동 또는 조각의 `onPause()` 메서드를 사용하여 오디오 녹음 및 예측 처리를 중지합니다.

오디오 녹음 및 분류를 중지하려면:

- `AudioFragment` 클래스에서 아래와 같이 `AudioClassificationHelper.stopAudioClassification()` 메서드를 사용하여 녹음 및 모델 실행을 중지합니다.
    ```
    override fun onPause() {
      super.onPause()
      if (::audioHelper.isInitialized ) {
        audioHelper.stopAudioClassification()
      }
    }
    ```

## 모델 출력 처리하기

Android 앱에서 오디오 클립을 처리한 후 모델은 추가 비즈니스 로직을 실행하거나 사용자에게 결과를 표시하거나 기타 작업을 수행하여 앱 코드가 처리해야 하는 예측 목록을 생성합니다. 주어진 TensorFlow Lite 모델의 출력은 생성하는 예측 수(하나 또는 여러 개)와 각 예측에 대한 설명 정보에 따라 다릅니다. 예제 앱의 모델의 경우 예측은 인식된 소리 또는 단어의 목록입니다. 코드 예제에 사용된 AudioClassifier 옵션 객체를 사용하면 <a>ML 모델 초기화</a> 섹션에 표시된 대로 <code>setMaxResults()</code> 메서드로 최대 예측 수를 설정할 수 있습니다.

모델에서 예측 결과를 얻으려면:

1. [AudioClassifier](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier) 객체의 `classify()` 메서드 결과를 가져와 리스너 객체(코드 참조)에 전달합니다.
    ```
    private fun classifyAudio() {
      ...
      val output = classifier.classify(tensorAudio)
      listener.onResult(output[0].categories, inferenceTime)
    }
    ```
2. 리스너의 onResult() 함수를 사용하여 비즈니스 로직을 실행하거나 사용자에게 결과를 표시하여 출력을 처리합니다.
    ```
    private val audioClassificationListener = object : AudioClassificationListener {
      override fun onResult(results: List<Category>, inferenceTime: Long) {
        requireActivity().runOnUiThread {
          adapter.categoryList = results
          adapter.notifyDataSetChanged()
          fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
            String.format("%d ms", inferenceTime)
        }
      }
    ```

이 예에서 사용된 모델은 분류된 소리 또는 단어에 대한 레이블이 있는 예측 목록을 생성하고 예측의 신뢰도를 나타내는 Float로 0과 1 사이의 예측 점수를 생성합니다(1이 가장 높은 신뢰도 등급). 일반적으로 점수가 50%(0.5) 미만인 예측은 결정적이지 않은 것으로 간주됩니다. 그러나 낮은 값의 예측 결과를 처리하는 방식은 사용자와 애플리케이션의 요구 사항에 달려 있습니다.

모델이 일련의 예측 결과를 반환하면 애플리케이션은 결과를 사용자에게 제공하거나 추가 논리를 실행하여 이러한 예측에 따라 조치를 취할 수 있습니다. 예제 코드의 경우 애플리케이션은 앱 사용자 인터페이스에서 식별된 소리나 단어를 나열합니다.

## 다음 단계

[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding,audio-pitch-extraction,audio-event-classification,audio-stt) 및 [사전 학습된 모델 가이드](https://www.tensorflow.org/lite/models/trained) 페이지를 통해 오디오 처리를 위한 추가 TensorFlow Lite 모델을 찾을 수 있습니다. TensorFlow Lite를 사용하여 모바일 애플리케이션에서 머신 러닝을 구현하는 방법에 대한 자세한 내용은 [TensorFlow Lite 개발자 가이드](https://www.tensorflow.org/lite/guide)를 참조하세요.

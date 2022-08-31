# 오디오 분류기 통합

오디오 분류는 소리 유형을 분류하기 위한 머신 러닝의 일반적인 사용 사례입니다. 예를 들어 노래로 새의 종을 식별할 수 있습니다.

작업 라이브러리 `AudioClassifier` API를 사용하여 사용자 지정 오디오 분류기 또는 사전 훈련된 분류기를 모바일 앱에 배포할 수 있습니다.

## AudioClassifier API의 주요 기능

- 입력 오디오 처리(예: PCM 16비트 인코딩을 PCM 부동 인코딩으로 변환 및 오디오 링 버퍼 조작)

- 레이블 맵 로케일

- 멀티 헤드 분류 모델 지원

- 단일 레이블 및 다중 레이블 분류를 모두 지원

- 결과를 필터링하기 위한 스코어 임계값

- Top-k 분류 결과

- 레이블 허용 목록 및 거부 목록

## 지원되는 오디오 분류기 모델

다음 모델이 `AudioClassifier` API와 호환이 보장됩니다.

- [오디오 분류를 위해 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier)에서 만든 모델

- [TensorFlow Hub에서 사전 훈련된 오디오 이벤트 분류 모델](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)

- [모델 호환성 요구 사항](#model-compatibility-requirements)을 충족하는 사용자 정의 모델

## Java에서 추론 실행하기

Android 앱에서 <code>AudioClassifier</code>를 사용하는 예는 <a>오디오 분류 참조 앱</a>을 참조하세요.

### 1단계: Gradle 종속성 및 기타 설정 가져오기

`.tflite` 모델 파일을 모델이 실행될 Android 모듈의 assets 디렉토리에 복사합니다. 파일을 압축하지 않도록 지정하고 TensorFlow Lite 라이브러리를 모듈의 `build.gradle` 파일에 추가합니다.

```java
android {
    // Other settings

    // Specify that the tflite file should not be compressed when building the APK package.
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // Other dependencies

    // Import the Audio Task Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.4.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
}
```

참고: Android Gradle 플러그인 버전 4.1부터 기본적으로 .tflite가 noCompress 목록에 추가되며 위의 aaptOptions는 더 이상 필요하지 않습니다.

### 2단계: 모델 사용하기

```java
// Initialization
AudioClassifierOptions options =
    AudioClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
AudioClassifier classifier =
    AudioClassifier.createFromFileAndOptions(context, modelFile, options);

// Start recording
AudioRecord record = classifier.createAudioRecord();
record.startRecording();

// Load latest audio samples
TensorAudio audioTensor = classifier.createInputTensorAudio();
audioTensor.load(record);

// Run inference
List<Classifications> results = audioClassifier.classify(audioTensor);
```

<code>AudioClassifier</code> 구성을 위한 추가 옵션은 <a>소스 코드 및 javadoc</a>을 참조하세요.

## Python에서 추론 실행하기

### 1단계: pip 패키지 설치하기

```
pip install tflite-support
```

참고: 작업 라이브러리의 오디오 API는 [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html)를 사용하여 장치의 마이크에서 오디오를 녹음합니다. 오디오 녹음에 작업 라이브러리의 [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord)를 사용하려면 시스템에 PortAudio를 설치해야 합니다.

- Linux: `sudo apt-get update && apt-get install libportaudio2` 실행
- Mac 및 Windows: `tflite-support` pip 패키지를 설치할 때 PortAudio가 자동으로 설치됩니다.

### 2단계: 모델 사용하기

```python
# Imports
from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
classification_options = processor.ClassificationOptions(max_results=2)
options = audio.AudioClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = audio.AudioClassifier.create_from_options(options)

# Alternatively, you can create an audio classifier in the following manner:
# classifier = audio.AudioClassifier.create_from_file(model_path)

# Run inference
audio_file = audio.TensorAudio.create_from_wav_file(audio_path, classifier.required_input_buffer_size)
audio_result = classifier.classify(audio_file)
```

<code>AudioClassifier</code> 구성을 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## C++에서 추론 실행하기

```c++
// Initialization
AudioClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<AudioClassifier> audio_classifier = AudioClassifier::CreateFromOptions(options).value();

// Create input audio buffer from your `audio_data` and `audio_format`.
// See more information here: tensorflow_lite_support/cc/task/audio/core/audio_buffer.h
int input_size = audio_classifier->GetRequiredInputBufferSize();
const std::unique_ptr<AudioBuffer> audio_buffer =
    AudioBuffer::Create(audio_data, input_size, audio_format).value();

// Run inference
const ClassificationResult result = audio_classifier->Classify(*audio_buffer).value();
```

<code>AudioClassifier</code> 구성을 위한 추가 옵션은 <a>소스 코드</a>를 참조하세요.

## 모델 호환성 요구 사항

`AudioClassifier` API는 필수 [TFLite 모델 메타데이터](../../models/convert/metadata.md)가 있는 TFLite 모델을 예상합니다. [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers)를 사용하여 오디오 분류자에 대한 메타데이터를 생성하는 예를 참조하세요.

호환되는 오디오 분류기 모델은 다음 요구 사항을 충족해야 합니다.

- 입력 오디오 텐서(kTfLiteFloat32)

    - `[batch x samples]` 크기의 오디오 클립
    - 배치 추론은 지원되지 않습니다(`batch`는 1이어야 함).
    - 다중 채널 모델의 경우 채널을 인터리브해야 합니다.

- 출력 점수 텐서(kTfLiteFloat32)

    - `[1 x N]` 배열(`N`은 클래스 번호를 나타냄)
    - 선택적(그러나 권장함) 레이블 맵은 한 줄에 하나의 레이블을 포함하여 TENSOR_VALUE_LABELS 유형의 AssociatedFile-s로 매핑할 수 있습니다. 첫 번째 AssociatedFile(있는 경우)은 결과의 `label` 필드(C++에서 `class_name`으로 명명됨)를 채우는 데 사용됩니다. `display_name` 필드는 생성 시 사용된 `ImageClassifierOptions`의 `display_names_locale` 필드와 로케일이 일치하는 AssociatedFile(있는 경우)로부터 채워집니다(기본적으로 "en", 즉 영어). 이들 중 어느 것도 사용할 수 없는 경우, 결과의 `index` 필드만 채워집니다.

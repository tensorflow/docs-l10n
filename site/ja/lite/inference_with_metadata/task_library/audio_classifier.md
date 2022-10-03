# 音声分類器の統合

音声分類は、機械学習の一般的なユースケースであり、音声の種類を分類します。たとえば、鳴き声から鳥の種類を特定できます。

Task Library `AudioClassifier` APIを使用して、カスタム音声分類器または事前トレーニング済みモデルをモバイルアプリにデプロイできます。

## AudioClassifier API の主な機能

- 入力音声処理。例: PCM 16 ビットエンコーディングを PCM 浮動小数点数エンコーディングに変換、音声リングバッファの操作。

- マップロケールのラベル付け

- マルチヘッド分類モデルのサポート。

- シングルラベルおよびマルチラベル分類の両方のサポート。

- 結果をフィルタリングするスコアしきい値。

- Top-k 分類結果。

- 許可リストと拒否リストのラベルを付け

## サポートされている音声分類器モデル

次のモデルは、`AudioClassifier` API との互換性が保証されています。

- [TensorFlow Lite Model Maker による音声分類](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier)によって作成されたモデル。

- [TensorFlow Hub の事前トレーニング済み音声分類モデル](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)。

- [モデルの互換性要件](#model-compatibility-requirements)を満たすカスタムモデル。

## Java で推論を実行する

Android アプリで `AudioClassifier` を使用する例については、[音声分類リファレンスアプリ](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)を参照してください。

### ステップ 1: Gradle の依存関係とその他の設定をインポートする

`.tflite`モデルファイルを、モデルが実行される Android モジュールのアセットディレクトリにコピーします。ファイルを圧縮しないように指定し、TensorFlow Lite ライブラリをモジュールの`build.gradle`ファイルに追加します。

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

注：Android Gradle プラグインのバージョン 4.1 以降、.tflite はデフォルトで noCompress リストに追加され、上記の aaptOptions は不要になりました。

### ステップ 2: モデルを使用する

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

[ソースコードと javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier.java) を参照し、`AudioClassifier` を構成するその他のオプションについてご覧ください。

## iOS で推論を実行する

### ステップ 1: 依存関係をインストールする

タスクライブラリは、CocoaPods を使用したインストールをサポートしています。CocoaPods がシステムにインストールされていることを確認してください。手順については、[CocoaPods インストールガイド](https://guides.cocoapods.org/using/getting-started.html#getting-started)を参照してください。

ポッドを Xcode プロジェクトに追加する詳細な方法については、[CocoaPods ガイド](https://guides.cocoapods.org/using/using-cocoapods.html)を参照してください。

Podfile に `TensorFlowLiteTaskText` ポッドを追加します。

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskAudio'
end
```

推論で使用する `.tflite` モデルがアプリバンドルに存在することを確認します。

### ステップ 2: モデルを使用する

#### Swift

```swift
// Imports
import TensorFlowLiteTaskAudio
import AVFoundation

// Initialization
guard let modelPath = Bundle.main.path(forResource: "sound_classification",
                                            ofType: "tflite") else { return }

let options = AudioClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try AudioClassifier.classifier(options: options)

// Create Audio Tensor to hold the input audio samples which are to be classified.
// Created Audio Tensor has audio format matching the requirements of the audio classifier.
// For more details, please see:
// https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/core/audio_tensor/sources/TFLAudioTensor.h
let audioTensor = classifier.createInputAudioTensor()

// Create Audio Record to record the incoming audio samples from the on-device microphone.
// Created Audio Record has audio format matching the requirements of the audio classifier.
// For more details, please see:
https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/core/audio_record/sources/TFLAudioRecord.h
let audioRecord = try classifier.createAudioRecord()

// Request record permissions from AVAudioSession before invoking audioRecord.startRecording().
AVAudioSession.sharedInstance().requestRecordPermission { granted in
    if granted {
        DispatchQueue.main.async {
            // Start recording the incoming audio samples from the on-device microphone.
            try audioRecord.startRecording()

            // Load the samples currently held by the audio record buffer into the audio tensor.
            try audioTensor.load(audioRecord: audioRecord)

            // Run inference
            let classificationResult = try classifier.classify(audioTensor: audioTensor)
        }
    }
}
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskAudio/TensorFlowLiteTaskAudio.h>
#import <AVFoundation/AVFoundation.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"sound_classification" ofType:@"tflite"];

TFLAudioClassifierOptions *options =
    [[TFLAudioClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLAudioClassifier *classifier = [TFLAudioClassifier audioClassifierWithOptions:options
                                                                          error:nil];

// Create Audio Tensor to hold the input audio samples which are to be classified.
// Created Audio Tensor has audio format matching the requirements of the audio classifier.
// For more details, please see:
// https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/core/audio_tensor/sources/TFLAudioTensor.h
TFLAudioTensor *audioTensor = [classifier createInputAudioTensor];

// Create Audio Record to record the incoming audio samples from the on-device microphone.
// Created Audio Record has audio format matching the requirements of the audio classifier.
// For more details, please see:
https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/core/audio_record/sources/TFLAudioRecord.h
TFLAudioRecord *audioRecord = [classifier createAudioRecordWithError:nil];

// Request record permissions from AVAudioSession before invoking -[TFLAudioRecord startRecordingWithError:].
[[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
    if (granted) {
        dispatch_async(dispatch_get_main_queue(), ^{
            // Start recording the incoming audio samples from the on-device microphone.
            [audioRecord startRecordingWithError:nil];

            // Load the samples currently held by the audio record buffer into the audio tensor.
            [audioTensor loadAudioRecord:audioRecord withError:nil];

            // Run inference
            TFLClassificationResult *classificationResult =
                [classifier classifyWithAudioTensor:audioTensor error:nil];

        });
    }
}];
```

`TFLAudioClassifier` を構成するその他のオプションについては、[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h)を参照してください。

## Python で推論を実行する

### ステップ 1: pip パッケージをインストールする

```
pip install tflite-support
```

注意: Task Library の Audio API は [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html) を仕様して、デバイスのマイクから音声を記録します。音声記録で Task Library の [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord) を使用する場合は、システムに PortAudio をインストールする必要があります。

- Linux: `sudo apt-get update && apt-get install libportaudio2` を実行します。
- Mac および Windows: `tflite-support` pip パッケージをインストールするときに、PortAudio が自動的にインストールされます。

### ステップ 2: モデルを使用する

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

`AudioClassifier` を構成するその他のオプションについては、[ソースコード](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/audio/audio_classifier.py)を参照してください。

## C++ で推論を実行する

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

[ソースコードと](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/audio/audio_classifier.h)を参照し、`AudioClassifier` を構成するその他のオプションについてご覧ください。

## モデルの互換性要件

`AudioClassifier` API は、必須の [TFLite モデル メタデータ](../../models/convert/metadata.md)を持つ TFLite モデルを想定しています。[TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers) を使用して音声分類器のメタデータを作成する例をご覧ください。

互換性のある音声分類モデルは、次の要件を満たす必要があります。

- 入力音声テンソル (kTfLiteFloat32)

    - サイズ `[batch x samples]` の音声クリップ
    - バッチ推論はサポートされていません (`batch`は 1 である必要があります)。
    - マルチチャネルモデルでは、チャネルをインターリーブする必要があります。

- 出力スコアテンソル (kTfLiteFloat32)

    - `[1 x N]` 配列。`N` はクラス番号です。
    - TENSOR_AXIS_LABELS 型の AssociatedFile ラベルマップ (オプションですが推薦されます)。1 行に 1 つのラベルが含まれます。最初の AssociatedFile (存在する場合) は、結果の`label`フィールド (C ++では`class_name`と名付けられています) を入力ために使用されます。`display_name`フィールドは、AssociatedFile (存在する場合) から入力されます。そのロケールは、作成時に使用される`ImageClassifierOptions`の`display_names_locale`フィールドと一致します（デフォルトでは「en (英語)」）。これらのいずれも使用できない場合、結果の`index`フィールドのみが入力されます。

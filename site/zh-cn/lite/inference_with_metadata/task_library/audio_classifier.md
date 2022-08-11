# 集成音频分类器

音频分类是机器学习对声音类型进行分类的常见用例。例如，它可以通过鸟类的歌声来识别鸟类。

可以使用 Task Library `AudioClassifier` API 将您的自定义音频分类器或预训练的音频分类器部署到您的移动应用中。

## AudioClassifier API 的主要功能

- 输入音频处理，例如将 PCM 16 位编码转换为 PCM 浮点编码和处理音频环形缓冲区。

- 标注映射区域。

- 支持多头分类模型。

- 支持单标签和多标签分类。

- 筛选结果的分数阈值。

- Top-k 分类结果。

- 标注允许列表和拒绝列表。

## 支持的音频分类器模型

以下模型保证可与 `AudioClassifier` API 兼容。

- 由[适用于音频分类的 TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier) 创建的模型。

- [TensorFlow Hub 上的预训练音频分类模型](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)。

- 符合[模型兼容性要求](#model-compatibility-requirements)的自定义模型。

## 用 Java 运行推断

请参阅[音频分类参考应用](https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/android)，获得如何在 Android 应用中使用 `AudioClassifier` 的示例。

### 步骤 1：导入 Gradle 依赖项和其他设置

将 `.tflite` 模型文件复制到将要运行模型的 Android 模块的资源目录下。指定不压缩该文件，并将 TensorFlow Lite 库添加到模块的 `build.gradle` 文件中。

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

注：从 Android Gradle 插件的 4.1 版开始，默认情况下，.tflite 将被添加到 noCompress 列表中，不再需要上面的 aaptOptions。

### 第 2 步：使用模型

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

请参阅[源代码和 Javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier.java)，了解有关配置 `AudioClassifier` 的更多选项。

## 用 Python 运行推断

### 第 1 步：安装 pip 软件包

```
pip install tflite-support
```

注：Task Library 的 Audio API依靠 [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html) 来录制来自设备麦克风的音频。如果您打算使用 Task Library 的 [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord) 进行音频录制，则需要在您的系统上安装 PortAudio。

- Linux：运行 `sudo apt-get update && apt-get install libportaudio2`
- Mac 和 Windows：安装 `tflite-support` pip 软件包时会自动安装 PortAudio。

### 第 2 步：使用模型

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

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/audio/audio_classifier.py)，了解有关配置 `AudioClassifier` 的更多选项。

## 用 C++ 运行推断

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

请参阅[源代码](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/audio/audio_classifier.h)，了解有关配置 `AudioClassifier` 的更多选项。

## 模型兼容性要求

`AudioClassifier` API 需要具有强制性 [TFLite Model Metadata](../../models/convert/metadata.md) 的 TFLite 模型。请参阅使用 [TensorFlow Lite Metadata Writer API](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers) 为音频分类器创建元数据的示例。

兼容的音频分类器模型应满足以下要求：

- 输入音频张量 (kTfLiteFloat32)

    - 大小为 `[batch x samples]` 的音频剪辑。
    - 不支持批量推断（`batch` 必须为 1）。
    - 对于多通道模型，通道需要交错。

- 输出分数张量 (kTfLiteFloat32)

    - `[1 x N]` 数组，其中 `N` 表示类编号。
    - 可选（但推荐）标签映射可作为 AssociatedFile-s，类型为 TENSOR_AXIS_LABELS，每行包含一个标签。第一个此类 AssociatedFile（如果有）用于填充结果的 `label` 字段（在 C++ 中命名为 `class_name`）。`display_name` 字段由其区域与创建时所用的 `ImageClassifierOptions` 的 `display_names_locale` 字段（默认为“en”，即英语）相匹配的 AssociatedFile（如果有）填充。如果上述选项均不可用，将仅填充结果中的 `index` 字段。

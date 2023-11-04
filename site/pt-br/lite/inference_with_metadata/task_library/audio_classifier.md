# Integração de classificadores de áudio

A classificação de áudio é um caso de uso comum de aprendizado de máquina para classificar tipos de sons. Por exemplo: é possível identificar espécies de pássaro por meio de seus cantos.

A API `AudioClassifier` da biblioteca Task pode ser usada para implantar classificadores de áudio personalizados ou pré-treinados em seu aplicativo para dispositivos móveis.

## Principais recursos da API AudioClassifier

- Pré-processamento do áudio de entrada, como a conversão de codificação PCM de 16 bits para codificação PCM de ponto flutuante e a manipulação do buffer circular de áudio.

- Idioma do mapa de rótulos.

- Suporte à classificação multi-head.

- Suporte à classificação com um e vários rótulos.

- Limite de pontuação para filtrar resultados.

- Resultados de classificação top-k.

- Lista de permissão e proibição de rótulos.

## Modelos de classificador de áudio com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `AudioClassifier`.

- Modelos criados pelo [Model Maker do TensorFlow Lite para classificação de áudio](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier).

- [Modelos pré-treinados de classificação de eventos de áudio no TensorFlow Hub](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

Confira o [aplicativo de referência para classificação de áudio](https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android) para ver um exemplo de como usar o `AudioClassifier` em um aplicativo para Android.

### Etapa 1 – Importe a dependência e outras configurações do Gradle

Copie o arquivo do modelo `.tflite` para o diretório de ativos do módulo para Android no qual o modelo será executado. Especifique que o arquivo não deve ser compactado e adicione a biblioteca do TensorFlow Lite ao arquivo `build.gradle` do modelo:

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
    implementation 'org.tensorflow:tensorflow-lite-task-audio:0.4.4'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

### Etapa 2 – Use o modelo

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

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier.java) para ver mais opções de configuração do `AudioClassifier`.

## Execute a inferência no iOS

### Etapa 1 – Instale as dependências

A biblioteca Task oferece suporte à instalação usando o CocoaPods, que precisa estar instalado em seu sistema. Confira as instruções no [guia de instalação do CocoaPods](https://guides.cocoapods.org/using/getting-started.html#getting-started).

Confira os detalhes de como adicionar pods a um projeto do Xcode no [guia do CocoaPods](https://guides.cocoapods.org/using/using-cocoapods.html).

Adicione o pod `TensorFlowLiteTaskAudio` ao Podfile.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskAudio'
end
```

Confirme se o modelo `.tflite` que você usará para inferência está presente no pacote do aplicativo.

### Etapa 2 – Use o modelo

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

#### Objective-C

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h) para ver mais opções de configuração do `TFLAudioClassifier`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote via pip

```
pip install tflite-support
```

Observação: as APIs de áudio da biblioteca Task dependem do [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html) para gravar áudio usando o microfone do dispositivo. Se você deseja usar o [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord) da biblioteca Task para gravação de áudio, precisa instalar o PortAudio em seu sistema.

- Linux: execute `sudo apt-get update && apt-get install libportaudio2`
- Mac e Windows: o PortAudio é instalado automaticamente ao instalar o pacote `tflite-support` via pip.

### Etapa 2 – Use o modelo

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/audio/audio_classifier.py) para ver mais opções de configuração do `AudioClassifier`.

## Execute a inferência no C++

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/audio/audio_classifier.h) para ver mais opções de configuração do `AudioClassifier`.

## Requisitos de compatibilidade de modelos

A API `AudioClassifier` espera um modelo do TF Lite com os [metadados de modelo do TF Lite](../../models/convert/metadata.md) obrigatórios. Confira exemplos de criação dos metadados para classificadores de áudio na [API de gravação de metadados do TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers).

Os modelos compatíveis de classificador de áudio devem atender aos seguintes requisitos:

- Tensor de áudio de entrada (kTfLiteFloat32)

    - Clipe de áudio de tamanho `[batch x samples]` (lote x amostras).
    - Não há suporte à inferência de lote (`batch` precisa ser igual a 1).
    - Para modelos multicanais, os canais precisam ser intercalados.

- Tensor de pontuação de saída (kTfLiteFloat32)

    - Array `[1 x N]`, em que `N` representa o número de classes.
    - Mapa(s) de rótulos (opcionais, mas recomendados), como AssociatedFiles com tipo TENSOR_AXIS_LABELS, contendo um rótulo por linha. O primeiro AssociatedFile (se houver) é usado para preencher o campo `label` (com nome igual a `class_name` no C++) dos resultados. O campo `display_name` é preenchido a partir do AssociatedFile (se houver) cujo idioma coincida com o campo `display_names_locale` das opções `AudioClassifierOptions` usadas no momento da criação ("en" por padrão, ou seja, inglês). Se nenhum estiver disponível, somente o campo `index` dos resultados será preenchido.

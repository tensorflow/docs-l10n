# Integrar clasificadores de audio

La clasificación de audio es un caso común de uso del aprendizaje automático para clasificar los tipos de sonido. Por ejemplo, puede identificar las especies de aves por sus cantos.

La API `AudioClassifier` de la librería de tareas  puede usarse para implementar sus clasificadores de audio personalizados o preentrenados en su app móvil.

## Características principales de la API AudioClassifier

- Procesamiento del audio de entrada, por ejemplo la conversión de la codificación PCM de 16 bits a la codificación PCM Float y la manipulación de la memoria cíclica de audio.

- Etiquetar la localización del mapa.

- Admite el modelo de clasificación multicabecera.

- Admite tanto la clasificación monoetiqueta como la multietiqueta.

- Umbral de puntuaciones para filtrar los resultados.

- Resultados de la clasificación Top-k.

- Etiquetar en allowlist y denylist.

## Modelos de clasificador de audio compatibles

Se garantiza que los siguientes modelos son compatibles con la API `AudioClassifier`.

- Modelos creados por [Model Maker de TensorFlow Lite para la clasificación de audio](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/audio_classifier).

- Los [modelos de clasificación de eventos de audio preentrenados en TensorFlow Hub](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

Consulte la [app de referencia de Audio Classification](https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android) para ver un ejemplo usando `AudioClassifier` en una app Android.

### Paso 1: Importar la dependencia de Gradle y otras configuraciones

Copie el archivo del modelo `.tflite` en el directorio de activos del módulo Android en el que se ejecutará el modelo. Especifique que el archivo no debe comprimirse y añada la librería TensorFlow Lite al archivo `build.gradle` del módulo:

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

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no es necesario.

### Paso 2: Usar el modelo

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

Consulte el [código fuente y javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier/AudioClassifier.java) para ver más opciones para configurar `AudioClassifier`.

## Ejecutar la inferencia en iOS

### Paso 1: Instalar las dependencias

La librería de tareas admite la instalación usando CocoaPods. Asegúrese de que CocoaPods está instalado en su sistema. Consulte la [Guía de instalación de CocoaPods](https://guides.cocoapods.org/using/getting-started.html#getting-started) si desea instrucciones.

Consulte la [Guía de CocoaPods](https://guides.cocoapods.org/using/using-cocoapods.html) para más detalles sobre cómo añadir pods a un proyecto Xcode.

Añada el pod `TensorFlowLiteTaskAudio` en el Podfile.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskAudio'
end
```

Asegúrese de que el modelo `.tflite` que va a usar para la inferencia está presente en el paquete de su app.

### Paso 2: Usar el modelo

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

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h) para más opciones de configuración de `TFLAudioClassifier`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete pip

```
pip install tflite-support
```

Nota: Las API de audio de la librería de tareas dependen de [PortAudio](http://www.portaudio.com/docs/v19-doxydocs/index.html) para grabar audio desde el micrófono del dispositivo. Si desea usar [AudioRecord](/lite/api_docs/python/tflite_support/task/audio/AudioRecord) de la librería de tareas para grabar audio, deberá instalar PortAudio en su sistema.

- Linux: Ejecute `sudo apt-get update && apt-get install libportaudio2`
- Mac y Windows: PortAudio se instala automáticamente al instalar el paquete pip `tflite-support`.

### Paso 2: Usar el modelo

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

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/audio/audio_classifier.py) para ver más opciones para configurar `AudioClassifier`.

## Ejecutar la inferencia en C++

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

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/audio/audio_classifier.h) para ver más opciones para configurar `AudioClassifier`.

## Requisitos de compatibilidad del modelo

La API `AudioClassifier` espera un modelo TFLite con [Metadatos del modelo TFLite](../../models/convert/metadata.md) como requisito. Vea ejemplos de creación de metadatos para clasificadores de audio usando la [API de escritura de metadatos de TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#audio_classifiers).

Los modelos de clasificador de audio compatibles deben cumplir los siguientes requisitos:

- Tensor de audio de entrada (kTfLiteFloat32)

    - clip de audio de tamaño `[batch x samples]`.
    - no se admite la inferencia por lotes (`batch` debe ser 1).
    - para los modelos multicanal, es necesario intercalar los canales.

- Tensor de puntuaciones de salida (kTfLiteFloat32)

    - `[1 x N]` arreglo con `N` representando el número de clase.
    - mapa(s) de etiquetas opcional(es) (pero recomendables) como AssociatedFile-s con tipo TENSOR_AXIS_LABELS, conteniendo una etiqueta por línea. El primero de tales AssociatedFile (si existe) se usa para llenar el campo `label` (nombrado como `class_name` en C++) de los resultados. El campo `display_name` se llena a partir del AssociatedFile (si existe) cuya configuración regional coincida con el campo `display_names_locale` del `AudioClassifierOptions` usado en el momento de la creación ("en" por predeterminado, es decir, inglés). Si ninguno de ellos está disponible, sólo se llenará el campo `index` de los resultados.

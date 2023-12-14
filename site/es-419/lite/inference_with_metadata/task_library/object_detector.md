# Integrar detectores de objetos

Los detectores de objetos pueden identificar cuáles de un conjunto conocido de objetos pueden estar presentes y dar información sobre sus posiciones dentro de la imagen dada o de un flujo de vídeo. Un detector de objetos se entrena para detectar la presencia y la ubicación de múltiples clases de objetos. Por ejemplo, un modelo podría entrenarse con imágenes que contengan varias piezas de fruta, junto con una *etiqueta* que especifique la clase de fruta que representan (por ejemplo, una manzana, un plátano o una fresa), y datos que especifiquen dónde aparece cada objeto en la imagen. Consulte la [introducción a la detección de objetos](../../examples/object_detection/overview) para saber más sobre los detectores de objetos.

Use la API `ObjectDetector` de la librería de tareas para implementar sus detectores de objetos personalizados o preentrenados en sus apps móviles.

## Características principales de la API ObjectDetector

- Procesamiento de imágenes de entrada, incluyendo rotación, redimensionamiento y conversión del espacio de color.

- Etiquetar la localización del mapa.

- Umbral de puntuaciones para filtrar los resultados.

- Resultados de la detección Top-k.

- Etiquetar en allowlist y denylist.

## Modelos de detector de objetos compatibles

Se garantiza que los siguientes modelos son compatibles con la API `ObjectDetector`.

- Los [modelos preentrenados de detección de objetos en TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)[.](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1)

- Modelos creados por [Detección de objetos en los bordes de visión AutoML](https://cloud.google.com/vision/automl/object-detection/docs).

- Modelos creados por [Model Maker de TensorFlow Lite para el detector de objetos](https://www.tensorflow.org/lite/guide/model_maker).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

Consulte la [app de referencia de detección de objetos](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/) para ver un ejemplo de cómo usar `ObjectDetector` en una app de Android.

### Paso 1: Importar la dependencia de Gradle y otras configuraciones

Copie el archivo del modelo `.tflite` en el directorio de activos del módulo Android en el que se ejecutará el modelo. Especifique que el archivo no debe comprimirse y añada la librería TensorFlow Lite al archivo `build.gradle` del módulo:

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.

### Paso 2: Usar el modelo

```java
// Initialization
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

Consulte el [código fuente y javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/detector/ObjectDetector.java) para ver más opciones para configurar `ObjectDetector`.

## Ejecutar la inferencia en iOS

### Paso 1: Instalar las dependencias

La librería de tareas admite la instalación usando CocoaPods. Asegúrese de que CocoaPods está instalado en su sistema. Consulte la [Guía de instalación de CocoaPods](https://guides.cocoapods.org/using/getting-started.html#getting-started) si desea instrucciones.

Consulte la [Guía de CocoaPods](https://guides.cocoapods.org/using/using-cocoapods.html) para más detalles sobre cómo añadir pods a un proyecto Xcode.

Añada el pod `TensorFlowLiteTaskVision` en el Podfile.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

Asegúrese de que el modelo `.tflite` que va a usar para la inferencia está presente en el paquete de su app.

### Paso 2: Usar el modelo

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "ssd_mobilenet_v1",
                                            ofType: "tflite") else { return }

let options = ObjectDetectorOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let detector = try ObjectDetector.detector(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "cats_and_dogs.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let detectionResult = try detector.detect(mlImage: mlImage)
```

#### Objective-C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"ssd_mobilenet_v1" ofType:@"tflite"];

TFLObjectDetectorOptions *options = [[TFLObjectDetectorOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLObjectDetector *detector = [TFLObjectDetector objectDetectorWithOptions:options
                                                                     error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"dogs.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLDetectionResult *detectionResult = [detector detectWithGMLImage:gmlImage error:nil];
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLObjectDetector.h) para más opciones de configuración de `TFLObjectDetector`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete pip

```
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(max_results=2)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/object_detector.py) para ver más opciones para configurar `ObjectDetector`.

## Ejecutar la inferencia en C++

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const DetectionResult result = object_detector->Detect(*frame_buffer).value();
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/object_detector.h) para ver más opciones para configurar `ObjectDetector`.

## Resultados de ejemplo

Aquí tiene un ejemplo de los resultados de detección de [ssd mobilenet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1) de TensorFlow Hub.

<img src="images/dogs.jpg" width="50%" alt="perros">

```
Results:
 Detection #0 (red):
  Box: (x: 355, y: 133, w: 190, h: 206)
  Top-1 class:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1 (green):
  Box: (x: 103, y: 15, w: 138, h: 369)
  Top-1 class:
   index       : 17
   score       : 0.73047
   class name  : dog
```

Renderice los cuadros delimitadores en la imagen de entrada:

<img src="images/detection-output.png" width="50%" alt="salida de detección">

Pruebe la sencilla herramienta demo [CLI para ObjectDetector](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#object-detector) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `ObjectDetector` espera un modelo TFLite con [Metadatos del modelo TFLite](../../models/convert/metadata) como requisito. Vea ejemplos de creación de metadatos para clasificadores de audio usando la [API de escritura de metadatos de TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#object_detectors).

Los modelos del detector de objetos compatibles deben cumplir los siguientes requisitos:

- Tensor de imagen de entrada: (kTfLiteUInt8/kTfLiteFloat32)

    - entrada de imagen de tamaño `[batch x height x width x channels]`.
    - no se admite la inferencia por lotes (`batch` debe ser 1).
    - sólo se admiten entradas RGB (`channels` debe ser 3).
    - si el tipo es kTfLiteFloat32, se requiere adjuntar NormalizationOptions a los metadatos para la normalización de la entrada.

- Los tensores de salida deben ser las 4 salidas de una op `DetectionPostProcess`, es decir:

    - Tensor de ubicaciones (kTfLiteFloat32)

        - tensor de tamaño `[1 x num_results x 4]`, el arreglo interior que representa las cajas delimitadoras de la forma [arriba, izquierda, derecha, abajo].
        - Las BoundingBoxProperties deben adjuntarse a los metadatos y deben especificar `type=BOUNDARIES` y coordinate_type=RATIO.

    - Tensor de clases (kTfLiteFloat32)

        - tensor de tamaño `[1 x num_results]`, cada valor representa el índice entero de una clase.
        - se pueden adjuntar mapas de etiquetas opcionales (pero recomendables) como AssociatedFile-s con tipo TENSOR_VALUE_LABELS, que contienen una etiqueta por línea. Consulte el [archivo de etiquetas de ejemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/object_detector/labelmap.txt). El primero de estos AssociatedFile (si lo hay) se usa para llenar el campo `class_name` de los resultados. El campo `display_name` se llena a partir del AssociatedFile (si lo hay) cuya configuración regional coincida con el campo `display_names_locale` del `ObjectDetectorOptions` usado en el momento de la creación ("en" de forma predeterminada, es decir, inglés). Si no se dispone de ninguno de ellos, sólo se llenará el campo `index` de los resultados.

    - Tensor de puntuaciones (kTfLiteFloat32)

        - tensor de tamaño `[1 x num_results]`, cada valor representa la puntuación del objeto detectado.

    - Número del tensor de detección (kTfLiteFloat32)

        - entero num_results como un tensor de tamaño `[1]`.

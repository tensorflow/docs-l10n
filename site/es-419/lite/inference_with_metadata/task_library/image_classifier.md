# Integrar clasificadores de imágenes

La clasificación de imágenes es un uso común del aprendizaje automático para identificar lo que representa una imagen. Por ejemplo, podemos querer saber qué tipo de animal aparece en una imagen dada. La tarea de predecir lo que representa una imagen se denomina *clasificación de imágenes*. Un clasificador de imágenes se entrena para reconocer varias clases de imágenes. Por ejemplo, un modelo podría entrenarse para reconocer fotos que representen tres tipos diferentes de animales: conejos, hámsters y perros. Consulte la [visión general de la clasificación de imágenes](https://www.tensorflow.org/lite/examples/image_classification/overview) para saber más sobre los clasificadores de imágenes.

Use la API de la librería de tareas `ImageClassifier` para implementar sus clasificadores de imágenes personalizados o preentrenados en sus apps móviles.

## Características principales de la API ImageClassifier

- Procesamiento de imágenes de entrada, incluyendo rotación, redimensionamiento y conversión del espacio de color.

- Región de interés de la imagen de entrada.

- Etiquetar la localización del mapa.

- Umbral de puntuaciones para filtrar los resultados.

- Resultados de la clasificación Top-k.

- Etiquetar en allowlist y denylist.

## Modelos de clasificador de imágenes compatibles

Se garantiza que los siguientes modelos son compatibles con la API `ImageClassifier`.

- Modelos creados por [Model Maker de TensorFlow Lite para la clasificación de imágenes](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

- Los [modelos de clasificación de imágenes preentrenados en TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1).

- Modelos creados por [Clasificación de imágenes de borde de visión AutoML](https://cloud.google.com/vision/automl/docs/edge-quickstart).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

Consulte la [app de referencia de clasificación de imágenes](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) para ver un ejemplo de cómo usar `ImageClassifier` en una app de Android.

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

### Paso 2: Usar el modelo

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

Consulte el [código fuente y javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java) para ver más opciones para configurar `ImageClassifier`.

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
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective-C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h) para ver más opciones para configurar `TFLImageClassifier`.

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
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_classifier.py) para ver más opciones para configurar `ImageClassifier`.

## Ejecutar la inferencia en C++

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_classifier.h) para ver más opciones para configurar `ImageClassifier`.

## Resultados de ejemplo

Aquí tiene un ejemplo de los resultados de clasificación de un [clasificador de aves](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3).

<img src="images/sparrow.jpg" width="50%" alt="gorrión">

```
Results:
  Rank #0:
   index       : 671
   score       : 0.91406
   class name  : /m/01bwb9
   display name: Passer domesticus
  Rank #1:
   index       : 670
   score       : 0.00391
   class name  : /m/01bwbt
   display name: Passer montanus
  Rank #2:
   index       : 495
   score       : 0.00391
   class name  : /m/0bwm6m
   display name: Passer italiae
```

Pruebe la sencilla herramienta demo [CLI para ImageClassifier](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `ImageClassifier` espera un modelo TFLite con [Metadatos del modelo TFLite](https://www.tensorflow.org/lite/models/convert/metadata) obligatorios. Vea ejemplos de creación de metadatos para clasificadores de imágenes usando la [API escritora de metadatos de TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers).

Los modelos clasificadores de imágenes compatibles deben cumplir los siguientes requisitos:

- Tensor de imagen de entrada (kTfLiteUInt8/kTfLiteFloat32)

    - entrada de imagen de tamaño `[batch x height x width x channels]`.
    - no se admite la inferencia por lotes (`batch` debe ser 1).
    - sólo se admiten entradas RGB (`channels` debe ser 3).
    - si el tipo es kTfLiteFloat32, se requiere adjuntar NormalizationOptions a los metadatos para la normalización de la entrada.

- Tensor de puntuaciones de salida (kTfLiteUInt8/kTfLiteFloat32)

    - con `N` clases y 2 o 4 dimensiones, es decir, `[1 x N]` o `[1 x 1 x 1 x N]`
    - mapa(s) de etiquetas opcional(es) (pero recomendables) como AssociatedFile-s con tipo TENSOR_AXIS_LABELS, que contiene(n) una etiqueta por línea. Consulte el [archivo de etiquetas de ejemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt). El primero de tales AssociatedFile (si lo hay) se usa para llenar el campo `label` (nombrado `class_name` en C++) de los resultados. El campo `display_name` se llena a partir del AssociatedFile (si existe) cuya configuración regional coincida con el campo `display_names_locale` del `ImageClassifierOptions` usado en el momento de la creación ("en" de forma predeterminada, es decir, inglés). Si no se dispone de ninguno de ellos, sólo se llenará el campo `index` de los resultados.

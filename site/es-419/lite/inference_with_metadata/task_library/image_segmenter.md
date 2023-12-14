# Integrar segmentadores de imágenes

Los segmentadores de imágenes predicen si cada pixel de una imagen está asociado a una clase determinada. Esto contrasta con la <a href="../../examples/object_detection/overview">detección de objetos</a>, que detecta objetos en regiones rectangulares, y la <a href="../../examples/image_classification/overview">clasificación de imágenes</a>, que clasifica la imagen en su conjunto. Consulte la [visión general de la segmentación de imágenes](../../examples/segmentation/overview) para saber más sobre los segmentadores de imágenes.

Use la API de la librería de tareas `ImageSegmenter` para implementar sus segmentadores de imágenes personalizados o preentrenados en sus apps móviles.

## Características principales de la API ImageSegmenter

- Procesamiento de imágenes de entrada, incluyendo rotación, redimensionamiento y conversión del espacio de color.

- Etiquetar la localización del mapa.

- Dos tipos de salida, máscara de categoría y máscaras de confianza.

- Etiqueta de color para su visualización.

## Modelos de segmentadores de imágenes compatibles

Se garantiza que los siguientes modelos son compatibles con la API `ImageSegmenter`.

- Los [modelos de segmentación de imágenes preentrenados en TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en Java

Consulte la [app de referencia de segmentación de imágenes](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/) para ver un ejemplo de cómo usar `ImageSegmenter` en una app de Android.

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
ImageSegmenterOptions options =
    ImageSegmenterOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setOutputType(OutputType.CONFIDENCE_MASK)
        .build();
ImageSegmenter imageSegmenter =
    ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

Consulte el [código fuente y javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/segmenter/ImageSegmenter.java) para ver más opciones para configurar `ImageSegmenter`.

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
guard let modelPath = Bundle.main.path(forResource: "deeplabv3",
                                            ofType: "tflite") else { return }

let options = ImageSegmenterOptions(modelPath: modelPath)

// Configure any additional options:
// options.outputType = OutputType.confidenceMasks

let segmenter = try ImageSegmenter.segmenter(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "plane.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let segmentationResult = try segmenter.segment(mlImage: mlImage)
```

#### Objective C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"deeplabv3" ofType:@"tflite"];

TFLImageSegmenterOptions *options =
    [[TFLImageSegmenterOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.outputType = TFLOutputTypeConfidenceMasks;

TFLImageSegmenter *segmenter = [TFLImageSegmenter imageSegmenterWithOptions:options
                                                                      error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"plane.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLSegmentationResult *segmentationResult =
    [segmenter segmentWithGMLImage:gmlImage error:nil];
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageSegmenter.h) para más opciones de configuración de `ImageSegmenter`.

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
segmentation_options = processor.SegmentationOptions(
    output_type=processor.SegmentationOptions.OutputType.CATEGORY_MASK)
options = vision.ImageSegmenterOptions(base_options=base_options, segmentation_options=segmentation_options)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Alternatively, you can create an image segmenter in the following manner:
# segmenter = vision.ImageSegmenter.create_from_file(model_path)

# Run inference
image_file = vision.TensorImage.create_from_file(image_path)
segmentation_result = segmenter.segment(image_file)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_segmenter.py) para ver más opciones para configurar `ImageSegmenter`.

## Ejecutar la inferencia en C++

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const SegmentationResult result = image_segmenter->Segment(*frame_buffer).value();
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_segmenter.h) para ver más opciones para configurar `ImageSegmenter`.

## Resultados de ejemplo

Este es un ejemplo de los resultados de segmentación de [deeplab_v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/1), un modelo de segmentación genérico disponible en TensorFlow Hub.

<img src="images/plane.jpg" width="50%" alt="avión">

```
Color Legend:
 (r: 000, g: 000, b: 000):
  index       : 0
  class name  : background
 (r: 128, g: 000, b: 000):
  index       : 1
  class name  : aeroplane

# (omitting multiple lines for conciseness) ...

 (r: 128, g: 192, b: 000):
  index       : 19
  class name  : train
 (r: 000, g: 064, b: 128):
  index       : 20
  class name  : tv
Tip: use a color picker on the output PNG file to inspect the output mask with
this legend.
```

La máscara de categoría de segmentación debería tener el siguiente aspecto:

<img src="images/segmentation-output.png" width="30%" alt="salida de segmentación">

Pruebe la sencilla herramienta demo [CLI para BertQuestionAnswerer](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `ImageSegmenter` espera un modelo TFLite con [Metadatos del modelo TFLite](../../models/convert/metadata) obligatorios. Vea ejemplos de creación de metadatos para segmentadores de imágenes usando la [API escritora de metadatos de TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#image_segmenters).

- Tensor de imagen de entrada (kTfLiteUInt8/kTfLiteFloat32)

    - entrada de imagen de tamaño `[batch x height x width x channels]`.
    - no se admite la inferencia por lotes (`batch` debe ser 1).
    - sólo se admiten entradas RGB (`channels` debe ser 3).
    - si el tipo es kTfLiteFloat32, se requiere adjuntar NormalizationOptions a los metadatos para la normalización de la entrada.

- Tensor de máscaras de salida: (kTfLiteUInt8/kTfLiteFloat32)

    - tensor de tamaño `[batch x mask_height x mask_width x num_classes]`, donde `batch` debe ser 1, `mask_width` y `mask_height` son las dimensiones de las máscaras de segmentación producidas por el modelo, y `num_classes` es el número de clases admitidas por el modelo.
    - mapa(s) de etiquetas opcional(es) (pero recomendables) como AssociatedFile-s con tipo TENSOR_AXIS_LABELS, conteniendo una etiqueta por línea. El primero de tales AssociatedFile (si existe) se usa para llenar el campo `label` (nombrado como `class_name` en C++) de los resultados. El campo `display_name` se llena a partir del AssociatedFile (si existe) cuya configuración regional coincida con el campo `display_names_locale` del `ImageSegmenterOptions` usado en el momento de la creación ("en" por predeterminado, es decir, inglés). Si ninguno de ellos está disponible, sólo se llenará el campo `index` de los resultados.

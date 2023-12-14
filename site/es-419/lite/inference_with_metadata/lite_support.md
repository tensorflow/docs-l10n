# Procesar datos de entrada y salida con la biblioteca de soporte TensorFlow Lite

Nota: Actualmente, la librería de soporte de TensorFlow Lite sólo es compatible con Android.

Los desarrolladores de aplicaciones móviles suelen interactuar con objetos tipados, como mapas de bits, o primitivos, como números enteros. Sin embargo, la API del intérprete de TensorFlow Lite que ejecuta el modelo de aprendizaje automático en el dispositivo usa tensores en forma de ByteBuffer, que pueden ser difíciles de depurar y manipular. La [Librería de soporte de TensorFlow Lite para Android](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) está diseñada para ayudar a procesar la entrada y la salida de los modelos de TensorFlow Lite, y facilitar el uso del intérprete de TensorFlow Lite.

## Empecemos

### Importar la dependencia de Gradle y otras configuraciones

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

    // Import tflite dependencies
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // The GPU delegate library is optional. Depend on it as needed.
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly-SNAPSHOT'
}
```

Nota: a partir de la versión 4.1 del plugin Gradle para Android, .tflite se añadirá a la lista noCompress de forma predeterminada y el aaptOptions anterior ya no será necesario.

Explore el AAR de la [librería de soporte de TensorFlow Lite alojada en MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support) para conocer las diferentes versiones de la librería de soporte.

### Manipulación y conversión básicas de imágenes

La librería de soporte TensorFlow Lite dispone de un conjunto de métodos básicos de manipulación de imágenes, como recortar y redimensionar. Para usarlo, cree un `ImagePreprocessor` y añada las operaciones requeridas. Para convertir la imagen al formato de tensor requerido por el intérprete de TensorFlow Lite, cree un `TensorImage` para usarlo como entrada:

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
TensorImage tensorImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tensorImage.load(bitmap);
tensorImage = imageProcessor.process(tensorImage);
```

`DataType` de un tensor puede leerse a través de la [librería extractora de metadatos](../models/convert/metadata.md#read-the-metadata-from-models), así como otra información del modelo.

### Procesamiento básico de datos de audio

La librería de soporte de TensorFlow Lite también define una clase `TensorAudio` que encapsula algunos métodos básicos de procesamiento de datos de audio. Se usa principalmente junto con [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) y captura muestras de audio en un buffer de anillo.

```java
import android.media.AudioRecord;
import org.tensorflow.lite.support.audio.TensorAudio;

// Create an `AudioRecord` instance.
AudioRecord record = AudioRecord(...)

// Create a `TensorAudio` object from Android AudioFormat.
TensorAudio tensorAudio = new TensorAudio(record.getFormat(), size)

// Load all audio samples available in the AudioRecord without blocking.
tensorAudio.load(record)

// Get the `TensorBuffer` for inference.
TensorBuffer buffer = tensorAudio.getTensorBuffer()
```

### Crear objetos de salida y ejecutar el modelo

Antes de ejecutar el modelo, necesitamos crear los objetos contenedores que almacenarán el resultado:

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

Cargar el modelo y ejecutar la inferencia:

```java
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.InterpreterApi;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    InterpreterApi tflite = new InterpreterFactory().create(
        tfliteModel, new InterpreterApi.Options());
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != tflite) {
    tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
}
```

### Acceder al resultado

Los desarrolladores pueden acceder a la salida directamente a través de `probabilityBuffer.getFloatArray()`. Si el modelo produce una salida cuantizada, recuerde convertir el resultado. Para el modelo cuantificado MobileNet, el desarrollador necesita dividir cada valor de salida por 255 para obtener la probabilidad que va de 0 (menos probable) a 1 (más probable) para cada categoría.

### Opcional: Mapear resultados a etiquetas

Los desarrolladores también pueden mapear opcionalmente los resultados a etiquetas. En primer lugar, copie el archivo de texto que contiene las etiquetas en el directorio de activos del módulo. A continuación, cargue el archivo de etiquetas usando el siguiente código:

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List<String> associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

El siguiente fragmento demuestra cómo asociar las probabilidades a las etiquetas de categoría:

```java
import java.util.Map;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map<String, Float> floatMap = labels.getMapWithFloatValue();
}
```

## Cobertura actual de casos de uso

La versión actual de la librería de soporte TensorFlow Lite cubre:

- tipos de datos comunes (float, uint8, imágenes, audio y arreglos de estos objetos) como entradas y salidas de los modelos tflite.
- operaciones básicas de imagen (recortar imagen, redimensionar y rotar).
- normalización y cuantización
- herramientas de archivo

Las versiones futuras mejorarán la compatibilidad con las aplicaciones relacionadas con el texto.

## Arquitectura del ImageProcessor

El diseño del `ImageProcessor` permitió definir por adelantado las operaciones de manipulación de imágenes y optimizarlas durante el proceso de construcción. `ImageProcessor` admite actualmente tres operaciones básicas de preprocesamiento, como se describe en los tres comentarios del fragmento de código siguiente:

```java
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size, size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees / 90))
        .add(new NormalizeOp(127.5, 127.5))
        .add(new QuantizeOp(128.0, 1/128.0))
        .build();
```

Vea más detalles [aquí](../models/convert/metadata.md#normalization-and-quantization-parameters) sobre la normalización y la cuantización.

La meta final de la librería de soporte es soportar todas las transformaciones [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image). Esto significa que la transformación será la misma que TensorFlow y la implementación será independiente del sistema operativo.

Los desarrolladores también pueden crear procesadores personalizados. En estos casos es importante que estén alineados con el proceso de entrenamiento, es decir, que se aplique el mismo preprocesamiento tanto al entrenamiento como a la inferencia para aumentar la reproducibilidad.

## Cuantización

Al iniciar objetos de entrada o salida como `TensorImage` o `TensorBuffer` debe especificar que sus tipos sean `DataType.UINT8` o `DataType.FLOAT32`.

```java
TensorImage tensorImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

`TensorProcessor` puede usarse para cuantizar tensores de entrada o para decuantizar tensores de salida. Por ejemplo, al procesar una salida cuantizada `TensorBuffer`, el desarrollador puede usar `DequantizeOp` para decuantizar el resultado a una probabilidad en punto flotante entre 0 y 1:

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

Los parámetros de cuantización de un tensor pueden leerse a través de la [librería de extractor de metadatos](../models/convert/metadata.md#read-the-metadata-from-models).

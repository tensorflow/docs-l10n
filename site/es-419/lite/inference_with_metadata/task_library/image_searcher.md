# Integrar buscadores de imágenes

La búsqueda de imágenes permite buscar imágenes similares en una base de datos. Funciona incrustando la consulta de búsqueda en un vector de alta dimensión que representa el significado semántico de la consulta, seguido de una búsqueda por similitud en un índice predefinido y personalizado usando [ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Vecinos más cercanos escalables).

A diferencia de la [clasificación de imágenes](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier), ampliar el número de elementos que pueden reconocerse no requiere volver a entrenar todo el modelo. Se pueden añadir nuevos elementos simplemente reconstruyendo el índice. Esto también permite trabajar con bases de datos de imágenes más grandes (más de 100,000 elementos).

Use la API `ImageSearcher` de la librería de tareas para implementar su buscador de imágenes personalizado en sus apps móviles.

## Características principales de la API ImageSearcher

- Toma una sola imagen como entrada, realiza la extracción de la incrustación y la búsqueda del vecino más próximo en el índice.

- Procesamiento de imágenes de entrada, incluyendo rotación, redimensionamiento y conversión del espacio de color.

- Región de interés de la imagen de entrada.

## Requisitos previos

Antes de usar la API `ImageSearcher`, es necesario generar un índice basado en el corpus de imágenes personalizado en el que se va a buscar. Esto puede lograrse usando la API [Model Maker Searcher](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) siguiendo y adaptando el [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher).

Para ello necesitará:

- un modelo de incrustación de imágenes TFLite como [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1). Vea más modelos de incrustación preentrenados (también conocidos como modelos de vectores de características) en la [colección de módulos de imagen de Google en TensorFlow Hub](https://tfhub.dev/google/collections/image/1).
- su corpus de imágenes.

Después de este paso, debería tener un modelo de buscador TFLite independiente (por ejemplo, `mobilenet_v3_searcher.tflite`), que es el modelo de incrustación de imágenes original con el índice adjunto en los [Metadatos del modelo TFLite](https://www.tensorflow.org/lite/models/convert/metadata).

## Ejecutar la inferencia en Java

### Paso 1: Importar la dependencia de Gradle y otras configuraciones

Copie el archivo del modelo de buscador `.tflite` en el directorio de activos del módulo Android en el que se ejecutará el modelo. Especifique que el archivo no debe comprimirse y añada la librería TensorFlow Lite al archivo `build.gradle` del módulo:

```java
android {
    // Other settings

    // Specify tflite index file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.4'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
}
```

### Paso 2: Usar el modelo

```java
// Initialization
ImageSearcherOptions options =
    ImageSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
ImageSearcher imageSearcher =
    ImageSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = imageSearcher.search(image);
```

Consulte el [código fuente y el javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/searcher/ImageSearcher.java) para ver más opciones para configurar el `ImageSearcher`.

## Ejecutar la inferencia en C++

```c++
// Initialization
ImageSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<ImageSearcher> image_searcher = ImageSearcher::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const SearchResult result = image_searcher->Search(*frame_buffer).value();
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_searcher.h) para ver más opciones para configurar `ImageSearcher`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete Pypi de soporte de TensorFlow Lite.

Puede instalar el paquete TensorFlow Lite Support Pypi utilizando el siguiente comando:

```sh
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_searcher.py) para ver más opciones para configurar `ImageSearcher`.

## Resultados de ejemplo

```
Results:
 Rank#0:
  metadata: burger
  distance: 0.13452
 Rank#1:
  metadata: car
  distance: 1.81935
 Rank#2:
  metadata: bird
  distance: 1.96617
 Rank#3:
  metadata: dog
  distance: 2.05610
 Rank#4:
  metadata: cat
  distance: 2.06347
```

Pruebe la sencilla herramienta demo [CLI para ImageSearcher](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher) con su propio modelo y datos de prueba.

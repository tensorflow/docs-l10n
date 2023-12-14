# Integrar incrustadores de imágenes

Los incrustadores de imágenes permiten incrustar imágenes en un vector de características de alta dimensión que representa el significado semántico de una imagen, que luego puede compararse con el vector de características de otras imágenes para evaluar su similitud semántica.

A diferencia de la [búsqueda de imágenes](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher), el incrustador de imágenes permite calcular la similitud entre imágenes sobre la marcha en lugar de buscar a través de un índice predefinido construido a partir de un corpus de imágenes.

Use la API `ImageEmbedder` de la librería de tareas para implementar su incrustador de imágenes personalizado en sus apps móviles.

## Características principales de la API ImageEmbedder

- Procesamiento de imágenes de entrada, incluyendo rotación, redimensionamiento y conversión del espacio de color.

- Región de interés de la imagen de entrada.

- Función de utilidad incorporada para calcular la [similitud de coseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre vectores de características.

## Modelos de incrustadores de imágenes compatibles

Se garantiza que los siguientes modelos son compatibles con la API `ImageEmbedder`.

- Modelos de vectores de características de la colección [Módulos de imágenes de Google en TensorFlow Hub](https://tfhub.dev/google/collections/image/1).

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en C++

```c++
// Initialization
ImageEmbedderOptions options:
options.mutable_model_file_with_metadata()->set_file_name(model_path);
options.set_l2_normalize(true);
std::unique_ptr<ImageEmbedder> image_embedder = ImageEmbedder::CreateFromOptions(options).value();

// Create input frame_buffer_1 and frame_buffer_2 from your inputs `image_data1`, `image_data2`, `image_dimension1` and `image_dimension2`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer_1 = CreateFromRgbRawBuffer(
      image_data1, image_dimension1);
std::unique_ptr<FrameBuffer> frame_buffer_2 = CreateFromRgbRawBuffer(
      image_data2, image_dimension2);

// Run inference on two images.
const EmbeddingResult result_1 = image_embedder->Embed(*frame_buffer_1);
const EmbeddingResult result_2 = image_embedder->Embed(*frame_buffer_2);

// Compute cosine similarity.
double similarity = ImageEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector(),
    result_2.embeddings[0].feature_vector());
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_embedder.h) para ver más opciones para configurar `ImageEmbedder`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete Pypi de soporte de TensorFlow Lite.

Puede instalar el paquete TensorFlow Lite Support Pypi utilizando el siguiente comando:

```sh
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
from tflite_support.task import vision

# Initialization.
image_embedder = vision.ImageEmbedder.create_from_file(model_path)

# Run inference on two images.
image_1 = vision.TensorImage.create_from_file('/path/to/image1.jpg')
result_1 = image_embedder.embed(image_1)
image_2 = vision.TensorImage.create_from_file('/path/to/image2.jpg')
result_2 = image_embedder.embed(image_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = image_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_embedder.py) para ver más opciones para configurar `ImageEmbedder`.

## Resultados de ejemplo

La similitud del coseno entre los vectores de características normalizados devuelve una puntuación entre -1 y 1. Cuanto más alta, mejor, es decir, una similitud del coseno de 1 significa que los dos vectores son idénticos.

```
Cosine similarity: 0.954312
```

Pruebe la sencilla herramienta demo [CLI para ImageEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `ImageEmbedder` espera un modelo TFLite con [Metadatos del modelo TFLite](https://www.tensorflow.org/lite/models/convert/metadata) opcionales, pero muy recomendables.

Los modelos de incrustadores de imágenes compatibles deben cumplir los siguientes requisitos:

- Un tensor de imagen de entrada (kTfLiteUInt8/kTfLiteFloat32)

    - entrada de imagen de tamaño `[batch x height x width x channels]`.
    - no se admite la inferencia por lotes (`batch` debe ser 1).
    - sólo se admiten entradas RGB (`channels` debe ser 3).
    - si el tipo es kTfLiteFloat32, se requiere adjuntar NormalizationOptions a los metadatos para la normalización de la entrada.

- Al menos un tensor de salida (kTfLiteUInt8/kTfLiteFloat32)

    - con `N` componentes correspondientes a las `N` dimensiones del vector de características devuelto para esta capa de salida.
    - De 2 ó 4 dimensiones, es decir, `[1 x N]` o `[1 x 1 x 1 x N]`.

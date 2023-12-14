# Integrar buscadores de texto

La búsqueda de texto permite buscar texto semánticamente similar en un corpus. Funciona incrustando la consulta de búsqueda en un vector de alta dimensión que representa el significado semántico de la consulta, seguido de una búsqueda por similitud en un índice predefinido y personalizado usando [ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Vecinos más cercanos escalables).

A diferencia de la clasificación de texto (por ejemplo, [clasificador de lenguaje natural Bert](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier)), ampliar el número de elementos que pueden reconocerse no requiere volver a entrenar todo el modelo. Se pueden añadir nuevos elementos simplemente reconstruyendo el índice. Esto también permite trabajar con corpus más grandes (más de 100,000 elementos).

Use la API `TextSearcher` de la librería de tareas para implementar su buscador de texto personalizado en sus apps móviles.

## Características principales de la API TextSearcher

- Toma una sola cadena como entrada, realiza la extracción de la incrustación y la búsqueda del vecino más próximo en el índice.

- Procesamiento del texto de entrada, incluyendo tokenizaciones dentro o fuera del grafo [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) o [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) sobre el texto de entrada.

## Requisitos previos

Antes de usar la API `TextSearcher`, es necesario generar un índice basado en el corpus de imágenes personalizado en el que se va a buscar. Esto puede lograrse usando la API [Model Maker Searcher](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) siguiendo y adaptando el [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher).

Para ello necesitará:

- un modelo de incrustación de texto TFLite, como el codificador universal de frases. Por ejemplo,
    - [el que se ha reentrenado](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/searcher/text_to_image_blogpost/text_embedder.tflite) en este [Colab](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/colab/on_device_text_to_image_search_tflite.ipynb), que está optimizado para la inferencia en el dispositivo. Sólo se tarda 6 ms en consultar una cadena de texto en Pixel 6.
    - el [cuantizado](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1), que es más pequeño que el anterior pero tarda 38 ms por cada incrustación.
- su corpus de texto.

Después de este paso, debería tener un modelo de buscador TFLite independiente (por ejemplo, `mobilenet_v3_searcher.tflite`), que es el modelo de incrustación de texto original con el índice adjunto en los [Metadatos del modelo TFLite](https://www.tensorflow.org/lite/models/convert/metadata).

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
TextSearcherOptions options =
    TextSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
TextSearcher textSearcher =
    textSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = textSearcher.search(text);
```

Consulte el [código fuente y el javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/searcher/TextSearcher.java) para ver más opciones de configuración de `TextSearcher`.

## Ejecutar la inferencia en C++

```c++
// Initialization
TextSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<TextSearcher> text_searcher = TextSearcher::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
const SearchResult result = text_searcher->Search(input_text).value();
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_searcher.h) para más opciones de configuración de `TextSearcher`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete Pypi de soporte de TensorFlow Lite.

Puede instalar el paquete TensorFlow Lite Support Pypi utilizando el siguiente comando:

```sh
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
from tflite_support.task import text

# Initialization
text_searcher = text.TextSearcher.create_from_file(model_path)

# Run inference
result = text_searcher.search(text)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_searcher.py) para más opciones de configuración de `TextSearcher`.

## Resultados de ejemplo

```
Results:
 Rank#0:
  metadata: The sun was shining on that day.
  distance: 0.04618
 Rank#1:
  metadata: It was a sunny day.
  distance: 0.10856
 Rank#2:
  metadata: The weather was excellent.
  distance: 0.15223
 Rank#3:
  metadata: The cat is chasing after the mouse.
  distance: 0.34271
 Rank#4:
  metadata: He was very happy with his newly bought car.
  distance: 0.37703
```

Pruebe la sencilla [herramienta demo CLI para TextSearcher](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textsearcher) con su propio modelo y datos de prueba.

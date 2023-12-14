# Integrar incrustadores de texto.

Los incrustadores de texto permiten incrustar texto en un vector de características de alta dimensión que representa su significado semántico, que luego puede compararse con el vector de características de otros textos para evaluar su similitud semántica.

A diferencia de la [búsqueda de texto](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher), el incrustador de texto permite calcular la similitud entre textos sobre la marcha en lugar de buscar a través de un índice predefinido construido a partir de un corpus.

Use la API `TextEmbedder` de la librería de tareas para implementar su incrustador de texto personalizado en sus apps móviles.

## Características principales de la API TextEmbedder

- Procesamiento del texto de entrada, incluyendo tokenizaciones dentro o fuera del grafo [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) o [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) sobre el texto de entrada.

- Función de utilidad incorporada para calcular la [similitud de coseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre vectores de características.

## Modelos de incrustadores de texto compatibles

Se garantiza que los siguientes modelos son compatibles con la API `TextEmbedder`.

- El modelo [TFLite del codificador universal de frases de TensorFlow Hub](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

- Modelos personalizados que cumplen los [requisitos de compatibilidad de modelos](#model-compatibility-requirements).

## Ejecutar la inferencia en C++

```c++
// Initialization.
TextEmbedderOptions options:
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<TextEmbedder> text_embedder = TextEmbedder::CreateFromOptions(options).value();

// Run inference with your two inputs, `input_text1` and `input_text2`.
const EmbeddingResult result_1 = text_embedder->Embed(input_text1);
const EmbeddingResult result_2 = text_embedder->Embed(input_text2);

// Compute cosine similarity.
double similarity = TextEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector()
    result_2.embeddings[0].feature_vector());
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_embedder.h) para más opciones de configuración de `TextEmbedder`.

## Ejecutar la inferencia en Python

### Paso 1: Instalar el paquete Pypi de soporte de TensorFlow Lite.

Puede instalar el paquete TensorFlow Lite Support Pypi utilizando el siguiente comando:

```sh
pip install tflite-support
```

### Paso 2: Usar el modelo

```python
from tflite_support.task import text

# Initialization.
text_embedder = text.TextEmbedder.create_from_file(model_path)

# Run inference on two texts.
result_1 = text_embedder.embed(text_1)
result_2 = text_embedder.embed(text_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = text_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

Consulte el [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_embedder.py) para más opciones de configuración de `TextEmbedder`.

## Resultados de ejemplo

La similitud del coseno entre los vectores de características normalizados devuelve una puntuación entre -1 y 1. Cuanto más alta, mejor, es decir, una similitud del coseno de 1 significa que los dos vectores son idénticos.

```
Cosine similarity: 0.954312
```

Pruebe la sencilla [herramienta demo CLI para TextEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder) con su propio modelo y datos de prueba.

## Requisitos de compatibilidad del modelo

La API `TextEmbedder` espera un modelo TFLite con [Metadatos del modelo TFLite](https://www.tensorflow.org/lite/models/convert/metadata) como requisito.

Se admiten tres tipos principales de modelos:

- Modelos basados en BERT (véase [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h) para más detalles):

    - Exactamente 3 tensores de entrada (kTfLiteString)

        - Tensor de IDs, con nombre de metadatos "ids",
        - Tensor de máscara, con nombre de metadatos "mask".
        - Tensor de IDs de segmento, con nombre de metadatos "segment_ids"

    - Exactamente un tensor de salida (kTfLiteUInt8/kTfLiteFloat32)

        - con `N` componentes correspondientes a las `N` dimensiones del vector de características devuelto para esta capa de salida.
        - De 2 ó 4 dimensiones, es decir, `[1 x N]` o `[1 x 1 x 1 x N]`.

    - Un input_process_units para el tokenizador Wordpiece/Sentencepiece

- Modelos basados en el codificador universal de frases (véase [código fuente](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h) para más detalles):

    - Exactamente 3 tensores de entrada (kTfLiteString)

        - Tensor de texto de consulta, con nombre de metadatos "inp_text".
        - Tensor de contexto de respuesta, con nombre de metadatos "res_context".
        - Tensor de texto de respuesta, con nombre de metadatos "res_text".

    - Exactamente 2 tensores de salida (kTfLiteUInt8/kTfLiteFloat32)

        - Tensor de codificación de consultas, con nombre de metadatos "query_encoding".
        - Tensor de codificación de la respuesta, con nombre de metadatos "codificación_respuesta".
        - Ambos con `N` componentes correspondientes a las `N` dimensiones del vector de características devuelto para esta capa de salida.
        - Ambas con 2 o 4 dimensiones, es decir, `[1 x N]` o `[1 x 1 x 1 x N]`.

- Cualquier modelo de incrustador de texto con:

    - Un tensor de texto de entrada (kTfLiteString)

    - Al menos un tensor de incrustación de salida (kTfLiteUInt8/kTfLiteFloat32)

        - con `N` componentes correspondientes a las `N` dimensiones del vector de características devuelto para esta capa de salida.
        - de 2 ó 4 dimensiones, es decir, `[1 x N]` o `[1 x 1 x 1 x N]`.

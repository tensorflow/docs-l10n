# Guía paso a paso de texto

En esta página, encontrará un conjunto de guías y herramientas conocidas que resuelven problemas en el dominio del texto con TensorFlow Hub. Es un punto de partida para cualquiera que quiera resolver problemas típicos de aprendizaje automático con componentes de aprendizaje automático preentrenados en lugar de empezar desde cero.

## Clasificación

Cuando queremos predecir una clase para un ejemplo determinado, por ejemplo **sentimiento**, **toxicidad**, **categoría del artículo** o cualquier otra característica.

![Gráfico de clasificación de texto](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-classification.png)

Los tutoriales a continuación resuelven la misma tarea desde diferentes perspectivas y con diferentes herramientas.

### Keras

[Clasificación de texto con Keras](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub): ejemplo para crear un clasificador de sentimientos de IMDB con Keras y TensorFlow Datasets.

### Estimator

[Clasificación de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub.ipynb): ejemplo para crear un clasificador de sentimientos de IMDB con Estimator. Contiene múltiples consejos de mejora y una sección de comparación de módulos.

### BERT

[Predecir sentimientos de reseñas de películas con BERT en TF Hub](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb): se muestra cómo usar un módulo BERT para la clasificación. Incluye el uso de la biblioteca `bert` para la tokenización y el preprocesamiento.

### Kaggle

[Clasificación de IMDB en Kaggle](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub_on_kaggle.ipynb): se muestra cómo interactuar fácilmente con una competencia de Kaggle desde un Colab, incluida la descarga de datos y el envío de los resultados.

```
                                                                                                                                                                                     | Estimator                                                                                         | Keras                                                                                             | TF2                                                                                               | TF Datasets                                                                                       | BERT                                                                                              | Kaggle APIs
```

-------------------------------------------------- -------------------------------------------------- -------------------------------------------------- ---------------------------------- | -------------------------------------------------- ----------------------------------------- | -------------------------------------------------- ----------------------------------------- | -------------------------------------------------- ----------------------------------------- | -------------------------------------------------- ----------------------------------------- | -------------------------------------------------- ----------------------------------------- | ----------- [Clasificación de texto](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | | | | | [Clasificación de textos con Keras](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) | | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | | [Predecir la opinión de reseñas de películas con BERT en TF Hub](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | | | | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | [Clasificación de IMDB en Kaggle](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub_on_kaggle.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | | | | | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

### Tarea bengalía con incorporaciones de FastText

TensorFlow Hub no ofrece actualmente un módulo en todos los idiomas. El siguiente tutorial muestra cómo aprovechar TensorFlow Hub para una experimentación rápida y un desarrollo de aprendizaje automático modular.

[Clasificador de artículos en bengalí](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/bangla_article_classifier.ipynb): se muestra cómo crear una incorporación de texto reutilizable de TensorFlow Hub y usarla para entrenar un clasificador de Keras para el [conjunto de datos de artículos en bengalí de BARD](https://github.com/tanvirfahim15/BARD-Bangla-Article-Classifier).

## Similitudes semánticas

Cuando queremos saber qué oraciones se correlacionan entre sí en una configuración zero-shot (sin ejemplos de entrenamiento).

![Gráfico de similitud semántica](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

### Básico

[Similitud semántica](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder.ipynb): se muestra cómo usar el módulo de codificador de oraciones para calcular la similitud de oraciones.

### Entre idiomas

[Similitud semántica entre idiomas](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb): se muestra cómo usar uno de los codificadores de oraciones entre idiomas para calcular la similitud de oraciones entre idiomas.

### Recuperación semántica

[Recuperación semántica](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa.ipynb): se muestra cómo usar el codificador de oraciones de preguntas y respuestas para indexar una colección de documentos para su recuperación en función de la similitud semántica.

### Entrada SentencePiece

[Similitud semántica con el codificador universal lite](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb): se muestra cómo usar módulos de codificador de oraciones que aceptan identificadores [SentencePiece](https://github.com/google/sentencepiece) en la entrada en lugar de texto.

## Creación de módulos

En lugar de usar solo módulos en [tfhub.dev](https://tfhub.dev), existen formas de crear módulos propios. Esta puede ser una herramienta útil para mejorar la modularidad de la base de código de aprendizaje automático y para compartirlo.

### Envolver incorporaciones preentrenadas existentes

[Exportador de módulos de incorporaciones de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py): una herramienta para empaquetar una incorporación preentrenada existente en un módulo. Muestra cómo incluir operaciones de preprocesamiento de texto en el módulo. Esto permite crear un módulo de incorporación de oraciones a partir de incorporaciones de tokens.

[Exportador de módulo de incorporaciones de texto v2](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings_v2/export_v2.py): igual que el anterior, pero compatible con TensorFlow 2 y ejecución eager.

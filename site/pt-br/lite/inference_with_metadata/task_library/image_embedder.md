# Integração de incorporadores de imagem

Os incorporadores de imagem permitem incorporar imagens a um vetor de características de alta dimensão que representa o significado semântico de uma imagem. Em seguida, esse vetor pode ser comparado ao vetor de características de outras imagens para avaliar a similaridade semântica.

Ao contrário da [pesquisa de imagens](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher), o incorporador de imagem permite computar a similaridade entre imagens em tempo real em vez de pesquisar em um índice predefinido criado a partir de um corpus de imagens.

Use a API `ImageEmbedder` da biblioteca Task para implantar seu incorporador de imagem personalizado em seus aplicativos para dispositivos móveis.

## Principais recursos da API ImageEmbedder

- Processamento da imagem de entrada, incluindo rotação, redimensionamento e conversão do espaço de cores.

- Região de interesse da imagem de entrada.

- Função utilitária integrada para computar a [similaridade por cosseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre os vetores de características.

## Modelos de incorporador de imagem com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `ImageEmbedder`.

- Modelos de vetores de características da [coleção de módulos do Google Imagens no TensorFlow Hub](https://tfhub.dev/google/collections/image/1).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no C++

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_embedder.h) para ver mais opções de configuração do `ImageEmbedder`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote Pypi do TensorFlow Lite Support

É possível instalar o pacote Pypi do TensorFlow Lite Support usando o seguinte comando:

```sh
pip install tflite-support
```

### Etapa 2 – Use o modelo

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_embedder.py) para ver mais opções de configuração do `ImageEmbedder`.

## Exemplo de resultados

A similaridade por cosseno entre os vetores de características normalizados retorna uma pontuação entre -1 e 1. Quanto maior, melhor; uma similaridade por cosseno igual a 1 significa que os dois vetores são idênticos.

```
Cosine similarity: 0.954312
```

Experimente a [ferramenta CLI simples de demonstração para ImageEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `ImageEmbedder` espera um modelo do TF Lite com os [metadados de modelo do TFLite](https://www.tensorflow.org/lite/models/convert/metadata), que são opcionais, mas extremamente recomendados.

Os modelos compatíveis de incorporador de imagem devem atender aos seguintes requisitos:

- Um tensor de imagem de entrada (kTfLiteUInt8/kTfLiteFloat32).

    - Imagem de entrada de tamanho `[batch x height x width x channels]` (lote x altura x largura x canais).
    - Não há suporte à inferência de lote (`batch` precisa ser igual a 1).
    - Só há suporte a entradas RGB (`channels` precisa ser igual a 3).
    - Se o tipo for kTfLiteFloat32, as opções NormalizationOptions precisam ser adicionadas aos metadados para a normalização da entrada.

- Pelo menos um tensor de saída (kTfLiteUInt8/kTfLiteFloat32).

    - Com `N` componentes que correspondem às `N` dimensões do vetor de características retornado para essa camada de saída.
    - Duas ou quatro dimensões, ou seja, `[1 x N]` ou `[1 x 1 x 1 x N]`.

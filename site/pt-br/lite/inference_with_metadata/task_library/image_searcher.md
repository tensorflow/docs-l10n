# Integração de pesquisadores de imagem

A pesquisa de imagens permite procurar imagens similares em um banco de dados de imagens. Funciona pela incorporação de uma consulta em um vetor de alta dimensão que representa o significado semântico da consulta, seguida por pesquisa de similaridade em um índice personalizado e predefinido usando [ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Scalable Nearest Neighbors).

Ao contrário da [classificação de imagem](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier), expandir o número de itens que podem ser reconhecidos não requer treinar novamente todo o modelo. Novos itens podem ser adicionados simplesmente recriando o índice, o que permite trabalhar com bancos de imagens maiores (mais de 100 mil itens).

Use a API `ImageSearcher` da biblioteca Task para implantar seu pesquisador de imagem personalizado em seus aplicativos para dispositivos móveis.

## Principais recursos da API ImageSearcher

- Recebe uma única imagem como entrada, realiza a extração de embeddings e procura o vizinho mais próximo no índice.

- Processamento da imagem de entrada, incluindo rotação, redimensionamento e conversão do espaço de cores.

- Região de interesse da imagem de entrada.

## Pré-requisitos

Antes de usar a API `ImageSearcher`, é preciso criar um índice com base no corpus personalizado de imagens que será pesquisado. Para fazer isso, basta usar a [API Model Maker Searcher](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) seguindo e adaptando o [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher).

Para isso, você precisará de:

- Um modelo de incorporador de imagem do TF Lite, como [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1). Confira mais modelos de incorporador pré-treinados (também conhecidos como modelos de vetores) na [coleção Google Image Modules do TensorFlow Hub](https://tfhub.dev/google/collections/image/1).
- Seu corpus de imagens.

Após essa etapa, você terá um modelo de pesquisador do TF Lite independente (por exemplo: `mobilenet_v3_searcher.tflite`), que é o modelo de incorporador de imagem original, com o índice incluído nos [metadados do modelo do TF Lite](https://www.tensorflow.org/lite/models/convert/metadata).

## Execute a inferência no Java

### Etapa 1 – Importe a dependência e outras configurações do Gradle

Copie o arquivo do modelo de pesquisador `.tflite` para o diretório de ativos do módulo para Android no qual o modelo será executado. Especifique que o arquivo não deve ser compactado e adicione a biblioteca do TensorFlow Lite ao arquivo `build.gradle` do modelo:

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

### Etapa 2 – Use o modelo

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

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/searcher/ImageSearcher.java) para ver mais opções de configuração do `ImageSearcher`.

## Execute a inferência no C++

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_searcher.h) para ver mais opções de configuração do `ImageSearcher`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote Pypi do TensorFlow Lite Support

É possível instalar o pacote Pypi do TensorFlow Lite Support usando o seguinte comando:

```sh
pip install tflite-support
```

### Etapa 2 – Use o modelo

```python
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_searcher.py) para ver mais opções de configuração do `ImageSearcher`.

## Exemplo de resultados

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

Experimente a [ferramenta CLI simples de demonstração para ImageSearcher](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher) com seu próprio modelo e dados de teste.

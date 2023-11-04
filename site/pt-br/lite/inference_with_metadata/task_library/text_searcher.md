# Integração de pesquisadores de texto

A pesquisa de texto permite procurar textos semanticamente similares em um corpus. Funciona pela incorporação de uma consulta em um vetor de alta dimensão que representa o significado semântico da consulta, seguida por pesquisa de similaridade em um índice personalizado e predefinido usando [ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Scalable Nearest Neighbors).

Ao contrário da classificação de texto (como o [classificador de linguagem natural BERT](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier)), expandir o número de itens que podem ser reconhecidos não requer treinar novamente todo o modelo. Novos itens podem ser adicionados simplesmente recriando o índice, o que permite trabalhar com corpus maiores (mais de 100 mil itens).

Use a API `TextSearcher` da biblioteca Task para implantar seu pesquisador de texto personalizado em seus aplicativos para dispositivos móveis.

## Principais recursos da API TextSearcher

- Recebe um único texto como entrada, realiza a extração de embeddings e procura o vizinho mais próximo no índice.

- Processamento do texto de entrada, incluindo as tokenizações [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) ou [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) do texto de entrada fora do grafo.

## Pré-requisitos

Antes de usar a API `TextSearcher`, é preciso criar um índice com base no corpus personalizado de texto que será pesquisado. Para fazer isso, basta usar a [API Model Maker Searcher](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher) seguindo e adaptando o [tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher).

Para isso, você precisará de:

- Um modelo de incorporador de texto do TF Lite, como Universal Sentence Encoder. Por exemplo:
    - O [modelo](https://storage.googleapis.com/download.tensorflow.org/models/tflite_support/searcher/text_to_image_blogpost/text_embedder.tflite) treinado novamente neste [Colab](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/colab/on_device_text_to_image_search_tflite.ipynb), que é otimizado para inferência no dispositivo. Ele leva somente 6 milissegundos para consultar uma string de texto no Pixel 6.
    - O modelo [quantizado](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1), que é menor do que o acima, mas leva 38 milissegundos para cada embedding.
- Seu corpus de texto.

Após essa etapa, você terá um modelo de pesquisador do TF Lite independente (por exemplo: `mobilenet_v3_searcher.tflite`), que é o modelo de incorporador de texto original, com o índice incluído nos [TF Lite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata) (metadados do modelo).

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

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/searcher/TextSearcher.java) para ver mais opções de configuração do `TextSearcher`.

## Execute a inferência no C++

```c++
// Initialization
TextSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<TextSearcher> text_searcher = TextSearcher::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
const SearchResult result = text_searcher->Search(input_text).value();
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_searcher.h) para ver mais opções de configuração do `TextSearcher`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote Pypi do TensorFlow Lite Support

É possível instalar o pacote Pypi do TensorFlow Lite Support usando o seguinte comando:

```sh
pip install tflite-support
```

### Etapa 2 – Use o modelo

```python
from tflite_support.task import text

# Initialization
text_searcher = text.TextSearcher.create_from_file(model_path)

# Run inference
result = text_searcher.search(text)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_searcher.py) para ver mais opções de configuração do `TextSearcher`.

## Exemplo de resultados

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

Experimente a [ferramenta CLI simples de demonstração para TextSearcher](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textsearcher) com seu próprio modelo e dados de teste.

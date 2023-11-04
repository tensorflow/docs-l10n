# Integração de incorporadores de texto

Os incorporadores de texto permitem incorporar texto a um vetor de características de alta dimensão que representa seu significado semântico. Em seguida, esse vetor pode ser comparado ao vetor de características de outros textos para avaliar a similaridade semântica.

Ao contrário da [pesquisa de texto](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher), o incorporador de texto permite computar a similaridade entre textos em tempo real em vez de pesquisar em um índice predefinido criado a partir de um corpus.

Use a API `TextEmbedder` da biblioteca Task para implantar seu incorporador de texto personalizado em seus aplicativos para dispositivos móveis.

## Principais recursos da API TextEmbedder

- Processamento do texto de entrada, incluindo as tokenizações [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) ou [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) do texto de entrada fora do grafo.

- Função utilitária integrada para computar a [similaridade por cosseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre os vetores de características.

## Modelos de incorporador de texto com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `TextEmbedder`.

- [Modelo Universal Sentence Encoder do TF Lite no TensorFlow Hub](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no C++

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_embedder.h) para ver mais opções de configuração do `TextEmbedder`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote Pypi do TensorFlow Lite Support

É possível instalar o pacote Pypi do TensorFlow Lite Support usando o seguinte comando:

```sh
pip install tflite-support
```

### Etapa 2 – Use o modelo

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

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_embedder.py) para ver mais opções de configuração do `TextEmbedder`.

## Exemplo de resultados

A similaridade por cosseno entre os vetores de características normalizados retorna uma pontuação entre -1 e 1. Quanto maior, melhor; uma similaridade por cosseno igual a 1 significa que os dois vetores são idênticos.

```
Cosine similarity: 0.954312
```

Experimente a [ferramenta CLI simples de demonstração para TextEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `TextEmbedder` espera um modelo do TF Lite com os [TF Lite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata) (metadados de modelo) obrigatórios.

Há suporte a três tipos principais de modelos:

- Modelos baseados em BERT (confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h)):

    - Exatamente 3 tensores de entrada (kTfLiteString):

        - Tensor de IDs, com nome de metadados "ids".
        - Tensor de máscara, com nome de metadados "mask".
        - Tensor de IDs de segmentos, com nome de metadados "segment_ids".

    - Exatamente um tensor de saída (kTfLiteUInt8/kTfLiteFloat32):

        - Com `N` componentes que correspondem às `N` dimensões do vetor de características retornado para essa camada de saída.
        - Duas ou quatro dimensões, ou seja, `[1 x N]` ou `[1 x 1 x 1 x N]`.

    - input_process_units para o tokenizador Wordpiece/Sentencepiece.

- Modelos baseados em Universal Sentence Encoder (confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h)):

    - Exatamente 3 tensores de entrada (kTfLiteString):

        - Tensor de texto da consulta, com nome de metadados "inp_text".
        - Tensor de contexto da resposta, com nome de metadados "res_context".
        - Tensor de texto da resposta, com nome de metadados "res_text".

    - Exatamente dois tensores de saída (kTfLiteUInt8/kTfLiteFloat32):

        - Tensor de codificação da consulta, com nome de metadados "query_encoding".
        - Tensor de codificação da resposta, com nome de metadados "response_encoding".
        - Ambos com `N` componentes que correspondem às `N` dimensões do vetor de características retornado para essa camada de saída.
        - Ambos com duas ou quatro dimensões, ou seja, `[1 x N]` ou `[1 x 1 x 1 x N]`.

- Qualquer modelo de incorporador de texto com:

    - Um tensor de texto de entrada (kTfLiteString)

    - Pelo menos um tensor de embedding de saída (kTfLiteUInt8/kTfLiteFloat32)

        - Com `N` componentes que correspondem às `N` dimensões do vetor de características retornado para essa camada de saída.
        - Duas ou quatro dimensões, ou seja, `[1 x N]` ou `[1 x 1 x 1 x N]`.

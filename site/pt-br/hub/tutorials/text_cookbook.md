# Tutoriais para domínio de texto

Esta página lista um conjunto de guias e ferramentas conhecidos para resolver problemas no domínio de texto com o TensorFlow Hub. É um ponto de partida para qualquer pessoa que deseja resolver problemas típicos de aprendizado de máquina usando componentes pré-treinados em vez de começar do zero.

## Classificação

Quando desejamos prever a classe de um exemplo fornecido, como **sentimento**, **toxicidade**, **categoria de artigo** ou qualquer outra característica.

![Text Classification Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-classification.png)

Os tutoriais abaixo resolvem a mesma tarefa sob diferentes perspectivas e usando ferramentas diferentes.

### Keras

[Classificação de texto com o Keras](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub) – exemplo para criar um classificador de sentimentos do IMDB com o Keras e o TensorFlow Datasets.

### Estimator

[Classificação de texto](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub.ipynb) – exemplo para criar um classificador de sentimentos do IMDB com o Estimator (estimador). Contém várias dicas para melhorias e uma seção de comparação de módulos.

### BERT

[Previsão de sentimento em classificações de filmes com BERT no TF Hub](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) – mostra como usar um módulo BERT para classificação. Inclui o uso da biblioteca `bert` para tokenização e pré-processamento.

### Kaggle

[Classificação do IMDB no Kaggle](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub_on_kaggle.ipynb) – mostra como interagir facilmente com uma competição do Kaggle em um Colab, incluindo como baixar os dados e enviar os resultados.

```
                                                                                                                                                                                     | Estimator                                                                                         | Keras                                                                                             | TF2                                                                                               | TF Datasets                                                                                       | BERT                                                                                              | Kaggle APIs
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------- [Classificação de texto](https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub)                                                                                          | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |                                                                                                   |                                                                                                   |                                                                                                   |                                                                                                   | [Classificação de texto com o Keras](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)                                                                                |                                                                                                   | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |                                                                                                   | [Previsão de sentimento em classificações de filmes com BERT  no TF Hub](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)                          | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |                                                                                                   |                                                                                                   |                                                                                                   | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) | [Classificação do IMDB no Kaggle](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/text_classification_with_tf_hub_on_kaggle.ipynb) | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png) |                                                                                                   |                                                                                                   |                                                                                                   |                                                                                                   | ![done](https://www.gstatic.com/images/icons/material/system_gm/1x/bigtop_done_googblue_18dp.png)

### Tarefa em bengali com embeddings FastText

No momento, o TensorFlow Hub não oferece um módulo em cada idioma. O tutorial abaixo mostra como usar o TensorFlow Hub para experimentação rápida e desenvolvimento modular de aprendizado de máquina.

[Classificador de artigos em bengali](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/bangla_article_classifier.ipynb) – demonstra como criar um embedding de texto do TensorFlow Hub reutilizável e usá-lo para treinar um classificador do Keras para o [dataset Artigos em bengali BARD](https://github.com/tanvirfahim15/BARD-Bangla-Article-Classifier).

## Similaridade semântica

Quando queremos descobrir quais frases estão correlacionadas uma com a outra em uma configuração zero-shot (sem exemplos de treinamento).

![Semantic Similarity Graphic](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

### Básico

[Similaridade semântica](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder.ipynb) – mostra como usar o módulo de encoder de frases para computar a similaridade semântica.

### Interlíngua

[Similaridade semântica interlíngua](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb) – mostra como usar um dos encoders de frases interlínguas para computar a similaridade semântica entre idiomas.

### Busca semântica

[Busca semântica](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa.ipynb) – mostra como usar o encoder de frases de perguntas e respostas para indexar um conjunto de documentos para busca com base na similaridade semântica.

### Entrada SentencePiece

[Similaridade semântica com o encoder lite universal](https://github.com/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb) – mostra como usar os módulos de encoder de frases que aceitam IDs de [SentencePiece](https://github.com/google/sentencepiece) como entrada em vez de texto.

## Criação de módulos

Em vez de usar somente módulos em [tfhub.dev](https://tfhub.dev), existem formas de criar seus próprios módulos, o que pode ser útil para uma melhor modularidade do código-base de aprendizado de máquina e para futuro compartilhamento.

### Encapsulamento de embeddings pré-treinados existentes

[Exportador de módulos de embedding de texto](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings/export.py) – ferramenta para encapsular um embedding pré-treinado existente em um módulo. Mostra como incluir operações de pré-processamento de texto no módulo. Isso permite criar um módulo de embedding de frases a partir de embeddings de tokens.

[Exportador de módulos de embedding de texto v2](https://github.com/tensorflow/hub/blob/master/examples/text_embeddings_v2/export_v2.py) – igual ao acima, mas compatível com o TensorFlow 2 e execução eager.

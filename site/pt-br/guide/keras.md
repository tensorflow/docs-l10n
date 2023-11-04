# Keras: API de alto nível do TensorFlow

O Keras é a API de alto nível da plataforma TensorFlow e conta com uma interface acessível e altamente produtiva para resolver problemas de aprendizado de máquina (ML), com foco no aprendizado profundo moderno. O Keras abrange cada etapa do workflow de aprendizado de máquina, desde o processamento de dados, passando pela tunagem de hiperparâmetros até a implantação. O Keras foi desenvolvido com foco em experimentações rápidas.

Com o Keras, você tem acesso completo aos recursos de escalabilidade e interplataforma do TensorFlow. É possível executar o Keras em um Pod de TPUs ou clusters grandes de GPUs. Além disso, é possível exportar modelos do Keras para execução em navegadores ou dispositivos móveis. Por fim, você também pode disponibilizar modelos do Keras para acesso via API web.

O Keras foi desenvolvido para reduzir a carga cognitiva ao atingir os seguintes objetivos:

- Conta com interfaces simples e consistentes.
- Minimiza o número de ações necessárias para casos de uso comuns.
- Apresenta mensagens de erro claras e práticas.
- Segue o princípio da complexidade progressiva: é fácil começar, e você consegue desenvolver workflows mais avançados à medida que vai aprendendo.
- Ajuda a escrever códigos concisos e fáceis de ler.

## Quem deve usar o Keras

A resposta curta é que todo usuário do TensorFlow deve usar as APIs do Keras por padrão. Não importa se você seja engenheiro, pesquisador ou trabalhe com aprendizado de máquina, deve começar pelo Keras.

Existem alguns casos de uso (por exemplo, criar ferramentas baseadas no TensorFlow ou desenvolver sua própria plataforma de alto desempenho) em que é necessário usar as [APIs de baixo nível TensorFlow Core](https://www.tensorflow.org/guide/core). Mas, se o seu caso de uso não estiver incluído nas [aplicações da API Core ](https://www.tensorflow.org/guide/core#core_api_applications), você deve optar pelo Keras.

## Componentes da API do Keras

As estruturas de dados principais do Keras são [camadas](https://keras.io/api/layers/) e [modelos](https://keras.io/api/models/). Uma camada é uma transformação simples entrada/saída, enquanto um modelo é um grafo acíclico dirigido (DAG, na sigla em inglês) de camadas.

### Camadas

A classe `tf.keras.layers.Layer` é a abstração fundamental do Keras. Uma `Layer` encapsula um estado (pesos) e algumas computações (definidas no método `tf.keras.layers.Layer.call`).

Os pesos criados pelas camadas podem ser treináveis ou não treináveis. As camadas são combináveis recursivamente: se você atribuir a instância de uma camada como um atributo de outra camada, a camada externa começará a rastrear os pesos criados pela interna.

Além disso, é possível usar camadas para lidar com tarefas de processamento de dados, como normalização e vetorização de texto. As camadas de pré-processamento podem ser incluídas diretamente em um modelo, seja durante ou após o treinamento, o que proporciona portabilidade ao modelo.

### Modelos

Um modelo é um objeto que agrupa camadas e que pode ser treinado com dados.

O tipo mais simples de modelo é o [`Sequential`](https://www.tensorflow.org/guide/keras/sequential_model) (sequencial), que é um pilha linear de camadas. Para arquiteturas mais complexas, você pode usar a [API funcional do Keras](https://www.tensorflow.org/guide/keras/functional_api), que permite criar grafos de camadas arbitrários ou pode [usar subclasses para escrever modelos do zero](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing).

A classe `tf.keras.Model` conta com métodos integrados de treinamento e avaliação:

- `tf.keras.Model.fit`: treina o modelo com um número fixo de épocas.
- `tf.keras.Model.predict`: gera previsões de saída dadas amostras de entrada.
- `tf.keras.Model.evaluate`: retorna os valores de métrica e perda para o modelo. É configurado pelo método `tf.keras.Model.compile`.

Esses métodos oferecem acesso aos seguintes recursos integrados de treinamento:

- [Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks). Você pode usar callbacks integrados para paragem antecipada, criação de checkpoints do modelo e monitoramento do [TensorBoard](https://www.tensorflow.org/tensorboard). Além disso, você pode [implementar callbacks personalizados](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks).
- [Treinamento distribuído](https://www.tensorflow.org/guide/keras/distributed_training). É fácil dimensionar o treinamento para usar diversas GPUs, TPUs ou dispositivos.
- Combinação de passos. Com o argumento `steps_per_execution` de `tf.keras.Model.compile`, é possível processar diversos lotes em uma única chamada a `tf.function`, o que melhora consideravelmente o uso de dispositivos em TPUs.

Confira uma visão geral detalhada de como usar `fit` no [guia de treinamento e avaliação](https://www.tensorflow.org/guide/keras/training_with_built_in_methods). Para ver como personalizar os loops integrados de treinamento e avaliação, confira [Personalize o que acontece em `fit()`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).

### Outras APIs e ferramentas

O Keras conta com diversas outras APIs e ferramentas para aprendizado profundo, incluindo:

- [Otimizadores](https://keras.io/api/optimizers/)
- [Métricas](https://keras.io/api/metrics/)
- [Perdas](https://keras.io/api/losses/)
- [Utilitários de carregamento de dados](https://keras.io/api/data_loading/)

Confira a lista completa de APIs disponíveis na [referência da API do Keras](https://keras.io/api/). Para saber mais sobre outros projetos e iniciativas do Keras, confira o [ecossistema do Keras](https://keras.io/getting_started/ecosystem/).

## Próximos passos

Para começar a usar o Keras com o TensorFlow, confira os seguintes tópicos:

- [O modelo Sequential](https://www.tensorflow.org/guide/keras/sequential_model)
- [A API Functional](https://www.tensorflow.org/guide/keras/functional)
- [Treinamento e avaliação com os métodos integrados](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [Criando novas camadas e modelos via subclasses](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [Como serializar e salvar](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [Trabalhando com camadas de pré-processamento](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [Personalize o que acontece em fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [Escrevendo um loop de treinamento do zero](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [Trabalhando com RNNs](https://www.tensorflow.org/guide/keras/rnn)
- [Sobre mascaramento e preenchimento](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [Escrevendo seus próprios callbacks](https://www.tensorflow.org/guide/keras/custom_callback)
- [Aprendizado por transferência e ajuste fino](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Várias GPUs e treinamento distribuído](https://www.tensorflow.org/guide/keras/distributed_training)

Para saber mais sobre o Keras, confira os seguintes tópicos em [keras.io](http://keras.io):

- [Sobre o Keras](https://keras.io/about/)
- [Introdução ao Keras para engenheiros](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [Introdução ao Keras para pesquisadores](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Referência da API do Keras](https://keras.io/api/)
- [Ecossistema do Keras](https://keras.io/getting_started/ecosystem/)

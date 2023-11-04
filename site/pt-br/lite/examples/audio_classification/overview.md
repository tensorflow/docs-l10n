# Classificação de áudio

<img src="../images/audio.png" class="attempt-right">

A tarefa de identificar o que um áudio representa é chamada de *classificação de áudio*. Um modelo desse tipo é treinado para reconhecer diversos eventos de áudio. Por exemplo, é possível treinar um modelo para reconhecer eventos que representem três situações diferentes: palmas, estalo de dedos e digitação. O TensorFlow Lite conta com modelos pré-treinados e otimizados que podem ser implantados em seus aplicativos móveis. Saiba mais sobre o uso do TensorFlow para classificação de áudio [aqui](https://www.tensorflow.org/tutorials/audio/simple_audio).

A imagem abaixo mostra a saída do modelo de classificação de áudio no Android.

<img src="images/android_audio_classification.png" width="30%" alt="Screenshot of Android example">

Observação: (1) para integrar um modelo existente, use a [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/audio_classifier) (biblioteca de tarefas do TensorFlow Lite). (2) Para personalizar um modelo, use o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification) (criador de modelos do TensorFlow Lite).

## Como começar

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android, recomendamos conferir os exemplos de aplicativo abaixo que podem te ajudar a começar.

Você pode usar a API integrada da [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/audio_classifier) para integrar modelos de classificação de áudio com somente algumas linhas de código. Além disso, pode criar seu próprio pipeline de inferência personalizado usando a [TensorFlow Lite Support Library](../../inference_with_metadata/lite_support) (biblioteca de suporte do TensorFlow Lite).

O exemplo no Android abaixo demonstra a implementação usando a [TFLite Task Library](https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android">Ver exemplo do Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/sound_classification/ios">Ver exemplo do iOS</a>

Se você estiver usando outra plataforma que não o Android/iOS ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), baixe o modelo inicial e os arquivos de suporte (se aplicável).

<a class="button button-primary" href="https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite">Baixar modelo inicial do TensorFlow Hub</a>

## Descrição do modelo

YAMNet é um classificador de eventos de áudio que recebe a forma de onda do áudio como entrada e faz previsões independentes para cada um dos 521 eventos de áudio da ontologia [AudioSet](https://g.co/audioset). O modelo usa a arquitetura MobileNet v1 e foi treinado usando o corpus AudioSet. Originalmente, esse modelo foi lançado no TensorFlow Model Garden, onde estão o código fonte do modelo, o checkpoint original do modelo e documentações mais detalhadas.

### Como funciona

Existem duas versões do modelo YAMNet convertidas para o TF Lite:

- [YAMNet](https://tfhub.dev/google/yamnet/1) é o modelo original de classificação de áudio com tamanho de entrada dinâmico, adequado para fazer aprendizado por transferência e implantação web e em dispositivos móveis. Além disso, tem uma saída mais complexa.

- [YAMNet/classification](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1) é uma versão quantizada com entrada de tamanho fixo mais simples (15.600 amostras) que retorna um único vetor de pontuações para 521 classes de eventos de áudio.

### Entradas

O modelo recebe um Tensor `float32` unidimensional ou um array NumPy de tamanho igual a 15.600 contendo uma forma de onda de 0,975 segundo representada como amostras de 16 kHz mono no intervalo `[-1.0, +1.0]`.

### Saídas

O modelo retorna um Tensor `float32` bidimensional de formato (1, 521) contendo as pontuações previstas para cada uma das 521 classes na ontologia AudioSet disponíveis no YAMNet. O índice de coluna (0-520) do tensor de pontuações é mapeado para a classe AudioSet correspondente usando o YAMNet Class Map (mapeamento de classes YAMNet), que está disponível como um arquivo auxiliar `yamnet_label_list.txt`, encapsulado no arquivo do modelo. Confira os usos possíveis abaixo.

### Usos possíveis

O YAMNet pode ser usado como:

- Um classificador independente de eventos de áudio que fornece uma linha de base razoável para uma ampla variedade de eventos de áudio.
- Um extrator de características de alto nível: a saída de embedding de 1.024 dimensões do YAMNet pode ser usada como as características de entrada de outro modelo, que pode ser treinado com uma quantidade menor de dados para uma tarefa específica. Dessa forma, é possível criar classificadores de áudio especializados mais rapidamente, sem a necessidade de ter muitos dados rotulados e sem precisar treinar um modelo grande do início ao fim.
- Uma inicialização rápida: os parâmetros do modelo YAMNet podem ser usados para inicializar parte de um modelo maior, o que propicia maior velocidade de ajustes finos e exploração do modelo.

### Limitações

- As saídas do classificador do YAMNet não foram calibradas entre as classes, então não é possível tratar as saídas como probabilidades diretamente. Para uma determinada tarefa, é muito provável que você precise fazer uma calibração com dados específicos da tarefa em questão, o que permite atribuir dimensionamento e limites de pontuação por classe adequados.
- O YAMNet foi treinado com milhões de vídeos do YouTube e, embora sejam muito diversificados, ainda pode haver uma incompatibilidade de domínios entre um vídeo comum do YouTube e as entradas de áudio esperadas para uma determinada tarefa. É provável que você precise fazer ajustes finos e calibração para que o YAMNet possa ser utilizado em qualquer sistema que você criar.

## Personalização do modelo

Os modelos pré-treinados fornecidos foram treinados para detectar 521 classes de áudio diferentes. Para ver a lista completa de classes, confira o arquivo labels (rótulos) no <a href="https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv">repositório do modelo</a>.

Você pode usar a técnica aprendizado por transferência para treinar novamente um modelo a fim de reconhecer classes que não estão no dataset original. Por exemplo: você pode treinar novamente o modelo para detectar cantos de diversos pássaros. Para fazer isso, será necessário um conjunto de áudios de treinamento para cada um dos novos rótulos que você deseja treinar. É recomendável usar a biblioteca [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification), que simplifica o processo de treinamento de um modelo do TensorFlow Lite usando um dataset personalizado com algumas linhas de código. Essa biblioteca usa aprendizado por transferência para reduzir a quantidade de dados de treinamento necessários e o tempo gasto. Veja um exemplo de aprendizado por transferência em [Aprendizado por transferência para reconhecimento de áudio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio).

## Leituras e recursos adicionais

Confira os recursos abaixo para saber mais sobre os conceitos relacionados à classificação de áudio:

- [Classificação de áudio usando o TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Aprendizado por transferência para reconhecimento de áudio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Ampliação de dados de áudio](https://www.tensorflow.org/io/tutorials/audio)

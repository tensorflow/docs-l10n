# Aprendizagem federada

## Visão geral

Este documento apresenta interfaces que facilitam tarefas de aprendizado federado, como treinamento federado ou avaliação com modelos de aprendizado de máquina existentes implementados no TensorFlow. Ao projetar essas interfaces, nosso objetivo principal era tornar possível experimentar o aprendizado federado sem exigir o conhecimento de como ele funciona nos bastidores, e avaliar os algoritmos de aprendizado federado implementados numa variedade de modelos e dados existentes. Nós encorajamos você a contribuir de volta para a plataforma. O TFF foi projetado tendo em mente a extensibilidade e a capacidade de composição, e agradecemos contribuições; estamos ansiosos para ver o que você vai descobrir!

As interfaces oferecidas por esta camada consistem nas três partes principais a seguir:

- **Modelos**. Classes e funções auxiliares que permitem agrupar seus modelos existentes para uso com TFF. O encapsulamento de um modelo pode ser tão simples quanto chamar uma única função wrapper (por exemplo, `tff.learning.models.from_keras_model`) ou definindo uma subclasse da interface `tff.learning.models.VariableModel` para personalização total.

- **Construtores de computação federada**. Funções auxiliares que constroem computações federadas para treinamento ou avaliação, usando seus modelos existentes.

- **Datasets**. Coleções prontas de dados que você pode baixar e acessar em Python para usar na simulação de cenários de aprendizado federado. Embora a aprendizagem federada seja projetada para uso com dados descentralizados que não podem ser simplesmente baixados de um local centralizado, nos estágios de pesquisa e desenvolvimento muitas vezes é conveniente conduzir experimentos iniciais usando dados que podem ser baixados e manipulados localmente, especialmente para desenvolvedores novatos em relação a essa abordagem.

Essas interfaces são definidas principalmente no namespace `tff.learning`, exceto para datasets de pesquisa e outros recursos relacionados à simulação que foram agrupados em `tff.simulation`. Esta camada é implementada usando interfaces de baixo nível oferecidas pelo [Federated Core (FC)](federated_core.md), que também fornece um ambiente de runtime.

Antes de prosseguir, recomendamos que você primeiro revise os tutoriais sobre [classificação de imagens](tutorials/federated_learning_for_image_classification.ipynb) e [geração de texto](tutorials/federated_learning_for_text_generation.ipynb), pois eles apresentam a maioria dos conceitos descritos aqui usando exemplos concretos. Se você estiver interessado em aprender mais sobre como o TFF funciona, você pode dar uma olhada no tutorial [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) como uma introdução às interfaces de baixo nível que usamos para expressar a lógica das computações federadas e estudar a implementação existente das interfaces `tff.learning`.

## Modelos

### Suposições arquitetônicas

#### Serialização

O TFF visa oferecer suporte a uma variedade de cenários de aprendizado distribuído nos quais o código do modelo de aprendizado de máquina que você escreve pode estar sendo executado num grande número de clientes heterogêneos com capacidades diversas. Embora numa ponta do espectro, em algumas aplicações esses clientes possam ser servidores de banco de dados poderosos, muitos usos importantes que nossa plataforma pretende suportar envolvem dispositivos móveis e incorporados que têm recursos limitados. Não podemos presumir que esses dispositivos sejam capazes de hospedar runtimes do Python; a única coisa que podemos assumir neste momento é que eles são capazes de hospedar um runtime local do TensorFlow. Portanto, uma suposição arquitetônica fundamental que fazemos no TFF é que o código do seu modelo deve ser serializável como um grafo do TensorFlow.

Você ainda pode (e deve) ainda desenvolver seu código TF seguindo as práticas recomendadas mais recentes, como usar o modo eager. Entretanto, o código final deve ser serializável (por exemplo, pode ser encapsulado como um `tf.function` para código de modo eager). Isto garante que qualquer estado Python ou fluxo de controle necessário em tempo de execução possa ser serializado (possivelmente com a ajuda do [Autograph](https://www.tensorflow.org/guide/autograph)).

Atualmente, o TensorFlow não oferece suporte total à serialização e desserialização do TensorFlow em modo eager. Assim, a serialização no TFF atualmente segue o padrão TF 1.0, onde todo o código deve ser construído dentro de um `tf.Graph` que o TFF controla. Isto significa que atualmente o TFF não pode consumir um modelo já construído; em vez disso, a lógica de definição do modelo é empacotada em uma função sem argumentos que retorna um `tff.learning.models.VariableModel`. Esta função é então chamada pelo TFF para garantir que todos os componentes do modelo sejam serializados. Além disso, por ser um ambiente fortemente tipado, o TFF exigirá alguns *metadados* adicionais, como uma especificação do tipo de entrada do seu modelo.

#### Agregação

Recomendamos fortemente que a maioria dos usuários construa modelos usando Keras. Veja a seção [Conversores para Keras](#converters-for-keras) abaixo. Esses wrappers tratam automaticamente da agregação de atualizações do modelo, bem como de quaisquer métricas definidas para o modelo. No entanto, ainda pode ser útil entender como a agregação é manipulada para um `tff.learning.models.VariableModel` geral.

Sempre existem pelo menos duas camadas de agregação na aprendizagem federada: agregação local no dispositivo e agregação entre dispositivos (ou federada):

- **Agregação local**. Este nível de agregação refere-se à agregação em múltiplos lotes de exemplos pertencentes a um cliente individual. Aplica-se tanto aos parâmetros do modelo (variáveis), que continuam a evoluir sequencialmente à medida que o modelo é treinado localmente, quanto às estatísticas que você computa (como perda média, precisão e outras métricas), que seu modelo atualizará novamente localmente à medida que itera no stream de dados local de cada cliente individual.

    A realização da agregação nesse nível é responsabilidade do código do modelo e é feita usando construtos padrão do TensorFlow.

    A estrutura geral do processamento é a seguinte:

    - O modelo primeiro constrói `tf.Variable` para conter agregados, como o número de lotes ou o número de exemplos processados, a soma de perdas por lote ou por exemplo, etc.

    - O TFF chama o método `forward_pass` no seu `Model` várias vezes, sequencialmente em lotes subsequentes de dados do cliente, o que permite atualizar as variáveis ​​que contêm vários agregados como efeito colateral.

    - Por fim, o TFF invoca o método `report_local_unfinalized_metrics` no seu modelo para permitir que seu modelo compile todas as estatísticas resumidas coletadas num conjunto compacto de métricas a serem exportadas pelo cliente. É aqui que o código do seu modelo pode, por exemplo, dividir a soma das perdas pelo número de exemplos processados ​​para exportar a perda média, etc.

- **Agregação federada**. Este nível de agregação refere-se à agregação entre múltiplos clientes (dispositivos) no sistema. Novamente, isto se aplica tanto aos parâmetros do modelo (variáveis), cuja média está sendo calculada entre os clientes, quanto às métricas que seu modelo exportou como resultado da agregação local.

    A realização da agregação neste nível é responsabilidade do TFF. Como criador de modelo, entretanto, você pode controlar esse processo (mais sobre isso abaixo).

    A estrutura geral do processamento é a seguinte:

    - O modelo inicial e quaisquer parâmetros necessários para o treinamento são distribuídos por um servidor para um subconjunto de clientes que participarão de uma rodada de treinamento ou avaliação.

    - Em cada cliente, de forma independente e em paralelo, seu código de modelo é chamado repetidamente num stream de lotes de dados locais para produzir um novo conjunto de parâmetros de modelo (durante o treinamento) e um novo conjunto de métricas locais, conforme descrito acima (isto é agregação local).

    - A TFF executa um protocolo de agregação distribuída para acumular e agregar os parâmetros do modelo e métricas exportadas localmente por todo o sistema. Essa lógica é expressa de maneira declarativa usando a própria linguagem de *computação federada* do TFF (não do TensorFlow). Consulte o tutorial [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) para saber mais sobre a API de agregação.

### Interfaces abstratas

Este *construtor* básico + interface *de metadados* é representado pela interface `tff.learning.models.VariableModel`, como segue:

- Os métodos construtor, `forward_pass` e `report_local_unfinalized_metrics` devem construir variáveis ​​de modelo, passo para frente e estatísticas que você deseja relatar, respectivamente. O TensorFlow construído por esses métodos deve ser serializável, conforme discutido acima.

- A propriedade `input_spec`, bem como as 3 propriedades que retornam subconjuntos de suas variáveis ​​treináveis, não treináveis ​​e locais representam os metadados. O TFF usa essas informações para determinar como conectar partes do seu modelo aos algoritmos de otimização federados e para definir assinaturas de tipos internos para auxiliar na verificação da exatidão do sistema construído (para que seu modelo não possa ser instanciado sobre dados que não correspondam ao que o modelo foi projetado para consumir).

Além disso, a interface abstrata `tff.learning.models.VariableModel` expõe uma propriedade `metric_finalizers` que recebe os valores não finalizados de uma métrica (retornados por `report_local_unfinalized_metrics()`) e retorna os valores da métrica finalizados. Os métodos `metric_finalizers` e `report_local_unfinalized_metrics()` serão usados ​​juntos para construir um agregador de métricas entre clientes ao definir os processos de treinamento federados ou computações de avaliação. Por exemplo, um agregador `tff.learning.metrics.sum_then_finalize` simples primeiro somará os valores de métricas não finalizadas dos clientes e, em seguida, chamará as funções do finalizador no servidor.

Você poderá encontrar exemplos de como definir seu próprio `tff.learning.models.VariableModel` personalizado na segunda parte de nosso tutorial [classificação de imagens](tutorials/federated_learning_for_image_classification.ipynb), bem como nos modelos de exemplo que usamos para teste em [`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/models/model_examples.py).

### Conversores para Keras

Praticamente todas as informações exigidas pelo TFF podem ser derivadas chamando interfaces `tf.keras`, portanto, se você tiver um modelo Keras, poderá confiar em `tff.learning.models.from_keras_model` para construir um `tff.learning.models.VariableModel`.

Observe que o TFF ainda espera que você forneça um construtor - uma *função de modelo* sem argumentos, como a seguinte:

```python
def model_fn():
  keras_model = ...
  return tff.learning.models.from_keras_model(keras_model, sample_batch, loss=...)
```

Além do modelo em si, você fornece um lote de dados de amostra que o TFF usa para determinar o tipo e formato da entrada do seu modelo. Isto garante que o TFF possa instanciar corretamente o modelo para os dados que realmente estarão presentes nos dispositivos clientes (já que presumimos que esses dados não estão geralmente disponíveis no momento em que você está construindo o TensorFlow para ser serializado).

O uso de wrappers do Keras é ilustrado em nossos tutoriais de [classificação de imagens](tutorials/federated_learning_for_image_classification.ipynb) e [geração de texto](tutorials/federated_learning_for_text_generation.ipynb).

## Construtores de computação federada

O pacote `tff.learning` fornece diversos construtores para `tff.Computation` que executam tarefas relacionadas ao aprendizado; esperamos que o conjunto de tais computações se expanda no futuro.

### Suposições arquitetônicas

#### Execução

Existem duas fases distintas na execução de uma computação federada.

- **Compile**: o TFF primeiro *compila* algoritmos de aprendizagem federados numa representação serializada abstrata de toda a computação distribuída. É quando ocorre a serialização do TensorFlow, mas outras transformações podem ocorrer para oferecer suporte a uma execução mais eficiente. Referimo-nos à representação serializada emitida pelo compilador como uma *computação federada*.

- **Execute** O TFF fornece maneiras de *executar* essas computações. Por enquanto, a execução só é suportada através de simulação local (por exemplo, num notebook usando dados descentralizados simulados).

Uma computação federada gerada pela API Federated Learning do TFF, como um algoritmo de treinamento que usa [média de modelo federado](https://arxiv.org/abs/1602.05629) ou uma avaliação federada, inclui vários elementos, sendo os mais importantes:

- Uma forma serializada do código do seu modelo, bem como código adicional do TensorFlow construído pelo framework Federated Learning para conduzir o loop de treinamento/avaliação do seu modelo (como a construção de otimizadores, aplicação de atualizações de modelo, iteração sobre `tf.data.Dataset` e métricas de computação, e aplicar a atualização agregada no servidor, por exemplo).

- Uma especificação declarativa da comunicação entre os *clientes* e um *servidor* (tipicamente várias formas de *agregação* entre os dispositivos do cliente e *difusão* do servidor para todos os clientes) e como essa comunicação distribuída é intercalada com a execução client-local ou server-local do código do TensorFlow.

As *computações federadas* representadas neste formato serializado são expressas numa linguagem interna independente de plataforma, distinta do Python, mas para usar a API Federated Learning, você não precisará se preocupar com os detalhes dessa representação. As computações são representadas no seu código Python como objetos do tipo `tff.Computation`, que geralmente você pode tratar como objetos `callable` opacos do Python.

Nos tutoriais, você invocará essas computações federadas como se fossem funções comuns do Python, para serem executadas localmente. No entanto, o TFF foi projetado para expressar computações federadas de maneira independente da maioria dos aspectos do ambiente de execução, de modo que possam ser potencialmente implantáveis, por exemplo, para grupos de dispositivos que executam `Android` ou em clusters num datacenter. Novamente, a principal consequência disso são fortes suposições sobre [serialização](#serialization). Em particular, quando você chama um dos métodos `build_...` descritos abaixo, a computação é totalmente serializada.

#### Estado de modelagem

O TFF é um ambiente de programação funcional, mas muitos processos de interesse na aprendizagem federada são stateful. Por exemplo, um loop de treinamento que envolve múltiplas rodadas de obtenção de média federado de modelos é um exemplo do que poderíamos classificar como um *processo stateful*. Neste processo, o estado que evolui de rodada para rodada inclui o conjunto de parâmetros do modelo que estão sendo treinados e, possivelmente, um estado adicional associado ao otimizador (por exemplo, um vetor de momento).

Já que o TFF é funcional, os processos stateful são modelados no TFF como computações que aceitam o estado atual como entrada e então fornecem o estado atualizado como saída. Para definir completamente um processo stateful, também é necessário especificar de onde vem o estado inicial (caso contrário, não poderemos dar partida no processo). Isso é capturado na definição da classe auxiliar `tff.templates.IterativeProcess`, com as duas propriedades `initialize` e `next` correspondendo à inicialização e iteração, respectivamente.

### Construtores disponíveis

No momento, o TFF fornece várias funções construtoras que geram computações federadas para treinamento e avaliação federados. Dois exemplos notáveis ​​incluem:

- `tff.learning.algorithms.build_weighted_fed_avg`, que recebe como entrada uma *função de modelo* e um *otimizador de cliente* e retorna um `tff.learning.templates.LearningProcess` stateful (que estende a classe `tff.templates.IterativeProcess`).

- `tff.learning.build_federated_evaluation` usa uma *função de modelo* e retorna uma única computação federada para avaliação federada de modelos, uma vez que a avaliação não é stateful.

## Datasets

### Suposições arquitetônicas

#### Seleção de cliente

No cenário típico de aprendizagem federada, temos uma grande *população* de potencialmente centenas de milhões de dispositivos clientes, dos quais apenas uma pequena parte poderá estar ativa e disponível para treinamento num determinado momento (por exemplo, isto pode ser limitado a clientes que estão conectados a uma fonte de energia, não numa rede monitorada e talvez ociosa). Geralmente, o conjunto de clientes disponíveis para participar de treinamento ou avaliação está fora do controle do desenvolvedor. Além disso, como é impraticável coordenar milhões de clientes, uma rodada típica de treinamento ou avaliação incluirá apenas uma fração dos clientes disponíveis, que podem ser [amostrados aleatoriamente](https://arxiv.org/pdf/1902.01046.pdf).

A principal consequência disso é que os cálculos federados, por design, são expressos de uma maneira que ignora o conjunto exato de participantes; todo o processamento é expresso como operações agregadas sobre um grupo abstrato de *clientes* anônimos, e esse grupo pode variar de uma rodada de treinamento para outra. A vinculação real da computação aos participantes concretos e, portanto, aos dados concretos que eles alimentam na computação, é, portanto, modelada fora da própria computação.

Para simular uma implantação realista do seu código de aprendizado federado, você geralmente escreverá um loop de treinamento semelhante a este:

```python
trainer = tff.learning.algorithms.build_weighted_fed_avg(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  result = trainer.next(state, data_for_this_round)
  state = result.state
```

Para facilitar isso, ao usar TFF em simulações, os dados federados são aceitos como objetos `list` do Python, com um elemento por dispositivo cliente participante para representar o `tf.data.Dataset` local desse dispositivo.

### Interfaces abstratas

Para padronizar o tratamento de datasets federados simulados, o TFF fornece uma interface abstrata `tff.simulation.datasets.ClientData`, que permite enumerar o conjunto de clientes e construir um `tf.data.Dataset` que contém os dados de um determinado cliente. Esses `tf.data.Dataset` podem ser alimentados diretamente como entrada para as computações federadas geradas no modo eager.

Deve-se notar que a capacidade de acessar identidades de clientes é um recurso fornecido apenas pelos datasets para uso em simulações, onde pode ser necessária a capacidade de treinar dados de subconjuntos específicos de clientes (por exemplo, para simular a disponibilidade diurna de diferentes tipos de clientes). As computações compiladas e o runtime subjacente *não* envolvem qualquer noção de identidade do cliente. Depois que os dados de um subconjunto específico de clientes forem selecionados como entrada, por exemplo, em uma chamada para `tff.templates.IterativeProcess.next`, as identidades dos clientes não aparecerão mais nela.

### Datasets disponíveis

Dedicamos o namespace `tff.simulation.datasets` para datasets que implementam a interface `tff.simulation.datasets.ClientData` para uso em simulações e o semeamos com datasets para dar suporte aos tutoriais de [classificação de imagens](tutorials/federated_learning_for_image_classification.ipynb) e [geração de texto](tutorials/federated_learning_for_text_generation.ipynb). Gostaríamos de incentivá-lo a contribuir com seus próprios datasets para a plataforma.

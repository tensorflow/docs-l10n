# Federated Core

Este documento apresenta a camada core do TFF que serve como base para o [Federated Learning](federated_learning.md) (aprendizado federado) e possíveis futuros algoritmos federados não relacionados a aprendizado.

Para uma introdução suave ao Federated Core, leia os tutoriais a seguir, pois eles apresentam alguns dos conceitos fundamentais através de exemplos e demonstram,  passo a passo, a construção de um algoritmo federado simples de cálculo de média.

- [Algoritmos federados personalizados, parte 1: Introdução ao Federated Core](tutorials/custom_federated_algorithms_1.ipynb).

- [Algoritmos federados personalizados, parte 2: Implementando o cálculo federado de médias](tutorials/custom_federated_algorithms_2.ipynb).

Também recomendamos que você se familiarize com o [Federated Learning](federated_learning.md) e os tutoriais associados sobre [classificação de imagens](tutorials/federated_learning_for_image_classification.ipynb) e [geração de texto](tutorials/federated_learning_for_text_generation.ipynb), pois os usos da Federated Core API (FC API) para aprendizado federado trazem um contexto importante para algumas das escolhas que fizemos ao projetar esta camada.

## Visão geral

### Objetivos, usos pretendidos e escopo

O Federated Core (FC) é melhor entendido como um ambiente de programação para implementar computações distribuídas, ou seja, computações que envolvem múltiplos computadores (telefones celulares, tablets, dispositivos embarcados, computadores desktop, sensores, servidores de banco de dados, etc.) que podem executar processamento não trivial localmente e se comunicar através da rede para coordenar seu trabalho.

O termo *distribuído* é muito genérico e o TFF não se destina a todos os tipos possíveis de algoritmos distribuídos existentes, por isso preferimos usar o termo menos genérico *computação federada* para descrever os tipos de algoritmos que podem ser expressos nesse framework.

Embora a definição do termo *computação federada*, de uma maneira puramente formal, esteja fora do escopo deste documento, pense nos tipos de algoritmos que você talvez veja expressos em pseudocódigo numa [publicação científica](https://arxiv.org/pdf/1602.05629.pdf) que descreva um novo algoritmo de aprendizado distribuído.

O objetivo da FC, em poucas palavras, é permitir uma representação compacta semelhante, num nível de abstração semelhante ao pseudocódigo, da lógica do programa que *não* é pseudocódigo, mas sim executável numa variedade de ambientes-alvo.

A principal característica que define os tipos de algoritmos que a FC foi projetada para expressar é que as ações dos participantes do sistema são descritas de maneira coletiva. Assim, tendemos a falar sobre *cada dispositivo* que transforma dados localmente, e dispositivos que coordenam o trabalho através de um coordenador centralizado *que transmite*, *coleta* ou *agrega* seus resultados.

Embora o TFF tenha sido projetado para ir além das simples arquiteturas *cliente-servidor*, a noção de processamento coletivo é fundamental. Isto deve-se às origens do TFF na aprendizagem federada, uma tecnologia originalmente concebida para suportar computações sobre dados potencialmente sensíveis que permanecem sob controle de dispositivos clientes, e que não podem ser simplesmente baixados para um local centralizado por questões de privacidade. Embora cada cliente em tais sistemas contribua com dados e poder de processamento com o objetivo de computar um resultado pelo sistema (um resultado que geralmente esperamos que tenha valor para todos os participantes), também nos esforçamos para preservar a privacidade e o anonimato de cada cliente.

Assim, embora a maioria dos frameworks para computação distribuída sejam projetados para expressar o processamento a partir da perspectiva de participantes individuais - isto é, no nível de trocas individuais de mensagens ponto a ponto e da interdependência das transições de estado locais do participante, com mensagens recebidas e enviadas, o Federated Core do TFF foi projetado para descrever o comportamento do sistema a partir da sua perspectiva *global* (de forma similar a, por exemplo, [MapReduce](https://research.google/pubs/pub62/)).

Consequentemente, embora frameworks distribuídos de propósito geral possam oferecer operações como *envio* (send) e *recebimento* (receive) como blocos básicos de construção, o FC fornece blocos de construção tais como `tff.federated_sum`, `tff.federated_reduce` ou `tff.federated_broadcast` que encapsulam protocolos distribuídos simples.

## Linguagem

### Interface Python

O TFF usa uma linguagem interna para representar computações federadas, cuja sintaxe é definida pela representação serializável em [computation.proto](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). No entanto, os usuários da API FC geralmente não precisarão interagir diretamente com essa linguagem. Em vez disso, fornecemos uma API Python (o namespace `tff`) que a encapsula, como uma maneira de definir computações.

Especificamente, o TFF fornece decoradores de funções Python, como `tff.federated_computation`, que rastreiam os corpos das funções decoradas e produzem representações serializadas da lógica de computação federada na linguagem do TFF. Uma função decorada com `tff.federated_computation` atua como portadora de tal representação serializada e pode incorporá-la como um bloco básico de construção no corpo de outra computação ou executá-la sob demanda quando chamada.

Aqui está um exemplo (mais exemplos podem ser encontrados nos tutoriais [de algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb)).

```python
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

Os leitores não familiarizados com o TensorFlow acharão essa abordagem análoga a escrever código Python que usa funções como `tf.add` ou `tf.reduce_sum` num trechi de código Python que defina um gráfico do TensorFlow. Embora o código seja tecnicamente expresso em Python, seu objetivo é construir uma representação serializável de um `tf.Graph` subjacente, e é o gráfico, e não o código Python, que é executado internamente pelo tempo de execução do TensorFlow. Da mesma forma, pode-se pensar em `tff.federated_mean` como a inserção de uma *operação federada* numa computação federada representada por `get_average_temperature`.

Uma parte da razão para o FC definir uma linguagem tem a ver com o fato de que, como observado acima, as computações federadas especificam comportamentos coletivos distribuídos e, como tal, sua lógica é não local. Por exemplo, a TFF fornece operadores, cujas entradas e saídas podem existir em diferentes locais da rede.

Isso exige uma linguagem e um sistema de tipos que capturem essa noção de distribuição.

### Sistema de tipos

O Federated Core oferece as seguintes categorias de tipos. Ao descrever esses tipos, apontamos para os construtores de tipo e também introduzimos uma notação compacta, pois é uma forma prática de descrever tipos de computações e operadores.

Primeiro, aqui estão listadas as categorias de tipos que são conceitualmente semelhantes aos encontrados em linguagens convencionais existentes:

- **Tipos de tensor** (`tff.TensorType`). Assim como no TensorFlow, eles possuem `dtype` e `shape`. A única diferença é que objetos desse tipo não estão limitados a instâncias `tf.Tensor` em Python que representam saídas de operações do TensorFlow num grafo TensorFlow, mas também podem incluir unidades de dados que podem ser produzidas, por exemplo, como uma saída de um protocolo de agregação distribuído. Assim, o tipo de tensor TFF é simplesmente uma versão abstrata de uma representação física concreta desse tipo em Python ou TensorFlow.

    Os `TensorTypes` do TFF podem ser mais rigorosos no tratamento (estático) dos formatos do que o TensorFlow. Por exemplo, o sistema de tipos do TFF trata um tensor com classificação desconhecida como atribuível *a partir de* qualquer outro tensor do mesmo `dtype`, mas não atribuível *para* qualquer tensor com posto fixo. Este tratamento evita determinadas falhas de tempo de execução (por exemplo, tentativa de alterar o formato de um tensor de posto desconhecido para um formato com número incorreto de elementos), ao custo de maior rigor nas computações aceitas pelo TFF como válidas.

    A notação compacta para tipos de tensor é `dtype` ou `dtype[formato]`. Por exemplo, `int32` e `int32[10]` são os tipos para inteiros e vetores de inteiros, respectivamente.

- **Tipos de sequência** (`tff.SequenceType`). Esses são o equivalente abstrato do TFF ao conceito concreto de um `tf.data.Dataset` do TensorFlow. Elementos de sequência podem ser consumidos de maneira sequencial e podem incluir tipos complexos.

    A representação compacta dos tipos de sequência é `T*` , onde `T` é o tipo dos elementos. Por exemplo `int32*` representa uma sequência de inteiros.

- **Tipos de tupla nomeada** (`tff.StructType`). Essa é a maneira do TFF construir tuplas e estruturas semelhantes a dicionários que possuem um número predefinido de *elementos* com tipos específicos, nomeados ou não. É importante ressaltar que o conceito de tupla nomeada do TFF abrange o equivalente abstrato das tuplas de argumento do Python, ou seja, coleções de elementos dos quais alguns, mas não todos, são nomeados, e alguns são posicionais.

    A notação compacta para tuplas nomeadas é `<n_1=T_1, ..., n_k=T_k>`, onde `n_k` são nomes de elementos opcionais e `T_k` são tipos de elementos. Por exemplo, `<int32,int32>` é uma notação compacta para um par de números inteiros sem nome e `<X=float32,Y=float32>` é uma notação compacta para um par de pontos flutuantes chamados `X` e `Y` que podem representar um ponto num plano. Tuplas podem ser aninhadas e também misturadas com outros tipos, por exemplo, `<X=float32,Y=float32>*` seria uma notação compacta para uma sequência de pontos.

- **Tipos de função** (`tff.FunctionType`). O TFF é um framework de programação funcional, onde as funções são tratadas como [valores de primeira classe](https://en.wikipedia.org/wiki/First-class_citizen). As funções têm no máximo um argumento e exatamente um resultado.

    A notação compacta para funções é `(T -> U)`, onde `T` é o tipo de um argumento e `U` é o tipo do resultado, ou `( -> U)` se não houver argumentos (embora funções sem argumento sejam um conceito degenerado que existe geralmente apenas no nível do Python). Por exemplo `(int32* -> int32)` é a notação para um tipo de função que reduz uma sequência inteira a um único valor inteiro.

Os tipos a seguir abordam o aspecto de sistemas distribuídos das computações TFF. Como esses conceitos são exclusivos do TFF, recomendamos que você consulte o tutorial sobre [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) para comentários e exemplos adicionais.

- **Tipo de posicionamento**. Este tipo ainda não está exposto na API pública, exceto na forma de 2 literais `tff.SERVER` e `tff.CLIENTS` que você pode considerar como constantes desse tipo. No entanto, ele é usado internamente e será introduzido na API pública em versões futuras. A representação compacta deste tipo é `placement`.

    Um *placement* (posicionamento) representa um coletivo de participantes do sistema que desempenham uma função específica. A versão inicial tem como alvo computações cliente-servidor, nos quais existem 2 grupos de participantes: *clientes* e um *servidor* (você pode pensar neste último como um grupo de singletons). No entanto, em arquiteturas mais elaboradas, pode haver outras funções, como agregadores intermediários num sistema multicamadas, que podem executar diferentes tipos de agregação ou usar tipos diferentes de compactação/descompactação de dados daqueles usados ou ​​pelo servidor ou pelos clientes.

    O objetivo principal da definição da noção de posicionamentos é servir de base para a definição de *tipos federados*.

- **Tipos federados** (`tff.FederatedType`). O valor de um tipo federado é aquele que é hospedado por um grupo de participantes do sistema definido por um posicionamento específico (como `tff.SERVER` ou `tff.CLIENTS`). Um tipo federado é definido pelo valor do seu *placement* (posicionamento) (portanto, é um [tipo dependente](https://en.wikipedia.org/wiki/Dependent_type)), pelo tipo de *membros constituintes* (que tipo de conteúdo cada um dos participantes hospeda localmente) e pelo bit adicional `all_equal` que especifica se todos os participantes estão localmente hospedando o mesmo item.

    A notação compacta para tipos federados de valores que incluem itens (membros constituintes) do tipo `T`, cada um hospedado pelo grupo (placement) `G` é `T@G` ou `{T}@G` com o bit `all_equal` definido ou não definido, respectivamente.

    Por exemplo:

    - `<int>@CLIENTS` representa um *valor federado* que consiste de um conjunto de números inteiros potencialmente distintos, um por dispositivo cliente. Observe que estamos falando de um único *valor federado* que abrange múltiplos itens de dados que aparecem em múltiplos locais da rede. Uma maneira de pensar nisso é como uma espécie de tensor com dimensão de “rede”, embora esta analogia não seja perfeita porque o TFF não permite [acesso aleatório](https://en.wikipedia.org/wiki/Random_access) aos membros constituintes de um valor federado.

    - `{<X=float32,Y=float32>*}@CLIENTS` representa um *dataset federado*  um valor que consiste de diversas sequências de coordenadas `XY`, uma sequência por dispositivo cliente.

    - `<weights=float32[10,5],bias=float32[5]>@SERVER` representa uma tupla nomeada de tensores de peso e bias no servidor. Como eliminamos as chaves, isto indica que o bit `all_equal` está definido, ou seja, existe apenas uma única tupla (independentemente de quantas réplicas de servidor possam existir num cluster que hospeda esse valor).

### Blocos de construção

A linguagem do Federated Core é um tipo de [cálculo lambda](https://en.wikipedia.org/wiki/Lambda_calculus), com alguns elementos adicionais.

Ela fornece as seguintes abstrações de programação, atualmente expostas na API pública:

- Computações **do TensorFlow** ( `tff.tf_computation`). Estas são seções do código do TensorFlow encapsuladas como componentes reutilizáveis ​​no TFF usando o decorador `tff.tf_computation`. Eles sempre têm tipos funcionais e, diferentemente das funções do TensorFlow, podem receber parâmetros estruturados ou retornar resultados estruturados de um tipo de sequência.

    Aqui está um exemplo, uma computação TF do tipo `(int32* -> int)` que usa o operador `tf.data.Dataset.reduce` para computar uma soma de números inteiros:

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

- <strong>Intrínsecos</strong> ou *operadores federados{/em} (`tff.federated_...`). Esta é uma biblioteca de funções como `tff.federated_sum` ou `tff.federated_broadcast` que constitui a maior parte da API FC, a maioria das quais representa operadores de comunicação distribuída para uso com o TFF.*

    Chamamos de *intrínsecos* porque, assim como as [funções intrínsecas](https://en.wikipedia.org/wiki/Intrinsic_function), eles são um conjunto aberto e extensível de operadores que são compreendidos pelo TFF e compilados para código de nível mais baixo.

    A maioria desses operadores possui parâmetros e resultados de tipos federados e a maior parte são modelos que podem ser aplicados a vários tipos de dados.

    Por exemplo, `tff.federated_broadcast` pode ser pensado como um operador de modelo de um tipo funcional `T@SERVER -> T@CLIENTS`.

- **Expressões lambda** (`tff.federated_computation`). Uma expressão lambda em TFF é equivalente a `lambda` ou `def` em Python; consiste do nome do parâmetro e um corpo (expressão) que contém referências a esse parâmetro.

    No código Python, elas podem ser criadas decorando funções Python com `tff.federated_computation` e definindo um argumento.

    Eis um exemplo de expressão lambda que já mencionamos anteriormente:

    ```python
    @tff.federated_computation(tff.type_at_clients(tf.float32))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

- **Literais de posicionamento**. Por enquanto, apenas `tff.SERVER` e `tff.CLIENTS` permitem definir computações cliente-servidor simples.

- **Chamadas de função** (`__call__`). Qualquer coisa que tenha um tipo funcional pode ser chamada usando a sintaxe `__call__` padrão do Python. A chamada é uma expressão cujo tipo é igual ao tipo do resultado da função que está sendo invocada.

    Por exemplo:

    - `add_up_integers(x)` representa uma chamada da computação do TensorFlow definida anteriormente sobre um argumento `x`. O tipo desta expressão é `int32`.

    - `tff.federated_mean(sensor_readings)` representa uma chamada do operador de média federado em `sensor_readings`. O tipo desta expressão é `float32@SERVER` (assumindo o contexto do exemplo acima).

- Formando **tuplas** e **selecionando** seus elementos. Expressões Python no formato `[x, y]`, `x[y]` ou `xy` que aparecem nos corpos das funções decoradas com `tff.federated_computation`.

# Depuração de problemas numéricos nos programas do TensorFlow com o TensorBoard Debugger V2

> *OBSERVAÇÃO*: tf.debugging.experimental.enable_dump_debug_info() é uma API experimental e está sujeita a mudanças que causam quebras no futuro.

Às vezes, podem ocorrer eventos catastróficos envolvendo [NaN](https://en.wikipedia.org/wiki/NaN)s em um programa do TensorFlow, prejudicando os processos de treinamento do modelo. A causa raiz desses eventos são geralmente obscuras, especialmente para modelos de tamanho e complexidade não triviais. Para facilitar a depuração desses tipos de bugs de modelos, o TensorBoard 2.3+ (junto com o TensorFlow 2.3+) oferece um painel de controle especializado chamado "Debugger V2". Demonstramos aqui como usar essa ferramenta ao resolver um bug real envolvendo NaNs em uma rede neural escrita no TensorFlow.

As técnicas ilustradas neste tutorial são aplicáveis a outros tipos de atividades de depuração, como a inspeção de formatos de tensores de runtime em programas complexos. Este tutorial foca em NaNs devido à frequência relativamente alta de ocorrência.

## Observação do bug

O código-fonte do programa TF2 que vamos depurar está [disponível no GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/v2/debug_mnist_v2.py). O programa de exemplo também está empacotado no pacote pip do TensorFlow (versão 2.3+) e pode ser invocado com:

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2
```

Este programa TF2 cria uma percepção de várias camadas (MLP) e a treina para reconhecer imagens [MNIST](https://en.wikipedia.org/wiki/MNIST_database). Este exemplo usa propositalmente a API de baixo nível do TF2 para definir construtos de camadas personalizadas, função de perda e loop de treinamento, porque a probabilidade de bugs NaN é mais alta quando usamos essa API mais flexível, mas mais propensa a erros, do que quando usamos APIs de alto nível que são mais fáceis de usar, mas um pouco menos flexíveis, como [tf.keras](https://www.tensorflow.org/guide/keras).

O programa imprime uma exatidão de teste após cada passo de treinamento. Podemos ver no console que a exatidão de teste fica presa em um nível próximo do acaso (~0,1) depois do primeiro passo. Esse certamente não é o comportamento esperado para o treinamento do modelo: esperamos que a exatidão se aproxime gradualmente de 1,0 (100%) com o aumento de passos.

```
Accuracy at step 0: 0.216
Accuracy at step 1: 0.098
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
...
```

Um palpite seria que esse problema é causado por uma instabilidade numérica, como NaN ou infinito. No entanto, como confirmamos se essa é realmente a causa e como encontramos a operação (op) do TensorFlow responsável por gerar a instabilidade numérica? Para responder a essas perguntas, vamos instrumentar o programa cheio de bugs com o Debugger V2.

## Instrumentação de código do TensorFlow com o Debugger V2

[`tf.debugging.experimental.enable_dump_debug_info()`](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info) é o ponto de entrada da API do Debugger V2. Ele instrumenta um programa TF2 com uma única linha de código. Por exemplo, ao adicionar a linha seguinte perto do início do programa, informações de depuração são escritas no diretório de log (logdir) em /tmp/tfdbg2_logdir. As informações de depuração abrangem vários aspectos do runtime do TensorFlow. No TF2, ele inclui o histórico completo da eager execution, a criação de grafos realizada pela [@tf.function](https://www.tensorflow.org/api_docs/python/tf/function), a graph execution, os valores de tensores gerados pelos eventos de execução, bem como a localização do código (stack traces do Python) desses eventos. A riqueza das informações de depuração permite que os usuários foquem em bugs obscuros.

```py
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
```

O argumento `tensor_debug_mode` controla quais informações o Debugger V2 extrai de cada tensor eager ou no grafo. "FULL_HEALTH" é um modo que captura as seguintes informações sobre cada tensor de tipo flutuante (por exemplo, o float32 mais comum e o dtype [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) menos comum):

- DType
- Posto
- Número total de elementos
- Classificação dos elementos de tipo flutuante nas seguintes categorias: negativo finito (`-`), zero (`0`), positivo finito (`+`), negativo infinito (`-∞`), positivo infinito (`+∞`) e `NaN`.

O modo "FULL_HEALTH" é adequado para depurar bugs envolvendo NaN e infinito. Veja abaixo outros `tensor_debug_mode`s compatíveis.

O argumento `circular_buffer_size` controla quantos eventos de tensor são salvos no logdir. O padrão é 1000, o que faz com que apenas os últimos 1000 tensores antes do final do programa TF2 instrumentado sejam salvos no disco. Esse comportamento padrão reduz o overhead do depurador ao sacrificar a completude dos dados de depuração. Se a completude for preferível, como nesse caso, podemos desativar o buffer circular ao definir o argumento como um valor negativo (por exemplo, -1 aqui).

O exemplo debug_mnist_v2 invoca `enable_dump_debug_info()` ao passar flags de linha de comando a ele. Para executar nosso programa TF2 problemático novamente com essa instrumentação de depuração ativada, execute:

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2 \
    --dump_dir /tmp/tfdbg2_logdir --dump_tensor_debug_mode FULL_HEALTH
```

## Inicialização da GUI do Debugger V2 no TensorBoard

A execução do programa com a instrumentação do depurador cria um logdir em /tmp/tfdbg2_logdir. Podemos inicializar o TensorBoard e apontá-lo ao logdir com:

```sh
tensorboard --logdir /tmp/tfdbg2_logdir
```

No navegador da Web, acesse a página do TensorBoard em http://localhost:6006. O plugin "Debugger V2" estará inativo por padrão, então selecione-o no menu "Inactive plugins" (Plugins inativos) no canto superior direito. Depois disso, você verá algo assim:

![Captura de tela da visão completa do Debugger V2](./images/debugger_v2_1_full_view.png)

## Uso da GUI do Debugger V2 para encontrar a causa raiz de NaNs

A GUI do Debugger V2 no TensorBoard é organizada em seis seções:

- **Alerts** (Alertas): essa seção no canto superior esquerdo contém uma lista de eventos de "alerta" detectados pelo depurador nos dados de depuração do programa TensorFlow instrumentado. Cada alerta indica uma determinada anomalia que merece atenção. Em nosso caso, essa seção destaca 499 eventos NaN/∞ com uma cor rosa avermelhada vibrante. Isso confirma nossa suspeita de que o modelo falha em aprender devido à presença de NaNs e/ou infinitos nos valores de tensor internos. Vamos nos aprofundar nesses alertas em seguida.
- **Python Execution Timeline** (Linha de tempo de execução do Python): essa é a metade superior da seção no meio do topo. Ela apresenta o histórico completo da eager execution de ops e grafos. Cada caixa da linha de tempo é marcada com a letra inicial do nome da op ou do grafo (por exemplo, "T" para a op "TensorSliceDataset" ou "m" para o "modelo" `tf.function`). Podemos navegar por essa linha de tempo usando os botões de navegação e a barra de rolagem acima dela.
- **Graph Execution** (Execução do grafo): localizada no canto superior direito da GUI, essa seção será fundamental para nossa tarefa de depuração. Ela contém um histórico de todos os tensores dtype flutuantes computados dentro dos grafos (ou seja, compilados por `@tf-function`s).
- Inicialmente, a **Graph Structure**, ou Estrutura do grafo, (metade inferior da seção no centro do topo), o **Source Code**, ou Código-fonte, (seção no canto inferior esquerdo) e o **Stack Trace** (seção no canto inferior direito) ficarão vazios. O conteúdo será preenchido quando interagirmos com a GUI. Essas três seções também terão funções importantes na nossa tarefa de depuração.

Depois de nos orientarmos com a organização da interface do usuário, vamos realizar os seguintes passos para entender a fundo por que apareceram NaNs. Primeiro, clique no alerta **NaN/∞** na seção Alerts. Isso rola automaticamente a lista de 600 tensores de grafo na seção Graph Execution e foca no nº 88, que é um tensor chamado `Log:0` gerado por uma op `Log` (logaritmo natural). Uma cor rosa avermelhada vibrante destaca um elemento -∞ entre os 1000 elementos do tensor float32 2D. Esse é o primeiro tensor no histórico de runtime do programa TF2 que contém qualquer NaN ou ∞. Vários dos tensores (na verdade, a maioria) computados depois contêm NaNs. Podemos confirmar isso ao rolar para cima e para baixo na lista do Graph Execution. Essa observação oferece uma forte dica que a op `Log` é a origem da instabilidade numérica nesse programa TF2.

![Debugger V2: alertas Nan / infinito e lista da graph execution](./images/debugger_v2_2_nan_inf_alerts.png)

Por que essa op `Log` gera um -∞? Para responder essa pergunta, é preciso examinar a entrada da op. Ao clicar no nome do tensor (`Log:0`), aparece uma visualização simples, mas informativa, das proximidades da op `Log` no grafo do TensorFlow na seção Graph Structure. Observe a direção de cima para baixo do fluxo de informações. A própria op é mostrada em negrito no meio. Imediatamente acima dela, podemos ver que uma op de marcador de posição fornece a única entrada para a op `Log`. Onde está o tensor gerado por esse marcador de posição `probs` na lista Graph Execution? Ao usar a cor de plano de fundo amarelo como um recurso visual, podemos ver que o tensor `probs:0` está três linhas acima do tensor `Log:0`, ou seja, na linha 85.

![Debugger V2: visão da Graph Structure e tracing do tensor de entrada](./images/debugger_v2_3_graph_input.png)

Uma análise mais cuidadosa do detalhamento numérico do tensor `probs:0` na linha 85 revela por que o consumidor `Log:0` produz um -∞: entre os 1000 elementos de `probs:0`, um elemento tem um valor de 0. O -∞ é resultado da computação do logaritmo natural de 0! Se conseguirmos garantir de alguma forma que a op `Log` seja exposta somente a entradas positivas, poderemos evitar a ocorrência de NaN/∞. Isso é possível ao aplicar o recorte (por exemplo, usando o [`tf.clip_by_value()`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)) no tensor de marcador de posição `probs`.

Estamos mais perto de solucionar o bug, mas ainda não chegamos lá. Para aplicar a correção, precisamos saber onde no código-fonte do Python a op `Log` e a entrada do marcador de posição se originaram. O Debugger V2 fornece suporte de primeira classe para o tracing de ops de grafos e eventos de execução até a fonte. Quando clicamos no tensor `Log:0` em Graph Execution, a seção Stack Trace foi preenchida com o stack trace original da criação da op `Log`. O stack trace é um pouco grande porque inclui vários frames do código interno do TensorFlow (por exemplo, gen_math_ops.py e dumping_callback.py), que podemos ignorar com segurança para a maioria das tarefas de depuração. O frame de interesse está na linha 216 de debug_mnist_v2.py (ou seja, o arquivo Python que estamos tentando depurar). Ao clicar em "Line 216", aparece uma visão da linha correspondente do código na seção Source Code.

![Debugger V2: código-fonte e stack trace](./images/debugger_v2_4_source_code.png)

Por fim, isso nos leva ao código-fonte que criou a op `Log` problemática da entrada `probs`. Essa é nossa função de perda de entropia cruzada categórica e personalizada, que é decorada com `@tf.function` e, portanto, convertida em um grafo do TensorFlow. A op de marcador de posição `probs` corresponde ao primeiro argumento de entrada da função de perda. A op `Log` é criada com a chamada de API tf.math.log().

A correção de recorte de valor para esse bug será algo assim:

```py
  diff = -(labels *
           tf.math.log(tf.clip_by_value(probs), 1e-6, 1.))
```

Isso resolverá a instabilidade numérica nesse programa TF2 e fará com que o MLP treine com sucesso. Outra abordagem possível de correção da instabilidade numérica é usar [`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy).

Isso conclui nossa jornada, desde a observação do bug de um modelo TF2 até a mudança no código que o corrige, com a ajuda da ferramenta Debugger V2, que oferece visibilidade total do histórico de eager e graph execution do programa TF2 instrumentado, incluindo os resumos numéricos dos valores de tensor e a associação entre ops, tensores e o código-fonte original.

## Compatibilidade de hardware do Debugger V2

O Debugger V2 é compatível com hardware de treinamento convencional, incluindo CPU e GPU. O treinamento de várias GPUs com [tf.distributed.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) também é compatível. O suporte à [TPU](https://www.tensorflow.org/guide/tpu) ainda está em fase inicial e exige a chamada de

```py
tf.config.set_soft_device_placement(True)
```

antes da chamada de `enable_dump_debug_info()`. Também pode haver outras limitações em TPUs. Se você tiver problemas usando o Debugger V2, relate os bugs na nossa [página de issues do GitHub](https://github.com/tensorflow/tensorboard/issues).

## Compatibilidade de API do Debugger V2

O Debugger V2 é implementado em um stack de software do TensorFlow de nível relativamente baixo. Por isso, é compatível com [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras), [tf.data](https://www.tensorflow.org/guide/data) e outras APIs desenvolvidas com base nos níveis inferiores do TensorFlow. O Debugger V2 também é compatível com versões anteriores do TF1, embora a Eager Execution Timeline (Linha do tempo da eager execution) fique vazia para logdirs de depuração gerados por programas TF1.

## Dicas de uso da API

Uma pergunta frequente sobre essa API de depuração é onde a chamada de `enable_dump_debug_info()` deve ser inserida no código do TensorFlow. Geralmente, a API deve ser chamada o mais cedo possível no seu programa TF2, preferencialmente após as linhas de importação do Python e antes do desenvolvimento e execução do grafo começar. Isso garantirá a cobertura completa de todas as ops e grafos que alimentam seu modelo e o treinamento dele.

Os tensor_debug_modes compatíveis no momento são: `NO_TENSOR`, `CURT_HEALTH`, `CONCISE_HEALTH`, `FULL_HEALTH` e `SHAPE`. Eles variam em relação à quantidade de informações extraídas de cada tensor e ao overhead de desempenho para o programa depurado. Consulte a [seção args](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info) da documentação de `enable_dump_debug_info()`.

## Overhead de desempenho

A API de depuração apresenta o overhead de desempenho ao programa do TensorFlow instrumentado. O overhead varia por `tensor_debug_mode`, tipo de hardware e natureza do programa TensorFlow instrumentado. Como ponto de referência, em uma GPU, o modo `NO_TENSOR` adiciona um overhead de 15% durante o treinamento de um [modelo Transformer](https://github.com/tensorflow/models/tree/master/official/legacy/transformer) de tamanho de lote 64. A porcentagem de overhead para outros tensor_debug_modes é mais alta: cerca de 50% para os modos `CURT_HEALTH`, `CONCISE_HEALTH`, `FULL_HEALTH` e `SHAPE`. Em CPUs, o overhead é levemente mais baixo. Em TPUs, no momento, o overhead é mais alto.

## Relação com outras APIs de depuração do TensorFlow

Observe que o TensorFlow oferece outras ferramentas e APIs para depuração. Você pode pesquisar essas APIs no [namespace `tf.debugging.*`](https://www.tensorflow.org/api_docs/python/tf/debugging) na página de documentação da API. Entre essas APIs, a [`tf.print()`](https://www.tensorflow.org/api_docs/python/tf/print) é usada com mais frequência. Quando usar o Debugger V2 e quando usar a `tf.print()`? A `tf.print()` é conveniente quando

1. sabemos exatamente quais tensores imprimir,
2. sabemos onde exatamente inserir essas declarações `tf.print()` no código-fonte,
3. o número desses tensores não é muito grande.

Para outros casos (por exemplo, examinando vários valores de tensores, examinando valores de tensores gerados pelo código interno do TensorFlow e pesquisando a origem da instabilidade numérica como mostramos acima), o Debugger V2 oferece uma maneira mais rápida de depurar. Além disso, o Debugger V2 fornece uma abordagem unificada para a inspeção de tensores eager e de grafo. Além disso, disponibiliza informações sobre a estrutura do grafo e localizações de código, que estão além da capacidade de `tf.print()`.

Outra API que pode ser usada para depurar problemas envolvendo ∞ e NaN é [`tf.debugging.enable_check_numerics()`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics). Ao contrário de `enable_dump_debug_info()`, `enable_check_numerics()` não salva as informações de depuração no disco. Em vez disso, ela simplesmente monitora ∞ e NaN durante o runtime do TensorFlow e falha na localização do código de origem assim que uma op gera esses valores numéricos ruins. Ela tem um overhead de desempenho mais baixo em comparação com `enable_dump_debug_info()`, mas não permite um trace completo do histórico de execução do programa nem possui uma interface gráfica do usuário como o Debugger V2.

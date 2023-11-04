# Otimize o desempenho do TensorFlow com o Profiler

[TOC]

Este guia demonstra como usar as ferramentas disponíveis no TensorFlow Profiler para monitorar o desempenho dos seus modelos do TensorFlow. Você verá como avaliar o desempenho do seu modelo no host (CPU), no dispositivo (GPU) ou em uma combinação de host e dispositivo(s).

Fazer o profiling ajuda a entender o uso de recursos de hardware (tempo e memória) das diversas operações (ops) do TensorFlow no seu modelo, resolver os gargalos de desempenho e, por fim, acelerar a execução do modelo.

Este guia mostrará como instalar o Profiler, as diversas ferramentas disponíveis, os diferentes modos que o Profiler usa para coletar dados de desempenho e algumas práticas recomendadas para otimizar o desempenho do modelo.

Se você deseja fazer o profiling do desempenho do seu modelo em TPUs na nuvem, confira o [guia de TPU na nuvem](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

## Instale o Profiler e os pré-requisitos de GPU

Use o pip para instalar o Profiler para o TensorBoard. Um dos requisitos do Profiler são as versões mais recentes do TensorFlow e TensorBoard (a partir da 2.2).

```shell
pip install -U tensorboard_plugin_profile
```

Para fazer o profiling de GPUs, você precisa:

1. Atender aos requisitos de drivers de GPU da NVIDIA® e do Kit de ferramentas CUDA® indicados nos [requisitos de software para suporte a GPUs no TensorFlow](https://www.tensorflow.org/install/gpu#linux_setup).

2. Confirmar se a [interface de ferramentas de profiling da NVIDIA®](https://developer.nvidia.com/cupti) (CUPTI) existe no caminho:

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

Se você não tiver a CUPTI no caminho, adicione o diretório de instalação ao começo da variável de ambiente `$LD_LIBRARY_PATH` por meio da execução de:

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Em seguida, execute o comando `ldconfig` acima novamente para verificar se a biblioteca CUPTI é encontrada.

### Resolva problemas de privilégio de acesso

Ao executar o profiling com o Kit de ferramentas CUDA® em um ambiente Docker no Linux, talvez você observe problemas relacionados a privilégios CUPTI insuficientes (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). Acesse a [Documentação de Desenvolvedor da NVIDIA](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters){:.external} para ver como resolver esses problemas no Linux.

Para resolver problemas de privilégio da CUPTI em um ambiente Docker, execute:

```shell
docker run option '--privileged=true'
```

<a name="profiler_tools"></a>

## Ferramentas do Profiler

Acesse o Profiler na guia **Profile** (Profiling) no TensorBoard, que é exibida somente ao capturar dados do modelo.

Observação: o Profile requer acesso à Internet para carregar as [bibliotecas do Google Chart](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading). Alguns gráficos e tabelas poderão estar ausentes se você executar o TensorBoard inteiramente offline em sua máquina local, atrás de um firewall corporativo ou em um data center.

O Profiler tem várias ferramentas para ajudar com a análise de desempenho:

- Página Overview (Visão geral)
- Input Pipeline Analyzer (Analisador do pipeline de entrada)
- TensorFlow Stats (estatísticas do TensorFlow)
- Trace Viewer (visualizador de tracing)
- GPU Kernel Stats (estatísticas de kernels de GPU)
- Memory Profile Tool (ferramenta de análise de memória)
- Pod Viewer (visualizador de pods)

<a name="overview_page"></a>

### Página Overview (Visão geral)

A página de visão geral apresenta uma visão de alto nível do desempenho do seu modelo durante uma execução da análise. É exibida uma página de visão geral agregada para o host e todos os dispositivos, além de algumas recomendações para melhorar o desempenho de treinamento do modelo. Além disso, é possível selecionar hosts específicos no menu suspenso Host.

A página de visão geral exibe os dados da seguinte forma:

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/overview_page.png?raw=true)

- **Performance Summary** (resumo de desempenho): exibe um resumo geral do desempenho do modelo, que tem duas partes:

    1. Step-time breakdown (detalhamento do tempo dos passos): detalha o tempo médio dos passos em diversas categorias indicando onde o tempo foi gasto:

        - Compilation (compilação): tempo gasto compilando kernels.
        - Input (entrada): tempo gasto lendo dados de entrada.
        - Output (saída): tempo gasto lendo dados de saída.
        - Kernel launch (iniciação de kernels): tempo gasto pelo host para iniciar kernels.
        - Host compute time (tempo de computação do host).
        - Device-to-device communication time (tempo de comunicação dispositivo para dispositivo).
        - On-device compute time (tempo de computação nos dispositivos).
        - All others (todos os outros), incluindo sobrecarga do Python.

    2. Device compute precisions (precisões de computação dos dispositivos): indica a porcentagem do tempo de computação dos dispositivos que usa computações de 16 e 32 bits.

- **Step-time Graph** (gráfico do tempo dos passos): exibe um gráfico do tempo dos passos dos dispositivos (em milissegundos) para todos os passos amostrados. Cada passo é dividido em diversas categorias (com cores diferentes) indicando onde o tempo foi gasto. A área vermelha corresponde à parte do tempo dos passos em que os dispositivos estavam ociosos aguardando dados de entrada do host. A área verde mostra quanto tempo o dispositivo passou efetivamente trabalhando.

- **Top 10 TensorFlow operations on device (e.g. GPU)** [10 principais operações do TensorFlow nos dispositivos (por exemplo, GPU): exibe as operações nos dispositivos que foram executados pelo maior tempo.

    Cada linha exibe o tempo de uma operação (como a porcentagem do tempo levado por todas as operações), o tempo cumulativo, a categoria e o nome.

- **Run Environment** (ambiente de execução): mostra um resumo geral do ambiente de execução do modelo, incluindo:

    - Número de host usados.
    - Tipo de dispositivos (GPU/TPU).
    - Número de núcleos dos dispositivos.

- **Recommendation for Next Step** (recomendação para o próximo passo): indica quanto um modelo está vinculado à entrada e recomenda ferramentas para identificar e resolver gargalos de desempenho do modelo.

<a name="input_pipeline_analyzer"></a>

### Input Pipeline Analyzer (Analisador do pipeline de entrada)

Quando um programa do TensorFlow lê dados de um arquivo, começa pelo topo do grafo do TensorFlow em pipeline. O processo de leitura é dividido em diversas fases de processamento de dados conectadas em série, em que a saída de uma fase é a entrada da próxima. Esse sistema de leitura de dados é chamado de *pipeline de entrada*.

Um pipeline típico para ler registros dos arquivos conta com as seguintes fases:

1. Leitura dos arquivos.
2. Pré-processamento dos arquivos (opcional).
3. Transferência de arquivos do host para os dispositivos.

Um pipeline de entrada ineficiente pode causar uma grave lentidão em sua aplicação, Uma aplicação é considerada **vinculada à entrada** quando passa uma parte considerável do tempo no pipeline de entrada. Use as informações do analisador do pipeline de entrada para entender em que parte o pipeline de entrada é ineficiente.

O analisador do pipeline de entrada indica imediatamente se o seu programa é vinculado à entrada e fornece análises dos dispositivos e do host para depurar gargalos de desempenho em qualquer fase do pipeline de entrada.

Confira as orientações sobre desempenho do pipeline de entrada para ver as práticas recomendadas para otimizá-lo.

#### Painel do pipeline de entrada

Para abrir o analisador do pipeline de entrada, selecione **Profile** (Profiling) e depois **input_pipeline_analyzer** no menu suspenso **Tools** (Ferramentas).

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/input_pipeline_analyzer.png?raw=true)

O painel contém três seções:

1. **Summary** (resumo): resume o pipeline de entrada, indicando se a sua aplicação está vinculada à entrada e, em caso afirmativo, em que nível.
2. **Device-side analysis** (análise dos dispositivos): exibe resultados de análise detalhados dos dispositivos, incluindo o tempo dos passos dos dispositivos e o intervalo de tempo gasto esperando dados de entrada nos núcleos de cada passo.
3. **Host-side analysis** (análise do host): mostra uma análise detalhada do host, incluindo o detalhamento do tempo de processamento da entrada no host.

#### Resumo do pipeline de entrada

O **resumo** indica se o seu programa está vinculado à entrada, apresentando a porcentagem do tempo dos dispositivos gasto esperando a entrada do host. Se você estiver usando um pipeline de entrada padrão que tenha sido instrumentado, a ferramenta indicará onde a maior parte do tempo de processamento da entrada foi gasto.

#### Análise dos dispositivos

Essa análise apresenta informações sobre o tempo gasto nos dispositivos em comparação ao gasto no host e quanto tempo os dispositivos gastaram esperando os dados de entrada do host.

1. **Step time plotted against step number** (gráfico do tempo dos passos versus número do passo): exibe um gráfico do tempo dos passos dos dispositivos (em milissegundos) para todos os passos amostrados. Cada passo é dividido em diversas categorias (com cores diferentes) indicando onde o tempo foi gasto. A área vermelha corresponde à parte do tempo dos passos em que os dispositivos estavam ociosos aguardando dados de entrada do host. A área verde mostra quanto tempo o dispositivo passou efetivamente trabalhando.
2. **Step time statistics** (estatísticas do tempo dos passos): indica a média, o desvio padrão e o intervalo ([mínimo, máximo]) do tempo dos passos dos dispositivos.

#### Análise do host

Esse relatório apresenta um detalhamento do tempo de processamento da entrada (o tempo gasto em operações da API `tf.data`) no host em diversas categorias:

- **Reading data from files on demand** (leitura de dados dos arquivos sob demanda): tempo gasto na leitura de dados dos arquivos sem fazer cache, pré-busca ou intercalação.
- **Reading data from files in advance** (leitura de dados dos arquivos de antemão): tempo gasto na leitura de arquivos, incluindo cache, pré-busca e intercalação.
- **Data preprocessing** (pré-processamento dos dados): tempo gasto nas operações de pré-processamento, como descompactação de imagens.
- **Enqueuing data to be transferred to device** (enfileiramento de dados para transferência aos dispositivos): tempo gasto colocando dados em uma fila interna antes de transferi-los aos dispositivos.

Expanda **Input Op Statistics** (estatísticas das operações de entrada) para ver operações de entrada específicas e suas categorias, detalhadas por tempo de execução.

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/input_op_stats.png?raw=true)

Uma tabela de dados da fonte é exibida, e cada entrada contém as seguintes informações:

1. **Input Op** (operação da entrada): mostra o nome da operação do TensorFlow referente à operação da entrada.
2. **Count** (quantidade): mostra o número total de instâncias de execução de operação durante o período de profiling.
3. **Total Time (in ms)** [tempo total (em ms)]: mostra a soma cumulativa do tempo gasto em cada uma dessas instâncias.
4. **Total Time %** (tempo total percentual): mostra o tempo total gasto em uma operação como uma fração do tempo total gasto no processamento da entrada.
5. **Total Self Time (in ms)** [tempo total próprio (em ms)]: mostra a soma cumulativa do tempo próprio gasto em cada uma dessas instâncias. O tempo próprio mensura o tempo gasto dentro do corpo da função, excluindo-se o tempo gasto na função que ela chama.
6. **Total Self Time %** (tempo total próprio percentual): mostra o tempo total próprio gasto como uma fração do tempo total próprio gasto no processamento da entrada.
7. **Category** (categoria): mostra a categoria do processamento referente à operação da entrada.

<a name="tf_stats"></a>

### TensorFlow Stats (estatísticas do TensorFlow)

A ferramenta TensorFlow Stats mostra o desempenho de cada operação (op) do TensorFlow executada no host ou nos dispositivos durante uma sessão de profiling.

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/tf_stats.png?raw=true)

A ferramenta exibe informações de desempenho em dois painéis:

- O painel superior exibe até quatro gráficos de pizza:

    1. A distribuição do tempo de autoexecução de cada operação no host.
    2. A distribuição do tempo de autoexecução de cada tipo de operação no host.
    3. A distribuição do tempo de autoexecução de cada operação nos dispositivos.
    4. A distribuição do tempo de autoexecução de cada tipo de operação nos dispositivos.

- O painel inferior mostra uma tabela indicando dados sobre operações do TensorFlow, com uma linha para cada operação e uma coluna para cada tipo de dados (para ordenar as colunas, basta clicar no título delas). Clique no **botão Export as CSV** (Exportar como CSV) no lado direito do painel superior para exportar os dados da tabela como um arquivo CSV.

    Observações:

    - Se alguma operação têm operações filha:

        - O tempo total "cumulativo" de uma operação inclui o tempo gasto dentro das operações filha.
        - O tempo total "próprio" de uma operação não inclui o tempo gasto dentro das operações filha.

    - Se uma operação é executada no host:

        - A porcentagem do tempo total próprio nos dispositivos incorrido pela operação será 0.
        - A porcentagem cumulativa do tempo total próprio nos dispositivos até essa operação (incluindo-a) será 0.

    - Se uma operação é executada nos dispositivos:

        - A porcentagem do tempo total próprio no host incorrido pela operação será 0.
        - A porcentagem cumulativa do tempo total próprio no host até essa operação (incluindo-a) será 0.

Você pode optar por incluir ou excluir o tempo de ociosidade nos gráficos de pizza e na tabela.

<a name="trace_viewer"></a>

### Trace Viewer (visualizador de tracing)

O visualizador de tracing exibe uma linha do tempo que mostra:

- Durações das operações que foram executadas por seu modelo do TensorFlow.
- Qual parte do sistema (host ou dispositivo) executou uma operação. Geralmente, o host executa operações de entrada, pré-processa dados de treinamento e os transfere ao dispositivo, enquanto o dispositivo executa o treinamento do modelo em si.

Com o visualizador de tracing, você pode identificar problemas de desempenho em seu modelo e tomar ações para resolvê-los. Por exemplo, de forma geral, você pode identificar se a entrada ou o treinamento do modelo está tomando a maior parte do tempo. Ao detalhar, você pode identificar quais operações demoram mais tempo para executar. Observe que o visualizador de tracing está limitado a 1 milhão de eventos por dispositivo.

#### Interface do visualizador de tracing

Ao abrir o visualizador de tracing, ele parece exibir a execução mais recente:

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/trace_viewer.png?raw=true)

Essa tela contém os principais elementos abaixo:

1. **Timeline pane** (painel de linha do tempo): mostra as operações que os dispositivos e o host executaram ao longo do tempo.
2. **Details pane** (painel de detalhes): mostra informações adicionais sobre as operações selecionadas no painel de linha do tempo.

O painel de linha do tempo contém os elementos abaixo:

1. **Top bar** (barra superior): contém vários controles auxiliares.
2. **Time axis** (eixo do tempo): mostra o tempo relativo ao começo do tracing.
3. **Section and track labels** (rótulos Seção e Monitoramentos): cada seção contém diversos monitoramentos e tem um triângulo à esquerda – clique nele para expandir ou recolher a seção. Existe uma seção para cada elemento de processamento no sistema.
4. **Tool selector** (seletor de ferramentas): contém diversas ferramentas para interagir com o visualizador de tracing, como Zoom, Pan (panorâmica), Select (selecionar) e Timing (intervalo). Use a ferramenta Timing para marcar um intervalo de tempo.
5. **Events** (eventos): mostram o tempo durante o qual uma operação foi executada ou a duração de meta-eventos, como os passos de treinamento.

##### Seções e Monitoramentos

O visualizador de tracing contém as seguintes seções:

- **Uma seção para cada nó do dispositivo**, com o número do chip do dispositivo e nó do dispositivo dentro do chip (por exemplo, `/device:GPU:0 (pid 0)`). Cada seção de nó do dispositivo contém os seguintes monitoramentos:
    - **Step** (passo): mostra a duração dos passos de treinamento que estavam executando nos dispositivos.
    - **TensorFlow Ops** (operações do TensorFlow): mostra as operações executadas nos dispositivos.
    - **XLA Ops** (operações do XLA): mostra as operações (ops) do [XLA](https://www.tensorflow.org/xla/) que executaram no dispositivo se o XLA tiver sido o compilador usado (cada operação do TensorFlow traduz-se eu uma ou várias operações do XLA. O compilador do XLA traduz as operações do XLA em código que é executado no dispositivo).
- **Uma seção para threads executados na CPU da máquina host**, chamada **"Host Threads"** (threads do host). Essa seção contém um monitoramento para cada thread da CPU. Observação: você pode ignorar as informações exibidas junto com os rótulos de seção.

##### Eventos

Os eventos dentro da linha do tempo são exibidos em cores diferentes. As cores em si não têm significado específico.

O visualizador de tracing também pode exibir tracings das chamadas às funções do Python em seu programa do TensorFlow. Se você usa a API `tf.profiler.experimental.start`, pode ativar o tracing do Python pela tupla chamada `ProfilerOptions` ao iniciar o profiling. Por outro lado, se você usa o modo de amostragem para profiling, pode selecionar o nível de tracing nas opções do menu suspenso em **Capture Profile** (Capturar profiling).

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/python_tracer.png?raw=true)

<a name="gpu_kernel_stats"></a>

### GPU Kernel Stats (estatísticas de kernels de GPU)

Essa ferramenta mostra as estatísticas de desempenho e a operação de origem para cada kernel acelerado por GPU.

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/gpu_kernel_stats.png?raw=true)

A ferramenta exibe informações em dois painéis:

- O painel superior exibe um gráfico de pizza, que mostra os kernels CUDA com o maior tempo total decorrido.

- O painel inferior exibe uma tabela com os seguintes dados para cada par kernel-operação único:

    - Uma classificação em ordem decrescente da duração total de GPU, agrupada por par kernel-operação.
    - O nome do kernel iniciado.
    - O número de registros de GPU usados pelo kernel.
    - O tamanho total da memória compartilhada (estática + dinâmica compartilhada) usada, em bytes.
    - As dimensões de bloco, expressas como `blockDim.x, blockDim.y, blockDim.z`.
    - As dimensões de grid, expressas como `gridDim.x, gridDim.y, gridDim.z`.
    - Se a operação pode usar [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/).
    - Se o kernel contém instruções do Tensor Core.
    - O nome da operação que iniciou esse kernel.
    - O número de ocorrências desse par kernel-op.
    - O tempo total de GPU decorrido, em microssegundos.
    - O tempo médio de GPU decorrido, em microssegundos.
    - O tempo mínimo de GPU decorrido, em microssegundos.
    - O tempo máximo de GPU decorrido, em microssegundos.

<a name="memory_profile_tool"></a>

### Memory Profile Tool (ferramenta de análise da memória) {: id = 'memory_profile_tool'}

A ferramenta **Memory Profile** monitora o uso de memória dos seus dispositivos durante o intervalo do profiling. Você pode usar essa ferramenta para:

- Depurar problemas de falta de memória (OOM, na sigla em inglês para out of memory) ao identificar o pico de uso da memória e a alocação de memória correspondente às operações do TensorFlow. Você também pode depurar problemas de OOM que surgem ao executar a inferência [multilocatário](https://arxiv.org/pdf/1901.06887.pdf).
- Depurar problemas de fragmentação da memória.

A ferramenta de profiling de memória exibe dados em três seções:

1. **Memory Profile Summary (Resumo de análise da memória)**
2. **Memory Timeline Graph (Gráfico da linha de tempo da memória)**
3. **Memory Breakdown Table (Tabela de detalhamento da memória)**

#### Memory Profile Summary (Resumo de análise da memória)

Essa seção exibe um resumo geral da análise da memória do seu programa do TensorFlow, conforme exibido abaixo:

&lt;img src="./images/tf_profiler/memory_profile_summary.png" width="400", height="450"&gt;

O resumo da análise da memória tem seis campos:

1. **Memory ID** (ID de memória): menu suspenso que lista todos os sistemas de memória de dispositivo disponíveis. Selecione o sistema de memória que você deseja ver no menu suspenso.
2. **#Allocation** (nº de alocações): número de alocações de memória feitas durante o intervalo do profiling.
3. **#Deallocation** (nº de desalocações): número de desalocações de memória feitas durante o intervalo do profiling.
4. **Memory Capacity** (capacidade de memória): capacidade total (em GiBs) do sistema de memória que você selecionou.
5. **Peak Heap Usage** (pico de heap da memória): pico de uso da memória (em GiBs) desde que o modelo começou a executar.
6. **Peak Memory Usage** (pico de uso da memória): pico de uso da memória (em GiBs) no intervalo do profiling. Esse campo contém os seguintes subcampos:
    1. **Timestamp**: carimbo de data e hora do momento em que o pico de uso de memória ocorreu no gráfico de linha do tempo.
    2. **Stack Reservation**: (reserva da pilha): quantidade de memória reservada na pilha (em GiBs).
    3. **Heap Allocation**: (alocação do heap): quantidade de memória reservada no heap (em GiBs).
    4. **Free Memory** (memória livre): quantidade de memória livre (em GiBs). A Memory Capacity (capacidade de memória) é a soma total de Stack Reservation (reserva da pilha), Heap Allocation (alocação do heap) e Free Memory (memória livre).
    5. **Fragmentation** (fragmentação): porcentagem de framentação (quanto menor, melhor). É calculado como porcentagem de `(1 - Size of the largest chunk of free memory / Total free memory)` (1 - Tamanho do maior conjunto de memória livre/Memória total livre).

#### Memory Timeline Graph (Gráfico da linha de tempo da memória)

Essa seção exibe um gráfico do uso de memória (em GiBs) e a porcentagem de fragmentação versus tempo (em ms).

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/memory_timeline_graph.png?raw=true)

O eixo X representa a linha do tempo (em ms) do intervalo de profiling. O eixo Y à esquerda representa o uso de memória (em GiBs) e o eixo Y à direita representa a porcentagem de fragmentação. Em cada ponto do tempo no eixo X, o total de memória é dividido em três categorias: pilha (em vermelho), heap (em laranja) e livre (em verde). Passe o cursor do mouse por um timestamp específico para ver os detalhes sobre a alocação/desalocação de memória nesse ponto, conforme mostrado abaixo:

![image](./images/tf_profiler/memory_timeline_graph_popup.png)

A janela mostra as seguintes informações:

- **timestamp(ms)**: a localização do evento selecionado na linha do tempo.
- **event** (evento): tipo de evento (alocação ou desalocação).
- **requested_size(GiBs)** (tamanho solicitado): quantidade de memória solicitada. Será um número negativo para eventos de desalocação.
- **allocation_size(GiBs)** (tamanho alocado: quantidade de memória alocada. Será um número negativo para eventos de desalocação.
- **tf_op**: operação do TensorFlow que solicita a alocação/desalocação.
- **step_id** (ID do passo): passo do treinamento em que esse evento ocorreu.
- **region_type** (tipo de região): tipo de entidade de dados que usará essa memória alocada. Os valores possíveis são `temp` para temporários, `output` para ativações e gradientes, e `persist`/`dynamic` para pesos e constantes.
- **data_type** (tipo de dado): tipo do elemento tensor (por exemplo: uint8 para inteiro de 8 bits sem sinal).
- **tensor_shape** (formato do tensor): formato do tensor que está sendo alocado/desalocado.
- **memory_in_use(GiBs)** (memória em uso): total de memória em uso nesse momento.

#### Memory Breakdown Table (Tabela de detalhamento da memória)

Essa tabela mostra as alocações de memória ativas no momento de pico de memória de uso no intervalo do profiling.

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/memory_breakdown_table.png?raw=true)

Há uma linha para cada operação do TensorFlow, e cada linha tem as seguintes colunas:

- **Op Name** (nome da operação): nome da operação do TensorFlow.
- **Allocation Size (GiBs)** [tamanho da alocação (em GiBs)]: total de memória alocada a essa operação.
- **Requested Size (GiBs)** [tamanho solicitado (GiBs)]: total de memória solicitada para essa operação.
- **Occurrences** (ocorrências): número de alocações para essa operação.
- **Region type** (tipo de região): tipo de entidade de dados que usará essa memória alocada. Os valores possíveis são `temp` para temporários, `output` para ativações e gradientes, e `persist`/`dynamic` para pesos e constantes.
- **Data type** (tipo de dado): tipo do elemento tensor.
- **Shape** (formato): formato dos tensores alocados.

Observação: você pode ordenar qualquer coluna da tabela e também filtrar linhas por nome da operação.

<a name="pod_viewer"></a>

### Pod Viewer (visualizador de pods)

A ferramenta Pod Viewer mostra o detalhamento de um passo de treinamento para todos os workers.

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/pod_viewer.png?raw=true)

- O painel superior tem um controle deslizante para selecionar o número do passo.
- O painel inferior exibe um gráfico de colunas empilhadas. É uma visão geral de categorias de tempo dos passos colocadas uma em cima da outra. Cada coluna empilhada representa um worker único.
- Ao passar o cursor do mouse por uma coluna empilhada, a ficha no lado esquerdo mostra o detalhamento do passo.

<a name="tf_data_bottleneck_analysis"></a>

### Análise de gargalo de tf.data

Aviso: essa ferramenta é experimental. Abra um [issue no GitHub](https://github.com/tensorflow/profiler/issues) se o resultado da análise parecer incorreto.

A ferramenta de análise de gargalo de `tf.data` detecta automaticamente gargalos nos pipelines de entrada de `tf.data` em seu programa e fornece recomendações de como resolvê-los. Funciona com qualquer programa que utilize `tf.data`, não importa a plataforma (CPU/GPU/TPU). A análise e as recomendações são baseadas [neste guia](https://www.tensorflow.org/guide/data_performance_analysis).

A ferramenta segue as etapas abaixo para detectar um gargalo:

1. Encontra o host mais vinculado à entrada.
2. Encontra a execução mais lenta de um pipeline de entrada de `tf.data`.
3. Reconstrói o grafo do pipeline de entrada usando o tracing do Profiler.
4. Encontra o caminho crítico no grafo do pipeline de entrada.
5. Identifica a transformação mais lenta no caminho crítico como um gargalo.

A interface gráfica é dividida em três seções: **Performance Analysis Summary** (Resumo da análise do desempenho), **Summary of All Input Pipelines** (Resumo de todos os pipelines de entrada) e **Input Pipeline Graph** (Grafo do pipeline de entrada).

#### Performance Analysis Summary (Resumo da análise do desempenho)

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/tf_data_summary.png?raw=true)

Essa seção apresenta o resumo da análise, indicando pipelines de entrada de `tf.data` lentos no profiling. Essa seção também mostra o host mais vinculado à entrada e seu pipeline de entrada mais lento com a latência máxima. Mais importante ainda, identifica qual parte do pipeline de entrada é o gargalo e como resolvê-lo. As informações de gargalo são fornecidas com o tipo de iterador e seu nome longo.

##### Como ler o nome longo do iterador de tf.data

O formato do nome longo é `Iterator::<Dataset_1>::...::<Dataset_n>`. No nome longo, `<Dataset_n>` coincide com o tipo de iterador, e os outros datasets no nome longo representam transformações downstream.

Por exemplo, considere o seguinte dataset com pipeline de entrada:

```python
dataset = tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)
```

Os nomes longos para os iteradores do dataset acima serão:

Tipo de iterador | Nome longo
:-- | :--
Range | Iterator::Batch::Repeat::Map::Range
Map | Iterator::Batch::Repeat::Map
Repeat | Iterator::Batch::Repeat
Batch | Iterator::Batch

#### Summary of All Input Pipelines (Resumo de todos os pipelines de entrada)

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/tf_data_all_hosts.png?raw=true)

Essa seção apresenta o resumo de todos os pipelines de entrada em todos os hosts. Geralmente, há um pipeline de entrada. Ao usar a estratégia de distribuição, há um pipeline de entrada de host executando o código `tf.data` do programa e vários pipelines de entrada de dispositivo recuperando dados do pipeline de entrada do host e transferindo-os aos dispositivos.

Para cada pipeline de entrada, o resumo mostra as estatísticas do tempo de execução. Uma chamada é considerada lenta se demorar mais de 50 μs.

#### Input Pipeline Graph (Grafo do pipeline de entrada)

![image](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/guide/images/tf_profiler/tf_data_graph_selector.png?raw=true)

Essa seção mostra o grafo do pipeline de entrada com as informações do tempo de execução. Você pode usar "Host" e "Input Pipeline" (pipeline de entrada) para escolher qual host e pipeline de entrada deseja ver. As execuções do pipeline de entrada são ordenadas pelo tempo de execução em ordem decrescente, que você pode escolher usando o menu suspenso **Rank** (Classificação).

![image](./images/tf_profiler/tf_data_graph.png)

Os nós no caminho crítico têm contornos em negrito. O nó do gargalo, que é o nó com o maior tempo próprio no caminho crítico, tem um contorno vermelho. Os outros nós não críticos têm um contorno cinza pontilhado.

Em cada nó, o **Start Time** (tempo de início) indica o tempo de início de cada execução. O mesmo nó pode ser executado diversas vezes, por exemplo, quando há uma operação `Batch` (lote) no pipeline de entrada. Se for executado diversas vezes, é o tempo de início da primeira execução.

A **Total Duration** (duração total) é o tempo total da execução. Se for executado diversas vezes, é a soma do tempo total de todas as execuções.

O **Self Time** (tempo próprio) é o **tempo total** sem o tempo de sobreposição com seus nós filho imediatos.

"# Calls" (nº de chamadas) é o número de vezes que o pipeline de entrada foi executado.

<a name="collect_performance_data"></a>

## Coleta de dados de desempenho

O TensorFlow Profiler coleta atividades do host e rastreamentos de GPU do seu modelo do TensorFlow. É possível configurar o Profiler para coletar dados de desempenho pelo modo programático ou pelo modo de amostragem.

### APIs para profiling

As APIs abaixo podem ser usadas para profiling.

- Modo programático usando o callback do Keras do TensorBoard (`tf.keras.callbacks.TensorBoard`)

    ```python
    # Profile from batches 10 to 15
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch='10, 15')

    # Train the model and use the TensorBoard Keras callback to collect
    # performance profiling data
    model.fit(train_data,
              steps_per_epoch=20,
              epochs=5,
              callbacks=[tb_callback])
    ```

- Modo programático usando a API de função do `tf.profiler`

    ```python
    tf.profiler.experimental.start('logdir')
    # Train the model here
    tf.profiler.experimental.stop()
    ```

- Modo programático usando o gerenciador de contexto

    ```python
    with tf.profiler.experimental.Profile('logdir'):
        # Train the model here
        pass
    ```

Observação: executar o Profiler por tempo demais pode fazê-lo ficar sem memória. É recomendável criar o profiling de até 10 passos de cada vez. Evite criar o profiling dos primeiros lotes para evitar imprecisões devido à sobrecarga de inicialização.

<a name="sampling_mode"></a>

- Modo de amostragem: crie o profiling sob demanda usando `tf.profiler.experimental.server.start` para iniciar um servidor gRPC com a execução do seu modelo do TensorFlow. Após iniciar o servidor gRPC e executar seu modelo, você pode capturar um profiling usando o botão **Capture Profile** (Cappturar profiling) no plug-in de profiling do TensorBoard. Use o script na seção "Instale o Profiler" acima para iniciar uma instância do TensorBoard, caso ainda não esteja em execução.

    Veja um exemplo:

    ```python
    # Start a profiler server before your model runs.
    tf.profiler.experimental.server.start(6009)
    # (Model code goes here).
    #  Send a request to the profiler server to collect a trace of your model.
    tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                          'gs://your_tb_logdir', 2000)
    ```

    Veja um exemplo para criar o profiling com vários workers:

    ```python
    # E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
    # would like to profile for a duration of 2 seconds.
    tf.profiler.experimental.client.trace(
        'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
        'gs://your_tb_logdir',
        2000)
    ```

<a name="capture_dialog"></a>

&lt;img src="./images/tf_profiler/capture_profile.png" width="400", height="450"&gt;

Use a janela **Capture Profile** (Capturar profiling) para especificar:

- Uma lista separada por vírgulas com as URLs ou nomes de TPU do serviço de profiling.
- Uma duração do profiling.
- O nível de tracing de dispositivos, host e chamadas a funções do Python.
- Quantas vezes você quer que o Profiler tente capturar profilings novamente se não tiver sucesso na primeira tentativa.

### Criar profilings de loops de treinamento personalizados

Para criar o profiling de loops de treinamento personalizados em seu código do TensorFlow, faça a instrumentação do loop de treinamento com a API `tf.profiler.experimental.Trace` para identificar os limites de passos para o Profiler.

O argumento `name` (nome) é usado como prefixo para os nomes dos passos, o argumento palavra-chave `step_num` (número de passos) é anexado aos nomes dos passos, e o argumento palavra-chave `_r` faz esse evento de tracing ser processado como um evento de passo pelo Profiler.

Veja um exemplo:

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

Isso ativará a análise de desempenho baseada em passos do Profiler e fará os eventos de passos serem exibidos no visualizador de tracing.

Lembre-se de incluir o iterador de dataset dentro do contexto de `tf.profiler.experimental.Trace` para que a análise do pipeline de entrada seja exata.

O trecho de código abaixo vai contra o padrão:

Aviso: isso resultará na análise imprecisa do pipeline de entrada.

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### Como criar profilings para seus casos de uso

O Profiler abarca diversos casos de uso em quatro eixos diferentes. Há suporte para algumas das combinações atualmente, enquanto outras receberão suporte no futuro. Veja alguns casos de uso:

- *Criação de profiling local versus remota*: são as duas formas comuns de configurar seu ambiente de profiling. Na criação de profilings local, a API de profiling é chamada na mesma máquina em que seu modelo está sendo executado, por exemplo, uma workstation local com GPUs. Na criação de profilings remota, a API de profiling é chamada em uma máquina diferente da que seu modelo está sendo executado, por exemplo, uma TPU na nuvem.
- *Como criar profilings com vários workers*: você pode criar o profiling de diversas máquinas ao usar os recursos de treinamento distribuído do TensorFlow.
- *Plataforma de hardware*: crie profilings de CPUs, GPUs e TPUs.

A tabela abaixo mostra uma visão geral rápida dos casos de uso com suporte do TensorFlow mencionados acima:

<a name="profiling_api_table"></a>

| API de profiling                | Local     | Remoto    | Vários  | Hardware  | :                              :           :           : workers   : Plataformas : | :--------------------------- | :-------- | :-------- | :-------- | :-------- | | **Keras do TensorBoard          | Com suporte | Não       | Não       | CPU, GPU  | : Callback**                   :           : Com suporte : Com suporte :           : | **`tf.profiler.experimental` | Com suporte | Não       | Não       | CPU, GPU  | : [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2) start/stop**    :           : Com suporte : Com suporte :           : | **`tf.profiler.experimental` | Com suporte | Com suporte | Com suporte | CPU, GPU, | : [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace) client.trace**  :           :           :           : TPU       : |  **API de gerenciador de contexto**      | Com suporte | Não       | Não       | CPU, GPU  | :                              :           : Com suporte : Com suporte :           :

<a name="performance_best_practices"></a>

## Práticas recomendadas para um desempenho ideal do modelos

Use as recomendações abaixo conforme aplicável para seus modelos do TensorFlow a fim de atingir o desempenho ideal.

De forma geral, realize todas as transformações no dispositivo e use a versão compatível mais recente das bibliotecas, como cuDNN e Intel MKL, para sua plataforma.

### Otimize o pipeline de entrada de dados

Use os dados do [#input_pipeline_analyzer] (analisador do pipeline de entrada) para otimizar seu pipeline de entrada. Um pipeline de entrada de dados eficiente pode aumentar dramaticamente a velocidade de execução do seu modelo por meio da redução do tempo de ociosidade dos dispositivos. Tente incorporar as práticas recomendadas detalhadas no guia [Desempenho melhor com a API tf.data](https://www.tensorflow.org/guide/data_performance) e abaixo para deixar seu pipeline de entrada de dados mais eficiente.

- De forma geral, paralelizar qualquer operação que não precise ser executada sequencialmente pode otimizar consideravelmente o pipeline de entrada de dados.

- Em diversos casos, faz sentido alterar a ordem de algumas chamadas ou ajustar os argumentos para que funcionem da melhor forma para seu modelo. Ao otimizar o pipeline de entrada de dados, faça o comparativo somente do carregador de dados sem os passos de treinamento e retropropagação para quantificar o efeito das otimizações de forma independente.

- Tente executar seu modelo com dados sintéticos para verificar se o pipeline de entrada é um gargalo de desempenho.

- Use `tf.data.Dataset.shard` para treinamento com várias GPUs. Você deve fragmentar bem no começo do loop de entrada para evitar reduções da taxa de transferência. Ao trabalhar com TFRecords, você deve fragmentar a lista de TFRecords, e não seu conteúdo.

- Paralelize diversas operações por meio da definição dinâmica do valor de `num_parallel_calls` (número de chamadas paralelas) usando `tf.data.AUTOTUNE` (tunagem automática).

- Considere limitar o uso de `tf.data.Dataset.from_generator`, pois ele é mais lento em comparação às operações puramente do TensorFlow.

- Considere limitar o uso de `tf.py_function`, pois ele não pode ser serializado e não há suporte para a execução no TensorFlow distribuído.

- Use `tf.data.Options` para controlar as otimizações estáticas do pipeline de entrada.

Além disso, leia o [guia](https://www.tensorflow.org/guide/data_performance_analysis) sobre análise do desempenho de `tf.data` para conferir mais orientações sobre a otimização do pipeline de entrada.

#### Otimize a ampliação de dados

Ao trabalhar com dados de imagens, aumente a eficiência da [ampliação de dados](https://www.tensorflow.org/tutorials/images/data_augmentation) fazendo a conversão em diferentes tipos de dados <b><i>após</i></b> aplicar transformações espaciais, como inverter, recortar, girar, etc.

Observação: algumas operações, como `tf.image.resize`, alteram de forma transparente o `dtype` (tipo de dados) para `fp32`. Você deve normalizar seus dados de modo a ficarem no intervalo `0` e `1`, caso isso não seja feito automaticamente. Se essa etapa for pulada, pode causar erros `NaN` se você tiver ativado o [AMP](https://developer.nvidia.com/automatic-mixed-precision).

#### Use a DALI da NVIDIA®

Em alguns casos, como quando você tem um sistema com alta proporção GPU/CPU, talvez todas as otimizações acima não sejam suficientes para eliminar gargalos no carregador de dados causados devido à limitações dos ciclos de CPU.

Se você estiver usando GPUs da NVIDIA® para aplicações de aprendizado profundo para visão e áudio computacionais, considere usar a Biblioteca de Carregamento de Dados ([DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting%20started.html), na sigla em inglês) para acelerar o pipeline de dados.

Confira a documentação de [Operações da DALI da NVIDIA®](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html) para ver a lista de operações DALI com suporte.

### Use threading e execução paralela

Execute operações em diversos threads de CPU com a API `tf.config.threading` para executá-las mais rápido.

O TensorFlow define automaticamente o número de threads de paralelismo por padrão. O conjunto de threads disponíveis para executar operações do TensorFlow depende do número de threads de CPU disponíveis.

Controle a aceleração paralela máxima de uma única operação usando `tf.config.threading.set_intra_op_parallelism_threads`. Atenção: se você executar diversas operações em paralelo, todas vão compartilhar o mesmo conjunto de threads disponíveis.

Se você tiver operações independentes que não causam bloqueio (operações sem um caminho direcionado entre elas no grafo), use `tf.config.threading.set_inter_op_parallelism_threads` para executá-las simultaneamente usando o conjunto de threads disponíveis.

### Diversos

Ao trabalhar com modelos menores em GPUs da NVIDIA®, você pode definir `tf.compat.v1.ConfigProto.force_gpu_compatible=True` para forçar todos os tensores em CPUs a terem memória fixada CUDA alocada para melhorar bastante o desempenho do modelo. Entretanto, tenha cuidado ao usar essa opção para modelos desconhecidos/muito grandes, pois isso pode impactar negativamente o desempenho do host (CPU).

### Aumente o desempenho dos dispositivos

Siga as práticas recomendadas detalhadas aqui e no [guia de otimização de desempenho de GPUs](https://www.tensorflow.org/guide/gpu_performance_analysis) para otimizar o desempenho de modelos do TensorFlow em dispositivos.

Se você estiver usando GPUs da NVIDIA, crie um log com a utilização de GPU e memória em um arquivo CSV executando:

```shell
nvidia-smi
--query-gpu=utilization.gpu,utilization.memory,memory.total,
memory.free,memory.used --format=csv
```

#### Configure o layout de dados

Ao trabalhar com dados que contêm informações de canal (como imagens), otimize o formato do layout de dados, optando por canais por último (prefira NHWC em vez de NCHW).

Os formatos de dados com canal por último melhoram a utilização dos [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/) e trazem melhorias de desempenho consideráveis, especialmente em modelos convolucionais usados em conjunto com AMP. Os layouts de dados NCHW podem ser operados por Tensor Cores, mas trazem uma sobrecarga adicional devido às operações de transposição automáticas.

Para otimizar o layout de dados, optando por layouts NHWC, basta definir `data_format="channels_last"` para camadas como `tf.keras.layers.Conv2D`, `tf.keras.layers.Conv3D` e `tf.keras.layers.RandomRotation`.

Use `tf.keras.backend.set_image_data_format` para definir o formato de layout de dados padrão para a API de back-end do Keras.

#### Maximize o cache L2

Ao trabalhar com GPUs da NVIDIA®, execute o trecho de código abaixo antes do loop de treinamento para maximizar a granularidade de busca L2 para 128 bytes.

```python
import ctypes

_libcudart = ctypes.CDLL('libcudart.so')
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
assert pValue.contents.value == 128
```

#### Configure o uso de threads de GPUs

O modo de threads de GPUs decide quantos threads de GPUs são usados.

Defina o modo de threads como `gpu_private` para garantir que o pré-processamento não "roube" todos os threads de GPUs, o que reduz o atraso de iniciação de kernels durante o treinamento. Além disso, você pode definir o número de threads por GPUs. Defina esses valores por meio de variáveis de ambiente.

```python
import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'
```

#### Configure opções de memória de GPUs

De forma geral, aumente o tamanho do lote e dimensione o modelo para utilizar melhor as GPUs e obter uma taxa de transferência maior. Observação: aumentar o tamanho do lote vai alterar a exatidão do modelo e, portanto, o modelo precisa ser dimensionado por meio da tunagem de hiperparâmetros, como a taxa de aprendizado, para atingir a exatidão desejada.

Além disso, use `tf.config.experimental.set_memory_growth` para permitir que a memória de GPUs expanda a fim de evitar que toda a memória disponível seja totalmente alocada às operações que exijam somente uma fração da memória, o que permite a outros processos usar a memória de GPUs para executar no mesmo dispositivo.

Saiba mais nas orientações de [Limitação do crescimento de memória de GPUs](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth) no guia sobre GPUs.

#### Diversos

- Aumente o tamanho do minilote de treinamento (número de amostras de treinamento usadas por dispositivo em uma iteração do loop de treinamento) para o valor máximo que caiba sem causar um erro de falta de memória (OOM, na sigla em inglês de out of memory) na GPU. Aumentar o tamanho do lote impacta a exatidão do modelo, então dimensione o modelo por meio da tunagem de hiperparâmetros para atingir a exatidão desejada.

- Desative os relatórios de erros OOM durante a alocação de tensores no código de produção. Defina `report_tensor_allocations_upon_oom=False` em `tf.compat.v1.RunOptions`.

- Para modelos com camadas convolucionais, remova a adição de bias ao usar normalização de lote. A normalização de lote desloca os valores pela média, o que remove a necessidade de se ter um termo de bias constante.

- Use TF Stats para descobrir a eficiência da execução de operações nos dispositivos.

- Use `tf.function` para fazer computações e, opcionalmente, ative o sinalizador `jit_compile=True` (`tf.function(jit_compile=True`). Saiba mais no guia [Use tf.function do XLA](https://www.tensorflow.org/xla/tutorials/jit_compile).

- Minimize as operações do Python no host entre os passos e reduza os callbacks. Calcule métricas a cada determinado número passos em vez de em todo passo.

- Mantenha as unidades computacionais dos dispositivos ocupadas.

- Envie dados a vários dispositivos em paralelo.

- Considere [usar representações numéricas com 16 bits](https://www.tensorflow.org/guide/mixed_precision), como `fp16` — o formato de ponto flutuante de alta precisão especificado pelo IEEE — ou o formato de ponto flutuante do Brain [bfloat16](https://cloud.google.com/tpu/docs/bfloat16).

## Recursos adicionais

- Tutorial do [TensorFlow Profiler: Crie profilings do desempenho de modelos](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) com o Keras e o TensorBoard, em que você pode aplicar as orientações deste guia.
- Conversa [Análises de desempenho no TensorFlow 2](https://www.youtube.com/watch?v=pXHAQIhhMhI), do TensorFlow Dev Summit de 2020.
- [Demonstração do TensorFlow Profiler](https://www.youtube.com/watch?v=e4_4D7uNvf8) do TensorFlow Dev Summit de 2020.

## Limitações conhecidas

### Análises de várias GPUs no TensorFlow 2.2 e TensorFlow 2.3

O TensorFlow 2.2 e 2.3 têm suporte a análises de várias GPUs somente para sistemas com um único host. Não há suporte à análise de várias GPUs em sistemas com vários hosts. Para fazer a análise em configurações com GPUs com vários workers, cada worker precisa ser analisado de maneira independente. A partir do TensorFlow 2.4, é possível fazer a análise de vários workers usando a API `tf.profiler.experimental.client.trace`.

O Kit de ferramentas CUDA® 10.2 ou posterior é necessário para analisar várias GPUs. Como o TensorFlow 2.2 e 2.3 têm suporte ao Kit de ferramentas CUDA® somente até a versão 10.1, você precisa criar links simbólicos para `libcudart.so.10.1` e `libcupti.so.10.1`:

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```

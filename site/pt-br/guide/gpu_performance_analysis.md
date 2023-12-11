# Otimize o desempenho de GPUs do TensorFlow com o TensorFlow Profiler

## Visão geral

Este guia mostrará como usar o TensorFlow Profiler com o TensorBoard para ver o desempenho e conseguir obter o desempenho máximo das GPUs, além de identificar quando uma ou mais GPUs estiverem sendo subutilizadas.

Se você ainda não conhecer o Profiler muito bem:

- Comece pelo notebook [TensorFlow Profiler: Analise o desempenho do modelo](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras), com um exemplo do Keras e o [TensorBoard](https://www.tensorflow.org/tensorboard).
- Aprenda os diferentes métodos e ferramentas de análise disponíveis para otimizar o desempenho no host (CPU) no guia [Otimize o desempenho do TensorFlow usando o Profiler](https://www.tensorflow.org/guide/profiler#profiler_tools).

Lembre-se de que descarregar as computações na GPU nem sempre será benéfico, especialmente para modelos pequenos. Pode haver sobrecarga:

- Devido à transferência entre o host (CPU) e o dispositivo (GPU).
- Devido à latência que ocorre quando o host inicia kernels nas GPUs.

### Workflow de otimização de desempenho

Este guia mostra como depurar problemas de desempenho, começando com uma única GPU e passando para um único host com várias GPUs.

É recomendável depurar problemas de desempenho na seguinte ordem:

1. Otimize e depure o desempenho em uma GPU:
    1. Verifique se o pipeline de entrada é um gargalo.
    2. Depure o desempenho em uma GPU.
    3. Ative a precisão mista (com `fp16` (float16)) e, opcionalmente, ative o [XLA](https://www.tensorflow.org/xla).
2. Otimize e depure o desempenho em um host com várias GPUs.

Por exemplo, se você estiver usando uma [estratégia de distribuição](https://www.tensorflow.org/guide/distributed_training) do TensorFlow para treinar um modelo em um único host com várias GPUs e observar uma subutilização das GPUs, primeiro deve otimizar e depurar o desempenho em uma GPU antes de depurar o sistema com várias GPUs.

Este guia é uma linha de base para conseguir código com bom desempenho ao usar GPUs e pressupõe que você esteja usando `tf.function`. As APIs `Model.compile` e `Model.fit` do Keras utilizarão `tf.function` automaticamente por baixo dos panos. Ao escrever um loop de treinamento personalizado com `tf.GradientTape`, confira o guia [Desempenho melhor com tf.function](https://www.tensorflow.org/guide/function) para ver como ativar `tf.function`s.

As próximas seções discutem sugestões de estratégias para cada cenário acima a fim de ajudar a identificar e corrigir gargalos de desempenho.

## 1. Otimize o desempenho em uma GPU:

Idealmente, seu programa deve ter uma alta utilização das GPUs, uma comunicação mínima entre a CPU (host) e a GPU (dispositivo) e nenhuma sobrecarga do pipeline de entrada.

A primeira etapa ao analisar o desempenho é traçar um perfil para um modelo sendo executado com uma única GPU.

A [página de visão geral](https://www.tensorflow.org/guide/profiler#overview_page) do TensorBoard Profiler, que mostra uma visão de alto nível do desempenho do seu modelo em uma execução, pode dar uma ideia de quão longe o seu programa está do cenário ideal.

![TensorFlow Profiler Overview Page](images/gpu_perf_analysis/overview_page.png "The overview page of the TensorFlow Profiler")

Os principais números que você deve avaliar na página de visão geral são:

1. Qual parcela do tempo de execução advém da execução no dispositivo.
2. A porcentagem de operações feitas no dispositivo versus host.
3. Quantos kernels usam `fp16`.

Para atingir o desempenho ideal, é preciso maximizar esses números nos três casos. Para compreender profundamente o seu programa, você precisará conhecer o [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) do TensorBoard Profiler. As próximas seções mostram alguns padrões comuns do visualizador de tracing que você deve identificar ao diagnosticar gargalos de desempenho.

Veja abaixo uma imagem do visualizador de tracing de um modelo sendo executado em uma GPU. Nas seções *TensorFlow Name Scope* (Escopo de nomes do TensorFlow) e *TensorFlow Ops* (Operações do TensorFlow), você pode identificar diferentes partes do modelo, como o passo para frente, a perda de função, o passo para trás/cálculo de gradientes e a atualização dos pesos do otimizador. Além disso, você pode deixar as operações sendo executadas na GPU próximas de cada *Stream*, que refere-se a streams CUDA. Cada stream é usado para tarefas específicas. Nesse tracing, o *Stream#118* é usado para iniciar kernels de computação e cópias dispositivo para dispositivo. O *Stream#119* é usado para cópias host para dispositivo, e o *Stream#120* é usado para cópias dispositivo para host.

O tracing abaixo mostra características comuns de um modelo com bom desempenho.

![image](images/gpu_perf_analysis/traceview_ideal.png "An example TensorFlow Profiler trace view")

Por exemplo, a linha de tempo de computações na GPU (*Stream#118*) parece "ocupada", com pouquíssimas lacunas. Há cópias mínimas do host para o dispositivo (*Stream #119*) e do dispositivo para o host (*Stream #120*), bem como lacunas mínimas entre os passos. Quando você executar o Profiler em seu programa, talvez não consiga identificar essas características ideias na visualização de tracing. O restante deste guia discute cenários comuns e como corrigir os gargalos.

### 1. Depure o pipeline de entrada

A primeira etapa para depurar o desempenho na GPU é determinar se o seu programa é vinculado à entrada. A maneira mais fácil de descobrir é usando o [analisador de pipeline de entrada](https://www.tensorflow.org/guide/profiler#input_pipeline_analyzer) do Profiler no TensorBoard, que mostra uma visão geral do tempo gasto no pipeline de entrada.

![image](images/gpu_perf_analysis/input_pipeline_analyzer.png "TensorFlow Profiler Input-Analyzer")

Você pode realizar as possíveis ações abaixo se o seu pipeline de entrada contribuir com o tempo do passo de maneira significativa:

- Leia o [guia](https://www.tensorflow.org/guide/data_performance_analysis) específico sobre <code>tf.data</code> para aprender a depurar o pipeline de entrada.
- Outra maneira rápida de verificar se o pipeline de entrada é o gargalo é usando dados de entrada gerados aleatoriamente que não precisem de pré-processamento. [Veja um exemplo](https://github.com/tensorflow/models/blob/4a5770827edf1c3974274ba3e4169d0e5ba7478a/official/vision/image_classification/resnet/resnet_runnable.py#L50-L57) do uso dessa técnica para um modelo ResNet. Se o pipeline de entrada for ideal, você deverá observar um desempenho similar com dados reais e com dados aleatórios/sintéticos gerados. A única sobrecarga no caso de dados sintéticos ocorrerá devido à cópia dos dados de entrada, e é possível fazer a pré-busca e otimização.

Além disso, confira as [práticas recomendadas para otimizar o pipeline de dados de entrada](https://www.tensorflow.org/guide/profiler#optimize_the_input_data_pipeline).

### 2. Depure o desempenho em uma GPU

Diversos fatores podem contribuir para utilização baixa das GPUs. Veja abaixo alguns cenários comuns observados ao conferir o [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) e as possíveis soluções.

#### 1. Analise lacunas entre os passos

Uma observação comum quando o programa não tem execução ideal são lacunas entre os passos de treinamento. Na imagem do visualizador de tracing abaixo, há uma grande lacuna entre os passos 8 e 9, ou seja, a GPU está ociosa durante esse período.

![image](images/gpu_perf_analysis/traceview_step_gaps.png "TensorFlow Profile trace view showing gaps between steps")

Se o visualizador de tracing mostrar lacunas grandes entre os passos, pode ser um indicativo de que o seu programa está vinculado à entrada. Nesse caso, confira a seção anterior para depurar o pipeline de entrada, caso ainda não o tenha feito.

Entretanto, mesmo com um pipeline de entrada otimizado, ainda poderá haver lacunas entre o final de um passo e o começo de outro devido à contenção de threads da CPU. `tf.data` utiliza threads em segundo plano para paralelizar o processamento do pipeline. Esses threads podem interferir nas atividades do host com GPUs que ocorrem no começo de cada passo, como a cópia de dados ou o agendamento de operações nas GPUs.

Se você observar lacunas grandes no host, que agenda essas operações na GPU, pode definir a variável de ambiente `TF_GPU_THREAD_MODE=gpu_private`, que garante que os kernels de GPU sejam iniciados a partir de seus próprios threads dedicados e que não sejam colocados na fila atrás de trabalhos do `tf.data`.

As lacunas entre os passos também podem ser causadas por cálculos de métricas, callbacks do Keras ou operações fora de `tf.function` que não têm a execução feita no host. Essas operações não têm um desempenho tão bom quanto as operações dentro de um grafo do TensorFlow. Além disso, algumas dessas operações são executadas na CPU e copiam tensores para a GPU e da GPU.

Se, após otimizar o pipeline de entrada, você ainda observar lacunas entre os passos no visualizador de tracing, deve avaliar o código do modelo entre os passos e verificar se o desempenho aumenta ao desativar callbacks/métricas. Alguns detalhes dessas operações também são exibidos no visualizador de tracing (tanto no dispositivo quanto no host). Nesse cenário, a recomendação é reduzir a sobrecarga dessas operações fazendo a execução delas após um número fixo de passos em vez de fazê-la a cada passo. Ao usar o método `Model.compile` da API `tf.keras`, isso é feito automaticamente ao definir o sinalizador `steps_per_execution`. Para loops de treinamento personalizados, use `tf.while_loop`.

#### 2. Consiga uma utilização maior do dispositivo

##### 1. Kernels de GPU pequenos e atrasos ao iniciar kernels no host

O host enfileira a execução de kernels na GPU, mas existe uma latência (de cerca de 20 a 40 μs) antes de os kernels serem executados na GPU. Idealmente, o host enfileira kernels suficientes na GPU de forma que a GPU passe a maior parte do tempo executando em vez de esperando que o host enfileire mais kernels.

A [página de visão geral](https://www.tensorflow.org/guide/profiler#overview_page) do Profiler no TensorBoard mostra quanto tempo a GPU ficou ociosa devido à espera do início de kernels pelo host. Na imagem abaixo, a GPU fica ociosa cerca de 10% do tempo do passo aguardando que os kernels sejam iniciados.

![image](images/gpu_perf_analysis/performance_summary.png "Summary of performance from TensorFlow Profile")

O [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) para esse mesmo programa mostra lacunas entre os kernels, em que o host está ocupado iniciando kernels na GPU.

![image](images/gpu_perf_analysis/traceview_kernel_gaps.png "TensorFlow Profile trace view demonstrating gaps between kernels")

Ao iniciar várias operações pequenas na GPU (como adição de escalares, por exemplo), talvez o host não consiga acompanhar o ritmo da GPU. A ferramenta [TensorFlow Stats](https://www.tensorflow.org/guide/profiler#tensorflow_stats) do TensorBoard para o mesmo Profile mostra 126.224 operações Mul que demoram 2,77 segundos. Portanto, cada kernel demora cerca de 21,9 μs, o que é muito pouco (cerca do mesmo tempo que a latência de iniciação) e pode resultar em atrasos ao iniciar kernels do host.

![image](images/gpu_perf_analysis/tensorflow_stats_page.png "TensorFlow Profile stats page")

Se o [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) mostrar muitas lacunas pequenas entre as operações na GPU, como na imagem acima, você pode:

- Concatenar tensores pequenos e usar operações vetorizadas ou usar um tamanho de lote maior para que cada kernel iniciado realize mais trabalho, o que manterá a GPU ocupada por mais tempo.
- Confirmar se você está usando `tf.function` para criar grafos do TensorFlow para que não execute operações em um modo adiantado (eager) puro. Se você estiver usando `Model.fit` (em contraste a um loop de treinamento personalizado com `tf.GradientTape`), então `tf.keras.Model.compile` fará isso automaticamente.
- Combine kernels usando o XLA com `tf.function(jit_compile=True)` ou clustering automático. Confira mais detalhes na seção [Ative a precisão mista e o XLA](#3._enable_mixed_precision_and_xla) abaixo para ver como ativar o XLA para conseguir um desempenho maior. Esse recurso pode levar a uma utilização maior do dispositivo.

##### 2. Colocação das operações do TensorFlow

A [página de visão geral](https://www.tensorflow.org/guide/profiler#overview_page) do Profiler mostra a porcentagem de operações colocadas no host versus no dispositivo (você também pode conferir a colocação de operações específicas no [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer)). Como na imagem abaixo, você vai querer que a porcentagem de operações no host seja muito pequena em comparação com o dispositivo.

![image](images/gpu_perf_analysis/opp_placement.png "TF Op Placement")

Idealmente, a maioria das operações de computação interna devem ser colocadas na GPU.

Para descobrir a quais dispositivos as operações e os tensores do seu programa estão atribuídos, defina `tf.debugging.set_log_device_placement(True)` como a primeira declaração do seu programa.

Em alguns casos, mesmo se você especificar que uma operação deve ser colocada em um dispositivo específico, a implementação poderá sobrescrever esse comportamento (exemplo: `tf.unique`). Mesmo com treinamento em uma única GPU, se você especificar uma estratégia de distribuição, como `tf.distribute.OneDeviceStrategy`, poderá resultar em uma colocação mais determinísticas das operações em seu dispositivo.

Um motivo para que a maioria das operações sejam colocadas na GPU é evitar cópias excessivas de memória entre o host e o dispositivo (são esperadas cópias de memória referentes aos dados de saída/entrada do modelo entre o host e o dispositivo). Um exemplo de excesso de cópias é demonstrado no visualizador de tracing abaixo nos streams *#167*, *#168* e *#169* da GPU.

![image](images/gpu_perf_analysis/traceview_excessive_copy.png "TensorFlow Profile trace view demonstrating excessive H2D/D2H copies")

Às vezes, essas cópias podem afetar negativamente o desempenho se elas bloquearem a execução de kernels na GPU. As operações de cópia de memória no [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) têm mais informações sobre as operações que são a origem desses tensores copiados, mas nem sempre será fácil associar um memCopy a uma operação. Nesses casos, vale a pena verificar as operações próximas para conferir se a cópia de memória ocorre no mesmo local em todo passo.

#### 3. Kernels mais eficientes em GPUs

Quando a utilização de GPU do seu programa estiver aceitável, o próximo passo é tentar aumentar a eficiência dos kernels na GPU utilizando Tensor Cores ou combinação de operações.

##### 1. Utilize Tensor Cores

As GPUs modernas da NVIDIA® têm [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/) especializados que podem aumentar significativamente o desempenho de kernels compatíveis.

Você pode usar as [estatísticas de kernel de GPU](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) do TensorBoard para verificar quais kernels são compatíveis com Tensor Cores e quais kernels estão realmente utilizando Tensor Cores. Ativar `fp16` (Confira a seção "Ative a precisão mista" abaixo) é uma maneira de fazer seus kernels de Multiplicação de Matriz Geral (GEMM, na sigla em inglês) (operações matmul) utilizarem os Tensor Cores. Os kernels de GPU utilizam os Tensor Cores com eficiência quando a precisão é fp16 e quando as dimensões dos tensores de entrada/saída são divisíveis por 8 ou 16 (para `int8`).

Observação: com o cuDNN v.7.6.3 e posteriores, as dimensões de convolução serão preenchidas automaticamente quando necessário para utilizar os Tensor Cores.

Confira outras recomendações detalhadas de como tornar os kernels eficientes para GPUs no [guia de desempenho de aprendizado profundo da NVIDIA®](https://docs.nvidia.com/deeplearning/performance/index.html#perf-guidelines).

##### 2. Combine operações

Use `tf.function(jit_compile=True)` para combinar operações menores, formando kernels maiores, o que leva a ganhos de desempenho significativos. Saiba mais no guia [XLA](https://www.tensorflow.org/xla).

### 3. Ative a precisão mista e o XLA

Após seguir as etapas acima, ativar a precisão mista e o XLA são duas etapas opcionais para aumentar ainda mais o desempenho. A estratégia sugerida é ativá-los um de cada vez e verificar se os benefícios de desempenho são os que você esperava.

#### 1. Ative a precisão mista

O guia [Precisão mista](https://www.tensorflow.org/guide/keras/mixed_precision) do TensorFlow mostra como ativar a precisão `fp16` em GPUs. Ative o [AMP](https://developer.nvidia.com/automatic-mixed-precision) nas GPUs da NVIDIA® para usar os Tensor Cores e aumentar a velocidade geral em até 3 vezes em comparação com o uso da precisão `fp32` (float32) na arquitetura Volta e em arquiteturas de GPU mais novas.

Confirme se as dimensões de matrizes/tensores atendem aos requisitos de chamada de kernels que usam Tensor Cores. Os kernels de GPU usam Tensor Cores de maneira eficiente quando a precisão é fp16 e quando as dimensões de entrada/saída são divisíveis por 8 ou 16 (para int8).

Observação: com o cuDNN v.7.6.3 e posteriores, as dimensões de convolução serão preenchidas automaticamente quando necessário para utilizar os Tensor Cores.

Siga as práticas recomendadas abaixo para maximizar os benefícios de desempenho da precisão `fp16`.

##### 1. Use kernels fp16 ideais

Com o `fp16` ativado, os kernels de multiplicação de matriz (GEMM) do seu programa deverão usar a versão `fp16` correspondente que utiliza os Tensor Cores. Porém, em alguns casos, isso não acontece, e você não observa a aceleração esperada ao ativar o `fp16`, pois seu programa usa a implementação ineficiente no lugar dele.

![image](images/gpu_perf_analysis/gpu_kernels.png "TensorFlow Profile GPU Kernel Stats page")

A [página de estatísticas de kernels de GPU](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) mostra quais operações são compatíveis com os Tensor Cores e quais kernels estão realmente utilizando os Tensor Cores eficientes. O [guia de desempenho de aprendizado profundo da NVIDIA®](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores) apresenta outras sugestões de como usar os Tensor Cores. Além disso, os benefícios de usar o `fp16` também são válidos para kernels que eram vinculados à memória anteriormente, pois agora as operações demorarão metade do tempo.

##### 2. Dimensionamento de perda dinâmico versus estático

É necessário fazer o dimensionamento da perda ao usar o `fp16` para evitar um fluxo baixo devido à baixa precisão. Existem dois tipos de dimensionamento de perda – dinâmico e estático – e os dois são explicados com mais detalhes no [guia de precisão mista](https://www.tensorflow.org/guide/keras/mixed_precision). Você pode usar a política `mixed_float16` para ativar automaticamente o dimensionamento de perda no otimizador do Keras.

Observação: por padrão, a API de precisão mista do Keras faz a avaliação de operações softmax independentes (operações que não fazem parte de uma função de perda do Keras) como `fp16`, o que pode levar a problemas numéricos e uma convergência ruim. Converta essas operações em `fp32` para atingir o desempenho ideal.

Ao tentar otimizar o desempenho, é importante lembra que o dimensionamento de perda dinâmico pode gerar operações condicionais adicionais executadas no host e levar a lacunas entre os passos visíveis no visualizador de tracing. Por outro lado, o dimensionamento de perda estático não tem essas sobrecargas e pode ser uma opção melhor quanto ao desempenho, com a ressalva de que você precisa especificar o valor de escala estática de perda correto.

#### 2. Ative o XLA com tf.function(jit_compile=True) ou clustering automático

Uma etapa final para obter o melhor desempenho com uma única GPU é ativar o XLA, que combinará operações e levará a uma melhor utilização dos dispositivos e a um menor consumo de memória. Confira mais detalhes de como ativar o XLA em seu programa com `tf.function(jit_compile=True)` ou clustering automático no guia [XLA](https://www.tensorflow.org/xla).

Você pode definir o nível JIT global como `-1` (desativado), `1` ou `2`. Um nível mais alto é mais intenso e pode diminuir o paralelismo e usar mais memória. Defina o valor como `1` se você tiver restrições de memória. Observação: o XLA não tem bom desempenho para modelos com formatos variáveis de tensores de entrada, pois o compilador do XLA teria que compilar novamente os kernels sempre que encontrasse novos formatos.

## 2. Otimize o desempenho em um host com várias GPUs

A API `tf.distribute.MirroredStrategy` pode fazer o treinamento com várias GPUs em um único host em vez de usar somente uma GPU (para saber mais sobre como fazer treinamento distribuído com o TensorFlow, confira os guias [Treinamento distribuído com o TensorFlow](https://www.tensorflow.org/guide/distributed_training), [Use uma GPU](https://www.tensorflow.org/guide/gpu) e [Use TPUs](https://www.tensorflow.org/guide/tpu), além do tutorial [Treinamento distribuído com o Keras](https://www.tensorflow.org/tutorials/distribute/keras)).

Embora a transição de uma GPU para várias deve, idealmente, ser possível de forma integrada, às vezes você poderá observar problemas de desempenho.

Ao sair do treinamento em uma única GPU para várias GPUs no mesmo host, idealmente você deverá observar escalabilidade do desempenho, com apenas a sobrecarga adicional de comunicação de gradientes e o aumento da utilização de threads no host. Devido a essa sobrecarga, você não terá exatamente o dobro de aceleração se passar de 1 para 2 GPUs, por exemplo.

A visualização de tracing abaixo mostra um exemplo da sobrecarga extra de comunicação ao fazer o treinamento em várias GPUs. Há uma certa sobrecarga para concatenar os gradientes, comunicá-los entre as réplicas e dividi-los antes de fazer a atualização dos pesos.

![image](images/gpu_perf_analysis/traceview_multi_gpu.png "TensorFlow Profile trace view for single host multi GPU scenario")

As etapas abaixo ajudarão a atingir o desempenho ideal ao fazer a otimização para o caso com várias GPUs:

1. Tente maximizar o tamanho do lote, o que levará a uma maior utilização dos dispositivos e reduzirá os custos de comunicação entre as várias GPUs. Usar o [profiler de memória](https://www.tensorflow.org/guide/profiler#memory_profile_summary) ajuda a ver se o seu programa está perto do pico de utilização de memória. Embora um tamanho de lote maior possa afetar a convergência, geralmente os benefícios de desempenho são maiores.
2. Ao passar a usar várias GPUs em vez de uma só, agora o mesmo host precisa processar muito mais dados de entrada. Portanto, após a etapa 1, é recomendável verificar novamente o desempenho do pipeline de entrada e garantir que não seja um gargalo.
3. Verifique a linha de tempo das GPUs no visualizador de tracing do seu programa para ver se há chamadas AllReduce desnecessárias, pois isso resultaria em sincronização entre todos os dispositivos. Na visualização de tracing exibida acima, o AllReduce é feito pelo kernel da [NCCL](https://developer.nvidia.com/nccl), e há somente uma chamada NCCL em cada GPU para os gradientes em cada passo.
4. Verifique se há operações de cópia dispositivo para host, host para dispositivo e dispositivo para dispositivo desnecessárias que podem ser minimizadas.
5. Confira o tempo do passo para confirmar se cada réplica está fazendo o mesmo trabalho. Por exemplo, às vezes, uma GPU (geralmente, `GPU0`) faz mais trabalho porque o host acaba passando mais trabalho a ela por engano.
6. Por fim, verifique o passo de treinamento em todas as GPUs no visualizador de tracing para ver se há operações sendo executadas sequencialmente. Geralmente, isso acontece quando o programa inclui dependências de controle de uma GPU para outra. No passado, a depuração do desempenho nessas situações era feita caso a caso. Se você observar esse comportamento em seu programa, [abra um issue no GitHub](https://github.com/tensorflow/tensorflow/issues/new/choose) e inclua imagens do seu visualizador de tracing.

### 1. Otimize o AllReduce dos gradientes

Ao fazer o treinamento usando uma estratégia síncrona, cada dispositivo recebe uma parte dos dados de entrada.

Após computar os passos para frente e para trás no modelo, os gradientes calculados em cada dispositivo precisam ser agregados e reduzidos. Esse *AllReduce dos gradientes* ocorre após o cálculo dos gradientes em cada dispositivo e antes de o otimizador atualizar os pesos do modelo.

Primeiro, cada GPU concatena os gradientes das camadas do modelo, comunica-os para as GPUs usando `tf.distribute.CrossDeviceOps` (`tf.distribute.NcclAllReduce` é o padrão) e depois retorna os gradientes após a redução por camada.

O otimizador usará esses gradientes reduzidos para atualizar os pesos do modelo. Idealmente, esse processo deve ocorrer ao mesmo tempo em todas as GPUs para evitar sobrecargas.

O tempo para fazer o AllReduce deve ser aproximadamente igual:

```
(number of parameters * 4bytes)/ (communication bandwidth)
```

Esse cálculo é útil para verificar rapidamente se o desempenho obtido ao executar um trabalho de treinamento distribuído é o esperado ou se você precisa depurar mais o desempenho. É possível obter o número de parâmetros do modelo usando `Model.summary`.

Cada parâmetro do modelo tem tamanho igual a 4 bytes, já que o TensorFlow usa `fp32` (float32) para comunicar os gradientes. Mesmo que você tenha ativado `fp16`, o ALLReduce da NCCL utiliza parâmetros `fp32`.

Para obter os benefícios da escalabilidade, o tempo do passo precisa ser bem maior comparado a essas sobrecargas. Uma forma de conseguir isso é usando um tamanho de lote maior, pois esse tamanho afeta o tempo do passo, mas não impacta a sobrecarga de comunicação.

### 2. Contenção de threads no host com GPUs

Ao executar várias GPUs, o trabalho da CPU é manter todos os dispositivos ocupados iniciando os kernels de GPU neles de forma eficiente.

Entretanto, quando há muitas operações independentes que a CPU pode agendar em uma GPU, a CPU pode decidir usar muitos de seus threads para manter uma GPU ocupada e depois iniciar kernels em outra GPU em uma ordem não determinística. Isso pode causar um desvio ou escalabilidade negativa, o que pode afetar o desempenho negativamente.

O [visualizador de tracing](https://www.tensorflow.org/guide/profiler#trace_viewer) mostra a sobrecarga quando a CPU inicia os kernels em GPUs de forma ineficiente, pois a `GPU1` está ociosa e depois começa a executar operações após a `GPU2` ter iniciado.

![image](images/gpu_perf_analysis/traceview_gpu_idle.png "TensorFlow Profile device trace view demonstrating inefficient kernel launch")

A visualização de tracing do host mostra que ele está iniciando kernels na `GPU2` antes de iniciá-las na `GPU1` (observe que as operações `tf_Compute*` abaixo não são indicativas de threads na CPU).

![image](images/gpu_perf_analysis/traceview_host_contention.png "TensorFlow Profile host trace view demonstrating inefficient kernel launch")

Se você observar esse tipo de distribuição não uniforme de kernels nas GPUs no visualizador de tracing do seu programa, é recomendável:

- Definir a variável de ambiente `TF_GPU_THREAD_MODE` do TensorFlow como `gpu_private`. Essa variável dirá ao host que ele deve manter os threads de uma GPU privados.
- Por padrão, `TF_GPU_THREAD_MODE=gpu_private` define o número de threads como 2, o que é suficiente para a maioria dos casos. Entretanto, esse número pode ser alterado ao definir a variável de ambiente `TF_GPU_THREAD_COUNT` do TensorFlow como o número de threads desejado.

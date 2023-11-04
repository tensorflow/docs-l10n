# Análise do desempenho de `tf.data` com o TF Profiler

## Visão geral

Este guia pressupõe conhecimentos do TensorFlow [Profiler](https://www.tensorflow.org/guide/profiler) e [`tf.data`](https://www.tensorflow.org/guide/data). O objetivo é fornecer instruções passo a passo com exemplos para ajudar a diagnosticar e corrigir problemas no desempenho de pipelines de entrada.

Para começar, colete um perfil do seu trabalho do TensorFlow. Confira as instruções para [CPUs/GPUs](https://www.tensorflow.org/guide/profiler#collect_performance_data) e [TPUs na nuvem](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

![TensorFlow Trace Viewer](images/data_performance_analysis/trace_viewer.png "The trace viewer page of the TensorFlow Profiler")

O foco do workflow de análise abaixo é a ferramenta trace viewer (visualizador de rastreamento) do Profiler. Essa ferramenta exibe uma linha do tempo mostrando a duração das operações executadas por seu programa do TensorFlow e permite identificar quais operações levam mais tempo para executar. Confira mais informações sobre o trace viewer [nesta seção](https://www.tensorflow.org/guide/profiler#trace_viewer) do guia do TF Profiler. De forma geral, os eventos do `tf.data` são exibidos na linha do tempo de CPU do host.

## Workflow da análise

*Pedimos que siga o workflow abaixo. Se você tiver algum comentário sobre como melhorá-lo, [crie um Issue no GitHub](https://github.com/tensorflow/tensorflow/issues/new/choose) com o rótulo “comp:data”.*

### 1. O seu pipeline de `tf.data` está gerando dados rápido o suficiente?

Comece identificando se o pipeline de entrada é o gargalo do seu programa do TensorFlow.

Basta procurar operações `IteratorGetNext::DoCompute` no trace viewer. De forma geral, espera-se que você veja essas operações no começo de um passo. Esses fragmentos representam o tempo que seu pipeline de entrada demora para gerar um lote de elementos, quando solicitado. Se você estiver usando o Keras ou fazendo a iteração do dataset em uma função `tf.function`, essas operações deverão estar nos threads `tf_data_iterator_get_next`.

Se você estiver usando uma [estratégia de distribuição](https://www.tensorflow.org/guide/distributed_training), poderá ver eventos `IteratorGetNextAsOptional::DoCompute` em vez de `IteratorGetNext::DoCompute`(a partir do TF 2.3).

![image](images/data_performance_analysis/get_next_fast.png "If your IteratorGetNext::DoCompute calls return quickly, `tf.data` is not your bottleneck.")

**Se as chamadas retornarem rapidamente (&lt;= 50 us),** os seus dados ficam disponíveis quando solicitados. O pipeline de entrada não é o seu gargalo. Confira dicas de análise de desempenho mais genéricas no [guia do Profiler](https://www.tensorflow.org/guide/profiler).

![image](images/data_performance_analysis/get_next_slow.png "If your IteratorGetNext::DoCompute calls return slowly, `tf.data` is not producing data quickly enough.")

**Se as chamadas demorarem para retornar,** o `tf.data` não consegue acompanhar as solicitações do consumidor. Prossiga para a próxima seção.

### 2. Você está fazendo a pré-busca dos dados?

A prática recomendada para um bom desempenho do pipeline de entrada é inserir uma transformação `tf.data.Dataset.prefetch` no final do pipeline de `tf.data`. Essa transformação faz a sobreposição da computação do pré-processamento do pipeline de entrada com o próximo passo da computação do modelo e é necessária para um desempenho ideal do pipeline de entrada ao treinar o seu modelo. Se você estiver fazendo a pré-busca dos dados, deverá ver um segmento `Iterator::Prefetch` no mesmo thread da operação `IteratorGetNext::DoCompute`.

![image](images/data_performance_analysis/prefetch.png "If you're prefetching data, you should see a `Iterator::Prefetch` slice in the same stack as the `IteratorGetNext::DoCompute` op.")

**Se você não tiver uma `pré-busca` no final do pipeline**, precisa adicionar uma. Confira mais recomendações de desempenho do `tf.data` no [guia de desempenho do tf.data](https://www.tensorflow.org/guide/data_performance#prefetching).

**Se você já estiver fazendo a pré-busca dos dados** e o pipeline de entrada ainda for o garglo, prossiga para a próxima seção para analisar o desempenho mais detalhadamente.

### 3. Está ocorrendo alta utilização da CPU?

`tf.data` consegue uma alta taxa de transferência tentando fazer o melhor uso possível dos recursos disponíveis. De forma geral, até mesmo ao executar o seu modelo em um acelerador, como uma GPU ou TPU, os pipelines do `tf.data` são executados na CPU. Para verificar a utilização, você pode usar ferramentas como [sar](https://linux.die.net/man/1/sar) e [htop](https://en.wikipedia.org/wiki/Htop), ou pode conferir no [console de monitoramento na nuvem](https://cloud.google.com/monitoring/docs/monitoring_in_console) se estiver executando na GCP.

**Se a utilização for baixa,** isso indica que seu pipeline de entrada pode não ser aproveitando ao máximo a CPU do host. Consulte as práticas recomendadas no [guia de desempenho do tf.data](https://www.tensorflow.org/guide/data_performance). Se você tiver seguido as práticas recomendadas e a utilização e taxa de transferência permanecerem baixas, prossiga para a seção [Análise do gargalo](#4_bottleneck_analysis) abaixo.

**Se a utilização estiver se aproximando do limite de recursos**, para melhorar o desempenho, você precisará aumentar a eficiência do pipeline de entrada (por exemplo, evitando computações desnecessárias) ou descarregar as computações.

Para aumentar a eficiência do pipeline de entrada, você pode evitar computações desnecessárias no `tf.data`. Uma forma de fazer isso é inserir uma transformação [`tf.data.Dataset.cache`](https://www.tensorflow.org/guide/data_performance#caching) após trabalhos que fazem muita computação se os dados couberem na memória, pois isso reduz as computações, ao custo do aumento de uso da memória. Além disso, desativar o paralelismo intraoperações no `tf.data` tem o potencial de aumentar a eficiência em mais de 10%. Para desativá-lo, defina a seguinte opção em seu pipeline de entrada:

```python
dataset = ...
options = tf.data.Options()
options.experimental_threading.max_intra_op_parallelism = 1
dataset = dataset.with_options(options)
```

### 4. Análise do gargalo

Esta seção explica como analisar os eventos do `tf.data` no trace viewer para entender onde o gargalo está e as possíveis estratégias de mitigação.

#### Sobre os eventos do `tf.data` no Profiler

Cada evento do `tf.data` no Profiler tem o nome `Iterator::<Dataset>`, em que `<Dataset>` é o nome da fonte ou transformação do dataset. Cada evento também tem o nome longo `Iterator::<Dataset_1>::...::<Dataset_n>`, que pode ser visto clicando no evento do `tf.data`. No nome longo, `<Dataset_n>` corresponde a `<Dataset>` do nome curto, e os outros datasets no nome longo representam transformações dowstream.

![image](images/data_performance_analysis/map_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)")

Por exemplo, a captura de tela acima foi gerada pelo seguinte código:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
```

Aqui, o evento `Iterator::Map` tem o nome longo `Iterator::BatchV2::FiniteRepeat::Map`. Observe que o nome dos datasets pode ser ligeiramente diferente do nome da API do Python (por exemplo, FiniteRepeat em vez de Repeat), mas deve ser intuitivo para sua análise.

##### Transformações síncronas e assíncronas

Para transformações síncronas do `tf.data` (como `Batch` e `Map`), você verá eventos de transformações upstream no mesmo thread. No exemplo acima, como todas as transformações usadas são síncronas, todos os eventos aparecem no mesmo thread.

Para transformações assíncronas (como `Prefetch`, `ParallelMap`, `ParallelInterleave` e `MapAndBatch`), os eventos das transformações upstreams ficarão em um thread diferente. Nesses casos, o nome longo pode ajudar a identificar a qual transformação em um pipeline um determinado evento corresponde.

![image](images/data_performance_analysis/async_long_name.png "tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5).prefetch(1)")

Por exemplo, a captura de tela acima foi gerada pelo seguinte código:

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x: x)
dataset = dataset.repeat(2)
dataset = dataset.batch(5)
dataset = dataset.prefetch(1)
```

Aqui, os eventos `Iterator::Prefetch` estão nos threads `tf_data_iterator_get_next`. Como `Prefetch` é assíncrono, seus eventos de entrada (`BatchV2`) estarão em um thread diferente e podem ser localizados procurando-se o nome longo `Iterator::Prefetch::BatchV2`. Neste caso, eles estão no thread `tf_data_iterator_resource`. Pelo nome longo, você pode deduzir que `BatchV2` é o upstream do evento `Prefetch`. Além disso, o `parent_id` do evento `BatchV2` corresponderá ao ID do evento `Prefetch`.

#### Identificação do gargalo

De forma geral, para identificar o gargalo em seu pipeline de entrada, percorra-o da transformação mais externa até a fonte. Começando pela transformação final em seu pipeline, percorra as transformações upstream até encontrar uma transformação lenta ou se deparar com um dataset fonte, como `TFRecord`. No exemplo acima, você começaria por `Prefetch`, depois seguiria upstream até `BatchV2`, `FiniteRepeat`, `Map` e por fim `Range`.

De forma geral, uma transformação lenta tem eventos longos, mas com eventos de entrada curtos. Confira alguns exemplos abaixo.

Observe que a transformação final (mais externa) na maioria dos pipelines de entrada de host é o evento `Iterator::Model`. A transformação de modelos é introduzida automaticamente pelo runtime do `tf.data` e é usada para instrumentação e tunagem automática do desempenho do pipeline de entrada.

Se o seu trabalho estiver usando uma [estratégia de distribuição](https://www.tensorflow.org/guide/distributed_training), o trace viewer conterá eventos adicionais que correspondem ao pipeline de entrada do dispositivo. A transformação mais externa do pipeline do dispositivo (aninhada em `IteratorGetNextOp::DoCompute` ou `IteratorGetNextAsOptionalOp::DoCompute`) será um evento `Iterator::Prefetch` com um evento upstream `Iterator::Generator`. Para encontrar o pipeline do host correspondente, procure eventos `Iterator::Model`.

##### Exemplo 1

![image](images/data_performance_analysis/example_1_cropped.png "Example 1")

A captura de tela acima foi gerada a partir do seguinte pipeline de entrada:

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

Na captura de tela, observe que (1) os eventos `Iterator::Map` são longos, mas (2) seus eventos de entrada (`Iterator::FlatMap`) retornam rapidamente, indicando que a transformação Map sequencial é o gargalo.

Observe que, na captura de tela, o evento `InstantiatedCapturedFunction::Run` corresponde ao tempo que demora para executar a função Map.

##### Exemplo 2

![image](images/data_performance_analysis/example_2_cropped.png "Example 2")

A captura de tela acima foi gerada a partir do seguinte pipeline de entrada:

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(parse_record, num_parallel_calls=2)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

Esse exemplo é similar ao acima, mas usa um ParallelMap em vez de Map. Note aqui que (1) os eventos `Iterator::ParallelMap` são longos, mas (2) seus eventos de entrada `Iterator::FlatMap` (que estão em um thread diferente, já que ParallelMap é assíncrono) são curtos. Isso indica que a transformação ParallelMap é o gargalo.

#### Como resolver o gargalo

##### Datasets fonte

Se você tiver identificado um dataset fonte como o gargalo, como ao ler arquivos TFRecord, pode melhorar o desempenho fazendo a paralelização da extração de dados. Para fazer isso, garanta que os dados sejam fragmentos em diversos arquivos e use `tf.data.Dataset.interleave` com o parâmetro `num_parallel_calls` definido como `tf.data.AUTOTUNE`. Se determinismo não for importante para o seu programa, é possível melhorar ainda mais o desempenho definindo o sinalizador `deterministic=False` de `tf.data.Dataset.interleave` a partir do TF 2.2. Por exemplo, se você estiver lendo TFRecords, pode fazer o seguinte:

```python
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.interleave(tf.data.TFRecordDataset,
  num_parallel_calls=tf.data.AUTOTUNE,
  deterministic=False)
```

Observe que os arquivos fragmentados devem ser grandes o suficiente para compensar a sobrecarga de abrir um arquivo. Confira mais detalhes sobre extração paralela de dados [nesta seção](https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction) do guia de desempenho do `tf.data`.

##### Datasets de transformação

Se você tiver identificado uma transformação intermediária do `tf.data` como gargalo, pode resolver esse problema paralelizando a transformação ou [fazendo o cache das computações](https://www.tensorflow.org/guide/data_performance#caching) se os dados couberem na memória e se for apropriado. Algumas transformações, como `Map`, têm contrapartidas de paralelização: o <a href="https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation" data-md-type="link">guia de desempenho do `tf.data` demonstra</a> como fazer a paralelização. Outras transformações, como `Filter`, `Unbatch` e `Batch`, são inerentemente sequenciais. Para paralelizá-las, você pode introduzir “paralelismo externo”. Por exemplo, supondo que seu pipeline de entrada pareça inicialmente como o abaixo, com `Batch` sendo o gargalo:

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)
dataset = filenames_to_dataset(filenames)
dataset = dataset.batch(batch_size)
```

Você pode incorporar o “paralelismo externo” executando diversas cópias do pipeline de entrada em entradas fragmentadas e combinando os resultados:

```python
filenames = tf.data.Dataset.list_files(file_path, shuffle=is_training)

def make_dataset(shard_index):
  filenames = filenames.shard(NUM_SHARDS, shard_index)
  dataset = filenames_to_dataset(filenames)
  Return dataset.batch(batch_size)

indices = tf.data.Dataset.range(NUM_SHARDS)
dataset = indices.interleave(make_dataset,
                             num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

## Recursos adicionais

- [Guia de desempenho do tf.data](https://www.tensorflow.org/guide/data_performance) sobre como criar pipelines de entrada do `tf.data` que tenham bom desempenho
- [Vídeo "Por Dentro do TensorFlow": práticas recomendadas para `tf.data`](https://www.youtube.com/watch?v=ZnukSLKEw34)
- [Guia do Profiler](https://www.tensorflow.org/guide/profiler)
- [Tutorial do Profiler com o Colab](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)

# Pré-processamento de dados para ML com o Google Cloud

Este tutorial mostra como usar o [TensorFlow Transform](https://github.com/tensorflow/transform){: .external} (a biblioteca `tf.Transform`) para implementar o pré-processamento de dados para aprendizado de máquina (ML). A biblioteca `tf.Transform` do TensorFlow permite definir transformações de dados em nível de instância e de passagem completa (full pass) por meio de pipelines de pré-processamento de dados. Esses pipelines são executados de forma eficiente com o [Apache Beam](https://beam.apache.org/){: .external} e criam como subprodutos um grafo do TensorFlow para aplicar as mesmas transformações durante a previsão como quando o modelo é servido.

Este tutorial fornece um exemplo completo usando o [Dataflow](https://cloud.google.com/dataflow/docs){: .external } como executor para o Apache Beam. Presume-se que você esteja familiarizado com o [BigQuery](https://cloud.google.com/bigquery/docs) {: .external }, o Dataflow, o [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external } e a API TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview). Também pressupomos que você tenha alguma experiência no uso de Jupyter Notebooks, como o [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction){: .external }.

Este tutorial também pressupõe que você esteja familiarizado com os conceitos de tipos de pré-processamento, desafios e opções no Google Cloud, conforme descrito em [Pré-processamento de dados para ML: opções e recomendações](../../guide/tft_bestpractices).

## Objetivos

- Implementar o pipeline do Apache Beam usando a biblioteca `tf.Transform`.
- Executar o pipeline no Dataflow.
- Implementar o modelo TensorFlow usando a biblioteca `tf.Transform`.
- Treinar e usar o modelo para previsões.

## Custos

Este tutorial usa os seguintes componentes faturáveis ​​do Google Cloud:

- [Vertex AI](https://cloud.google.com/vertex-ai/pricing){: .external}
- [Cloud Storage](https://cloud.google.com/storage/pricing){: .external}
- [BigQuery](https://cloud.google.com/bigquery/pricing){: .external}
- [Dataflow](https://cloud.google.com/dataflow/pricing){: .external}

<!-- This doc uses plain text cost information because the pricing calculator is pre-configured -->

Para estimar o custo de execução deste tutorial, supondo que você use todos os recursos durante um dia inteiro, use a [calculadora de preços](/products/calculator/#id=fad4d8-dd68-45b8-954e-5a56a5d148) pré-configurada {: .external }.

## Antes de começar

1. No console do Google Cloud, na página do seletor de projetos, selecione ou [crie um projeto do Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects).

Observação: se você não planeja manter os recursos criados neste procedimento, crie um projeto em vez de selecionar um projeto existente. Após concluir essas etapas, você poderá excluir o projeto, removendo todos os recursos associados ao projeto.

[Go to project selector](https://console.cloud.google.com/projectselector2/home/dashboard){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

1. Certifique-se de que o faturamento esteja ativado para seu projeto do Cloud. Saiba como [verificar se o faturamento está ativado em um projeto](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled).

2. Ative as APIs Dataflow, Vertex AI e Notebooks. [Enable the APIs](https://console.cloud.google.com/flows/enableapi?apiid=dataflow,aiplatform.googleapis.com,notebooks.googleapis.com) {: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

## Notebooks Jupyter para esta solução

Os seguintes notebooks Jupyter mostram o exemplo de implementação:

- [Notebook 1](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb) {: .external } cobre o pré-processamento de dados. Os detalhes posteriormente são fornecidos na seção [Implementando o pipeline do Apache Beam](#implement-the-apache-beam-pipeline).
- [Notebook 2](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb) {: .external } cobre o treinamento do modelo. Os detalhes são posteriormente fornecidos na seção [Implementando o modelo do TensorFlow](#implement-the-tensorflow-model).

Nas seções a seguir, você clonará esses notebooks e, em seguida, executará os notebooks para aprender como funciona o exemplo de implementação.

## Inicie uma instância de notebooks gerenciados pelo usuário

1. No console do Google Cloud, acesse a página **Vertex AI Workbench**.

    [Go to Workbench](https://console.cloud.google.com/ai-platform/notebooks/list/instances){: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. Na aba **User-managed notebooks**, clique em **+New notebook**.

3. Selecione **TensorFlow Enterprise 2.8 (with LTS) without GPUs** como tipo de instância.

4. Clique em **Create**.

Depois de criar o notebook, aguarde até que o proxy do JupyterLab termine a inicialização. Quando estiver pronto, **Open JupyterLab** será exibido ao lado do nome do notebook.

## Clone o notebook

1. Na aba **User-managed notebooks**, ao lado do nome do notebook, clique em **Open JupyterLab**. A interface do JupyterLab abrirá numa nova aba.

    Se o JupyterLab exibir uma caixa de diálogo **Build Recommended**, clique em **Cancel** para rejeitar o build sugerido.

2. Na aba **Launcher**, clique em **Terminal**.

3. Na janela do terminal, clone o notebook:

    ```sh
    git clone https://github.com/GoogleCloudPlatform/training-data-analyst
    ```

## Implemente o pipeline do Apache Beam

Esta seção e a próxima seção [Execute o pipeline no Dataflow](#run-the-pipeline-in-dataflow) {: track-type="solution" track-name="internalLink" track-metadata-position="body" } fornecem uma visão geral e um contexto para o Notebook 1. O notebook fornece um exemplo prático para descrever como usar a biblioteca `tf.Transform` para pré-processar dados. Este exemplo usa o dataset Natality, que é usado para prever pesos do bebês com base em várias entradas. Os dados são armazenados na tabela pública [natality](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=natality&page=table&_ga=2.267763789.2122871960.1676620306-376763843.1676620306) {: target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" } no BigQuery.

### Execute o Notebook 1

1. Na interface do JupyterLab, clique em **File &gt; Open from path** e insira o seguinte caminho:

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_01.ipynb
    ```

2. Clique em **Edit &gt; Clear all outputs**.

3. Na seção **Install required packages**, execute a primeira célula para executar o comando `pip install apache-beam`.

    A última parte da saída é a seguinte:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    ```

    Você pode ignorar erros de dependência na saída. Você não precisa reiniciar o kernel ainda.

4. Execute a segunda célula para executar o comando `pip install tensorflow-transform` . A última parte da saída é a seguinte:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    Você pode ignorar erros de dependência na saída.

5. Clique em **Kernel &gt; Restart Kernel**.

6. Execute as células nas seções **Confirm the installed packages** e **Create setup.py to install packages to Dataflow containers**.

7. Na seção **Set global flags**, ao lado de `PROJECT` e `BUCKET`, substitua <var><code>your-project</code></var> pelo ID do projeto do Cloud e execute a célula.

8. Execute todas as células restantes até a última célula do notebook. Para obter informações sobre o que fazer em cada célula, consulte as instruções no caderno.

### Visão geral do pipeline

No exemplo do notebook, o Dataflow executa o pipeline `tf.Transform` em escala para preparar os dados e produzir os artefatos de transformação. As seções posteriores deste documento descrevem as funções que executam cada etapa do pipeline. As etapas gerais do pipeline são as seguintes:

1. Ler os dados de treinamento do BigQuery.
2. Analisar e transformar os dados de treinamento usando a biblioteca `tf.Transform`.
3. Gravar os dados de treinamento transformados no Cloud Storage no formato [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord){: target="external" class="external" track-type="solution" track-name="externalLink" track-metadata-position="body" }.
4. Ler os dados de avaliação do BigQuery.
5. Transformar os dados de avaliação usando o grafo `transform_fn` produzido na etapa 2.
6. Gravar os dados de treinamento transformados no Cloud Storage no formato TFRecord.
7. Gravar os artefatos de transformação no Cloud Storage que serão usados ​​posteriormente para criar e exportar o modelo.

O exemplo a seguir mostra o código Python para o pipeline geral. As seções a seguir fornecem explicações e listagens de códigos para cada etapa.

```py{:.devsite-disable-click-to-copy}
def run_transformation_pipeline(args):

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

    runner = args['runner']
    data_size = args['data_size']
    transformed_data_location = args['transformed_data_location']
    transform_artefact_location = args['transform_artefact_location']
    temporary_dir = args['temporary_dir']
    debug = args['debug']

    # Instantiate the pipeline
    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(temporary_dir):

            # Preprocess train data
            step = 'train'
            # Read raw train data from BigQuery
            raw_train_dataset = read_from_bq(pipeline, step, data_size)
            # Analyze and transform raw_train_dataset
            transformed_train_dataset, transform_fn = analyze_and_transform(raw_train_dataset, step)
            # Write transformed train data to sink as tfrecords
            write_tfrecords(transformed_train_dataset, transformed_data_location, step)

            # Preprocess evaluation data
            step = 'eval'
            # Read raw eval data from BigQuery
            raw_eval_dataset = read_from_bq(pipeline, step, data_size)
            # Transform eval data based on produced transform_fn
            transformed_eval_dataset = transform(raw_eval_dataset, transform_fn, step)
            # Write transformed eval data to sink as tfrecords
            write_tfrecords(transformed_eval_dataset, transformed_data_location, step)

            # Write transformation artefacts
            write_transform_artefacts(transform_fn, transform_artefact_location)

            # (Optional) for debugging, write transformed data as text
            step = 'debug'
            # Write transformed train data as text if debug enabled
            if debug == True:
                write_text(transformed_train_dataset, transformed_data_location, step)
```

### Leia dados brutos de treinamento do BigQuery{: id="read_raw_training_data"}

O primeiro passo é ler os dados brutos de treinamento do BigQuery usando a função `read_from_bq`. Esta função retorna um objeto `raw_dataset` extraído do BigQuery. Você passa um valor `data_size` e um valor de `step` `train` ou `eval`. A consulta de fontes do BigQuery é construída usando a função `get_source_query`, conforme mostrado no exemplo a seguir:

```py{:.devsite-disable-click-to-copy}
def read_from_bq(pipeline, step, data_size):

    source_query = get_source_query(step, data_size)
    raw_data = (
        pipeline
        | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
                           beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
        | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
    )

    raw_metadata = create_raw_metadata()
    raw_dataset = (raw_data, raw_metadata)
    return raw_dataset
```

Antes de executar o pré-processamento `tf.Transform`, talvez seja necessário executar o processamento típico baseado no Apache Beam, incluindo processamento de mapa, filtro, grupo e janela. No exemplo, o código limpa os registros lidos do BigQuery usando o método `beam.Map(prep_bq_row)`, onde `prep_bq_row` é uma função personalizada. Esta função personalizada converte o código numérico de uma característica categórica em rótulos legíveis por humanos.

Além disso, para usar a biblioteca `tf.Transform` para analisar e transformar o objeto `raw_data` extraído do BigQuery, você precisa criar um objeto `raw_dataset`, que é uma tupla de objetos `raw_data` e `raw_metadata`. O objeto `raw_metadata` é criado usando a função `create_raw_metadata`, como mostrado a seguir:

```py{:.devsite-disable-click-to-copy}
CATEGORICAL_FEATURE_NAMES = ['is_male', 'mother_race']
NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']
TARGET_FEATURE_NAME = 'weight_pounds'

def create_raw_metadata():

    feature_spec = dict(
        [(name, tf.io.FixedLenFeature([], tf.string)) for name in CATEGORICAL_FEATURE_NAMES] +
        [(name, tf.io.FixedLenFeature([], tf.float32)) for name in NUMERIC_FEATURE_NAMES] +
        [(TARGET_FEATURE_NAME, tf.io.FixedLenFeature([], tf.float32))])

    raw_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))

    return raw_metadata
```

Ao executar a célula no notebook que segue imediatamente a célula que define esse método, o conteúdo do objeto `raw_metadata.schema` é exibido. Ele inclui as seguintes colunas:

- `gestation_weeks` (tipo: `FLOAT`)
- `is_male` (tipo: `BYTES`)
- `mother_age` (tipo: `FLOAT`)
- `mother_race` (tipo: `BYTES`)
- `plurality` (tipo: `FLOAT`)
- `weight_pounds` (tipo: `FLOAT`)

### Transforme dados brutos de treinamento

Imagine que você deseja aplicar transformações típicas de pré-processamento às características brutas de entrada dos dados de treinamento para prepará-las para o ML. Essas transformações incluem operações de full-pass e de nível de instância, conforme mostradas na tabela a seguir:

<table>
<thead>
  <tr>
    <th>Característica de entrada</th>
    <th>Transformação</th>
    <th>Estatísticas necessárias</th>
    <th>Tipo</th>
    <th>Característica de saída</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><code>weight_pound</code></td>
    <td>Nenhuma</td>
    <td>Nenhuma</td>
    <td>N/A</td>
    <td><code>weight_pound</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Normalização</td>
    <td>mean, var</td>
    <td>Full-pass</td>
    <td><code>mother_age_normalized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Bucketização de tamanho equivalente</td>
    <td>quantiles</td>
    <td>Full-pass</td>
    <td><code>mother_age_bucketized</code></td>
  </tr>
  <tr>
    <td><code>mother_age</code></td>
    <td>Computação do log</td>
    <td>Nenhuma</td>
    <td>Nível de instância</td>
    <td>
        <code>mother_age_log</code>
    </td>
  </tr>
  <tr>
    <td><code>plurality</code></td>
    <td>Indicar se são bebês únicos ou múltiplos</td>
    <td>Nenhuma</td>
    <td>Nível de instância</td>
    <td><code>is_multiple</code></td>
  </tr>
  <tr>
    <td><code>is_multiple</code></td>
    <td>Converter valores nominais em índice numérico</td>
    <td>vocab</td>
    <td>Full-pass</td>
    <td><code>is_multiple_index</code></td>
  </tr>
  <tr>
    <td><code>gestation_weeks</code></td>
    <td>Escala entre 0 e 1</td>
    <td>min, max</td>
    <td>Full-pass</td>
    <td><code>gestation_weeks_scaled</code></td>
  </tr>
  <tr>
    <td><code>mother_race</code></td>
    <td>Converter valores nominais em índice numérico</td>
    <td>vocab</td>
    <td>Full-pass</td>
    <td><code>mother_race_index</code></td>
  </tr>
  <tr>
    <td><code>is_male</code></td>
    <td>Converter valores nominais em índice numérico</td>
    <td>vocab</td>
    <td>Full-pass</td>
    <td><code>is_male_index</code></td>
  </tr>
</tbody>
</table>

Essas transformações são implementadas numa função `preprocess_fn`, que espera um dicionário de tensores (`input_features`) e retorna um dicionário de características processadas ​​(`output_features`).

O código a seguir mostra a implementação da função `preprocess_fn`, usando as APIs `tf.Transform` de transformação full-pass  (prefixadas com `tft.`) e operações em nível de instância do TensorFlow (prefixadas com `tf.`):

```py{:.devsite-disable-click-to-copy}
def preprocess_fn(input_features):

    output_features = {}

    # target feature
    output_features['weight_pounds'] = input_features['weight_pounds']

    # normalization
    output_features['mother_age_normalized'] = tft.scale_to_z_score(input_features['mother_age'])

    # scaling
    output_features['gestation_weeks_scaled'] =  tft.scale_to_0_1(input_features['gestation_weeks'])

    # bucketization based on quantiles
    output_features['mother_age_bucketized'] = tft.bucketize(input_features['mother_age'], num_buckets=5)

    # you can compute new features based on custom formulas
    output_features['mother_age_log'] = tf.math.log(input_features['mother_age'])

    # or create flags/indicators
    is_multiple = tf.as_string(input_features['plurality'] > tf.constant(1.0))

    # convert categorical features to indexed vocab
    output_features['mother_race_index'] = tft.compute_and_apply_vocabulary(input_features['mother_race'], vocab_filename='mother_race')
    output_features['is_male_index'] = tft.compute_and_apply_vocabulary(input_features['is_male'], vocab_filename='is_male')
    output_features['is_multiple_index'] = tft.compute_and_apply_vocabulary(is_multiple, vocab_filename='is_multiple')

    return output_features
```

O <a>framework</a> <code>tf.Transform</code>{: .external } possui diversas outras transformações além daquelas do exemplo anterior, incluindo as listadas na tabela a seguir:

<table>
<thead>
  <tr>
  <th>Transformação</th>
  <th>Aplica-se a</th>
  <th>Descrição</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td><code>scale_by_min_max</code></td>
    <td>Características numéricas</td>
    <td>       Redimensiona uma coluna numérica no intervalo [<code>output_min</code>, <code>output_max</code>]</td>
  </tr>
  <tr>
    <td><code>scale_to_0_1</code></td>
    <td>Características numéricas</td>
    <td>       Retorna uma coluna que é a coluna de entrada redimensionada para ter intervalo [<code>0</code>, <code>1</code>]</td>
  </tr>
  <tr>
    <td><code>scale_to_z_score</code></td>
    <td>Características numéricas</td>
    <td>Retorna uma coluna padronizada com média 0 e variância 1</td>
  </tr>
  <tr>
    <td><code>tfidf</code></td>
    <td>Características de texto</td>
    <td>       Mapeia os termos em <i>x</i> para sua frequência de termo * frequência inversa do documento</td>
  </tr>
  <tr>
    <td><code>compute_and_apply_vocabulary</code></td>
    <td>Características categóricas</td>
    <td>       Gera um vocabulário para uma característica categórica e o mapeia para um número inteiro com este vocabulário</td>
  </tr>
  <tr>
    <td><code>ngrams</code></td>
    <td>Características de texto</td>
    <td>Cria um <code>SparseTensor</code> de n-grams</td>
  </tr>
  <tr>
    <td><code>hash_strings</code></td>
    <td>Características categóricas</td>
    <td>Faz hash de strings para buckets</td>
  </tr>
  <tr>
    <td><code>pca</code></td>
    <td>Características numéricas</td>
    <td>Computa o PCA no dataset usando covariância com viés</td>
  </tr>
  <tr>
    <td><code>bucketize</code></td>
    <td>Características numéricas</td>
    <td>       Retorna uma coluna agrupada de tamanho igual (baseada em quantis), com um índice de intervalo atribuído a cada entrada</td>
  </tr>
</tbody>
</table>

Para aplicar as transformações implementadas na função `preprocess_fn` ao objeto `raw_train_dataset` produzido na etapa anterior do pipeline, você usa o método `AnalyzeAndTransformDataset`. Este método espera o objeto `raw_dataset` como entrada, aplica a função `preprocess_fn` e produz o objeto `transformed_dataset` e o grafo `transform_fn`. O código a seguir ilustra esse processamento:

```py{:.devsite-disable-click-to-copy}
def analyze_and_transform(raw_dataset, step):

    transformed_dataset, transform_fn = (
        raw_dataset
        | '{} - Analyze & Transform'.format(step) >> tft_beam.AnalyzeAndTransformDataset(
            preprocess_fn, output_record_batches=True)
    )

    return transformed_dataset, transform_fn
```

As transformações são aplicadas nos dados brutos em duas fases: a fase de análise e a fase de transformação. A Figura 3 mais adiante neste documento mostra como o método `AnalyzeAndTransformDataset` é decomposto no método `AnalyzeDataset` e no método `TransformDataset`.

#### A fase de análise

Na fase de análise, os dados brutos de treinamento são analisados num processo completo para calcular as estatísticas necessárias para as transformações. Isto inclui calcular a média, variância, mínimo, máximo, quantis e vocabulário. O processo de análise espera um dataset bruto (dados brutos mais metadados brutos) e produz duas saídas:

- `transform_fn`: um grafo do TensorFlow que contém as estatísticas computadas da fase de análise e a lógica de transformação (que usa as estatísticas) como operações em nível de instância. Conforme discutido posteriormente em [Salve o grafo](#save_the_graph) {: track-type="solution" track-name="internalLink" track-metadata-position="body" }, o grafo `transform_fn` é salvo para ser anexado à função `serving_fn` do modelo. Isto torna possível aplicar a mesma transformação aos pontos de dados de previsão online.
- `transform_metadata`: um objeto que descreve o esquema esperado dos dados após a transformação.

A fase de análise é ilustrada no diagrama a seguir, figura 1:

<figure id="tf-transform-analyze-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-analyze-phase.svg"
    alt="The tf.Transform analyze phase.">
  <figcaption><b>Figure 1.</b> The <code>tf.Transform</code> analyze phase.</figcaption>
</figure>

Os <a>analisadores</a> <code>tf.Transform</code> {: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" } incluem `min`, `max`, `sum`, `size`, `mean`, `var`, `covariance`, `quantiles`, `vocabulary` e `pca`.

#### A fase de transformação

Na fase de transformação, o grafo `transform_fn` produzido pela fase de análise é usado para transformar os dados brutos de treinamento num processo em nível de instância para produzir os dados de treinamento transformados. Os dados de treinamento transformados são emparelhados com os metadados transformados (produzidos pela fase de análise) para produzir o dataset `transformed_train_dataset`.

A fase de transformação é ilustrada no diagrama a seguir, figura 2:

<figure id="tf-transform-transform-phase">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-transform-phase.svg"
    alt="The tf.Transform transform phase.">
  <figcaption><b>Figure 2.</b> The <code>tf.Transform</code> transform phase.</figcaption>
</figure>

Para pré-processar as características, você chama as transformações `tensorflow_transform` necessárias (importadas como `tft` no código) em sua implementação da função `preprocess_fn`. Por exemplo, quando você chama as operações `tft.scale_to_z_score`, a biblioteca `tf.Transform` traduz essa chamada de função em analisadores de média e variância, calcula as estatísticas na fase de análise e, em seguida, aplica essas estatísticas para normalizar a característica numérica na fase de transformação. Tudo isso é feito automaticamente chamando o método `AnalyzeAndTransformDataset(preprocess_fn)`.

A entidade `transformed_metadata.schema` produzida por esta chamada inclui as seguintes colunas:

- `gestation_weeks_scaled` (tipo: `FLOAT`)
- `is_male_index` (tipo: `INT`, is_categorical: `True`)
- `is_multiple_index` (tipo: `INT`, is_categorical: `True`)
- `mother_age_bucketized` (tipo: `INT`, is_categorical: `True`)
- `mother_age_log` (tipo: `FLOAT`)
- `mother_age_normalized` (tipo: `FLOAT`)
- `mother_race_index` (tipo: `INT`, is_categorical: `True`)
- `weight_pounds` (tipo: `FLOAT`)

Conforme explicado em [Operações de pré-processamento](data-preprocessing-for-ml-with-tf-transform-pt1#preprocessing_operations) na primeira parte desta série, a transformação de características converte características categóricas numa representação numérica. Após a transformação, as características categóricas são representadas por valores inteiros. Na entidade `transformed_metadata.schema`, o sinalizador `is_categorical` para colunas do tipo `INT` indica se a coluna representa uma característica categórica ou uma característica numérica verdadeira.

### Escreva dados de treinamento transformados{: id="step_3_write_transformed_training_data"}

Depois que os dados de treinamento forem pré-processados ​​com a função `preprocess_fn` durante as fases de análise e transformação, você poderá gravar os dados num coletor para ser usado no treinamento do modelo do TensorFlow. Quando você executa o pipeline do Apache Beam usando o Dataflow, o coletor é o Cloud Storage. Caso contrário, o coletor será o disco local. Embora você possa gravar os dados como um arquivo CSV de arquivos formatados com largura fixa, o formato de arquivo recomendado para datasets do TensorFlow é o formato TFRecord. Este é um formato binário simples orientado a registros que consiste em mensagens de buffer de protocolo `tf.train.Example`.

Cada registro `tf.train.Example` contém uma ou mais características. Elas são convertidas em tensores quando são alimentadas no modelo para treinamento. O código a seguir grava o dataset transformado em arquivos TFRecord no local especificado:

```py{:.devsite-disable-click-to-copy}
def write_tfrecords(transformed_dataset, location, step):
    from tfx_bsl.coders import example_coder

    transformed_data, transformed_metadata = transformed_dataset
    (
        transformed_data
        | '{} - Encode Transformed Data'.format(step) >> beam.FlatMapTuple(
                            lambda batch, _: example_coder.RecordBatchToExamples(batch))
        | '{} - Write Transformed Data'.format(step) >> beam.io.WriteToTFRecord(
                            file_path_prefix=os.path.join(location,'{}'.format(step)),
                            file_name_suffix='.tfrecords')
    )
```

### Leia, transforme e grave dados de avaliação

Depois de transformar os dados de treinamento e produzir o grafo `transform_fn`, você poderá usá-lo para transformar os dados de avaliação. Primeiro, você lê e limpa os dados de avaliação do BigQuery usando a função `read_from_bq` descrita anteriormente em [Leia os dados brutos de treinamento do BigQuery](#read-raw-training-data-from-bigquery) {: track-type="solution" track-name="internalLink" track-metadata-position="body" } e passando um valor de `eval` para o parâmetro `step`. Em seguida, use o código a seguir para transformar o dataset de avaliação bruto (`raw_dataset`) no formato transformado esperado (`transformed_dataset`):

```py{:.devsite-disable-click-to-copy}
def transform(raw_dataset, transform_fn, step):

    transformed_dataset = (
        (raw_dataset, transform_fn)
        | '{} - Transform'.format(step) >> tft_beam.TransformDataset(output_record_batches=True)
    )

    return transformed_dataset
```

Quando você transforma os dados de avaliação, somente as operações em nível de instância se aplicam, usando a lógica no grafo `transform_fn` e as estatísticas calculadas na fase de análise nos dados de treinamento. Em outras palavras, você não analisa os dados de avaliação de maneira completa para calcular novas estatísticas, como a média e a variância para normalização de pontuação z de características numéricas em dados de avaliação. Em vez disso, você usa as estatísticas computadas dos dados de treinamento para transformar os dados de avaliação em nível de instância.

Portanto, você usa o método `AnalyzeAndTransform` no contexto de dados de treinamento para calcular as estatísticas e transformar os dados. Ao mesmo tempo, você usa o método `TransformDataset` no contexto de transformação de dados de avaliação para transformar apenas os dados usando as estatísticas calculadas nos dados de treinamento.

Em seguida, você grava os dados num coletor (Cloud Storage ou disco local, dependendo do executor) no formato TFRecord para avaliar o modelo do TensorFlow durante o processo de treinamento. Para fazer isso, você usa a função `write_tfrecords` discutida em [Grave os dados de treinamento transformados](#step_3_write_transformed_training_data) {: track-type="solution" track-name="internalLink" track-metadata-position="body" }. O diagrama a seguir, figura 3, mostra como o grafo `transform_fn` produzido na fase de análise dos dados de treinamento é usado para transformar os dados de avaliação.

<figure id="transform-eval-data-using-transform-fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-transforming-eval-data-using-transform_fn.svg"
    alt="Transforming evaluation data using the transform_fn graph.">
  <figcaption><b>Figure 3.</b> Transforming evaluation data using the <code>transform_fn</code> graph.</figcaption>
</figure>

### Salve o grafo

Uma etapa final no pipeline de pré-processamento `tf.Transform` é armazenar os artefatos, que incluem o grafo `transform_fn` produzido pela fase de análise nos dados de treinamento. O código para armazenar os artefatos é mostrado na seguinte função `write_transform_artefacts`:

```py{:.devsite-disable-click-to-copy}
def write_transform_artefacts(transform_fn, location):

    (
        transform_fn
        | 'Write Transform Artifacts' >> transform_fn_io.WriteTransformFn(location)
    )
```

Esses artefatos serão usados ​​posteriormente para treinamento de modelo e exportação para o serviço. Os seguintes artefatos também são produzidos, conforme mostrado na próxima seção:

- `saved_model.pb`: representa o grafo do TensorFlow que inclui a lógica de transformação (o grafo `transform_fn`), que deve ser anexado à interface de serviço do modelo para transformar os pontos de dados brutos no formato transformado.
- `variables`: inclui as estatísticas calculadas durante a fase de análise dos dados de treinamento e é usada na lógica de transformação no artefato `saved_model.pb`.
- `assets`: inclui arquivos de vocabulário, um para cada característica categórica processada com o método `compute_and_apply_vocabulary`, para serem usados ​​durante o serviço para converter um valor nominal bruto de entrada em um índice numérico.
- `transformed_metadata`: um diretório que contém o arquivo `schema.json` que descreve o esquema dos dados transformados.

## Execute o pipeline no Dataflow{:#run_the_pipeline_in_dataflow}

Depois de definir o pipeline `tf.Transform` , execute-o usando o Dataflow. O diagrama a seguir, figura 4, mostra o grafo de execução do Dataflow do pipeline `tf.Transform` descrito no exemplo.

<figure id="dataflow-execution-graph">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-dataflow-execution-graph.png"
    alt="Dataflow execution graph of the tf.Transform pipeline." class="screenshot">
  <figcaption><b>Figure 4.</b> Dataflow execution graph
     of the <code>tf.Transform</code> pipeline.</figcaption>
</figure>

Depois de executar o pipeline do Dataflow para pré-processar os dados de treinamento e avaliação, você poderá explorar os objetos produzidos no Cloud Storage executando a última célula do notebook. Os snippets de código nesta seção mostram os resultados, onde <var><code>YOUR_BUCKET_NAME</code></var> é o nome do seu bucket no Cloud Storage.

Os dados de treinamento e avaliação transformados no formato TFRecord são armazenados no seguinte local:

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed
```

Os artefatos de transformação são produzidos no seguinte local:

```none{:.devsite-disable-click-to-copy}
gs://YOUR_BUCKET_NAME/babyweight_tft/transform
```

A lista a seguir é a saída do pipeline, mostrando os objetos de dados e artefatos produzidos:

```none{:.devsite-disable-click-to-copy}
transformed data:
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/eval-00000-of-00001.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00000-of-00002.tfrecords
gs://YOUR_BUCKET_NAME/babyweight_tft/transformed/train-00001-of-00002.tfrecords

transformed metadata:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/asset_map
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transformed_metadata/schema.pbtxt

transform artefact:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/saved_model.pb
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/variables/

transform assets:
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_male
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/is_multiple
gs://YOUR_BUCKET_NAME/babyweight_tft/transform/transform_fn/assets/mother_race
```

## Implemente o modelo TensorFlow{: id="implementing_the_tensorflow_model"}

Esta seção e a próxima seção, [Treine e use o modelo para previsões](#train_and_use_the_model_for_predictions){: track-type="solution" track-name="internalLink" track-metadata-position="body" }, fornecem uma visão geral e contexto para o Notebook 2. O notebook fornece um exemplo de modelo de ML para prever o peso de bebês. Neste exemplo, um modelo TensorFlow é implementado usando a API Keras. O modelo usa os dados e artefatos produzidos pelo pipeline de pré-processamento `tf.Transform` explicado anteriormente.

### Execute o Notebook 2

1. Na interface do JupyterLab, clique em **File &gt; Open from path** e insira o seguinte caminho:

    ```sh
    training-data-analyst/blogs/babyweight_tft/babyweight_tft_keras_02.ipynb
    ```

2. Clique em **Edit &gt; Clear all outputs**.

3. Na seção **Install required packages**, execute a primeira célula para executar o comando `pip install tensorflow-transform`.

    A última parte da saída é a seguinte:

    ```none{:.devsite-disable-click-to-copy}
    Successfully installed ...
    Note: you may need to restart the kernel to use updated packages.
    ```

    Você pode ignorar erros de dependência na saída.

4. No menu **Kernel**, selecione **Restart Kernel** .

5. Execute as células nas seções **Confirm the installed packages** e **Create setup.py to install packages to Dataflow containers**.

6. Na seção **Set global flags**, ao lado de `PROJECT` e `BUCKET`, substitua <var><code>your-project</code></var> pelo ID do projeto do Cloud e execute a célula.

7. Execute todas as células restantes até a última célula do notebook. Para obter informações sobre o que fazer em cada célula, consulte as instruções no caderno.

### Visão geral da criação do modelo

As etapas para criar o modelo são as seguintes:

1. Crie colunas de características usando as informações de esquema armazenadas no diretório `transformed_metadata`.
2. Crie o modelo amplo e profundo com a API Keras usando as colunas de características como entrada para o modelo.
3. Crie a função `tfrecords_input_fn` para ler e analisar os dados de treinamento e avaliação usando os artefatos de transformação.
4. Treine e avalie o modelo.
5. Exporte o modelo treinado definindo uma função `serving_fn` que possua o grafo `transform_fn` anexado a ela.
6. Inspecione o modelo exportado usando a ferramenta [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model).
7. Use o modelo exportado para previsão.

Este documento não explica como construir o modelo, portanto não discute detalhadamente como o modelo foi construído ou treinado. No entanto, as seções a seguir mostram como as informações armazenadas no diretório `transform_metadata` — que é produzido pelo processo `tf.Transform` — são usadas para criar as colunas de características do modelo. O documento também mostra como o grafo `transform_fn` – que também é produzido pelo processo `tf.Transform` – é usado na função `serving_fn` quando o modelo é exportado para o serviço.

### Use os artefatos de transformação gerados no treinamento do modelo

Ao treinar o modelo do TensorFlow, você usa os objetos `train` e `eval` transformados produzidos na etapa anterior de processamento de dados. Esses objetos são armazenados como arquivos fragmentados no formato TFRecord. As informações de esquema no diretório `transformed_metadata` gerado na etapa anterior podem ser úteis na análise dos dados (objetos `tf.train.Example`) para alimentar o modelo para treinamento e avaliação.

#### Processe os dados

Como você lê arquivos no formato TFRecord para alimentar o modelo com dados de treinamento e avaliação, é necessário analisar cada objeto `tf.train.Example` nos arquivos para criar um dicionário de características (tensores). Isso garante que as características sejam mapeadas para a camada de entrada do modelo usando as colunas de características, que atuam como interface de treinamento e avaliação do modelo. Para analisar os dados, use o objeto `TFTransformOutput` criado a partir dos artefatos gerados na etapa anterior:

1. Crie um objeto `TFTransformOutput` a partir dos artefatos gerados e salvos na etapa de pré-processamento anterior, conforme descrito na seção [Salve o grafo](#save_the_graph) {: track-type="solution" track-name="internalLink" track-metadata-position="body" }:

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. Extraia um objeto `feature_spec` do objeto `TFTransformOutput`:

    ```py
    tf_transform_output.transformed_feature_spec()
    ```

3. Use o objeto `feature_spec` para especificar as  características contidas no objeto `tf.train.Example` como na função `tfrecords_input_fn`:

    ```py
    def tfrecords_input_fn(files_name_pattern, batch_size=512):

        tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
        TARGET_FEATURE_NAME = 'weight_pounds'

        batched_dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=files_name_pattern,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            label_key=TARGET_FEATURE_NAME,
            shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)

        return batched_dataset
    ```

#### Crie as colunas de características

O pipeline produz as informações de esquema no diretório `transformed_metadata` que descreve o esquema dos dados transformados esperados pelo modelo para treinamento e avaliação. O esquema contém o nome da característica e o tipo de dados, como a seguir:

- `gestation_weeks_scaled` (tipo: `FLOAT`)
- `is_male_index` (tipo: `INT` , is_categorical: `True`)
- `is_multiple_index` (tipo: `INT`, is_categorical: `True`)
- `mother_age_bucketized` (tipo: `INT`, is_categorical: `True`)
- `mother_age_log` (tipo: `FLOAT`)
- `mother_age_normalized` (tipo: `FLOAT`)
- `mother_race_index` (tipo: `INT` , is_categorical: `True`)
- `weight_pounds` (tipo: `FLOAT`)

Para ver essas informações, use os seguintes comandos:

```sh
transformed_metadata = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR).transformed_metadata
transformed_metadata.schema
```

O código a seguir mostra como você usa o nome da característica para criar colunas de características:

```py
def create_wide_and_deep_feature_columns():

    deep_feature_columns = []
    wide_feature_columns = []
    inputs = {}
    categorical_columns = {}

    # Select features you've checked from the metadata
    # Categorical features are associated with the vocabulary size (starting from 0)
    numeric_features = ['mother_age_log', 'mother_age_normalized', 'gestation_weeks_scaled']
    categorical_features = [('is_male_index', 1), ('is_multiple_index', 1),
                            ('mother_age_bucketized', 4), ('mother_race_index', 10)]

    for feature in numeric_features:
        deep_feature_columns.append(tf.feature_column.numeric_column(feature))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='float32')

    for feature, vocab_size in categorical_features:
        categorical_columns[feature] = (
            tf.feature_column.categorical_column_with_identity(feature, num_buckets=vocab_size+1))
        wide_feature_columns.append(tf.feature_column.indicator_column(categorical_columns[feature]))
        inputs[feature] = layers.Input(shape=(), name=feature, dtype='int64')

    mother_race_X_mother_age_bucketized = tf.feature_column.crossed_column(
        [categorical_columns['mother_age_bucketized'],
         categorical_columns['mother_race_index']],  55)
    wide_feature_columns.append(tf.feature_column.indicator_column(mother_race_X_mother_age_bucketized))

    mother_race_X_mother_age_bucketized_embedded = tf.feature_column.embedding_column(
        mother_race_X_mother_age_bucketized, 5)
    deep_feature_columns.append(mother_race_X_mother_age_bucketized_embedded)

    return wide_feature_columns, deep_feature_columns, inputs
```

O código cria uma coluna `tf.feature_column.numeric_column` para característica numéricas e uma coluna `tf.feature_column.categorical_column_with_identity` para características categóricas.

Você também pode criar colunas de características estendidas, conforme descrito na [Opção C: TensorFlow](/architecture/data-preprocessing-for-ml-with-tf-transform-pt1#option_c_tensorflow) {: track-type="solution" track-name="internalLink" track-metadata-position="body" } na primeira parte desta série. No exemplo usado para esta série, uma nova característica é criada, `mother_race_X_mother_age_bucketized`, cruzando as características `mother_race` e `mother_age_bucketized` usando a coluna de características `tf.feature_column.crossed_column`. A representação densa e de baixa dimensão dessa característica cruzada é criada usando a coluna de características `tf.feature_column.embedding_column`.

O diagrama a seguir, figura 5, mostra os dados transformados e como os metadados transformados são usados ​​para definir e treinar o modelo TensorFlow:

<figure id="training-tf-with-transformed-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-training-tf-model-with-transformed-data.svg"
    alt="Training the TensorFlow model with transformed data.">
  <figcaption><b>Figure 5.</b> Training the TensorFlow model with
    the transformed data.</figcaption>
</figure>

### Exporte o modelo para servir previsões

Depois de treinar o modelo do TensorFlow com a API Keras, você exporta o modelo treinado como um objeto SavedModel, para que ele possa servir novos pontos de dados para previsão. Ao exportar o modelo, você deve definir sua interface, ou seja, o esquema de características de entrada esperado durante o serviço. Este esquema de características de entrada é definido na função `serving_fn`, conforme mostrado no código a seguir:

```py{:.devsite-disable-click-to-copy}
def export_serving_model(model, output_dir):

    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    # The layer has to be saved to the model for Keras tracking purposes.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serveing_fn(uid, is_male, mother_race, mother_age, plurality, gestation_weeks):
        features = {
            'is_male': is_male,
            'mother_race': mother_race,
            'mother_age': mother_age,
            'plurality': plurality,
            'gestation_weeks': gestation_weeks
        }
        transformed_features = model.tft_layer(features)
        outputs = model(transformed_features)
        # The prediction results have multiple elements in general.
        # But we need only the first element in our case.
        outputs = tf.map_fn(lambda item: item[0], outputs)

        return {'uid': uid, 'weight': outputs}

    concrete_serving_fn = serveing_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='uid'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='is_male'),
        tf.TensorSpec(shape=[None], dtype=tf.string, name='mother_race'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='mother_age'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='plurality'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='gestation_weeks')
    )
    signatures = {'serving_default': concrete_serving_fn}

    model.save(output_dir, save_format='tf', signatures=signatures)
```

Durante o serviço, o modelo espera os pontos de dados em sua forma bruta (ou seja, características brutas antes de qualquer transformação). Portanto, a função `serving_fn` recebe as características brutas e as armazena num objeto `features` como um dicionário Python. No entanto, conforme discutido anteriormente, o modelo treinado espera os pontos de dados no esquema transformado. Para converter as características brutas em objetos `transformed_features` esperados pela interface do modelo, aplique o grafo `transform_fn` salvo ao objeto `features` com as seguintes etapas:

1. Crie o objeto `TFTransformOutput` a partir dos artefatos gerados e salvos na etapa de pré-processamento anterior:

    ```py
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_ARTEFACTS_DIR)
    ```

2. Crie um objeto `TransformFeaturesLayer` a partir do objeto `TFTransformOutput`:

    ```py
    model.tft_layer = tf_transform_output.transform_features_layer()
    ```

3. Aplique o grafo `transform_fn` usando o objeto `TransformFeaturesLayer`:

    ```py
    transformed_features = model.tft_layer(features)
    ```

O diagrama a seguir, figura 6, ilustra a etapa final da exportação de um modelo para servir:

<figure id="exporting-model-for-serving-with-transform_fn">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-exporting-model-for-serving-with-transform_fn.svg"
    alt="Exporting the model for serving with the transform_fn graph attached.">
  <figcaption><b>Figure 6.</b> Exporting the model for serving with the
    <code>transform_fn</code> graph attached.</figcaption>
</figure>

## Treine e use o modelo para previsões

Você pode treinar o modelo localmente executando as células do notebook. Para obter exemplos de como empacotar o código e treinar seu modelo em escala usando o Vertex AI Training, consulte os exemplos e guias no repositório GitHub do Google Cloud [cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples){: .external }.

Ao inspecionar o objeto SavedModel exportado usando a ferramenta `saved_model_cli`, você verá que os elementos `inputs` da definição de assinatura `signature_def` incluem as características brutas, conforme mostrado no exemplo a seguir:

```py{:.devsite-disable-click-to-copy}
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['gestation_weeks'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_gestation_weeks:0
    inputs['is_male'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_is_male:0
    inputs['mother_age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_mother_age:0
    inputs['mother_race'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_mother_race:0
    inputs['plurality'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: serving_default_plurality:0
    inputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_uid:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['uid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: StatefulPartitionedCall_6:0
    outputs['weight'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_6:1
  Method name is: tensorflow/serving/predict
```

As células restantes do notebook mostram como usar o modelo exportado para uma previsão local e como implantar o modelo como um microsserviço usando o Vertex AI Prediction. É importante destacar que o ponto de dados de entrada (amostra) está no esquema bruto em ambos os casos.

## Limpeza

Para evitar cobranças adicionais na sua conta do Google Cloud pelos recursos usados ​​neste tutorial, exclua o projeto que contém os recursos.

### Exclua o projeto

  <aside class="caution">     <strong>Atenção</strong>: A exclusão de um projeto tem os seguintes efeitos:<ul>
<li> <strong>Tudo no projeto é excluído.</strong> Se você usou um projeto existente para este tutorial, ao excluí-lo, você também excluirá qualquer outro trabalho realizado no projeto.</li>
<li> <strong>Os IDs de projeto personalizados são perdidos.</strong> Ao criar este projeto, você pode ter criado um ID de projeto personalizado que deseja usar no futuro. Para preservar as URLs que usam o ID do projeto, como uma URL <code translate="no" dir="ltr">appspot.com</code>, exclua os recursos selecionados dentro do projeto em vez de excluir o projeto inteiro.</li>
</ul>
<p> Se você planeja explorar vários tutoriais e guias de início rápido, a reutilização de projetos pode ajudá-lo a evitar exceder os limites de cota do projeto.</p></aside>


1. No console do Google Cloud, acesse a página **Manage resources**.

    [Vá para Manage resources](https://console.cloud.google.com/iam-admin/projects) {: class="button button-primary" target="console" track-type="solution" track-name="consoleLink" track-metadata-position="body" }

2. Na lista de projetos, selecione o projeto que deseja excluir e clique em **Delete** .

3. Na caixa de diálogo, digite o ID do projeto e clique em **Shut down** para excluir o projeto.

## Quais são os próximos passos?

- Para saber mais sobre os conceitos, desafios e opções de pré-processamento de dados para aprendizado de máquina no Google Cloud, consulte o primeiro artigo desta série, [Pré-processamento de dados para ML: opções e recomendações](../guide/tft_bestpractices).
- Para mais informações sobre como implementar, empacotar e executar um pipeline tf.Transform no Dataflow, consulte o exemplo [Previção de renda com o dataset do censo](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/tftransformestimator){: .external }.
- Faça a especialização do Coursera em ML com [TensorFlow on Google Cloud](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external }.
- Aprenda sobre as práticas recomendadas para engenharia de ML em [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }.
- Para mais arquiteturas de referência, diagramas e práticas recomendadas, explore o [Cloud Architecture Center](https://cloud.google.com/architecture).

<devsite-mathjax config="TeX-AMS-MML_SVG"></devsite-mathjax>

# Pré-processamento de dados para aprendizado de máquina: opções e recomendações

Este documento é o primeiro de uma série de duas partes que explora o tópico de engenharia de dados e engenharia de características para aprendizado de máquina (ML), com foco em tarefas de aprendizado supervisionado. Esta primeira parte discute as práticas recomendadas para pré-processamento de dados em um pipeline de ML no Google Cloud. O documento foca no uso das bibiotecas de código aberto TensorFlow e [TensorFlow Transform](https://github.com/tensorflow/transform){: target="github" class="external" track-type="solution" track-name="gitHubLink" track-metadata-position="body" } (`tf.Transform`) para preparar dados, treinar o modelo e servir o modelo para fazer previsões. Este documento destaca os desafios do pré-processamento de dados para ML e descreve as opções e cenários para realizar a transformação de dados no Google Cloud de maneira eficaz.

Este documento pressupõe que você esteja familiarizado com o [BigQuery](https://cloud.google.com/bigquery/docs){: .external }, o [Dataflow](https://cloud.google.com/dataflow/docs){: .external }, o [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform){: .external } e a API TensorFlow [Keras](https://www.tensorflow.org/guide/keras/overview).

O segundo documento, [Pré-processamento de dados para ML com Google Cloud](../tutorials/transform/data_preprocessing_with_cloud), fornece um tutorial passo a passo sobre como implementar um pipeline `tf.Transform`.

## Introdução

O aprendizado de máquina (ML) ajuda você a encontrar automaticamente padrões complexos e potencialmente úteis em dados. Esses padrões são condensados ​​num modelo de ML que pode então ser aplicado a novos pontos de dados – um processo chamado de *fazer previsões* ou *realizar inferências*.

A construção de um modelo de ML é um processo de múltiplas etapas. Cada etapa apresenta seus próprios desafios técnicos e conceituais. Esta série de duas partes concentra-se nas tarefas de aprendizagem supervisionada e no processo de seleção, transformação e ampliação dos dados originais para criar sinais preditivos poderosos para a variável-alvo. Essas operações combinam conhecimento do domínio com técnicas de ciência de dados. Essasa operações são a essência da [engenharia de características](https://developers.google.com/machine-learning/glossary/#feature_engineering){: .external }.

O tamanho dos datasets de treinamento para modelos de ML do mundo real pode facilmente ser igual ou superior a um terabyte (TB). Portanto, são necessários frameworks de processamento de dados de larga escala para processar esses datasets de forma eficiente e distribuída. Ao usar um modelo de ML para fazer previsões, você deve aplicar as mesmas transformações usadas para os dados de treinamento nos novos pontos de dados. Ao aplicar as mesmas transformações, você apresenta o dataset ao modelo de ML da maneira que o modelo espera.

Este documento discute esses desafios para diferentes níveis de granularidade das operações de engenharia de características: agregações em janela de tempo, em nível de instância, e full-pass. Este documento também descreve as opções e cenários para realizar a transformação de dados para ML no Google Cloud.

Este documento também fornece uma visão geral do [TensorFlow Transform](https://github.com/tensorflow/transform){: .external } (`tf.Transform`), uma biblioteca para TensorFlow que permite definir a transformação de dados em nível de instância e com full-pass através de pipelines de pré-processamento de dados. Esses pipelines são executados com o [Apache Beam](https://beam.apache.org/){: .external } e criam artefatos que permitem aplicar as mesmas transformações durante a previsão como quando o modelo é disponibilizado como serviço.

## Pré-processamento de dados para aprendizado de máquina

Esta seção introduz operações de pré-processamento de dados e estágios de preparação de dados. Também discute os tipos de operações de pré-processamento e sua granularidade.

### Engenharia de dados comparada à engenharia de características

O pré-processamento dos dados para ML envolve engenharia de dados e engenharia de características. Engenharia de dados é o processo de conversão de *dados brutos* em *dados preparados*. A engenharia de características então ajusta os dados preparados para criar as características esperadas pelo modelo de ML. Esses termos têm os seguintes significados:

**Dados brutos** (ou simplesmente **dados**): Os dados em sua forma original, sem qualquer preparação prévia para ML. Neste contexto, os dados podem estar na sua forma bruta (num data lake) ou numa forma transformada (num data warehouse). Os dados transformados que estão num data warehouse podem ter sido convertidos de sua forma bruta original para serem usados ​​em análises. No entanto, neste contexto, *dados brutos* significam que os dados não foram preparados especificamente para a sua tarefa de ML. Os dados também são considerados dados brutos se forem enviados de sistemas de streaming que eventualmente chamam modelos de ML para previsões.

**Dados preparados**: o dataset no formato pronto para sua tarefa de ML: as fontes de dados foram analisadas, unidas e colocadas num formato tabular. Os dados preparados são agregados e resumidos com a granularidade correta – por exemplo, cada linha no dataset representa um cliente único e cada coluna representa informações resumidas do cliente, como por exemplo, o total gasto nas últimas seis semanas. Numa tabela de dados preparada, colunas irrelevantes foram eliminadas e registros inválidos foram filtrados. Para tarefas de aprendizagem supervisionada, a característica-alvo está presente.

**Características projetadas**: o dataset com características ajustadas esperadas pelo modelo, ou seja, características criadas pela execução de determinadas operações específicas de ML nas colunas do dataset preparado e pela criação de novas características para seu modelo durante o treinamento e a previsão, conforme descrito posteriormente em [Operações de pré-processamento](#preprocessing_operations). Exemplos dessas operações incluem dimensionamento de colunas numéricas para um valor entre 0 e 1, valores de recorte e características categóricas de [one-hot encoding](https://developers.google.com/machine-learning/glossary/#one-hot_encoding){: .external }.

O diagrama a seguir, figura 1, mostra as etapas envolvidas na preparação de dados pré-processados:

<figure id="data-flow-raw-prepared-engineered-features">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-data-preprocessing-flow.svg"
    alt="Diagrama de fluxo mostrando dados brutos se transformando em dados preparados se transformando em características projetadas.">
  <figcaption><b>Figura 1.</b> O fluxo de dados de dados brutos para dados preparados e para dados de características projetadas para aprendizado de máquina.</figcaption>
</figure>

Na prática, os dados da mesma fonte estão frequentemente em diferentes estágios de preparação. Por exemplo, um campo de uma tabela no seu data warehouse pode ser usado diretamente como uma característica projetada. Ao mesmo tempo, outro campo na mesma tabela pode precisar passar por transformações antes de se tornar uma característica projetadaq. Da mesma forma, as operações de engenharia de dados e engenharia de características podem ser combinadas na mesma etapa de pré-processamento de dados.

### Operações de pré-processamento

O pré-processamento de dados inclui diversas operações. Cada operação é projetada para ajudar o ML a construir modelos preditivos melhores. Os detalhes destas operações de pré-processamento estão fora do escopo deste documento, mas algumas operações são brevemente descritas nesta seção.

Para dados estruturados, as operações de pré-processamento de dados incluem o seguinte:

- **Limpeza de dados:** remoção ou correção de registros que possuem valores corrompidos ou inválidos de dados brutos e remoção de registros nos quais estão faltando uma grande quantidade de colunas.
- **Seleção e particionamento de instâncias:** seleção de pontos de dados do dataset de entrada para criar [datasets de treinamento, avaliação (validação) e teste](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets){: .external }. Este processo inclui técnicas para amostragem aleatória repetível, sobreamostragem de classes minoritárias e particionamento estratificado.
- **Ajuste de características:** melhora a qualidade de uma característica para ML, que inclui dimensionamento e normalização de valores numéricos, imputação de valores ausentes, recorte de valores discrepantes e ajuste de valores com distribuições distorcidas.
- **Transformação de características:** conversão de uma característica numérico numa característica categórica (via [bucketização](https://developers.google.com/machine-learning/glossary/#bucketing){: .external }) e conversão de características categóricas numa representação numérica (via one-hot encoding, [aprendizagem com contagens](https://dl.acm.org/doi/10.1145/3326937.3341260){: .external }, incorporação de características esparsas, etc. .). Alguns modelos funcionam apenas com características numéricas ou categóricas, enquanto outros podem lidar com características de tipos mistos. Mesmo quando os modelos lidam com ambos os tipos, eles podem se beneficiar de diferentes representações (numéricas e categóricas) da mesma característica.
- **Extração de características:** redução do número de características criando representações de dados mais poderosas e de menor dimensão usando técnicas como [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis){: .external }, extração de [embedding](https://developers.google.com/machine-learning/glossary/#embeddings){: .external } e [hashing](https://medium.com/value-stream-design/introducing-one-of-the-best-hacks-in-machine-learning-the-hashing-trick-bf6a9c8af18f){: .external }.
- **Seleção de características:** seleção de um subconjunto de características de entrada para treinar o modelo e ignorar as características irrelevantes ou redundantes, usando [métodos de filtro ou wrapper](https://en.wikipedia.org/wiki/Feature_selection){: .external }. A seleção de características também pode envolver a simples eliminação de características se nas características estiver faltando um grande número de valores.
- **Construção de características:** criação de novas características usando técnicas típicas, como [expansão polinomial](https://en.wikipedia.org/wiki/Polynomial_expansion){: .external } (usando funções matemáticas univariadas) ou [cruzamento de características](https://developers.google.com/machine-learning/glossary/#feature_cross){: .external } (para capturar interações de características). As características também podem ser construídas usando lógica de negócios do domínio do caso de uso de ML.

Quando você trabalha com dados não estruturados (por exemplo, imagens, áudio ou documentos de texto), o aprendizado profundo substitui a engenharia de características baseada em conhecimento de domínio, integrando-as à arquitetura do modelo. Uma [camada convolucional](https://developers.google.com/machine-learning/glossary/#convolutional_layer){: .external } é um pré-processador automático de características. Construir a arquitetura do modelo correto requer algum conhecimento empírico dos dados. Além disso, é necessária uma determinada quantidade de pré-processamento, como o seguinte:

- Para documentos de texto: [lematização](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){: .external }, cálculo [TF-IDF](https://en.wikipedia.org/wiki/Tf%e2%80%93idf) {: .external } e extração de [n-grams](https://en.wikipedia.org/wiki/N-gram) {: .external }, pesquisa de embeddings.
- Para imagens: recorte, redimensionamento, corte, desfoque gaussiano e filtros canário.
- Para todo tipo de dados (incluindo texto e imagens): [aprendizado por transferência](https://developers.google.com/machine-learning/glossary/#transfer_learning){: .external }, que trata as últimas camadas do modelo totalmente treinado como uma etapa de engenharia de características.

### Granularidade de pré-processamento

Esta seção discute a granularidade dos tipos de transformações de dados. Ela mostra por que essa perspectiva é crítica ao preparar novos pontos de dados para previsões usando transformações aplicadas em dados de treinamento.

As operações de pré-processamento e transformação podem ser categorizadas da seguinte forma, com base na granularidade da operação:

- **Transformações em nível de instância durante treinamento e previsão**. Estas são transformações simples, onde apenas valores da mesma instância são necessários para a transformação. Por exemplo, as transformações em nível de instância podem incluir o recorte do valor de uma característica para algum limiar, a expansão polinomial de outra característica, a multiplicação de duas características ou a comparação de duas características para criar uma flag booleana.

    Essas transformações precisam ser aplicadas de forma idêntica durante o treinamento e previsão, porque o modelo será treinado nas características transformadas, não nos valores brutos de entrada. Se os dados não forem transformados de forma idêntica, o modelo se comportará mal porque serão apresentados dados que possuem uma distribuição de valores com os quais não foi treinado. Para mais informações, consulte a discussão sobre desvio treinamento-serviço na seção [Desafios do pré-processamento](#preprocessing_challenges).

- **Transformações full-pass durante o treinamento, mas transformações em nível de instância durante a previsão**. Neste cenário, as transformações são stateful, porque utilizam algumas estatísticas pré-computadas para realizar a transformação. Durante o treinamento, você analisa todo o corpo de dados de treinamento para calcular quantidades como mínimo, máximo, média e variância para transformar dados de treinamento, dados de avaliação e novos dados no momento da previsão.

    Por exemplo, para normalizar uma característica numérica para treinamento, você calcula sua média (μ) e seu desvio padrão (σ) em todos os dados de treinamento. Este cálculo é chamado de operação *full-pass* (ou operação de *análise*). Quando você fornece o modelo para previsão, o valor de um novo ponto de dados é normalizado para evitar desvios treinamento-serviço. Portanto, os valores μ e σ calculados durante o treinamento são usados ​​para ajustar o valor da característica, que é a seguinte operação simples *em nível de instância* :

    <div> $$ value_{scaled} = (value_{raw} - \mu) \div \sigma $$</div>

    Transformações full-pass incluem o seguinte:

    - Redimensionamento MinMax de características numéricas usando valores *mínimo* e *máximo* computados a partir do dataset de treinamento.
    - Redimensionamento padrão (normalização z-score) de características numéricas usando μ e σ computados no dataset de treinamento.
    - Bucketização de características numéricas usando quantis.
    - Imputação de valores faltantes usando a mediana (características numéricas) ou moda (características categóricas).
    - Conversão de strings (valores nominais) para inteiros (índices) extraindo todos os valores distintos (vocabulário) de uma característica categórica de entrada.
    - Contagem da ocorrência de um termo (valor da característica) em todos os documentos (instâncias) para cálculo do TF-IDF.
    - Computação do PCA das características de entrada para projetar os dados num espaço dimensional inferior (com características linearmente dependentes).

    Você deve usar apenas os dados de treinamento para calcular estatísticas como μ, σ, *min* e *max*. Se você adicionar os dados de teste e avaliação para essas operações, estará [vazando informações](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742){: .external } dos dados de avaliação e teste para treinar o modelo. Isso afeta a confiabilidade dos resultados do teste e da avaliação. Para garantir que você aplique uma transformação consistente a todos os datasets, use as mesmas estatísticas calculadas a partir dos dados de treinamento para transformar os dados de teste e avaliação.

- **Agregações históricas durante treinamento e previsão**. Isso envolve a criação de flags, derivações e agregações de negócios como sinais de entrada para a tarefa de previsão – por exemplo, a criação de métricas de [atualidade, frequência e monetárias (RFM)](https://en.wikipedia.org/wiki/RFM_(market_research)){: .external } para que os clientes construam modelos de propensão. Esses tipos de características podem ser pré-computados e armazenados em um banco de características para serem usados ​​durante o treinamento do modelo, pontuação em lote e serviço de previsão on-line. Você também pode realizar uma engenharia de características adicional (por exemplo, transformação e ajuste) nessas agregações antes do treinamento e da previsão.

- **Agregações históricas durante o treinamento, mas agregações em tempo real durante a previsão**. Essa abordagem envolve a criação de uma característica ao resumir valores em tempo real ao longo do tempo. Nesta abordagem, as instâncias a serem agregadas são definidas através de cláusulas de janela temporais. Por exemplo, você pode usar essa abordagem se quiser treinar um modelo que estime o tempo de viagem de táxi com base nas métricas de tráfego da rota nos últimos 5 minutos, nos últimos 10 minutos, nos últimos 30 minutos e em outros intervalos. Você também pode usar essa abordagem para prever a falha de uma peça do motor com base na média móvel dos valores de temperatura e vibração computados nos últimos 3 minutos. Embora essas agregações possam ser preparadas off-line para treinamento, elas são computadas em tempo real a partir de um fluxo de dados durante o serviço.

    Mais precisamente, quando você prepara dados de treinamento, se o valor agregado não estiver nos dados brutos, o valor será criado durante a fase de engenharia de dados. Os dados brutos geralmente são armazenados num banco de dados com formato `(entity, timestamp, value)`. Nos exemplos anteriores, `entity` é o identificador do segmento de rota para as rotas de táxi e o identificador da peça do motor para a falha do motor. Você pode usar operações de janelas para computar `(entity, time_index, aggregated_value_over_time_window)` e usar as características de agregação como entrada para o treinamento do seu modelo.

    Quando o modelo para previsões em tempo real (online) é disponibilizado como serviço, o modelo espera características derivadas dos valores agregados como entrada. Portanto, você pode usar uma tecnologia de processamento de streams como o Apache Beam para calcular as agregações dos pontos de dados em tempo real transmitidos para o seu sistema. A tecnologia de processamento de streams agrega dados em tempo real com base em janelas de tempo à medida que novos pontos de dados chegam. Você também pode realizar uma engenharia de características adicional (por exemplo, transformação e ajuste) nessas agregações antes do treinamento e da previsão.

## Pipeline de ML no Google Cloud{: id="machine_learning_pipeline_on_gcp" }

Esta seção discute os principais componentes de um pipeline típico de ponta a ponta para treinar e servir modelos do TensorFlow ML no Google Cloud usando serviços gerenciados. Ele também discute onde você pode implementar diferentes categorias de operações de pré-processamento de dados e desafios comuns que você pode enfrentar ao implementar tais transformações. A seção [Como funciona o tf.Transform](#how_tftransform_works) mostra como a biblioteca TensorFlow Transform ajuda a enfrentar esses desafios.

### Arquitetura de alto nível

O diagrama a seguir, figura 2, mostra uma arquitetura de alto nível de um pipeline de ML típico para treinar e servir modelos do TensorFlow. Os rótulos A, B e C no diagrama referem-se aos diferentes locais do pipeline onde o pré-processamento de dados pode ocorrer. Detalhes sobre essas etapas são fornecidos na seção a seguir.

<figure id="high-level-architecture-for-training-and-serving">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-ml-training-serving-architecture.svg"
    alt="Diagrama de arquitetura mostrando as etapas do processamento de dados.">
  <figcaption><b>Figura 2.</b> Arquitetura de alto nível para treinamento em aprendizado de máquina e disponibilização do serviço no Google Cloud.</figcaption>
</figure>

O pipeline consiste nas seguintes etapas:

1. Depois que os dados brutos são importados, os dados tabulares são armazenados no BigQuery, e outros dados, como imagens, áudio e vídeo, são armazenados no Cloud Storage. A segunda parte desta série usa dados tabulares armazenados no BigQuery como exemplo.
2. A engenharia de dados (preparação) e a engenharia de características são executadas em escala usando o Dataflow. Essa execução produz datasets de treinamento, avaliação e teste prontos para ML que são armazenados no Cloud Storage. Idealmente, esses datasets são armazenados como arquivos [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), que é o formato otimizado para computações do TensorFlow.
3. Um [pacote de treinamento](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container){: .external } de modelos do TensorFlow é enviado ao Vertex AI Training, que usa os dados pré-processados ​​das etapas anteriores para treinar o modelo. A saída desta etapa é um [SavedModel](https://www.tensorflow.org/guide/saved_model) treinado no TensorFlow que é exportado para o Cloud Storage.
4. O modelo TensorFlow treinado é implantado no Vertex AI Prediction como um serviço que tem uma API REST para que possa ser usado para previsões on-line. O mesmo modelo também pode ser usado para trabalhos de previsão em lote.
5. Depois que o modelo for implantado como uma API REST, os aplicativos cliente e os sistemas internos poderão invocar a API enviando solicitações com alguns pontos de dados e recebendo respostas do modelo com previsões.
6. Para orquestrar e automatizar esse pipeline, você pode usar o [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction){: .external } como um agendador para invocar as etapas de preparação de dados, treinamento de modelo e implantação de modelo.

Você também pode usar o [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/){: .external } para armazenar características de entrada para fazer previsões. Por exemplo, você pode criar periodicamente características projetadas a partir dos dados brutos mais recentes e armazená-las no Vertex AI Feature Store. Os aplicativos cliente buscam as características de entrada necessárias no Vertex AI Feature Store e os enviam ao modelo para receber previsões.

### Onde fazer o pré-processamento

Na figura 2, os rótulos A, B e C mostram que as operações de pré-processamento de dados podem ocorrer no BigQuery, Dataflow ou TensorFlow. As seções a seguir descrevem como cada uma dessas opções funciona.

#### Opção A: BigQuery{: id="option_a_bigquery"}

Normalmente, a lógica é implementada no BigQuery para as seguintes operações:

- Amostragem: seleção aleatória de um subconjunto dos dados.
- Filtragem: remoção de instâncias irrelevantes ou inválidas.
- Particionamento: divisão dos dados para produzir datasets de treinamento, avaliação e teste.

Os scripts SQL do BigQuery podem ser usados ​​como uma consulta de fonte para o pipeline de pré-processamento do Dataflow, que é a etapa de processamento de dados na figura 2. Por exemplo, se um sistema for usado no Canadá e o data warehouse tiver transações de todo o mundo, filtrando para obter dados de treinamento somente no Canadá é melhor feito no BigQuery. A engenharia de características no BigQuery é simples e escalonável e oferece suporte à implementação de transformações de características de agregações históricas e em nível de instância.

No entanto, recomendamos que você use o BigQuery para engenharia de características somente se usar seu modelo para previsão em lote (pontuação) ou se as características forem pré-computadas no BigQuery, mas armazenadas no Vertex AI Feature Store para serem usadas ​​durante a previsão on-line. Se você planeja implantar o modelo para previsões on-line e não tiver a característica projetada num repositório de características on-line, será necessário replicar as operações de pré-processamento SQL para transformar os pontos de dados brutos gerados por outros sistemas. Em outras palavras, você precisa implementar a lógica duas vezes: uma vez no SQL para pré-processar os dados de treinamento no BigQuery e uma segunda vez na lógica do aplicativo que consome o modelo para pré-processar pontos de dados on-line para previsão.

Por exemplo, se seu aplicativo cliente for escrito em Java, você precisará reimplementar a lógica em Java. Isto pode introduzir erros devido a discrepâncias de implementação, conforme descrito na seção de desvio de treinamento/serviço em [Desafios do pré-processamento](#preprocessing_challenges), mais adiante neste documento. Também é uma sobrecarga extra manter duas implementações diferentes. Sempre que você alterar a lógica no SQL para pré-processar os dados de treinamento, será necessário alterar a implementação Java de acordo para pré-processar os dados no momento de disponibilizar o serviço.

Se você estiver usando seu modelo apenas para previsão em lote (por exemplo, usando a [previsão em lote](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions) da Vertex AI {: .external }) e se os dados para pontuação forem provenientes do BigQuery, você poderá implementar essas operações de pré-processamento como parte do script SQL do BigQuery. Nesse caso, você pode usar o mesmo script SQL de pré-processamento para preparar dados de treinamento e de pontuação.

As transformações full-pass stateful não são adequadas para implementação no BigQuery. Se você usar o BigQuery para transformações full-pass, precisará de tabelas auxiliares para armazenar as quantidades necessárias para transformações stateful, como médias e variâncias para dimensionar recursos numéricos. Além disso, a implementação de transformações full-pass usando SQL no BigQuery introduz maior complexidade nos scripts SQL e cria uma dependência complexa entre o treinamento e os scripts SQL de pontuação.

#### Opção B: Dataflow{: id="option_b_cloud_dataflow"}

Conforme mostrado na Figura 2, você pode implementar operações de pré-processamento computacionalmente caras no Apache Beam e executá-las em escala usando o Dataflow. O Dataflow é um serviço de escalonamento automático totalmente gerenciado para processamento de dados em lote e stream. Ao usar o Dataflow, você também pode usar bibliotecas externas especializadas para processamento de dados, diferentemente do BigQuery.

O Dataflow pode realizar transformações em nível de instância e transformações de recursos de agregação histórica e em tempo real. Em particular, se seus modelos de ML esperam um recurso de entrada como `total_number_of_clicks_last_90sec`, [as funções de janelamento](https://beam.apache.org/documentation/programming-guide/#windowing) do Apache Beam {: .external } podem computar essas características com base na agregação dos valores de janelas de tempo de dados de eventos em tempo real (streaming) (por exemplo, eventos de clique ). Na discussão anterior sobre [granularidade de transformações](#preprocessing_granularity), isso foi mencionado como "agregações históricas durante o treinamento, mas agregações em tempo real durante a previsão".

O diagrama a seguir, figura 3, mostra a função do Dataflow no processamento de dados de stream para previsões quase em tempo real.

<figure id="high-level-architecture-for-stream-data">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-streaming-data-with-dataflow-architecture.svg"
    alt="Arquitetura para uso de dados de stream para previsões.">
  <figcaption><b>Figura 3.</b> Architetura de alto nível usando dados de stream
    para previsões no Dataflow.</figcaption>
</figure>

Conforme mostrado na figura 3, durante o processamento, eventos chamados *pontos de dados* são ingeridos no [Pub/Sub](https://cloud.google.com/pubsub/docs){: .external }. O Dataflow consome esses pontos de dados, computa características com base em agregações ao longo do tempo e, em seguida, chama a API do modelo de ML implantado para fazer previsões. As previsões são então enviadas para uma fila de saída do tipo Pub/Sub. No Pub/Sub, as previsões podem ser consumidas por sistemas downstream, como monitoramento ou controle, ou podem ser enviadas de volta (por exemplo, como notificações) ao cliente solicitante original. As previsões também podem ser armazenadas em um armazenamento de dados de baixa latência, como [Cloud Bigtable](https://cloud.google.com/bigtable/docs){: .external }, para busca em tempo real. O Cloud Bigtable também pode ser usado para acumular e armazenar essas agregações em tempo real para que possam ser consultadas quando necessário para previsões.

A mesma implementação do Apache Beam pode ser usada para processar em lote dados de treinamento provenientes de um armazenamento de dados off-line, como o BigQuery, e processar dados em tempo real para fornecer previsões on-line.

Em outras arquiteturas típicas, como a mostrada na Figura 2, o aplicativo cliente chama diretamente a API do modelo implantado para previsões on-line. Nesse caso, se as operações de pré-processamento forem implementadas no Dataflow para preparar os dados de treinamento, as operações não serão aplicadas aos dados de previsão que vão diretamente para o modelo. Portanto, transformações como essas devem ser integradas ao modelo durante a disponibilização do serviço de previsões online.

O Dataflow pode ser usado para realizar a transformação full-pass, computando as estatísticas necessárias em escala. No entanto, essas estatísticas precisam ser armazenadas em algum lugar para serem usadas durante a previsão para transformar os pontos de dados de previsão. Usando a biblioteca TensorFlow Transform (`tf.Transform`), você pode incorporar diretamente essas estatísticas no modelo em vez de armazená-las em outro lugar. Essa abordagem é explicada posteriormente em [Como funciona o tf.Transform](#how_tftransform_works).

#### Opção C: TensorFlow{: id="option_c_tensorflow"}

Conforme mostrado na Figura 2, você pode implementar operações de pré-processamento e transformação de dados no próprio modelo do TensorFlow. Conforme mostrado na figura, o pré-processamento implementado para treinar o modelo do TensorFlow torna-se parte integrante do modelo quando o modelo é exportado e implantado para previsões. As transformações no modelo do TensorFlow podem ser realizadas de uma das seguintes maneiras:

- Implementar toda a lógica de transformação em nível de instância na função `input_fn` e na função `serving_fn`. A função `input_fn` prepara um dataset usando a [API `tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) para treinar um modelo. A função `serving_fn` recebe e prepara os dados para previsões.
- Colocar o código de transformação diretamente em seu modelo do TensorFlow usando [camadas de pré-processamento Keras](https://keras.io/guides/preprocessing_layers/){: .external } ou[criando camadas personalizadas](https://keras.io/guides/making_new_layers_and_models_via_subclassing/){: .external }.

O código lógico de transformação na função [`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators) define a interface de serviço do seu SavedModel para previsão online. Se você implementar as mesmas transformações que foram usadas para preparar dados de treinamento no código lógico de transformação da função `serving_fn`, isso garantirá que as mesmas transformações sejam aplicadas a novos pontos de dados de previsão quando eles forem disponibilizados via serviço.

No entanto, como o modelo do TensorFlow processa cada ponto de dados de forma independente ou num lote pequeno, não é possível calcular agregações de todos os pontos de dados. Como resultado, as transformações full-pass não podem ser implementadas no seu modelo TensorFlow.

### Desafios do pré-processamento

A seguir estão os principais desafios da implementação do pré-processamento de dados:

- **Desvio de treinamento/serviço**. [O desvio de treinamento/serviço](https://developers.google.com/machine-learning/guides/rules-of-ml/#training-serving_skew){: .external } refere-se a uma diferença entre a eficácia (desempenho preditivo) durante o treinamento e durante o serviço. Esse desvio pode ser causado por uma discrepância entre como você lida com os dados no treinamento e nos pipelines de serviço. Por exemplo, se o seu modelo for treinado numa característica transformada logaritmicamente, mas for apresentado com a característica bruta durante o serviço, a saída da previsão poderá não ser precisa.

    Se as transformações se tornarem parte do próprio modelo, poderá ser simples lidar com as transformações no nível da instância, conforme descrito anteriormente na [Opção C: TensorFlow](#option_c_tensorflow). Nesse caso, a interface de serviço do modelo (a função [`serving_fn`](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators)) espera dados brutos, enquanto o modelo transforma internamente esses dados antes de calcular a saída. As transformações são as mesmas que foram aplicadas nos pontos de dados brutos de treinamento e previsão.

- **Transformações full-pass**. Não é possível implementar transformações full-pass, como transformações de dimensionamento e normalização, no seu modelo TensorFlow. Nas transformações full-pass, algumas estatísticas (por exemplo, valores `max` e `min` para dimensionar características numéricas) precisam ser computadas previamente nos dados de treinamento, conforme descrito na [Opção B: Dataflow](#option_b_dataflow). Os valores então devem ser armazenados em algum lugar para serem usados ​​durante a disponibilização do serviço do modelo para previsão para transformar os novos pontos de dados brutos como transformações em nível de instância, o que evita desvios treinamento-serviço. Você pode usar a biblioteca TensorFlow Transform (`tf.Transform`) para incorporar diretamente as estatísticas em seu modelo do TensorFlow. Essa abordagem é explicada posteriormente em [Como funciona o tf.Transform](#how_tftransform_works).

- **Preparação antecipada dos dados para melhorar a eficiência do treinamento**. A implementação de transformações no nível da instância como parte do modelo pode degradar a eficiência do processo de treinamento. Essa degradação ocorre porque as mesmas transformações são aplicadas repetidamente aos mesmos dados de treinamento em cada época. Imagine que você tem dados de treinamento brutos com 1.000 características e aplica uma combinação de transformações em nível de instância para gerar 10.000 características. Se você implementar essas transformações como parte do seu modelo e, em seguida, alimentar o modelo com os dados brutos de treinamento, essas 10.000 operações serão aplicadas *N* vezes em cada instância, onde *N* é o número de épocas. Além disso, se você estiver usando aceleradores (GPUs ou TPUs), eles ficarão ociosos enquanto a CPU executa essas transformações, o que não é um uso eficiente de seus aceleradores caros.

    Idealmente, os dados de treinamento são transformados antes do treinamento, usando a técnica descrita na [Opção B: Dataflow](#option_b_dataflow), onde as 10.000 operações de transformação são aplicadas apenas uma vez em cada instância de treinamento. Os dados de treinamento transformados são então apresentados ao modelo. Nenhuma outra transformação é aplicada e os aceleradores ficam ocupados o tempo todo. Além disso, usar o Dataflow ajuda a pré-processar grandes quantidades de dados em escala, usando um serviço totalmente gerenciado.

    A preparação antecipada dos dados de treinamento pode melhorar a eficiência do treinamento. No entanto, implementar a lógica de transformação fora do modelo (as abordagens descritas na [Opção A: BigQuery](#option_a_bigquery) ou [Opção B: Dataflow](#option_b_dataflow)) não resolve o problema do desvio treinamento-serviço. A menos que você armazene a característica projetada no armazenamento de características para ser usada tanto para treinamento quanto para previsão, a lógica de transformação deverá ser implementada em algum lugar para ser aplicada em novos pontos de dados que chegam para previsão, porque a interface do modelo espera dados transformados. A biblioteca TensorFlow Transform (`tf.Transform`) pode ajudar você a resolver esse problema, conforme descrito na seção a seguir.

## Como funciona o tf.Transform{:#how_tftransform_works}

A biblioteca `tf.Transform` é útil para transformações que exigem um passo completo (full pass). A saída da biblioteca `tf.Transform` é exportada como um grafo TensorFlow que representa a lógica de transformação no nível da instância e as estatísticas são computadas a partir de transformações full-pass, para serem usadas para treinamento e serviço. Usar o mesmo grafo para treinamento e serviço pode evitar desvios, porque as mesmas transformações são aplicadas em ambos os estágios. Além disso, a biblioteca `tf.Transform` pode ser executada em escala num pipeline de processamento em lote no Dataflow para preparar os dados de treinamento antecipadamente e melhorar a eficiência do treinamento.

O diagrama a seguir, figura 4, mostra como a biblioteca `tf.Transform` pré-processa e transforma dados para treinamento e previsão. O processo é descrito nas seções a seguir.

<figure id="tf-Transform-preprocessing--transforming-data-for-training-and-prediction">
  <img src="images/data-preprocessing-for-ml-with-tf-transform-tf-transform-behavior-flow.svg"
    alt="Diagrama mostrando o fluxo de dados brutos passando por tf.Transform para previsões.">
  <figcaption><b>Figura 4.</b> Comportamento de <code>tf.Transform</code> para
    pré-processamento e transformação de dados.</figcaption>
</figure>

### Transformação de dados de treinamento e avaliação

Você pré-processa os dados brutos de treinamento usando a transformação implementada nas APIs `tf.Transform` do Apache Beam e executa-os em escala no Dataflow. O pré-processamento ocorre nas seguintes fases:

- **Fase de análise:** durante a fase de análise, as estatísticas necessárias (como médias, variâncias e quantis) para transformações stateful são computadas nos dados de treinamento com operações full-pass. Esta fase produz um conjunto de artefatos de transformação, incluindo o grafo `transform_fn`. O grafo `transform_fn` é um grafo TensorFlow que tem sua a lógica de transformação na forma de operações em nível de instância. Ele inclui as estatísticas calculadas na fase de análise como constantes.
- **Fase de transformação:** Durante a fase de transformação, o grafo `transform_fn` é aplicado aos dados brutos de treinamento, onde as estatísticas computadas são usadas para processar os registros de dados (por exemplo, para dimensionar colunas numéricas) no estilo "nível de instância".

Uma abordagem de duas fases como esta abrange o [desafio de pré-processamento](#preprocessing_challenges) de realizar transformações full-pass.

Quando os dados de avaliação são pré-processados, apenas as operações em nível de instância são aplicadas, usando a lógica no grafo `transform_fn` e as estatísticas computadas a partir da fase de análise, nos dados de treinamento. Em outras palavras, você não analisa os dados de avaliação de maneira full-pass para computar novas estatísticas, como μ e σ, para normalizar características numéricas nos dados de avaliação. Em vez disso, você usa as estatísticas computadas dos dados de treinamento para transformar os dados de avaliação de uma maneira "nível de instância".

Os dados de treinamento e avaliação transformados são preparados em escala usando o Dataflow, antes de serem usados ​​para treinar o modelo. Este processo de preparação de dados em lote aborda o [desafio de pré-processamento](#preprocessing_challenges) de preparar os dados antecipadamente para melhorar a eficiência do treinamento. Conforme mostrado na figura 4, a interface interna do modelo espera características transformadas.

### Anexe transformações ao modelo exportado

Conforme observado, o grafo `transform_fn` produzido pelo pipeline `tf.Transform` é armazenado como um grafo TensorFlow exportado. O gráfico exportado consiste na lógica de transformação como operações em nível de instância e em todas as estatísticas computadas nas transformações full-pass como constantes do grafo. Quando o modelo treinado é exportado para disponibilização do serviço, o grafo `transform_fn` é anexado ao SavedModel como parte de sua função `serving_fn`.

Enquanto serve o modelo para previsão, a interface de atendimento do modelo espera pontos de dados no formato bruto (ou seja, antes de qualquer transformação). A interface interna do modelo, no entanto, espera os dados no formato transformado.

O grafo `transform_fn`, que agora faz parte do modelo, aplica toda a lógica de pré-processamento no ponto de dados de entrada. Ele usa as constantes armazenadas (como μ e σ para normalizar as características numéricas) na operação em nível de instância durante a previsão. Portanto, o grafo `transform_fn` converte o ponto de dados brutos no formato transformado. O formato transformado é o esperado pela interface interna do modelo para produzir a previsão, conforme mostra a figura 4.

Esse mecanismo resolve o [desafio de pré-processamento](#preprocessing_challenges) do desvio de treinamento/serviço, já que a mesma lógica (implementação) usada para transformar os dados de treinamento e avaliação é aplicada para transformar os novos pontos de dados durante o serviço de previsão.

## Resumo das opções de pré-processamento

A tabela a seguir resume as opções de pré-processamento de dados discutidas neste documento. Na tabela, “N/A” significa “não aplicável”.

<table class="alternating-odd-rows">
<tbody>
<tr>
<th>Opção de pré-processamento de dados</th>
<th>Nível de instância<br> (transformações stateless)</th>
<th>
  <p>Full-pass durante o treinamento e nível de instância durante o serviço (transformações stateful)</p>
</th>
<th>
  <p>Agregações em tempo real (window) durante o treinamento e serviço (transformações de streaming)</p>
</th>
</tr>
<tr>
  <td>
    <p>       <b>BigQuery</b>          (SQL)</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: OK</b> — a mesma implementação de transformação é aplicada aos dados durante o treinamento e a pontuação em lote.</p>
    <p>       <b>Previsão on-line: não recomendada</b> — você pode processar dados de treinamento, mas isso resulta em desvio treinamento/serviço porque você processa dados de serviço usando ferramentas diferentes.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: não recomendada</b>.</p>
    <p>       <b>Previsão on-line: não recomendada</b>.</p>
    <p>       Embora você possa usar estatísticas computadas com o BigQuery para transformações on-line/em lote no nível da instância, isso não é fácil porque você precisa manter um armazenamento de estatísticas para ser preenchido durante o treinamento e usado durante a previsão.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: N/A</b> – agregados como esses são computados com base em eventos em tempo real.</p>
    <p>       <b>Previsão on-line: não recomendada</b> — você pode processar dados de treinamento, mas isso resulta em desvio treinamento/serviço porque você processa dados de serviço usando ferramentas diferentes.</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam)</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: OK</b> — a mesma implementação de transformação é aplicada aos dados durante o treinamento e a pontuação em lote.</p>
    <p>       <b>Previsão on-line: OK</b> : se os dados no momento do serviço vierem do Pub/Sub para serem consumidos pelo Dataflow. Caso contrário, resultará em desvio de treinamento/serviço.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: não recomendada</b>.</p>
    <p>       <b>Previsões on-line: não recomendadas</b>.</p>
    <p>       Embora seja possível usar estatísticas calculadas com o Dataflow para transformações on-line/em lote no nível da instância, isso não é fácil porque é necessário manter um armazenamento de estatísticas para ser preenchido durante o treinamento e usado durante a previsão.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: N/A</b> – agregados como esses são computados com base em eventos em tempo real.</p>
    <p>       <b>Previsão on-line: OK</b> — a mesma transformação do Apache Beam é aplicada aos dados durante o treinamento (lote) e serviço (stream).</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>Dataflow</b> (Apache Beam + TFT)</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: OK</b> — a mesma implementação de transformação é aplicada aos dados durante o treinamento e a pontuação em lote.</p>
    <p>       <b>Previsão on-line: recomendada</b> — evita desvios treinamento-serviço e prepara os dados de treinamento antecipadamente.</p>
  </td>
  <td>
    <p>       <b>Pontuação de lote: Recomendada</b>.</p>
    <p>       <b>Previsão on-line: recomendada</b>.</p>
    <p>       Ambos os usos são recomendados porque a lógica de transformação e as estatísticas computadas durante o treinamento são armazenadas como um grafo TensorFlow anexado ao modelo exportado para o serviço.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: N/A</b> – agregados como esses são computados com base em eventos em tempo real.</p>
    <p>       <b>Previsão on-line: OK</b> — a mesma transformação do Apache Beam é aplicada aos dados durante o treinamento (lote) e serviço (stream).</p>
  </td>
</tr>
<tr>
  <td>
    <p>       <b>TensorFlow</b> <sup>*</sup>       <br>       (<code>input_fn</code> &amp; <code>serving_fn</code>)</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: não recomendada</b>.</p>
    <p>       <b>Previsão on-line: não recomendada</b>.</p>
    <p>       Para eficiência do treinamento em ambos os casos, é melhor preparar os dados de treinamento antecipadamente.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: não é possível</b>.</p>
    <p>       <b>Previsão on-line: não é possível</b>.</p>
  </td>
  <td>
    <p>       <b>Pontuação em lote: N/A</b> – agregados como esses são computados com base em eventos em tempo real.</p>
<p>       <b>Previsão on-line: não é possível</b>.</p>
  </td>
</tr>
</tbody>
</table>

<sup>*</sup> Com o TensorFlow, transformações como cruzamento, embedding e one-hot encoding devem ser executadas declarativamente como colunas `feature_columns`.

## Quais são os próximos passos?

- Para implementar um pipeline `tf.Transform` e executá-lo usando o Dataflow, leia a segunda parte desta série, [Pré-processamento de dados para ML usando TensorFlow Transform](https://www.tensorflow.org/tfx/tutorials/transform/data_preprocessing_with_cloud).
- Faça a especialização do Coursera em ML com [TensorFlow on Google Cloud](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp){: .external }.
- Aprenda sobre as práticas recomendadas para engenharia de ML em [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml/){: .external }.

- Para mais arquiteturas de referência, diagramas e práticas recomendadas, explore as <a href="https://www.tensorflow.org/tfx/guide/solutions" track-type="tutorial" track-name="textLink" track-metadata-position="nextSteps">Soluções TFX Cloud</a>.

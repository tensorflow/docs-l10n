# TensorFlow Data Validation: verificando e analisando seus dados

Depois que seus dados estiverem num pipeline do TFX, você poderá usar componentes do TFX para analisá-los e transformá-los. Você pode usar essas ferramentas antes mesmo de treinar um modelo.

Existem muitos motivos para analisar e transformar seus dados:

- Para encontrar problemas em seus dados. Problemas comuns incluem:
    - Dados ausentes, como características com valores vazios.
    - Rótulos tratados como características, para que seu modelo possa dar uma olhada na resposta certa durante o treinamento.
    - Características com valores fora do intervalo esperado.
    - Anomalias de dados.
    - O modelo de transferência aprendida com pré-processamento que não corresponde aos dados de treinamento.
- Para projetar conjuntos de características mais eficazes. Por exemplo, você poderá identificar:
    - Características especialmente informativas.
    - Características redundantes.
    - Características que variam tanto em escala que podem retardar o aprendizado.
    - Características com pouca ou nenhuma informação preditiva exclusiva.

As ferramentas do TFX podem ajudar a encontrar bugs em seus dados e ajudar na engenharia de características.

## TensorFlow Data Validation

- [Visão geral](#overview)
- [Validação de exemplo baseada em esquema](#schema_based_example_validation)
- [Detecção de desvios (skew) entre treinamento e serviço](#skewdetect)
- [Detecção de deriva (drift) de dados](#drift_detection)

### Visão geral

O TensorFlow Data Validation identifica anomalias no treinamento e no fornecimento de dados e poderá criar automaticamente um esquema ao examinar os dados. O componente pode ser configurado para detectar diferentes classes de anomalias nos dados. Pode, por exemplo:

1. Realizar verificações de validade comparando estatísticas de dados com um esquema que codifica as expectativas do usuário.
2. Detectar desvios entre treinamento/serviço comparando dados do treinamento com dados do fornecimento do serviço.
3. Detectar deriva de dados ao observar uma série de dados.

Documentamos cada uma dessas funcionalidades de forma independente:

- [Validação de exemplo baseada em esquema](#schema_based_example_validation)
- [Detecção de desvios (skew) entre treinamento e serviço](#skewdetect)
- [Detecção de deriva (drift) de dados](#drift_detection)

### Validação de exemplo baseada em esquema

O TensorFlow Data Validation identifica quaisquer anomalias nos dados de entrada comparando estatísticas de dados com um esquema. O esquema codifica propriedades que se espera que os dados de entrada satisfaçam, como tipos de dados ou valores categóricos, e podem ser modificados ou substituídos pelo usuário.

O Tensorflow Data Validation normalmente é chamado várias vezes no contexto do pipeline TFX: (i) para cada divisão obtida do ExampleGen, (ii) para todos os dados pré-transformação usados ​​pelo Transform e (iii) para todos os dados pós-transformação gerados pelo Transform. Quando chamado no contexto de Transform (ii-iii), as opções de estatísticas e restrições baseadas em esquema podem ser definidas definindo [`stats_options_updater_fn`](tft.md). Isto é útil ao validar dados não estruturados (por exemplo, características em formato texto). Veja o [user code](https://github.com/tensorflow/tfx/blob/master/tfx/examples/bert/mrpc/bert_mrpc_utils.py), como exemplo.

#### Características em esquemas avançados

Esta seção aborda configurações mais avançadas para esquemas que podem ser úteis em determinadas situações.

##### Características esparsas

A inclusão de características esparsas em Exemplos geralmente introduz várias características que precisam ter a mesma valência para todos os Exemplos. Por exemplo, a característica esparsa:

<pre><code>
WeightedCategories = [('CategoryA', 0.3), ('CategoryX', 0.7)]
</code></pre>

seria codificada usando características separadas para índice e valor:

<pre><code>
WeightedCategoriesIndex = ['CategoryA', 'CategoryX']
WeightedCategoriesValue = [0.3, 0.7]
</code></pre>

com a restrição de que a valência da característica de índice e valor deve corresponder para todos os exemplos. Esta restrição pode ser explicitada no esquema definindo um sparse_feature:

<pre><code class="lang-proto">
sparse_feature {
  name: 'WeightedCategories'
  index_feature { name: 'WeightedCategoriesIndex' }
  value_feature { name: 'WeightedCategoriesValue' }
}
</code></pre>

A definição de característica esparsa requer uma ou mais características de índice e uma característica de valor que façam referência às características que existem no esquema. Definir explicitamente características esparsas permite que o TFDV verifique se as valências de todas as característica referidas correspondem.

Alguns casos de uso introduzem restrições de valência similar entre características, mas não necessariamente contém uma característica esparsa. O uso de uma característica esparsa permite eliminar a restrição, mas não é a solução ideal.

##### Ambientes (environments) de esquema

Por padrão, as validações assumem que todos os exemplos em um pipeline aderem a um único esquema. Em alguns casos, é necessária a introdução de pequenas variações de esquema, por exemplo, recursos usados ​​como rótulos são necessários durante o treinamento (e devem ser validados), mas faltam durante a o fornecimento do serviço. Ambientes (environments) podem ser usados ​​para expressar tais requisitos, em particular `default_environment()`, `in_environment()`, `not_in_environment()`.

Por exemplo, suponha que um recurso chamado 'LABEL' seja necessário para treinamento, mas espera-se que esteja ausente no fornecimento do serviço. Isto pode ser expresso por:

- Definir dois ambientes distintos no esquema: ["SERVING", "TRAINING"] e associar 'LABEL' apenas ao ambiente "TRAINING".
- Associar os dados de treinamento ao ambiente "TRAINING" e os dados de fornecimento do serviço ao ambiente "SERVING".

##### Geração de esquemas

O esquema de dados de entrada é especificado como uma instância do TensorFlow [Schema](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/schema.proto).

Em vez de construir um esquema manualmente do zero, um desenvolvedor pode contar com a construção automática de esquemas do TensorFlow Data Validation. Especificamente, o TensorFlow Data Validation constrói automaticamente um esquema inicial com base em estatísticas calculadas sobre dados de treinamento disponíveis no pipeline. Os usuários podem simplesmente revisar esse esquema gerado automaticamente, modificá-lo conforme necessário, registrá-lo num sistema de controle de versão e enviá-lo explicitamente ao pipeline para validação adicional.

O TFDV inclui `infer_schema()` para gerar um esquema automaticamente. Por exemplo:

```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
```

Isto aciona uma geração automática de esquema com base nas seguintes regras:

- Se um esquema já tiver sido gerado automaticamente, ele será usado como está.

- Caso contrário, o TensorFlow Data Validation examina as estatísticas de dados disponíveis e calcula um esquema adequado para os dados.

*Nota: O esquema gerado automaticamente é o de melhor esforço e tenta apenas inferir propriedades básicas dos dados. Espera-se que os usuários o revisem e modifiquem conforme seja necessário.*

### Detecção de desvios (skew) entre treinamento e serviço<a name="skewdetect"></a>

#### Visão geral

O TensorFlow Data Validation pode detectar desvios (skew) de distribuição entre o treinamento e o fornecimento (serving) de dados. A distorção de distribuição ocorre quando a distribuição de valores de características para dados de treinamento é significativamente diferente do fornecimento dos dados. Uma das principais causas do desvio na distribuição é o uso de um corpus completamente diferente para treinar a geração de dados para lidar com a falta de dados iniciais no corpus desejado. Outro motivo é um mecanismo de amostragem defeituoso que escolhe apenas uma subamostra dos dados fornecidos para o treinamento.

##### Cenário de exemplo

Observação: por exemplo, para compensar uma fatia de dados sub-representada, se uma amostragem tendenciosa for usada sem aumentar o peso dos exemplos com uma amostragem reduzida de forma adequada, haverá um desvio artificial na distribuição de valores de características entre o treinamento e o fornecimento de dados.

Veja o [Guia de introdução ao TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) para mais informações sobre como configurar a detecção de desvios (skew).

### Detecção de deriva (drift) de dados

A detecção de deriva é suportada entre spans consecutivos de dados (ou seja, entre o span N e o span N+1), como por exemplo, entre diferentes dias de dados de treinamento. Expressamos a deriva em termos de [distância L-infinito](https://en.wikipedia.org/wiki/Chebyshev_distance) para características categóricas e [divergência aproximada de Jensen-Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) para características numéricas. Você pode definir a distância limite para receber avisos quando a deriva for maior do que a aceitável. Definir a distância correta é normalmente um processo iterativo que requer conhecimento de domínio e experimentação.

Veja o [Guia de introdução ao TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) para mais informações sobre como configurar a detecção de derivas (drift).

## Usando visualizações para verificar seus dados

O TensorFlow Data Validation fornece ferramentas para visualizar a distribuição de valores de características. Ao examinar essas distribuições num notebook Jupyter usando [Facets](https://pair-code.github.io/facets/), você poderá detectar problemas comuns com seus dados.

![Estatísticas de características](images/feature_stats.png)

### Identificando distribuições suspeitas

Você pode identificar bugs comuns nos seus dados usando uma visualização de Facets Overview para procurar distribuições suspeitas de valores de características.

#### Dados desbalanceados

Uma característica desbalanceada é aquela para a qual um valor predomina. Características desbalanceadas podem ocorrer naturalmente, mas se uma característica sempre tiver o mesmo valor, você talvez tenha um bug nos seus dados. Para detectar características desbalanceadas numa Facets Overview, escolha "Non-uniformity" no menu suspenso "Sort by".

As características mais desbalanceadas serão listadas no topo de cada lista de tipo de característica. Por exemplo, a captura de tela a seguir mostra uma característica composta apenas por zeros e uma segunda característica altamente desbalanceada, no topo da lista "Numeric Features":

![Visualização de dados desbalanceados](images/unbalanced.png)

#### Dados distribuídos uniformemente

Uma característica uniformemente distribuída é aquela para a qual todos os valores possíveis aparecem quase com a mesma frequência. Tal como acontece com os dados desbalanceados, esta distribuição pode ocorrer naturalmente, mas também pode ser causada por bugs nos dados.

Para detectar características distribuídas uniformemente numa Facets Overview, escolha "Non-uniformity" no menu suspenso "Sort by" e marque a caixa de seleção "Reverse order":

![Histograma de dados uniformes](images/uniform.png)

Os dados em formato string são representados usando gráficos de barras se houver 20 ou menos valores exclusivos e como um grafo de distribuição cumulativa se houver mais de 20 valores exclusivos. Portanto, para dados em formato string, distribuições uniformes poderão aparecer como gráficos de barras planas como o mostrado acima ou linhas retas como mostrado abaixo:

![Gráfico de linhas: distribuição cumulativa de dados uniformes](images/uniform_cumulative.png)

##### Bugs que podem produzir dados distribuídos uniformemente

Aqui estão alguns bugs comuns que podem produzir dados distribuídos uniformemente:

- Usar strings para representar tipos de dados que não são strings, como datas. Por exemplo, você terá muitos valores exclusivos para uma característica de data e hora contendo representações do tipo "2017-03-01-11-45-03". Valores exclusivos serão distribuídos uniformemente.

- Incluir índices como "número da linha" como características. Aqui, novamente, você terá muitos valores exclusivos.

#### Dados ausentes

Para verificar se uma característica está totalmente sem valores:

1. Escolha "Amount missing/zero" no menu suspenso "Sort by".
2. Marque a caixa de seleção "Reverse order".
3. Observe a coluna "missing" para ver a porcentagem de instâncias com valores ausentes para uma característica.

Um bug de dados também pode ser a causa de valores de característica incompletos. Por exemplo, você pode esperar que a lista de valores de uma característica sempre tenha três elementos e descobrir depois que às vezes ela possui apenas um. Para verificar valores incompletos ou outros casos em que as listas de valores das características não possuem o número esperado de elementos, faça o seguinte:

1. Escolha “Value list length” no menu suspenso “Chart to show” à direita.

2. Observe o gráfico à direita de cada linha de característica. O gráfico mostra o intervalo de comprimentos da lista de valores para a característica. Por exemplo, a linha destacada na captura de tela abaixo mostra uma característica que possui algumas listas de valores de comprimento zero:

![Facets Overview contendo característica com listas de valores de características de comprimento zero](images/zero_length.png)

#### Grandes diferenças de escala entre características

Se suas características variarem muito em escala, o modelo poderá ter dificuldades de aprendizado. Por exemplo, se algumas características variam de 0 a 1 e outras variam de 0 a 1.000.000.000, você tem uma grande diferença de escala. Compare as colunas "max" e "min" entre as características para descobrir escalas que variam muito.

Considere normalizar os valores das características para reduzir essas variações grandes.

#### Rótulos inválidos

Os estimadores do TensorFlow têm restrições quanto ao tipo de dados que aceitam como rótulos. Por exemplo, classificadores binários normalmente funcionam apenas com rótulos {0, 1}.

Revise os valores dos rótulos no Facets Overview e certifique-se de que estejam em conformidade com os [requisitos dos Estimators](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/feature_columns.md) .

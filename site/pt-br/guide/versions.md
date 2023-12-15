# Compatibilidade das versões do TensorFlow

Este documento é para usuários que precisam da compatibilidade com versões anteriores em diferentes versões do TensorFlow (seja para código ou dados) e para desenvolvedores que querem modificar o TensorFlow enquanto preservam a compatibilidade.

## Versionamento Semântico 2.0

O TensorFlow segue o Versionamento Semântico 2.0 ([semver](http://semver.org)) para a API pública. Cada versão do TensorFlow tem o formato `MAJOR.MINOR.PATCH`. Por exemplo, a versão 1.2.3 do TensorFlow tem a versão 1 `MAJOR`, versão 2 `MINOR` e versão 3 `PATCH`. As mudanças em cada número tem o seguinte significado:

- **MAJOR**: mudanças possivelmente incompatíveis com versões anteriores. O código e os dados que funcionavam com uma versão major anterior não funcionarão necessariamente com a nova versão. No entanto, em alguns casos, os grafos e checkpoints existentes do TensorFlow podem ser migrados para a versão mais recente. Confira mais detalhes sobre a compatibilidade de dados em [Compatibilidade de grafos e checkpoints](#compatibility_of_graphs_and_checkpoints).

- **MINOR**: recursos compatíveis com versões anteriores, melhorias na velocidade etc. O código e os dados que funcionavam com uma versão minor anterior *e* que dependiam apenas da API pública não experimental continuarão a funcionar sem alterações. Para saber mais sobre o que é ou não a API pública, confira [O que está coberto](#what_is_covered).

- **PATCH**: correções de bugs compatíveis com versões anteriores.

Por exemplo, a versão 1.0.0 apresentou mudanças *incompatíveis* com versões anteriores a partir da versão 0.12.1. No entanto, a versão 1.1.1 era *compatível* com a versão 1.0.0. <a name="what_is_covered"></a>

## O que está coberto

Somente as APIs públicas do TensorFlow são compatíveis com versões anteriores minor e patch. As APIs públicas consistem em:

- Todas as funções e classes documentadas do [Python](https://www.tensorflow.org/api_docs/python) no módulo e submódulos do `tensorflow`, com exceção de

    - Símbolos particulares: qualquer função, classe, entre outros, cujo nome comece com `_`
    - Símbolos experimentais e `tf.contrib`, confira mais detalhes [abaixo](#not_covered).

    Observe que o código nos diretórios `examples/` e `tools/` não pode ser alcançado pelo módulo `tensorflow` do Python. Portanto, não é coberto pela garantia de compatibilidade.

    Se um símbolo estiver disponível pelo módulo ou submódulos do `tensorflow` do Python, mas não estiver documentado, então ele **não** será considerado parte da API pública.

- A API de compatibilidade (no Python, o módulo `tf.compat`). Em versões major, podemos lançar utilitários e endpoints adicionais para ajudar os usuários com a transição para uma nova versão major. Esses símbolos de API foram descontinuados e não são compatíveis (ou seja, não vamos adicionar recursos nem corrigir bugs além das vulnerabilidades), mas eles são abarcados pelas nossas garantias de compatibilidade.

- A API C do TensorFlow:

    - [tensorflow/c/c_api.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h))

- A API C do TensorFlow Lite:

    - [tensorflow/lite/c/c_api.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api.h)
    - [tensorflow/lite/c/c_api_types.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_types.h).

- Os seguintes arquivos de buffer de protocolo:

    - [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    - [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    - [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    - [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    - [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    - [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    - [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    - [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    - [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    - [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>

## O que *não* está coberto

Algumas partes do TensorFlow podem mudar de maneiras incompatíveis com versões anteriores a qualquer momento. Isso incluem:

- **APIs experimentais**: para facilitar o desenvolvimento, isentamos alguns símbolos de API claramente marcados como experimentais das garantias de compatibilidade. Especificamente, o seguinte não é coberto por quaisquer garantias de compatibilidade:

    - qualquer símbolo no módulo ou submódulos do `tf.contrib`;
    - qualquer símbolo (módulo, função, argumento, propriedade, classe ou constante) cujo nome contém `experimental` ou `Experimental`; ou
    - qualquer símbolo cujo nome totalmente qualificado inclui um módulo ou uma classe que é experimental. Isso inclui campos e submensagens de qualquer buffer de protocolo chamado `experimental`.

- **Outras linguagens**: as APIs TensorFlow em linguagens diferentes do Python e C, como:

    - [C++](../install/lang_c.ipynb) (exposta através de arquivos de cabeçalho em [`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc))
    - [Java](../install/lang_java_legacy.md)
    - [Go](https://github.com/tensorflow/build/blob/master/golang_install_guide/README.md)
    - [JavaScript](https://www.tensorflow.org/js)

- **Detalhes das operações compostas:** várias funções públicas no Python expandem para diversas operações primitivas no grafo, e esses detalhes farão parte de quaisquer grafos salvos no disco como `GraphDef`s. Esses detalhes podem mudar para versões minor. Especialmente, testes de regressão que verificam a correspondência exata entre grafos tendem a perder a compatibilidade entre versões minor, mesmo que o comportamento do grafo não seja alterado e os checkpoints existentes ainda funcionem.

- **Detalhes numéricos de ponto flutuante:** os valores de ponto flutuante específicos computados pelas operações podem mudar a qualquer momento. Os usuários devem contar somente com a exatidão aproximada e a estabilidade numérica, e não os bits específicos computados. As mudanças nas fórmulas numéricas em versões minor e patch devem resultar em exatidão comparável ou melhorada, com a ressalva de que a melhoria na exatidão de determinadas fórmulas no aprendizado de máquina pode resultar em menor exatidão do sistema em geral.

- **Números aleatórios:** os números aleatórios específicos computados podem mudar a qualquer momento. Os usuários devem contar apenas com as distribuições aproximadamente corretas e o poder estatístico, e não com os bits específicos computados. Confira mais detalhes em [geração de números aleatórios](random_numbers.ipynb).

- **Desvio de versão no Tensorflow distribuído:** a execução de duas versões diferentes do TensorFlow em um único cluster não é compatível. Não há garantias de compatibilidade do wire protocol com versões anteriores.

- **Bugs:** reservamos o direito de fazer alterações incompatíveis com versões anteriores no comportamento (embora não na API) se a implementação atual estiver claramente quebrada, ou seja, se contradiz a documentação ou se um comportamento pretendido que é bem definido e conhecido não for implementado corretamente devido a um bug. Por exemplo, se um otimizador afirma implementar um algoritmo de otimização bem conhecido, mas não corresponde a esse algoritmo devido a um bug, vamos corrigir o otimizador. Nossa correção pode quebrar o código que depende do comportamento errado para convergência. Vamos apontar essas mudanças nas notas da versão.

- **API não utilizada:** reservamos o direito de realizar mudanças incompatíveis com versões anteriores nas APIs que não tiverem usos documentados (ao realizar auditoria do uso do TensorFlow pela pesquisa do GitHub). Antes de fazer essas alterações, vamos anunciar nossa intenção de fazer as mudanças na [lista de e-mails announce@](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce), fornecendo instruções de como resolver qualquer quebra (se aplicável), e vamos esperar duas semanas para dar à comunidade a chance de compartilhar feedback.

- **Comportamento do erro:** podemos substituir erros pelo comportamento não errado. Por exemplo, podemos mudar uma função para computar um resultado em vez de gerar um erro, mesmo se esse erro estiver documentado. Também reservamos o direito de mudar o texto das mensagens de erro. Além disso, o tipo de um erro pode mudar, a menos que o tipo de exceção para uma determinada condição de erro seja especificado na documentação.

<a name="compatibility_of_graphs_and_checkpoints"></a>

## Compatibilidade de SavedModels, grafos e checkpoints

O SavedModel é o formato de serialização preferencial para o uso nos programas do TensorFlow. O SavedModel contém duas partes: um ou mais grafos codificados como `GraphDefs` e um Checkpoint. Os grafos descrevem os fluxos de dados de operações a serem executadas, e os checkpoints contêm os valores de tensor salvos das variáveis em um grafo.

Vários usuários do TensorFlow criam SavedModels e os carregam e executam com uma versão mais recente do TensorFlow. Em conformidade com o [semver](https://semver.org), os SavedModels escritos com uma versão do TensorFlow podem ser carregados e avaliados com uma versão mais recente do TensorFlow com o mesmo major.

Fazemos garantias adicionais para SavedModels *compatíveis*. Chamamos um SavedModel que foi criado usando **apenas APIs não compatíveis, não descontinuadas e não experimentais** na versão major `N` do TensorFlow um <em data-md-type="raw_html">SavedModel compatível com a versão `N`</em>. Qualquer SavedModel compatível com a versão major `N` do TensorFlow pode ser carregado e executado com a versão major `N+1` do TensorFlow. No entanto, a funcionalidade necessária para criar ou modificar esse modelo pode não estar mais disponível, então essa garantia só se aplica ao SavedModel não modificado.

Vamos nos empenhar para preservar a compatibilidade com versões anteriores o máximo possível, para que os arquivos serializados possam ser utilizados por um longo período.

### Compatibilidade de GraphDef

Os grafos são serializados pelo buffer de protocolo `GraphDef`. Para facilitar as mudanças incompatíveis com as versões anteriores nos grafos, cada `GraphDef` tem um número de versão separado da versão do TensorFlow. Por exemplo, a versão 17 do `GraphDef` descontinuou a operação `inv` em favor de `reciprocal`. A semântica é esta:

- Cada versão do TensorFlow é compatível com um intervalo de versões do `GraphDef`. Esse intervalo será constante em versões patch e só crescerá em versões minor. A interrupção do suporte para uma versão `GraphDef` só ocorrerá para uma versão major do TensorFlow (e apenas alinhado ao suporte com a versão garantido para SavedModels).

- Os grafos criados recentemente são atribuídos ao número de versão mais recente do `GraphDef`.

- Se uma determinada versão do TensorFlow for compatível com a versão `GraphDef` de um grafo, ela carregará e avaliará com o mesmo comportamento que a versão do TensorFlow usada na geração dela (exceto para detalhes numéricos de ponto flutuante e números aleatórios, conforme descrito acima), independentemente da versão major do TensorFlow. Especialmente, um GraphDef compatível com um arquivo de checkpoint em uma versão do TensorFlow (como é o caso em um SavedModel) permanecerá compatível com esse checkpoint em versões subsequentes, enquanto o GraphDef for compatível.

    Observe que isso se aplica apenas a grafos serializados em GraphDefs (e SavedModels): o *código* que lê um checkpoint talvez não consiga ler checkpoints gerados pelo mesmo código que executa uma versão diferente do TensorFlow.

- Se o limite *superior* do `GraphDef` for aumentado para X em uma versão (minor), o limite *inferior* será aumentado para X no mínimo seis meses depois. Por exemplo (estamos usando números de versões hipotéticas aqui):

    - O TensorFlow 1.2 talvez seja compatível com as versões 4 a 7 do `GraphDef`.
    - O TensorFlow 1.3 poderia adicionar a versão 8 do `GraphDef` e oferecer suporte às versões 4 a 8.
    - Pelo menos seis meses depois, o TensorFlow 2.0.0 poderia interromper o suporte para as versões 4 a 7, deixando apenas a versão 8.

    Como as versões major do TensorFlow são geralmente publicadas com mais de 6 meses de intervalo, as garantias para os SavedModels compatíveis detalhados acima são muito mais fortes do que a garantia de 6 meses para GraphDefs.

Por fim, quando o suporte a uma versão do `GraphDef` é interrompido, vamos tentar oferecer ferramentas para converter automaticamente os grafos para uma versão mais recente do `GraphDef` compatível.

## Compatibilidade de grafos e checkpoints ao estender o TensorFlow

Esta seção só é relevante ao fazer alterações incompatíveis com o formato `GraphDef`, como ao adicionar operações, remover operações ou mudar a funcionalidade de operações existentes. A seção anterior deve bastar para a maioria dos usuários.

<a id="backward_forward"></a>

### Compatibilidade total com versões anteriores e parcial com versões futuras

Nosso esquema de versionamento tem três requisitos:

- **Compatibilidade com versões anteriores** para dar suporte ao carregamento de grafos e checkpoints criados com versões mais antigas do TensorFlow.
- **Compatibilidade com versões futuras** para dar suporte aos cenários em que o produtor de um grafo ou checkpoint faz upgrade para uma versão mais recente do TensorFlow antes do consumidor.
- Evolução do TensorFlow de maneiras incompatíveis. Por exemplo, removendo operações e adicionando ou removendo atributos.

Embora o mecanismo de versão do `GraphDef` seja separado da versão do TensorFlow, as mudanças incompatíveis com versões anteriores no formato `GraphDef` ainda estão restritas ao Versionamento Semântico. Isso significa que a funcionalidade só pode ser removida ou alterada entre as versões `MAJOR` do TensorFlow (como `1.7` a `2.0`). Além disso, a compatibilidade com versões futuras é aplicada nas versões Patch (`1.x.1` a `1.x.2`, por exemplo).

Para atingir a compatibilidade com versões anteriores e futuras e para saber quando aplicar as mudanças nos formatos, os grafos e checkpoints têm metadados que descrevem quando foram produzidos. As seções abaixo detalham a implementação e as diretrizes do TensorFlow para a evolução das versões do `GraphDef`.

### Esquemas de versões de dados independentes

Há diferentes versões de dados para grafos e checkpoints. Os dois formatos de dados evoluem a taxas diferentes em comparação com um ao outro e também com o TensorFlow. Os dois sistemas de versionamento são definidos em [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h). Quando uma nova versão é adicionada, uma nota é incluída no cabeçalho detalhando as alterações e a data.

### Dados, produtores e consumidores

Distinguimos entre os seguintes tipos de informações sobre as versões dos dados:

- **produtores**: binários que produzem dados. Os produtores têm uma versão (`producer`) e são compatíveis com uma versão de consumidor mínima (`min_consumer`).
- **consumidores**: binários que consomem dados. Os consumidores têm uma versão (`consumer`) e são compatíveis com uma versão de produtor mínima (`min_producer`).

Cada parte dos dados versionados tem um campo [`VersionDef versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto) que registra o `producer` que criou os dados, o `min_consumer` compatível e a lista de versões `bad_consumers` proibidas.

Por padrão, quando um produtor cria alguns dados, os dados herdam as versões `producer` e `min_consumer` do produtor. `bad_consumers` pode ser definido se versões do consumidor específicas tiverem bugs conhecidos e precisarem ser evitadas. Um consumidor pode aceitar alguns dados se todas as afirmações a seguir forem verdadeiras.

- `consumer` &gt;= `min_consumer` dos dados
- `producer` dos dados &gt;= `min_producer` do consumidor
- `consumer` não está no `bad_consumers` dos dados

Como ambos os produtores e consumidores vêm da mesma base de código do TensorFlow, [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) contém uma versão de dados principal que é tratada como `producer` ou `consumer` dependendo do contexto, além de ambas as versões `min_consumer` e `min_producer` (exigidas por produtores e consumidores, respectivamente). Especificamente,

- Para versões `GraphDef`, temos `TF_GRAPH_DEF_VERSION`, `TF_GRAPH_DEF_VERSION_MIN_CONSUMER` e `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`.
- Para versões de checkpoint, temos `TF_CHECKPOINT_VERSION`, `TF_CHECKPOINT_VERSION_MIN_CONSUMER` e `TF_CHECKPOINT_VERSION_MIN_PRODUCER`.

### Adicione um novo atributo com uma operação existente como padrão

Seguindo as orientações abaixo, você só alcança a compatibilidade com versões futuras se o conjunto de operações não tiver mudado:

1. Se você desejar a compatibilidade com versões futuras, defina `strip_default_attrs` como `True` ao exportar o modelo usando os métodos `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables` e `tf.saved_model.SavedModelBuilder.add_meta_graph` da classe `SavedModelBuilder` ou `tf.estimator.Estimator.export_saved_model`
2. Isso retira os atributos de valor padrão no momento de produção/exportação dos modelos, garantindo que `tf.MetaGraphDef` não contenha o novo op-attribute quando o valor padrão é usado.
3. Ter esse controle pode permitir que consumidores desatualizados (por exemplo, que veiculam binários com atraso em relação aos binários de treinamento) continuem carregando modelos e evitem interrupções na veiculação do modelo.

### Evolução das versões do GraphDef

Esta seção explica como usar esse mecanismo de versionamento para fazer diferentes tipos de mudanças no formato `GraphDef`.

#### Adicione uma operação

Adicione a nova operação a ambos os consumidores e produtores ao mesmo tempo e não mude qualquer versão do `GraphDef`. Esse tipo de mudança é automaticamente compatível com versões anteriores e não afeta o plano de compatibilidade com as versões futuras, já que os scripts de produtores existentes não usarão a nova funcionalidade de repente.

#### Adicione uma operação e faça com que seja usada pelos wrappers existentes do Python

1. Implemente a nova funcionalidade de consumidor e incremente a versão do `GraphDef`.
2. Se for possível fazer os wrappers usarem a nova funcionalidade apenas em casos que não funcionavam antes, os wrappers podem ser atualizados agora.
3. Altere os wrappers do Python para que usem a nova funcionalidade. Não incremente `min_consumer`, já que os modelos que não usam essa operação não devem quebrar.

#### Remova ou restrinja a funcionalidade de uma operação

1. Corrija todos os scripts (e não o próprio TensorFlow) para que não usem a operação ou funcionalidade banida.
2. Incremente a versão do `GraphDef` e implemente uma nova funcionalidade de consumidor para banir a operação ou funcionalidade removida para GraphDefs na versão nova e mais recente. Se possível, faça com que o TensorFlow pare de produzir `GraphDefs` com a funcionalidade banida. Para isso, adicione [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009).
3. Aguarde uma versão major para fins de compatibilidade com versões anteriores.
4. Aumente `min_producer` para a versão do GraphDef de (2) e remova toda a funcionalidade.

#### Mude a funcionalidade de uma operação

1. Adicione uma nova operação parecida com o nome `SomethingV2` ou algo semelhante e passe pelo processo de adicioná-la e fazer com que seja usada pelos wrappers existentes do Python. Para garantir a compatibilidade com versões futuras, use as verificações sugeridas em [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py) ao mudar os wrappers do Python.
2. Remova a operação antiga (só pode ser feito com a mudança em uma versão major devido à compatibilidade com versões anteriores).
3. Aumente `min_consumer` para descartar consumidores com a operação antiga, adicione a operação antiga de volta como um alias para `SomethingV2` e faça com que ela seja usada pelos wrappers existentes do Python.
4. Conclua o processo de remoção de `SomethingV2`.

#### Proíba uma única versão de consumidor arriscada

1. Eleve a versão `GraphDef` e adicione a versão ruim a `bad_consumers` para todos os novos GraphDefs. Se possível, adicione a `bad_consumers` somente para GraphDefs que contém uma operação específica ou semelhante.
2. Se os consumidores existentes têm a versão ruim, eles devem ser migrados assim que possível.

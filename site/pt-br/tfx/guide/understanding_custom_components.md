# Como funcionam os componentes personalizados do TFX

Os pipelines TFX permitem orquestrar seu workflow de aprendizado de máquina (ML) em orquestradores, tais como: Apache Airflow, Apache Beam e Kubeflow Pipelines. Os pipelines organizam seu workflow através de uma sequência de componentes, onde cada componente executa uma etapa do seu workflow de ML. Os componentes padrão do TFX fornecem funcionalidade comprovada para ajudá-lo a começar a criar facilmente um workflow de ML. Você também pode incluir componentes personalizados no seu workflow. Componentes personalizados permitem que você estenda seu workflow de ML:

- Construindo componentes adaptados para atender às suas necessidades, como o consumo de dados de um sistema proprietário.
- Aplicando ampliação de dados, unsampling ou downsampling.
- Realizando a detecção de anomalias com base em intervalos de confiança ou erros de reprodução do autoencoder.
- Fornecendo uma interface para sistemas externos, como help desks para alertas e monitoração.
- Aplicando rótulos a exemplos não rotulados.
- Integrando ferramentas criadas com linguagens diferentes de Python no seu workflow de ML, como realizando análise de dados usando R.

Ao misturar componentes padrão e componentes personalizados, você pode criar um workflow de ML que atenda às suas necessidades e, ao mesmo tempo, aproveitar as práticas recomendadas incorporadas aos componentes padrão do TFX.

Este guia descreve os conceitos necessários para entender os componentes personalizados do TFX e mostra diferentes maneiras de criar componentes personalizados.

## Anatomia de um componente TFX

Esta seção fornece uma visão geral em alto nível da composição de um componente TFX. Se você é novo em relação aos pipelines do TFX, [aprenda os conceitos básicos lendo o guia para entender os pipelines do TFX](understanding_tfx_pipelines.md).

Os componentes TFX são compostos por uma especificação de componente e uma classe executora empacotados numa classe que fornece a interface do componente.

Uma *especificação de componente* define o contrato de entrada e saída do componente. Este contrato especifica os artefatos de entrada e saída do componente e os parâmetros usados ​​para a execução do componente.

A classe *executora* de um componente fornece a implementação do trabalho executado pelo componente.

Uma classe de *interface de componente* combina a especificação do componente com o executor para uso como um componente num pipeline TFX.

### Componentes TFX em tempo de execução

Quando um pipeline executa um componente TFX, o componente é executado em três fases:

1. Primeiro, o Driver usa a especificação do componente para recuperar os artefatos necessários do metadata store e passá-los para o componente.
2. Em seguida, o Executor executa o trabalho do componente.
3. Por fim, o Publisher usa a especificação do componente e os resultados do executor para armazenar as saídas do componente no metadata store.

![Anatomia de um componente](https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tfx/guide/images/component.png?raw=true)

A maioria das implementações de componentes customizados não exige que você personalize o Driver ou o Publisher. Normalmente, as modificações no Driver e no Publisher só deverão ser necessárias se você quiser alterar a interação entre os componentes do pipeline e o metadata store. Se você deseja alterar apenas as entradas, saídas ou parâmetros do seu componente, você só precisa modificar a *especificação do componente*.

## Tipos de componentes personalizados

Há três tipos de componentes personalizados: componentes baseados em funções Python, componentes baseados em container e componentes totalmente personalizados. As seções a seguir descrevem esses diferentes tipos de componentes e as situações em que você deve usar cada abordagem.

### Componentes baseados em funções Python

Componentes baseados em funções Python são mais fáceis de construir do que componentes baseados em container ou componentes totalmente personalizados. A especificação do componente é definida através dos argumentos da função Python usando anotações de tipo que descrevem se um argumento é um artefato de entrada, um artefato de saída ou um parâmetro. O corpo da função define o executor do componente. A interface do componente é definida adicionando o [`@component` decorator](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py){: .external} à sua função.

Ao decorar sua função com o decorador `@component` e definindo os argumentos da função com anotações de tipo, você pode criar um componente sem a complexidade da construção de uma especificação de componente, um executor e uma interface de componente.

Aprenda como [construir componentes baseados em funções Python](custom_function_component.md).

### Componentes baseados em container

Os componentes baseados em container fornecem flexibilidade para integrar código escrito em qualquer linguagem ao seu pipeline, desde que você possa executar esse código dentro de um container Docker. Para criar um componente baseado em container, você precisa criar uma imagem de container Docker que contenha o código executável do seu componente. Depois deve chamar a [função `create_container_component`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py){: .external} para definir:

- As entradas, saídas e parâmetros da especificação do seu componente.
- A imagem do container e o comando que o executor do componente executa.

Esta função retorna uma instância de um componente que você pode incluir na definição do pipeline.

Esta abordagem é mais complexa do que construir um componente baseado em função Python, pois requer empacotar seu código como uma imagem de container. Essa abordagem é mais adequada para incluir código não-Python no seu pipeline ou para criar componentes Python com dependências ou ambientes de execução complexos.

Aprenda como [construir componentes baseados em container](container_component.md).

### Componentes totalmente personalizados

Componentes totalmente personalizados permitem construir componentes definindo a especificação do componente, o executor e as classes de interface do componente. Essa abordagem permite reutilizar e estender um componente padrão para atender às suas necessidades.

Se um componente existente for definido com as mesmas entradas e saídas do componente personalizado que você está desenvolvendo, você poderá simplesmente substituir a classe Executor do componente existente. Isto significa que você pode reutilizar uma especificação de componente e implementar um novo executor derivado de um componente existente. Dessa forma, você reutiliza a funcionalidade incorporada nos componentes existentes e implementa apenas a funcionalidade necessária.

Se, no entanto, as entradas e saídas do seu novo componente forem exclusivas, você poderá definir uma *especificação de componente* totalmente nova.

Essa abordagem é melhor para reutilizar especificações de componentes e executores existentes.

Aprenda como [construir componentes totalmente personalizados](custom_component.md).

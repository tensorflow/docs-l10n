# Implantação

Além de definir computações, o TFF fornece ferramentas para executá-las. Embora o foco principal esteja nas simulações, as interfaces e ferramentas que fornecemos são mais gerais. Este documento descreve as opções de implantação em vários tipos de plataforma.

Observação: Este documento ainda está em construção.

## Visão geral

Existem dois modos principais de implantação para computações TFF:

- **Back-ends nativos**. Chamamos um back-end de *nativo* se ele for capaz de interpretar a estrutura sintática das computações TFF conforme definidas em [`computation.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). Um back-end nativo não precisa necessariamente oferecer suporte a todos os construtos ou intrínsecos da linguagem. Os back-ends nativos devem implementar uma das interfaces *executor* padrão do TFF, como [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor) para consumo por código Python ou a versão independente de linguagem definida em [`executor.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/executor.proto) exposta como um endpoint gRPC.

    Back-ends nativos que suportam as interfaces acima podem ser usados ​​interativamente no lugar do runtime de referência padrão, por exemplo, para executar notebooks ou scripts de experimentos. A maioria dos backends nativos irão operar no *modo interpretado*, ou seja, eles irão processar a definição de computação conforme ela é definida, e executá-la de forma incremental, mas isto nem sempre precisa ser o caso. Um back-end nativo também pode *transformar* (*compilar* ou compilar com JIT) uma parte da computação para melhor desempenho ou para simplificar sua estrutura. Um exemplo de uso comum seria reduzir o conjunto de operadores federados que aparecem numa computação, de modo que partes do fluxo downstream de back-end da transformação não precisem ser expostas ao conjunto completo.

- **Back-ends não nativos**. Os back-ends não nativos, em contraste com os nativos, não podem interpretar diretamente a estrutura de computação do TFF e exigem que ela seja convertida numa *representação de destino* diferente, compreendida pelo back-end. Um exemplo de tal back-end seria um cluster Hadoop ou uma plataforma semelhante para pipelines de dados estáticos. Para que uma computação seja implantada em tal back-end, ela deve primeiro ser *transformada* (ou *compilada*). Dependendo da configuração, isto pode ser feito de forma transparente para o usuário (ou seja, um back-end não nativo pode ser encapsulado numa interface de executor padrão, como [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor), que executa transformações nos bastidores), ou pode ser exposto como uma ferramenta que permita ao usuário converter manualmente uma computação, ou um conjunto de computações, na representação de destino apropriada e compreendida pela classe específica de back-ends. O código que oferece suporte a tipos específicos de back-ends não nativos pode ser encontrado no namespace [`tff.backends`](https://www.tensorflow.org/federated/api_docs/python/tff/backends). No momento em que este artigo foi escrito, o único tipo de suporte de back-ends não nativos era uma classe de sistemas capaz de executar MapReduce em rodada única.

## Back-ends nativos

Mais detalhes em breve.

## Back-ends não nativos

### MapReduce

Mais detalhes em breve.

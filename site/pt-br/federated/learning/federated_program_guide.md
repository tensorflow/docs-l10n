# Guia de Desenvolvimento de Programa Federado para aprendizado

Esta documentação é destinada a qualquer pessoa que esteja interessada em criar [lógica de programa federado ](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) no [`tff.learning`](https://www.tensorflow.org/federated/api_docs/python/tff/learning). Ela pressupõe que você conheça o `tff.learning` e o [Guia de Desenvolvimento de Programa Federado](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md).

[TOC]

## Lógica do programa

Esta seção define as diretrizes de como a [lógica do programa](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic) deve ser criada **no `tff.learning`**.

### Componentes de aprendizado

**Use** os componentes de aprendizado na lógica do programa (por exemplo, [`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess) e [`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager)).

## Programa

Geralmente, os [programas](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs) não são criados no `tff.learning`.

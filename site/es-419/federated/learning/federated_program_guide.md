# Guía del desarrollador de programas federados de aprendizaje

Esta documentación es para cualquier persona que esté interesada en escribir [lógica de programación federada](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic) en [`tff.learning`](https://www.tensorflow.org/federated/api_docs/python/tff/learning). Se da por supuesto un conocimiento previo de `tff.learning` y de la [Guía del desarrollador de programas federados](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md).

[TOC]

## Lógica de programación

En esta sección se definen las pautas sobre cómo se debería escribir la [lógica de programación](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic) <strong>en <code data-md-type="codespan">tff.learning</code></strong>.

### Componentes de aprendizaje

**Use** componentes de aprendizaje en la lógica de programación (p. ej., [`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess) y [`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager)).

## Programa

Normalmente, los [programas](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs) no se escriben en `tff.learning`.

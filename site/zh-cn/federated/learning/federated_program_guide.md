# 学习联合程序开发者指南

本文档适用于任何对在 [`tff.learning`](https://www.tensorflow.org/federated/api_docs/python/tff/learning) 中编写[联合程序逻辑](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)感兴趣的人。它假设读者已经了解 `tff.learning` 和[联合程序开发者指南](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)的知识。

[目录]

## 程序逻辑

本部分定义了在 **`tff.learning`** 中编写[程序逻辑](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic)的准则。

### 学习组件

在程序逻辑中**务必**使用学习组件（例如，[`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess) 和 [`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager)）。

## 程序

通常情况下，[程序](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs)不是在 `tff.learning` 中编写的。

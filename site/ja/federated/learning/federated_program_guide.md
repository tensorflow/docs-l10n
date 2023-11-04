# Learning の連合プログラム開発者ガイド

このドキュメントは、[`tff.learning`](https://www.tensorflow.org/federated/api_docs/python/tff/learning) で[連合プログラムロジック](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)を記述することに興味のある方を対象としています。`tff.learning` の知識と[連合プログラム開発者ガイド](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)の理解を前提としています。

[TOC]

## プログラムロジック

このセクションは、**in `tff.learning`** でどのように[プログラムロジック](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic)を記述すべきかに関するガイドラインを定義します。

### Learning コンポーネント

プログラムロジックでは Learning コンポーネント（[`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess)、[`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager) など）を**使用してください**。

## プログラム

通常、[プログラム](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs)は `tff.learning` で記述されません。

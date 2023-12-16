# 학습 페더레이션 프로그램 개발자 가이드

이 문서는 [<code>tff.learning</code>](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)에서 <a>페더레이션 프로그램 로직</a>을 작성하는 데 관심이 있는 모든 사람을 대상으로 합니다. 이 문서는 `tff.learning` 및 [페더레이션 프로그램 개발자 가이드](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)에 대한 지식을 전제로 합니다.

[목차]

## 프로그램 로직

이 섹션에서는 `tff.learning`에서 [프로그램 로직](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic)을 작성하는 방법에 대한 가이드라인을 정의합니다.

### 학습 구성 요소

프로그램 로직에서 학습 구성 요소를 **사용해야 합니다**(예: [`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess) 및 [`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager)).

## 프로그램

일반적으로 [프로그램](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs)은 `tff.learning`에서 작성되지 않습니다.

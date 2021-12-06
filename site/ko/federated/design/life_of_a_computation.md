# 계산의 처리 과정

[TOC]

## TFF에서 Python 함수 실행하기

다음 예는 Python 함수가 TFF 계산이 되는 방법과 계산이 TFF에 의해 평가되는 방법을 강조하기 위한 것입니다.

**사용자 관점:**

```python
tff.backends.native.set_local_python_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1. *Python* 함수를 작성합니다.

2. `@tff.tf_computation`으로 *Python* 함수를 데코레이팅합니다.

    참고: 지금은 데코레이터 자체의 고유 정보가 아니라 Python 함수가 데코레이팅된다는 것이 중요합니다. [아래](#tf-vs-tff-vs-python)에 자세히 설명되어 있습니다.

3. TFF [context](context.md)를 설정합니다.

4. *Python* 함수를 호출합니다.

**TFF 관점:**

Python이 **구문 분석**될 때 `@tff.tf_computation` 데코레이터는 Python 함수를 [추적](tracing.md)하고 TFF 계산을 구성합니다.

데코레이팅된 Python 함수가 **호출**될 때 TFF 계산이 호출되며, TFF는 설정된 [context](context.md)에서 계산을 [컴파일](compilation.md) 및 [실행](execution.md)합니다.

## TF vs. TFF vs. Python

```python
tff.backends.native.set_local_python_execution_context()

@tff.tf_computation(tf.int32)
def add_one(x):
  return x + 1

@tff.federated_computation(tff.type_at_clients(tf.int32))
def add_one_to_all_clients(values):
  return tff.federated_map(add_one, values)

values = [1, 2, 3]
values = add_one_to_all_clients(values)
values = add_one_to_all_clients(values)
>>> [3, 4, 5]
```

TODO(b/153500547): TF vs. TFF vs. Python 예를 설명합니다.

# Computation の寿命

[TOC]

## TFF で Python 関数を実行する

この例では、Python 関数が TFF computation になる過程と computation が TFF によってどのように評価されるかを簡単に説明します。

**ユーザーの観点から見た計算:**

```python
tff.backends.native.set_local_python_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1. *Python* 関数を書きます。

2. *Python* 関数を `@tff.tf_computation` でデコレートします。

    注意: 現時点では、Python 関数がデコレートされていることが重要であって、具体的なデコレータ自体は重要ではありません。これについては、[以下](#tf-vs-tff-vs-python)で詳しく説明します。

3. TFF の[コンテキスト](context.md)を設定します。

4. *Python* 関数を呼び出します。

**TFF の観点から見た計算:**

Python が**構文解析**される際、`@tff.tf_computation` デコレータは Python 関数をトレースして TFF computation を構築します。

デコレートされた Python 関数が**呼び出される**と、呼び出されるのは TFF computation であり、TFF はその computation を設定された[コンテキスト](context.md)で[コンパイル](compilation.md)して[実行](execution.md)します。

## TF と TFF と Python

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

TODO(b/153500547): TF と TFF と Python の例を説明してください。

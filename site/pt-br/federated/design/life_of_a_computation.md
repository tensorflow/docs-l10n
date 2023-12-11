# Ciclo de vida de uma computação

[TOC]

## Execução de uma função do Python no TFF

Este exemplo serve para destacar como uma função do Python se torna uma computação do TFF e como a computação é avaliada pelo TFF.

**Pela perspectiva de usuário:**

```python
tff.backends.native.set_sync_local_cpp_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1. Escreva uma função do *Python*.

2. Decore a função do *Python* com `@tff.tf_computation`.

    Observação: por enquanto, só é importante que a função do Python seja decorada, e não as especificidades do decorador em si. Isso será explicado com maiores detalhes [abaixo](#tf-vs-tff-vs-python).

3. Defina um [contexto](context.md) do TFF.

4. Invoque a função do *Python*.

**Pela perspectiva do TFF:**

Quando o código Python é **processado**, o decorador `@tff.tf_computation` faz o [tracing](tracing.md) da função do Python e constrói uma computação do TFF.

Quando a função do Python é **invocada**, a computação do TFF que é invocada, e o TFF vai [compilar](compilation.md) e [executar](execution.md) a computação no [contexto](context.md) definido.

## TF x TFF x Python

```python
tff.backends.native.set_sync_local_cpp_execution_context()

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

TODO: b/153500547 – Descrever o exemplo TF x TFF x Python.

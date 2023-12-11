# La vida de un cálculo

[TOC]

## Ejecución de una función de Python en TFF

Este ejemplo se presenta con el objetivo de mostrar cómo se convierte una función de Python en un cálculo de TFF y cómo es evaluado ese cálculo por TFF.

**Desde la perspectiva de usuario:**

```python
tff.backends.native.set_sync_local_cpp_execution_context()  # 3

@tff.tf_computation(tf.int32)  # 2
def add_one(x):  # 1
  return x + 1

result = add_one(2) # 4
```

1. Escribe una función de *Python*.

2. Decora la función de *Python* con `@tff.tf_computation`.

    Nota: Por ahora, solamente es importante que la función de Python esté decorada, no las especificaciones del decorador mismo. Esto se explica más en detalle [a continuación](#tf-vs-tff-vs-python).

3. Establece un [contexto](context.md) de TFF.

4. Invoca la función de *Python*.

**Desde la perspectiva de TFF:**

Cuando se **analice** Python, el decorador `@tff.tf_computation` [rastreará](tracing.md) la función de Python y construirá un cálculo de TFF.

Cuando se **invoca** la función de Python decorada, lo que en realidad se invoca es el cálculo de TFF, entonces, TFF [compilará](compilation.md) y [ejecutará](execution.md) el cálculo en el [contexto](context.md) que fue establecido.

## TF vs. TFF vs. Python

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

Pendiente: b/153500547 - Describir el ejemplo de TF vs. TFF vs. Python.

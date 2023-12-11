# Rastreo

[TOC]

El rastreo es el proceso de construir un [AST](compilation.md#ast) (árbol de sintaxis abstracta) a partir de una función de Python.

Pendiente: b/153500547. Describir y vincular los componentes individuales del sistema de rastreo.

## Rastreo de un cálculo federado

A un alto nivel, hay tres componentes para rastrear un cálculo federado.

### Empaquetado de los argumentos

Internamente, un cálculo TFF pocas veces tiene argumento cero o uno. Los argumentos provistos al decorador [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) describen la firma de tipo de los argumentos para el cálculo de TFF. TFF usa esta información para determinar cómo agrupar (<em>pack</em>) los argumentos de la función Python en una sola [structure.Struct](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/common_libs/structure.py).

Nota: El uso de `Struct` como una estructura de datos única para representar tanto a `args` como a `kwargs` de Python es el motivo por el que la <code>Struct</code> acepta tanto los campos nombrados como los no nombrados.

Para más información, consulte [function_utils.create_argument_unpacking_fn](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/computation/function_utils.py).

### Rastreo de la función

Cuando se rastrea un `federated_computation`, la función del usuario se llama con [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) como un reemplazo interino para cada argumento. `Value` intenta emular el comportamiento del tipo de argumento original mediante la implementación de métodos <em>dunder</em> de Python (p. ej., `__getattr__`).

Para ser más específicos, cuando hay exactamente un argumento, el rastreo se logra de la siguiente manera:

1. Se construye un [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) respaldado por una [building_blocks.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) con la firma de tipo apropiada para representar el argumento.

2. Se invoca la función en el `Value`. Esta acción hace que el tiempo de ejecución de Python invoque métodos <em>dunder</em> implementados por `Value`, que traduce esos métodos <em>dunder</em> como construcción AST. Cada método <em>dunder</em> construye un AST y devuelve un `Value` respaldado por ese AST.

Por ejemplo:

```python
def foo(x):
  return x[0]
```

En este caso, el parámetro de la función es una tupla en el cuerpo de la función en que se ha seleccionado el elemento "0". Esto invoca el método `__getitem__` de Python, que queda sobrescrito por `Value`. En el caso más simple, la implementación de `Value.__getitem__` crea una [building_blocks.Selection](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) para representar la invocación de `__getitem__` y devuelve un `Value` respaldado por esta nueva `Selection`.

El rastreo continúa porque cada <em>dunder</em> devuelve un `Value` y deja la marca de cada operación que se produce en el cuerpo de la función que hace que se invoque uno de los métodos <em>dunder</em> sobrescritos.

### Construcción del AST

El resultado del rastreo de la función se empaqueta en un [building_blocks.Lambda](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) cuyos `parameter_name` y `parameter_type` mapean al [building_block.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) creado para representar los argumentos empaquetados. El `Lambda` resultante, entonces, se devuelve como un objeto de Python que representa plenamente la función Python del usuario.

## Rastreo de un cálculo de TensorFlow

Pendiente: b/153500547 - Describir el proceso de rastreo de un cálculo de TensorFlow.

## Limpieza de los mensajes de error de las excepciones durante el rastreo

En un momento de la historia de TFF, el proceso de rastreo del cálculo del usuario incluía el pasaje por varias funciones de encapsulamiento (<em>wrapper</em>) antes de llamar a la función del usuario. Este proceso tenía efectos indeseados, ya que producía mensajes de error como los siguientes:

```
Traceback (most recent call last):
  File "<user code>.py", in user_function
    @tff.federated_computation(...)
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<user code>", in user_function
    <some line of user code inside the federated_computation>
  File "<tff code>.py", tff_function
  ...
  File "<tff code>.py", tff_function
    <raise some error about something the user did wrong>
FederatedComputationWrapperTest.test_stackframes_in_errors.<locals>.DummyError
```

Resultaba bastante difícil hallar la línea final del código del usuario (la línea que realmente contenía el error) con este rastreo hacia atrás. Como resultado, los usuarios informaban estos problemas como errores de TFF y, por lo general, les complicaba la vida.

Hoy en día, TFF tiene algunas dificultades para garantizar que estas pilas de llamadas no tienen funciones TFF extra. Es el motivo por el que se usan generadores en el código de rastreo de TFF, con frecuencia en patrones como los que se muestran a continuación:

```
# Instead of writing this:
def foo(fn, x):
  return 5 + fn(x + 1)

print(foo(user_fn, 20))

# TFF uses this pattern for its tracing code:
def foo(x):
  result = yield x + 1
  yield result + 5

fooer = foo(20)
arg = next(fooer)
result = fooer.send(user_fn(arg))
print(result)
```

Este patrón permite llamar al código del usuario (arriba, `user_fn`) al nivel superior de la pila de llamadas. También permite la manipulación por funciones de encapsulamiento (<em>wrapping</em>) de sus argumentos, salidas y el contexto local del hilo.

Algunas versiones simples de este patrón pueden ser reemplazadas más sencillamente por funciones "<em>before</em>" (antes) o "<em>after</em>" (después). Por ejemplo, `foo` (arriba) se podría reemplazar por:

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

Se debería optar por este patrón en aquellos casos en los que no se requiera un estado compartido entre las porciones "<em>before</em>" y "<em>after</em>". Sin embargo, puede resultar engorroso expresar de este modo los casos menos simples que incluyen gestores de contexto y estados más complejos:

```
# With the `yield` pattern:
def in_ctx(fn):
  with create_ctx():
    yield
    ... something in the context ...
  ...something after the context...
  yield

# WIth the `before` and `after` pattern:
def before():
  new_ctx = create_ctx()
  new_ctx.__enter__()
  return new_ctx

def after(ctx):
  ...something in the context...
  ctx.__exit__()
  ...something after the context...
```

Queda mucho menos claro en el último ejemplo en el que el código se ejecuta dentro de un contexto y la situación se vuelve incluso menos clara cuando se comparten más bits de estado en las secciones "<em>before</em>" y "<em>after</em>"

Se intentó dar solución al problema general de "ocultar las funciones TFF de los mensajes de error de los usuarios" de muchas otras formas diferentes, incluso atrapando (<em>catching</em>) o propagando (<em>reraising</em>) excepciones (falló debido a la incapacidad de crear una excepción cuya pila incluyera solamente el nivel más bajo del código del usuario, sin incluir además el código que lo llamó). También se intentó atrapar (<em>catch</em>) las excepciones y reemplazar su rastreo hacia atrás con uno filtrado (que es específico de CPython y no es compatible con el lenguaje Python) y reemplazar el <em>handler</em> de la excepción (falla porque `absltest` no usa `sys.excepthook` y otros <em>frameworks</em> lo sobrescriben). A fin de cuentas, la inversión de control basada en generadores permite lograr la mejor experiencia posible en usuarios finales, pagando el costo de trabajar con cierta complejidad en la implementación de TFF.

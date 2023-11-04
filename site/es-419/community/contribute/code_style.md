# Guía de estilo del código de TensorFlow

## Estilo de Python

Siga la [guía de estilo de Python de PEP 8](https://www.python.org/dev/peps/pep-0008/), excepto que TensorFlow usa 2 espacios en lugar de 4. Péguese a lo escrito en la [Guía de estilo de Python de Google](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) y use [pylint](https://www.pylint.org/) para verificar sus cambios en Python.

### pylint

Para instalar `pylint`, escriba lo que sigue:

```bash
$ pip install pylint
```

Para comprobar un archivo con `pylint` desde el directorio raíz del código fuente de TensorFlow, ejecute lo siguiente:

```bash
$ pylint --rcfile=tensorflow/tools/ci_build/pylintrc tensorflow/python/keras/losses.py
```

### Versiones de Python compatibles

Para conocer las versiones de Python compatibles, consulte la [guía de instalación](https://www.tensorflow.org/install) de TensorFlow.

Consulte el [estado de compilación continua](https://github.com/tensorflow/tensorflow/blob/master/README.md#continuous-build-status) de TensorFlow para conocer las compilaciones oficiales y respaldadas por la comunidad.

## Estilo de codificación C++

Los cambios al código de TensorFlow C++ deben ajustarse a la [Guía de estilo de C++ de Google](https://google.github.io/styleguide/cppguide.html) y a los [detalles de estilo específicos de TensorFlow](https://github.com/tensorflow/community/blob/master/governance/cpp-style.md). Use `clang-format` para comprobar sus cambios en C/C++.

Para instalar en Ubuntu 16+, escriba lo que sigue:

```bash
$ apt-get install -y clang-format
```

Puede verificar el formato de un archivo C/C++ con el siguiente código:

```bash
$ clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
$ diff <my_cc_file> /tmp/my_cc_file.cc
```

## Otros lenguajes

- [Guía de estilo de Java de Google](https://google.github.io/styleguide/javaguide.html)
- [Guía de estilo de JavaScript de Google](https://google.github.io/styleguide/jsguide.html)
- [Guía de estilo de Shell de Google](https://google.github.io/styleguide/shell.xml)
- [Guía de estilo de Objective-C de Google](https://google.github.io/styleguide/objcguide.html)

## Convenciones y usos especiales de TensorFlow

### Operaciones de Python

Una *operación* de TensorFlow es una función que, en función de los tensores de entrada, devuelve tensores de salida (o agrega una operación a un gráfico cuando se crean gráficos).

- El primer argumento deben ser los tensores, seguidos de los parámetros básicos de Python. El último argumento es `name` con un valor predeterminado de `None`.
- Los argumentos de los tensores deben ser un único tensor o un iterable de tensores. Es decir, un "Tensor o lista de tensores" es demasiado amplio. Consulte `assert_proper_iterable`.
- Las operaciones que toman tensores como argumentos deberían llamar a `convert_to_tensor` para convertir entradas que no sean tensores en tensores si están usando operaciones C++. Tenga en cuenta que los argumentos todavía se describen como un objeto `Tensor` de un dtype específico en la documentación.
- Cada operación de Python debe tener un `name_scope`. Como se ve a continuación, se pasa el nombre de la operación como una cadena.
- Las operaciones deben contener un extenso comentario en Python con declaraciones de Args y Returns que expliquen tanto el tipo como el significado de cada valor. Las posibles formas, dtypes o clasificaciones deben especificarse en la descripción. Consulte los detalles de la documentación.
- Para aumentar la facilidad de uso, incluya un ejemplo de uso con entradas o salidas de la operación en la sección Ejemplo.
- Evite utilizar explícitamente `tf.Tensor.eval` o `tf.Session.run`. Por ejemplo, para escribir lógica que dependa del valor del Tensor, utilice el flujo de control de TensorFlow. Otra opción es restringir la operación para que se ejecute únicamente cuando el modo eager execution esté habilitado (`tf.executing_eagerly()`).

Ejemplo:

```python
def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
          output_collections=(), name=None):
  """My operation that adds two tensors with given coefficients.

  Args:
    tensor_in: `Tensor`, input tensor.
    other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
    my_param: `float`, coefficient for `tensor_in`.
    other_param: `float`, coefficient for `other_tensor_in`.
    output_collections: `tuple` of `string`s, name of the collection to
                        collect result of this op.
    name: `string`, name of the operation.

  Returns:
    `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

  Example:
    >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
              output_collections=['MY_OPS'], name='add_t1t2')
    [2.3, 3.4]
  """
  with tf.name_scope(name or "my_op"):
    tensor_in = tf.convert_to_tensor(tensor_in)
    other_tensor_in = tf.convert_to_tensor(other_tensor_in)
    result = my_param * tensor_in + other_param * other_tensor_in
    tf.add_to_collection(output_collections, result)
    return result
```

Uso:

```python
output = my_op(t1, t2, my_param=0.5, other_param=0.6,
               output_collections=['MY_OPS'], name='add_t1t2')
```

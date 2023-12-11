# Guia de estilo de código para o TensorFlow

## Estilo Python

Segue o [guia de estilo PEP 8 Python](https://www.python.org/dev/peps/pep-0008/), exceto que o TensorFlow usa 2 espaços em vez de 4. Siga o [Guia de estilo Python do Google](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) e use o [pylint](https://www.pylint.org/) para verificar suas alterações no Python.

### pilar

Para instalar o `pylint`:

```bash
$ pip install pylint
```

Para verificar um arquivo usando o `pylint` no diretório raiz do código-fonte do TensorFlow:

```bash
$ pylint --rcfile=tensorflow/tools/ci_build/pylintrc tensorflow/python/keras/losses.py
```

### Versões Python suportadas

Para versões do Python suportadas, consulte o [guia de instalação](https://www.tensorflow.org/install) do TensorFlow.

Consulte o [status de build contínuo](https://github.com/tensorflow/tensorflow/blob/master/README.md#continuous-build-status) do TensorFlow para builds oficiais e com suporte da comunidade.

## Estilo C++

As alterações no código C++ do TensorFlow devem estar em conformidade com o [Guia de estilo C++ do Google](https://google.github.io/styleguide/cppguide.html) e com os [detalhes de estilo específicos do TensorFlow](https://github.com/tensorflow/community/blob/master/governance/cpp-style.md). Use `clang-format` para verificar suas alterações em C/C++.

Para instalar no Ubuntu 16+:

```bash
$ apt-get install -y clang-format
```

Você pode verificar o formato de um arquivo C/C++ com:

```bash
$ clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
$ diff <my_cc_file> /tmp/my_cc_file.cc
```

## Outras linguagens

- [Guia de estilo Java do Google](https://google.github.io/styleguide/javaguide.html)
- [Guia de estilo JavaScript do Google](https://google.github.io/styleguide/jsguide.html)
- [Guia de estilo Shell do Google](https://google.github.io/styleguide/shell.xml)
- [Guia de estilo do Objective-C do Google](https://google.github.io/styleguide/objcguide.html)

## Convenções e usos especiais do TensorFlow

### Operações Python

Uma *operação* TensorFlow é uma função que, dados tensores de entrada, retorna tensores de saída (ou adiciona uma operação a um grafo ao construir grafos).

- O primeiro argumento deve consistir dos tensores, seguido por parâmetros básicos do Python. O último argumento é `name` com um valor padrão `None`.
- Argumentos de tensor devem ou ser um único tensor ou um iterável de tensores. Ou seja, um “Tensor ou lista de Tensores” é muito abrangente. Veja `assert_proper_iterable`.
- As operações que recebem tensores como argumentos devem chamar `convert_to_tensor` para converter entradas que não são de tensores em entradas de tensores se estiverem usando operações C++. Observe que os argumentos ainda são descritos como um objeto `Tensor` de um tipo específico na documentação.
- Cada operação Python deve ter um `name_scope`. Conforme visto abaixo, passe o nome da op como uma string.
- As operações devem conter um extenso comentário Python com declarações Args e Returns que explicam o tipo e o significado de cada valor. Possíveis formatos, dtypes ou postos devem ser especificados na descrição. Veja detalhes na documentação.
- Para maior usabilidade, inclua um exemplo de uso com entradas/saídas da operação na seção Exemplo.
- Evite fazer uso explícito de `tf.Tensor.eval` ou `tf.Session.run`. Por exemplo, para escrever uma lógica que dependa do valor do Tensor, use o fluxo de controle do TensorFlow. Como alternativa, restrinja a operação para seja executada apenas quando a execução antecipada (eager) estiver ativada (`tf.executing_eagerly()`).

Exemplo:

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

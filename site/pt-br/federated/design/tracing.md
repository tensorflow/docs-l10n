# Tracing

[TOC]

O tracing é o processo de construir uma [AST](compilation.md#ast) a partir de uma função do Python.

TODO: b/153500547 – Descrever os componentes individuais do sistema de tracing e adicionar os respectivos links.

## Tracing de uma computação federada

De forma geral, há três componentes ao fazer o tracing de uma computação federada.

### Encapsulamento dos argumentos

Internamente, uma computação do TFF tem apenas nenhum ou um argumento. Os argumentos fornecidos pelo decorador [federated_computation.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/federated_computation.py) descrevem a assinatura do tipo dos argumentos para a computação do TFF, que usa essas informações para determinar como encapsular os argumentos da função do Python em uma única [structure.Struct](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/common_libs/structure.py) (estrutura).

Observação: o uso de uma `Struct` como uma única estrutura de dados para representar tanto os `args` quanto os `kwargs` do Python é o motivo de a `Struct` aceitar campos com nome e sem nome.

Confira mais informações em [function_utils.create_argument_unpacking_fn](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/computation/function_utils.py).

### Tracing da função

Ao fazer o tracing de uma `federated_computation` (computação federada), a função do usuário é chamada usando-se [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) como uma substituição de cada argumento. `Value` tenta emular o comportamento do tipo de argumento original implementando os métodos dunder comuns do Python (por exemplo, `__getattr__`).

Quando há exatamente um argumento, o tracing é obtido:

1. Construindo um [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) associado a uma [building_blocks.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) com a assinatura de tipo apropriada para representar o argumento.

2. Invocando a função no `Value`, o que faz o runtime do Python invocar os métodos dunder implementados pelo `Value`, que traduz esses métodos dunder em uma construção da AST. Cada método dunder constrói uma AST e retorna um `Value` associado a essa AST.

Por exemplo:

```python
def foo(x):
  return x[0]
```

Aqui, o parâmetro da função é uma tupla e, no corpo da função, o elemento no índice 0 é selecionado. Isso invoca o método `__getitem__` do Python, que é sobrescrito no `Value`. No caso mais simples, a implementação de `Value.__getitem__` constrói uma [building_blocks.Selection](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) para representar a invocação de `__getitem__`e retorna um `Value` associado a essa nova `Selection`.

O tracing continua porque cada método dunder retorna um `Value`, eliminando toda operação no corpo da função, o que faz um dos métodos dunder sobrescritos ser invocado.

### Construção da AST

O resultado do tracing da função é encapsulado em um [building_blocks.Lambda](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py), cujos `parameter_name` e `parameter_type` mapeiam para a [building_block.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) criada para representar os argumentos encapsulados. O `Lambda` resultante é retornado como um objeto Python que representa totalmente a função do Python do usuário.

## Tracing de uma computação do TensorFlow

TODO: b/153500547 – Descrever o processo de fazer o tracing de uma computação do TensorFlow.

## Limpar as mensagens de erro das exceções durante o tracing

Em um dado momento na história do TFF, para fazer tracing da computação do usuário, era preciso passar por diversas funções encapsuladoras antes de chamar a função do usuário, o que trazia o efeito indesejado de gerar mensagens de erro como esta:

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

Era muito difícil encontrar a linha de código que continha o bug nesse traceback, o que fazia os usuários comunicarem esses problemas como bug do TFF e geralmente dificultava sua vida.

Atualmente, o TFF garante que essas pilhas de chamadas estejam livres de funções extras do TFF. Esse é o motivo para usar geradores no código de tracing do TFF, geralmente em padrões como este:

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

Com esse padrão, o código do usuário (`user_fn` acima) pode ser chamado no nível superior da pilha de chamadas, ao mesmo tempo em que permite que seus argumentos, saída e até mesmo contexto local dos threads sejam manipulados pelas funções encapsuladoras.

Algumas versões simples desse padrão podem ser substituídas por funções "before" (antes) e "after" (depois). Por exemplo, `foo` acima pode ser substituído por:

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

Deve-se optar por esse padrão em casos que não exijam estado compartilhado entre as partes "before" e "after". Porém, pode ser complicado expressar dessa forma casos mais complexos que envolvam estados complexos ou gerenciadores de contexto:

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

Neste último exemplo, fica bem menos claro qual código está sendo executado dentro de um contexto, e fica ainda menos claro quando mais estados são compartilhados entre as partes before e after.

Diversas outras soluções para o problema geral de "ocultar funções do TFF nas mensagens de erro de usuários" foram tentadas, incluindo capturar e gerar novamente exceções (isso falhou devido à incapacidade de criar uma exceção cuja pilha incluía somente o nível mais baixo do código de usuário, sem incluir o código que o chamou); capturar exceções e substituir o traceback por outro filtrado (o que é específico ao CPython e não é compatível com a linguagem Python); e substituir o manipulador de exceções (o que falhou porque `sys.excepthook` não é usado por `absltest` e é sobrescrito por outros frameworks). No fim, a inversão de controle baseada no gerador foi a que propiciou a melhor experiência para o usuário final ao custo de uma certa complexidade de implementação no TFF.

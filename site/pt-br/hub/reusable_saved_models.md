# SavedModels reutilizáveis

## Introdução

O TensorFlow Hub hospeda os SavedModels para o TensorFlow 2, entre outros ativos. Eles podem ser carregados de volta em um programa do Python por meio de `obj = hub.load(url)` [[saiba mais](tf2_saved_model)]. O `obj` retornado é o resultado de `tf.saved_model.load()` (confira o [guia sobre o SavedModel](https://www.tensorflow.org/guide/saved_model) do TensorFlow). Esse objeto pode ter atributos arbitrários que são tf.functions, tf.Variables (inicializadas com os valores pré-treinados), outros recursos e, recursivamente, mais objetos com esse.

Esta página descreve uma interface que pode ser implementada pelo `obj` carregado para ser *reutilizada* em um programa do Python no TensorFlow. Os SavedModels em conformidade com essa interface são chamados de *SavedModels reutilizáveis*.

Reutilizar significa criar um modelo maior usando `obj`, incluindo a capacidade de fazer ajustes finos. Fazer ajustes finos significa treinar ainda mais os pesos do `obj` carregado como parte do modelo adjacente. A função de perda e o otimizador são determinados pelo modelo adjacente; `obj` apenas define o mapeamento de atividades entrada-saída (o "passo para frente"), possivelmente incluindo técnicas como dropout ou normalização de lote.

**A equipe do TensorFlow Hub recomenda implementar a interface de SavedModel reutilizável** em todos os SavedModels que deverão ser reutilizados conforme explicado acima. Diversos utilitários da biblioteca `tensorflow_hub`, notadamente `hub.KerasLayer`, requerem SavedModels para implementá-la.

### Relação com SignatureDefs

Quanto às tf.functions e outros recursos do TF2, essa interface é separada das assinaturas de SavedModel, que estão disponíveis desde o TF1 e continuam sendo usadas no TF2 para inferência (como ao implantar SavedModels no TF Serving ou no TF Lite). As assinaturas para inferência não são expressivas o bastante para dar suporte aos ajustes finos, e [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) conta com uma [API do Python](https://www.tensorflow.org/tutorials/customization/performance) mais natural e expressiva para o modelo reutilizado.

### Relação com bibliotecas de criação de modelos

Um SavedModel reutilizável usa somente primitivos do TensorFlow 2, independentes de qualquer biblioteca específica de criação de modelos, como Keras ou Sonnet. Isso permite a reutilização entre as bibliotecas de criação de modelos, livres das dependências do código de criação de modelos original.

É necessário fazer uma certa adaptação para carregar SavedModels reutilizáveis ou salvá-los usando qualquer biblioteca de criação de modelos. Para o Keras, [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) fornece o carregamento, e o modo de salvamento integrado ao Keras no formato SavedModel foi reformulado para o TF2 com o objetivo de fornecer um superconjunto dessa interface (confira a [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md) de maio de 2019).

### Relação com "APIs comuns de SavedModel" para tarefas específicas

A definição da interface nesta página permite qualquer número e tipo de entradas e saídas. As [APIs comuns do SavedModel para o TF Hub](common_saved_model_apis/index.md) refinam essa interface geral com convenções de uso para tarefas específicas a fim de facilitar o intercambiamento de modelos.

## Definição da interface

### Atributos

Um SavedModel reutilizável é um SavedModel do TensorFlow 2 em que `obj = tf.saved_model.load(...)` retorna um objeto com os seguintes atributos:

- `__call__`. Obrigatório. Uma tf.function que implementa a computação do modelo (o "passo para frente"), sujeita à especificação abaixo.

- `variables`: uma lista de objetos tf.Variable, listando todas as variáveis usadas por qualquer possível invocação de `__call__`, incluindo variáveis treináveis e não treináveis.

    Essa lista pode ser omitida, se estiver vazia.

    Observação: convenientemente, esse nome coincide com o atributo sintetizado por `tf.saved_model.load(...)` ao carregar um SavedModel do TF1 para representar sua coleção de `GLOBAL_VARIABLES` (variáveis globais).

- `trainable_variables`: lista de objetos tf.Variable em que `v.trainable` é verdadeiro para todos os elementos. Essas variáveis precisam ser um subconjunto de `variables`. Essas são as variáveis a serem treinadas ao fazer os ajustes finos do objeto. O criador de SavedModel pode optar por omitir algumas variáveis que eram treináveis originalmente para indicar que elas não devem ser modificadas durante os ajustes finos.

    Essa lista pode ser omitida se estiver vazia, principalmente se o SavedModel não tive suporte a ajustes finos.

- `regularization_losses`: uma lista tf.functions, em que cada uma não recebe nenhuma entrada e retorna um único tensor float escalar. Para os ajustes finos, o usuário do SavedModel deve incluí-los como termos de regularização adicionais na perda (no caso mais simples, sem mais dimensionamento). Tipicamente, são usadas para representar regularizadores de pesos (por não receber entradas, essas tf.functions não podem expressar regularizadores de atividade).

    Essa lista pode ser omitida se estiver vazia, principalmente se o SavedModel não tive suporte a ajustes finos ou não quiser fazer regularização de pesos.

### Função `__call__`

Um `obj` SavedModel restaurado tem um atributo `obj.__call__` que é uma tf.function restaurada e permite que `obj` seja chamado da seguinte forma.

Sinopse (pseudocódigo):

```python
outputs = obj(inputs, trainable=..., **kwargs)
```

#### Argumentos

Os argumentos são os seguintes:

- Há um argumento posicional obrigatório com um lote de ativações de entrada do SavedModel. Seu tipo é um dos seguintes:

    - um único Tensor para uma única entrada
    - uma lista de Tensores para uma sequência ordenada de entradas sem nome
    - um dicionário (dict) de Tensores cujas chaves são um conjunto determinado de nomes de entradas

    (Versões futuras dessa interface poderão permitir aninhamentos mais gerais). O criador de SavedModel escolhe um desses argumentos, além dos dtypes e formatos do tensor. Quando for útil, algumas dimensões do formato devem ser indefinidas (notadamente o tamanho de lote).

- Pode haver um argumento palavra-chave opcional `training` que receba um booleano do Python, `True` ou `False`. O padrão é `False`. Se o modelo tiver suporte aos ajustes finos e se sua computação diferir entre os dois casos (por exemplo, dropout e normalização de lote), essa distinção será implementada com esse argumento. Caso contrário, o argumento poderá estar ausente.

    Não é obrigatório que `__call__` receba um argumento`training` com valor igual a um Tensor. Fica a critério do chamador usar `tf.cond()` se necessário para expedir entre eles.

- O criador de SavedModel pode optar por receber mais `kwargs` ou nomes específicos opcionais.

    - Para argumentos com valor igual a um Tensor, o criador de SavedModel define os dtypes e formatos admissíveis. `tf.function` recebe um valor padrão do Python em um argumento cujo tracing é feito com uma entrada tf.TensorSpec. Argumentos como esse podem ser usados para propiciar personalização de hiperparâmetros numéricos envolvidos em `__call__` (por exemplo, taxa de dropout).

    - Para argumentos com valor do Python, o criador de SavedModel define os valores admissíveis. Argumentos como esse podem ser usados como sinalizadores para fazer escolhas discretas na função com tracing (mas lembre-se da explosão combinacional de tracings).

A função `__call__` restaurada precisa fornecer tracings para todas as combinações de argumentos admissíveis. Mudar `training` entre `True` e `False` não deve alterar a admissibilidade de argumentos.

#### Resultado

As saídas (`outputs`) ao chamar `obj` podem ser:

- um único Tensor para uma única saída
- uma lista de Tensores para uma sequência ordenada de saídas sem nome
- um dicionário (dict) de Tensores cujas chaves são um conjunto determinado de nomes de saídas

(Versões futuras dessa interface poderão permitir aninhamentos mais gerais). O tipo de retorno pode variar dependendo dos kwargs com valor do Python. Isso permite que os sinalizadores gerem saídas extras. O criador de SavedModel define os formatos e dtypes de saída e sua dependência das entradas.

### Chamáveis com nome

Um SavedModel reutilizável pode fornecer diversas partes de modelo na forma descrita acima colocando-as em subobjetos com nome, por exemplo, `obj.foo`, `obj.bar` e assim por diante. Cada subobjeto conta com um método `__call__` e atributos subjacentes sobre as variáveis e etc. específicos à parte do modelo. No exemplo acima, haveria `obj.foo.__call__`, `obj.foo.variables` e assim por diante.

Observe que essa interface *não* abrange a estratégia de adicionar uma tf.function básica diretamente como `tf.foo`.

Espera-se que os usuários de SavedModels tratem somente um nível de aninhamento (`obj.bar`, mas não `obj.bar.baz`) (revisões futuras dessa interface poderão permitir um aninhamento mais profundo e poderão retirar o requisito de que o objeto de nível superior seja chamável).

## Considerações finais

### Relação com as APIs no processo

Este documento descreve uma interface de uma classe do Python que consiste de primitivos, como tf.function e tf.Variable, que sobrevivem a uma rodada por meio de serialização via `tf.saved_model.save()` e `tf.saved_model.load()`. Porém, a interface já estava presente no objeto original passado para `tf.saved_model.save()`. A adaptação dessa interface permite a troca de partes do modelo entre as APIs de criação de modelos dentro de um único programa do TensorFlow.

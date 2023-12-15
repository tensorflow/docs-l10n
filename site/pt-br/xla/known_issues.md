# Problemas conhecidos

A compilação com XLA pode melhorar muito o desempenho de seus programas, mas a interoperabilidade com o TensorFlow tem vários problemas conhecidos que devem ser considerados.

## `tf.Variable` em dispositivo diferente

*Mensagem de erro*: `INVALID_ARGUMENT: Trying to access resource <Variable> (defined @ <Loc>) located in device CPU:0 from device GPU:0`

O cluster XLA é executado em exatamente um dispositivo e não pode ler ou gravar numa `tf.Variable` localizada em dispositivo diferente. Geralmente, essa mensagem de erro indica que a variável não foi colocada no dispositivo correto. A mensagem de erro deve especificar precisamente a localização da variável ofensiva.

OBSERVAÇÃO: As `tf.Variable` do tipo `int32` são sempre colocadas num host e não podem ser colocadas numa GPU. Como solução alternativa, `int64` pode ser usado.

## Interconversão TensorArray TF/XLA não suportada

*Mensagem de erro*: `Support for TensorList crossing the XLA/TF boundary is not implemented`.

O XLA suporta `tf.TensorArray`. No entanto, a *interconversão* entre as representações TF e XLA ainda não está implementada. Este erro geralmente surge quando o `TensorArray` é usado dentro do bloco compilado, mas a derivada é obtida fora.

*Solução alternativa*: compile o escopo mais externo que está obtendo a derivada.

## Loops while do TensorFlow precisam ser limitados (ou ter a retropropagação desativada)

*Mensagem de erro*: `XLA compilation requires a fixed tensor list size. Set the max number of elements. This could also happen if you're using a TensorArray in a while loop that does not have its maximum_iteration set, you can fix this by setting maximum_iteration to a suitable value`.

[Loops](https://www.tensorflow.org/api_docs/python/tf/while_loop) while do TF criados usando `tf.while_loop` suportam retropropagação acumulando todos os resultados intermediários num `TensorArray`, mas XLA suporta apenas objetos `TensorArray` limitados.

*Solução alternativa*: todos os loops while compilados precisam ter o parâmetro `maximum_iterations` definido como um valor constante conhecido em tempo de compilação ou ter a retropropagação desabilitada usando `back_prop=False`.

## `tf.TensorArray` dinâmico não suportado

Gravações para `tf.TensorArray(..., dynamic_size=True)` não são compiláveis ​​com XLA, pois tais gravações exigem um número desconhecido de realocações quando o array exceder o limite original.

*Solução alternativa*: forneça um limite estaticamente conhecido para seus arrays.

## Geração de números aleatórios ignora a semente TF

Atualmente, o XLA ignora sementes TF para operações aleatórias. Isto afeta operações aleatórias de TF stateful, como `tf.random.normal` ou `tf.nn.dropout`. O XLA se comportará como se a compilação tivesse sido propagada com uma nova semente exclusiva a cada execução do mesmo processo (a primeira execução do processo sempre produzirá o mesmo resultado).

*Solução alternativa*: use [geradores de números aleatórios recomendados](https://www.tensorflow.org/guide/random_numbers#stateless_rngs), como `tf.random.stateless_uniform` ou `tf.random.Generator` diretamente.

## Entradas que precisam ser constantes e que são funções de variáveis ​​de indução não são suportadas

*Mensagem de erro*: `XLA compilation requires that operator arguments that represent shapes or dimensions be evaluated to concrete values at compile time. This error means that a shape or dimension argument could not be evaluated at compile time, usually because the value of the argument depends on a parameter to the computation, on a variable, or on a stateful operation such as a random number generator`

O XLA exige que certos valores sejam conhecidos em tempo de compilação, como o eixo de redução de uma operação reduce ou dimensões de transposição. Considere o caso quando, por exemplo, o eixo de redução é definido como uma função de uma variável de indução de `tf.range`: resolvê-lo estaticamente não é possível sem desenrolar todo o loop, o que pode não ser o que o usuário deseja.

*Solução alternativa*: desenrole os loops, por exemplo, convertendo `tf.range` para `range` do Python.

OBSERVAÇÃO: A mensagem de erro acima não é exclusiva deste problema e pode surgir devido a outras limitações ou bugs.

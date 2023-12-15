# Aliasing em XLA

Este documento descreve a API de aliasing para o XLA: ao construir um programa XLA, você pode especificar o aliasing desejado entre os buffers de entrada e saída.

## Definindo aliasing em tempo de compilação

Por exemplo, considere um módulo HLO trivial que simplesmente adiciona `1` à sua entrada:

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

Este módulo alocará dois buffers de 4 bytes: um para a entrada `%p` e outro para a saída `%out`.

No entanto, muitas vezes é desejável realizar a atualização no local (por exemplo, se no front-end que gera a expressão a variável de entrada não estiver mais ativa após a computação, como no incremento `p++`).

Para realizar essa atualização com eficiência, você pode especificar o aliasing de entrada:

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

O formato especifica que toda a saída (marcada por `{}`) tem o alias do parâmetro de entrada `0`.

Veja a API [`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) ​​para especificar o aliasing programaticamente.

## Definindo aliasing em tempo de execução

O aliasing definido na etapa anterior é especificado durante a *compilação*. Durante a execução, você pode escolher se realmente deseja doar o buffer usando a API [`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h).

Os buffers de entrada para o programa são agrupados em [`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h), que por sua vez contém uma árvore de `MaybeOwningDeviceMemory`. Se a memória for especificada como *proprietária* (a propriedade do buffer é passada para o runtime do XLA), o buffer é doado e a atualização é executada no local, conforme solicitado pela API de aliasing em tempo de compilação.

Se, no entanto, o buffer que tem aliasing em tempo de compilação *não* for doado em tempo de execução, *a proteção contra cópia* entra em ação: um buffer de saída extra `O` é alocado e o conteúdo do buffer de entrada `P` que deveria ter aliasing é copiado para `O` (de forma tão eficaz que o programa pode ser executado como se o buffer `O` tivesse sido doado em tempo de execução).

## Interoperação com o front-end

### TF/XLA

Em clusters do programa TensorFlow compilado com XLA, todas as atualizações de variáveis ​​de características recebem alias em tempo de compilação (o alias em tempo de execução depende se alguma outra coisa contém uma referência ao tensor de variável de característica).

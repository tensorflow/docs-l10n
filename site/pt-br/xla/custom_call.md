# Chamadas personalizadas XLA

Este documento descreve como escrever e usar "chamadas personalizadas" XLA. As chamadas personalizadas permitem invocar código escrito numa linguagem de programação como C++ ou CUDA a partir de um programa XLA.

Atenção: chamadas personalizadas são um recurso de baixo nível para usuários avançados. É fácil quebrar seu programa de uma forma difícil de depurar (e até mesmo difícil de perceber) usando chamadas personalizadas. Você não deve usar chamadas personalizadas, a menos que esteja preparado para depurar o XLA quando algo der errado, e você deve esperar relativamente menos assistência dos desenvolvedores do XLA se tiver problemas.

Atenção: a API/ABI de chamadas personalizadas não está estável no momento. Não pretendemos alterá-la muito, mas ela pode mudar. Algumas possíveis alterações futuras estão descritas abaixo.

## Chamada personalizada na CPU

Você pode criar uma instrução HLO que represente uma chamada personalizada por meio da API cliente do XLA. Isto ainda não é disponível via TensorFlow no momento em que este artigo foi escrito.

Por exemplo, o código a seguir usa uma chamada personalizada para calcular `A[i] = B[i % 128]+ C[i]` na CPU. (É claro que você poderia - e deveria! - fazer isso com o HLO no uso normal.)

```c++
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla::F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                      /*shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}));
}

void do_custom_call(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* in0 = reinterpret_cast<const float*>(in[0]);
  const float* in1 = reinterpret_cast<const float*>(in[1]);
  for (int i = 0; i < 2048; ++i) {
    out_buf[i] = in0[i % 128] + in1[i];
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "Host");
```

Observe que a função `do_custom_call` precisa saber as dimensões dos buffers sobre os quais opera. Neste exemplo, fixamos os tamanhos em 128 e 2048. Se você não quiser fazer isso, pode passar as dimensões como parâmetros para a chamada.

## Chamada personalizada na GPU

A estrutura de uma chamada personalizada na GPU é um pouco diferente dela na CPU. Aqui está um exemplo CUDA que faz a mesma computação `A[i] = B[i % 128] + C[i]` que o código na CPU acima.

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
```

Observe primeiro que a função de chamada personalizada da GPU *ainda é uma função executada na CPU*. Nossa função na CPU `do_custom_call` é responsável por enfileirar o trabalho na GPU. Aqui ele lança um kernel CUDA, mas também poderia fazer outra coisa, como chamar cublas.

`buffers` é um array de ponteiros que reside no host e cada elemento que contém aponta para a memória do dispositivo (ou seja, GPU). Os parâmetros vêm primeiro, seguidos pelo valor de saída. Isso é bem diferente da convenção de uma chamada na CPU, que possui dois parâmetros, `ins` e `out`. A principal razão pela qual divergimos é tornar possível lidar com entradas/saídas em forma de tupla de forma eficiente; veja a seção abaixo.

Como no exemplo da CPU, fixamos os tamanhos dos buffers de entrada e saída em nossa chamada personalizada. No entanto, ao contrário do caso da CPU, passar os tamanhos dos buffers como operandos para a chamada personalizada não funcionaria bem. Normalmente precisamos dos tamanhos de buffer disponíveis na CPU; por exemplo, ao lançar um kernel, precisamos saber as dimensões do bloco/grid a ser usado. Mas se passássemos os tamanhos dos buffers como operandos para nossa chamada personalizada, seus valores residiriam na memória da GPU. Teríamos então que fazer um memcpy síncrono do dispositivo para o host no início de nossa operação apenas para ler os tamanhos (e isto consome muitos recursos).

Para permitir que você contorne isso, fornecemos o parâmetro `opaque`. Você pode defini-lo como uma sequência arbitrária de bytes ao criar a chamada personalizada:

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
                opaque);
```

Como `xla::Shape` tem uma representação de buffer de protocolo, você pode armazenar esse proto serializado dentro de `opaque` e desserializá-lo em sua chamada personalizada de GPU. Observe, entretanto, que embora `xla::ShapeProto` não mude com frequência, ele às vezes *muda*. Verifique o log do git para ver como ele mudou no passado.

## Sinalizando um erro.

Se sua chamada personalizada encontrar um erro, você poderá sinalizar o erro para o runtime do XLA (em vez de, por exemplo, travar ou retornar texto sem sentido nos buffers de saída) usando a seguinte assinatura para sua função na CPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status);
```

... e na GPU:

```c++
#include "tensorflow/compiler/xla/service/custom_call_status.h"

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, xla::XlaCustomCallStatus* status);
```

Você pode sinalizar uma falha usando `XlaCustomCallStatusSetFailure`, por exemplo:

```c++
void do_custom_call(void* out, const void** in, XlaCustomCallStatus* status) {
  // ... do some work.

  if (bad_condition) {
    char* error_message = "An error occurred";
    XlaCustomCallStatusSetFailure(status, error_message, strlen(error_message));
    return;
  }

  // ... continue.
}
```

Você também pode usar `XlaCustomCallStatusSetSuccess` para indicar sucesso, mas o `XlaCustomCallStatus` está em estado de sucesso por padrão, portanto, ignorá-lo completamente também indicará sucesso.

Ao usar funções de chamada personalizadas com esta assinatura, você deve criar a operação `custom-call` correspondente com o conjunto apropriado de versões de API, por exemplo:

```c++
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/xla::ShapeUtil::MakeShape(F32, {2048}),
                opaque, /*has_side_effect=*/false,
                /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                /*api_version=*/API_VERSION_STATUS_RETURNING);
```

OBSERVAÇÃO: No futuro, todos os clientes serão obrigados a migrar suas funções de chamada personalizadas para a nova versão da API e a antiga será obsoleta. Para chamadas personalizadas que não podem falhar, basta adicionar o novo parâmetro `XlaCustomCallStatus*` e ignorá-la.

Em caso de falha, nenhuma das saídas de chamada customizadas será usada; o tempo de execução do XLA encerrará a computação. Não é possível, para uma computação HLO, se recuperar do erro (por exemplo, capturando-o e tratando-o).

## Passando tuplas para chamadas personalizadas

Considere a seguinte chamada personalizada.

```c++
using xla::ShapeUtil;
using xla::F32;
Shape p0_shape = ShapeUtil::MakeTuple({
    ShapeUtil::MakeShape(F32, {32}),
    ShapeUtil::MakeTuple({
        ShapeUtil::MakeShape(F32, {64}),
        ShapeUtil::MakeShape(F32, {128}),
    }),
    ShapeUtil::MakeShape(F32, {256}),
});
xla::XlaOp p0 = xla::Parameter(0, p0_shape, "p0");

Shape out_shape = ShapeUtil::MakeTuple({
  ShapeUtil::MakeShape(F32, {512}),
  ShapeUtil::MakeShape(F32, {1024}),
});
xla::CustomCall(&b, "do_custom_call", /*operands=*/{p0}, out_shape);
```

Tanto na CPU quanto na GPU, uma tupla é representada na memória como um array de ponteiros. No pseudocódigo C++, o parâmetro 0 acima é apresentado da seguinte forma.

```c++
// In-memory layout of parameter 0 from custom-call above.  True on both CPU
// and GPU.
float* subbuf0 = new float[32];
float* subbuf1 = new float[64];
float* subbuf2 = new float[128]
float* subbuf3 = new float[256];

void* subtuple = new void*[2];
(*subtuple)[0] = subbuf1;
(*subtuple)[1] = subbuf2;

void* p0 = new void*[3];
(*p0)[0] = subbuf0;
(*p0)[1] = subtuple;
(*p0)[2] = subbuf3;
```

Embora a representação de tuplas na memória seja a mesma na CPU e na GPU, elas são tratadas de maneira diferente nas convenções de chamadas personalizadas da CPU e da GPU.

### Saídas de tupla como buffers temporários

As entradas de tupla para chamadas personalizadas são uma conveniência, mas não são estritamente necessárias. Se não tivéssemos suporte a entradas de tuplas para chamadas personalizadas, você sempre poderia descompactar as tuplas usando get-tuple-element antes de passá-las para a chamada personalizada.

Por outro lado, as *saídas* de tupla permitem que você faça coisas que de outra forma não conseguiria.

A razão óbvia para ter saídas de tupla é que é assim que uma chamada personalizada (ou qualquer outra operação XLA) retorna vários arrays independentes.

Mas, de forma menos óbvia, uma saída de tupla também é uma maneira de fornecer memória temporária à sua chamada personalizada. Sim, uma *saída* pode representar um buffer temporário. Considere que um buffer de saída tem a propriedade de que a operação pode gravar nele e pode ler a partir dele depois de ter sido gravado. Isso é exatamente o que você espera de um buffer temporário.

No exemplo acima, suponha que quiséssemos usar `F32[1024]` como buffer temporário. Então escreveríamos o HLO como acima e simplesmente nunca leríamos o índice 1 da tupla na saída da chamada personalizada.

### Tuplas em chamadas personalizadas de CPU

No código da CPU, temos uma função `do_custom_call(const void** ins, void* out)`. `ins` é um array com apenas um elemento, que aponta para `param0`. Os sub-buffers de `param0` são acessíveis desreferenciando esse ponteiro, e os sub-buffers de `output_tuple` são acessíveis desreferenciando `out`.

### Tuplas em chamadas personalizadas de GPU

No código da GPU, temos uma função `do_custom_call(..., void** buffers, ...)`. Neste caso, `buffers` é um array de host de *seis* ponteiros de dispositivo, um para cada buffer-folha na entrada/saída. Para gerar a lista simples, iteramos sobre os parâmetros e a saída, e para cada um fazemos uma travessia de pré-ordem de seu formato. Concretamente:

```c++
// Layout of `buffers` parameter to GPU custom call function for custom-call
// above.
buffers[0] == subbuf0
buffers[1] == subbuf1
buffers[2] == subbuf2
buffers[3] == subbuf3
buffers[4] == output_subbuf0
buffers[5] == output_subbuf1
```

# Especificação da quantização de 8 bits do TensorFlow Lite

Este documento descreve a especificação para o esquema de quantização de 8 its do TensorFlow Lite. O objetivo dele é ajudar os desenvolvedores de hardware a oferecer suporte para a inferência com modelos do TensorFlow Lite quantizados.

## Resumo da especificação

Estamos fornecendo uma especificação e só podemos dar algumas garantias de comportamento se a especificação for seguida. Também entendemos que cada hardware pode ter preferências e restrições que causam leves desvios ao implementar a especificação, resultando em implementações de bits não exatos. Isso pode ser aceitável na maioria dos casos (e oferecemos um pacote de testes que, de acordo com nosso conhecimento, incluem tolerâncias por operação que reunimos de vários modelos), a natureza do aprendizado de máquina (e do aprendizado profundo no caso mais comum) torna impossível dar quaisquer garantias sólidas.

A quantização de 8 bits aproxima os valores de ponto flutuante usando a seguinte fórmula.

$$real_value = (int8_value - zero_point) \times scale$$

Os pesos por eixo (ou seja, por canal nas operações Conv) ou por tensor são representados por dois valores complementares `int8` no intervalo `[-127, 127]` com um ponto zero igual a 0. As ativações/entradas por tensor são representadas por dois valores complementares `int8` no intervalo `[-128, 127]`, com um ponto zero no intervalo `[-128, 127]`.

Há outras exceções para operações específicas documentadas abaixo.

Observação: no passado, nossas ferramentas de quantização usavam a quantização `uint8` assimétrica por tensor. As novas ferramentas, os kernels de referência e os kernels otimizados para a quantização de 8 bits usarão esta especificação.

## Números inteiros assinados x não assinados

A quantização do TensorFlow Lite priorizará principalmente as ferramentas e os kernels para a quantização `int8` de 8 bits. Isso é para a conveniência de a quantização simétrica ser representada pelo ponto zero igual a 0. Além disso, vários back-end têm otimizações adicionais para a acumulação `int8xint8`.

## Por eixo x por tensor

A quantização por tensor significa que haverá uma escala e/ou ponto zero para cada tensor. A quantização por eixo significa que haverá uma escala e/ou `zero_point` por fatia em `quantized_dimension`. A dimensão quantizada especifica a dimensão do formato do tensor a que as escalas e os pontos zeros correspondem. Por exemplo, um tensor `t`, de `dims=[4, 3, 2, 1]` com os parâmetros de quantização: `scale=[1.0, 2.0, 3.0]`, `zero_point=[1, 2, 3]`, `quantization_dimension=1` será quantizado em toda a segunda dimensão de `t`:

```
t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3
```

Geralmente, a `quantized_dimension` é o `output_channel` dos pesos das convoluções, mas, em teoria, pode ser que a dimensão corresponda a cada produto escalar na implementação do kernel, permitindo maior granularidade da quantização sem implicações no desempenho. Isso leva a grandes melhorias na exatidão.

O TFLite oferece o suporte por eixo a um número crescente de operações. No momento da publicação deste documento, há suporte para Conv2d e DepthwiseConv2d.

## Simétrico x assimétrico

As ativações são assimétricas: elas podem ter o ponto zero em qualquer lugar no intervalo `[-128, 127]` do  `int8` assinado. Várias ativações têm natureza assimétrica e o ponto zero é uma maneira relativamente barata de conseguir mais um bit binário de precisão de forma eficiente. Como as ativações só são multiplicadas por pesos constantes, o valor de ponto zero constante pode ser bastante otimizado.

Os pesos são assimétricos: eles são forçados a ter o ponto zero igual a 0. Os valores dos pesos são multiplicados pelos valores de ativação e entrada dinâmica. Isso significa que há o custo de runtime inevitável de multiplicar o ponto zero do peso com o valor de ativação. Ao impor o ponto zero como 0, podemos evitar esse custo.

A explicação da matemática: isso é semelhante à seção 2.3 em [arXiv:1712.05877](https://arxiv.org/abs/1712.05877), exceto que a diferença permitida para os valores escalares são por eixo. Isso é prontamente generalizado da seguinte maneira:

$A$ é uma matriz $m \times n$ das ativações quantizadas. <br> $B$ é uma matriz $n \times p$ dos pesos quantizados. <br> Considere multiplicar a linha $j$th de $A$, $a_j$ pela coluna $k$th de $B$, $b_k$, ambos de comprimento $n$. Os valores de números inteiros quantizados e de pontos zeros são $q_a$, $z_a$ e $q_b$, $z_b$, respectivamente.

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} = \sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) = \sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b - \sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

<!-- Don't change these `\\(` `\\)` to `$`. mathjax fails here with `$`-->

O termo \(\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)}\) é inevitável, já que fornece o valor escalar dos valores da entrada e do peso.

Os termos $$\sum_{i=0}^{n} q_{b}^{(i)} z_a$$ and $$\sum_{i=0}^{n} z_a z_b$$ são compostos de constantes que permanecem iguais por invocação de inferência e, por isso, podem ser pré-calculadas.

O termo \(\sum_{i=0}^{n} q_{a}^{(i)} z_b\) precisa ser computado a cada inferência, porque a ativação muda a cada inferência. Ao impor que os pesos sejam simétricos, podemos remover o custo desse termo.

## Especificações de operadores quantizados int8

Descrevemos abaixo os requisitos de quantização para nossos kernels tflite int8:

```
ADD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

AVERAGE_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONCATENATION
  Input ...:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 0)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

DEPTHWISE_CONV_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-axis (dim = 3)
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-axis
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

FULLY_CONNECTED
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1 (Weight):
    data_type  : int8
    range      : [-127, 127]
    granularity: per-tensor
    restriction: zero_point = 0
  Input 2 (Bias):
    data_type  : int32
    range      : [int32_min, int32_max]
    granularity: per-tensor
    restriction: (scale, zero_point) = (input0_scale * input1_scale[...], 0)
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

L2_NORMALIZATION
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

LOGISTIC
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

MAX_POOL_2D
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MUL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

RESHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

RESIZE_BILINEAR
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 256.0, -128)

SPACE_TO_DEPTH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TANH
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (1.0 / 128.0, 0)

PAD
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GATHER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

BATCH_TO_SPACE_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

SPACE_TO_BATCH_ND
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

TRANSPOSE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

MEAN
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUB
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SQUEEZE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LOG_SOFTMAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
    restriction: (scale, zero_point) = (16.0 / 256.0, 127)

MAXIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

ARG_MAX
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

MINIMUM
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

LESS
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

PADV2
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

GREATER
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

GREATER_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

LESS_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SLICE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  restriction: Input and outputs must all have same scale/zero_point

EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

NOT_EQUAL
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Input 1:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

SHAPE
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor

QUANTIZE (Requantization)
  Input 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
  Output 0:
    data_type  : int8
    range      : [-128, 127]
    granularity: per-tensor
```

## Referências

[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

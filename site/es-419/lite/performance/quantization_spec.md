# Especificación de la cuantización de 8 bits de TensorFlow Lite

El siguiente documento esboza la especificación para el esquema de cuantización de 8 bits de TensorFlow Lite. Con ello se pretende ayudar a los desarrolladores de hardware a proporcionar soporte de hardware para la inferencia con modelos cuantificados de TensorFlow Lite.

## Resumen de especificaciones

Facilitamos una especificación y sólo podemos ofrecer algunas garantías de comportamiento si se sigue la especificación. También entendemos que los distintos tipos de hardware pueden tener preferencias y restricciones que pueden causar ligeras desviaciones al implementar la especificación que resulten en implementaciones que no sean exactas en cuanto a bits. Aunque eso puede ser aceptable en la mayoría de los casos (y vamos a dar un conjunto de pruebas que, hasta donde sabemos, incluyen tolerancias por operación que hemos recopilado de varios modelos), la naturaleza del aprendizaje automático (y del aprendizaje profundo en el caso más común) hace que sea imposible dar garantías firmes.

La cuantización de 8 bits aproxima los valores de punto flotante usando la siguiente fórmula.

$$real_value = (int8_value - zero_point) \times scale$$

Las ponderaciones por eje (también llamadas "por canal" en las ops Conv) o por tensor se representan mediante valores de complemento a dos de `int8` en el intervalo `[-127, 127]` con un punto cero igual a 0. Las activaciones/entradas por tensor están representadas por valores de complemento a dos de `int8` en el rango `[-128, 127]`, con un punto cero en el rango `[-128, 127]`.

Hay otras excepciones para operaciones concretas que se documentan a continuación.

Nota: En el pasado, nuestras herramientas de cuantización usaban una cuantización por tensor, asimétrica, `uint8`. Las nuevas herramientas, kernels de referencia y kernels optimizados para la cuantización de 8 bits usarán esta especificación.

## Entero con signo vs entero sin signo

La cuantización de TensorFlow Lite priorizará principalmente herramientas y kernels para la cuantización `int8` para 8 bits. Esto es por la conveniencia de la cuantización simétrica que se representa por punto cero igual a 0. Además, muchos backends tienen optimizaciones adicionales para la acumulación de `int8xint8`.

## Por eje vs por tensor

La cuantización por tensor significa que habrá una escala y/o punto cero por tensor entero. La cuantización por eje significa que habrá una escala y/o `zero_point` por sección en la `quantized_dimension`. La dimensión cuantizada especifica la dimensión de la forma del tensor a la que corresponden las escalas y los puntos cero. Por ejemplo, un tensor `t`, con `dims=[4, 3, 2, 1]` con parámetros de cuantización `scale=[1.0, 2.0, 3.0]`, `zero_point=[1, 2, 3]`, `quantization_dimension=1` se cuantizará en la segunda dimensión de `t`:

```
t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
t[:, 1, :, :] will have scale[1]=2.0, zero_point[1]=2
t[:, 2, :, :] will have scale[2]=3.0, zero_point[2]=3
```

A menudo, la `quantized_dimension` es el `output_channel` de las ponderaciones de las convoluciones, pero en teoría puede ser la dimensión que corresponde a cada producto punto en la implementación del kernel, lo que permite una mayor granularidad de la cuantización sin implicaciones para el rendimiento. Esto supone grandes mejoras en la precisión.

TFLite tiene soporte por eje para un número creciente de operaciones. Al momento de redactar este documento, existe soporte para Conv2d y DepthwiseConv2d.

## Simétrico vs asimétrico

Las activaciones son asimétricas: pueden tener su punto cero en cualquier lugar dentro del rango de `int8` con signo `[-128, 127]`. Muchas activaciones son asimétricas por naturaleza y un punto cero es una forma relativamente barata de conseguir efectivamente hasta un bit binario extra de precisión. Dado que las activaciones sólo se multiplican por ponderaciones constantes, el valor constante del punto cero puede optimizarse bastante.

Las ponderaciones son simétricas: obligadas a tener el punto cero igual a 0. Los valores de ponderación se multiplican por los valores dinámicos de entrada y de activación. Esto significa que existe un costo de runtime inevitable de multiplicar el punto cero de la ponderación por el valor de activación. Al imponer que el punto cero sea 0 podemos evitar este costo.

Explicación matemática: es similar a la sección 2.3 de [arXiv:1712.05877](https://arxiv.org/abs/1712.05877), salvo por la diferencia de que permitimos que los valores de escala sean por eje. Esto se generaliza fácilmente, como sigue:

$A$ es una matriz de $m \veces n$ de activaciones cuantizadas. <br> $B$ es una matriz de $n \veces p$ de ponderaciones cuantizadas. <br> Consideremos multiplicar la fila $j$-ésima de $A$, $a_j$ por la columna $k$-ésima de $B$, $b_k$, ambas de longitud $n$. Los valores enteros cuantizados y los valores de los puntos cero son $q_a$, $z_a$ y $q_b$, $z_b$ respectivamente.

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} = \sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) = \sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b - \sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

<!-- Don't change these `\\(` `\\)` to `$`. mathjax fails here with `$`-->

El término \(\sum_{i=0}^{n} q_{a}^(i)} q_{b}^{(i)}\) no puede evitarse, ya que está realizando el producto escalar del valor de entrada y el valor de ponderación.

Los términos $$\sum_{i=0}^{n} q_{b}^(i)} z_a$$ y $$\sum_{i=0}^{n} z_a z_b$$ están formados por constantes que siguen siendo las mismas por cada invocación de inferencia, por lo que pueden calcularse previamente.

El término \(\sum_{i=0}^{n} q_{a}^(i)} z_b\) debe calcularse en cada inferencia, ya que la activación cambia en cada una. Al obligar a que las ponderaciones sean simétricas podemos eliminar el costo de este término.

## Especificaciones del operador cuantizado int8

Aquí describimos los requisitos de cuantización de nuestros kernels tflite int8:

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

## Referencias

[arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

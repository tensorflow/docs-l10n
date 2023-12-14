# Semántica de difusión

En este documento se describe cómo funciona la semántica de difusión en XLA.

## ¿Qué es la difusión?

La difusión es el proceso por el cual se consigue que arreglos con distintas formas tengan formas compatibles para las operaciones aritméticas. La terminología se toma de [difusión](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) Numpy.

Quizá se requiera la difusión para operaciones entre arreglos multidimensionales de diferentes rangos, o entre arreglos multidimensionales con formas diferentes pero compatibles. Pensemos en la suma `X+v` donde `X` es una matriz (un arreglo de rango 2) y `v` es un vector (un arreglo de rango 1). Para sumar los elementos, XLA debe "difundir" el vector `v` al mismo rango que la matriz `X`, para lo que debe replicar `v` un cierto número de veces. La longitud del vector debe coincidir con al menos una de las dimensiones de la matriz.

Por ejemplo:

```
|1 2 3| + |7 8 9|
|4 5 6|
```

Las dimensiones de la matriz son (2,3), las del vector son (3). Se difunde el vector al replicarlo sobre las filas para obtener lo que sigue:

```
|1 2 3| + |7 8 9| = |8  10 12|
|4 5 6|   |7 8 9|   |11 13 15|
```

En Numpy, esto se llama [difusión](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Principios

El lenguaje de XLA es lo más estricto y explícito posible, por lo que evita características implícitas y "mágicas". Estas características pueden hacer que algunos cálculos sean un poco más fáciles de definir, pero a cambio se introducen más suposiciones en el código de usuario que serán difíciles de cambiar a largo plazo. De ser necesario, se pueden agregar características implícitas y mágicas en envoltorios a nivel de cliente.

En cuanto a la difusión, se requieren especificaciones de difusión explícitas sobre operaciones entre arreglos de diferentes rangos. Esto es diferente de Numpy, que infiere la especificación siempre que sea posible.

## Cómo difundir un arreglo de rango inferior a un arreglo de rango superior

Los *escalares* siempre pueden difundirse a través de arreglos sin una especificación explícita de las dimensiones de difusión. Una operación binaria por elementos entre un escalar y un arreglo implica la aplicación de la operación con el escalar para cada elemento del arreglo. Por ejemplo, agregar un escalar a una matriz significa producir una matriz en la que cada elemento es una suma del escalar con el elemento de la matriz de entrada correspondiente.

```
|1 2 3| + 7 = |8  9  10|
|4 5 6|       |11 12 13|
```

La mayoría de las necesidades de difusión se pueden capturar con una tupla de dimensiones en una operación binaria. Cuando las entradas a la operación tienen distintos rangos, esta tupla de difusión especifica qué dimensiones del arreglo de **rango superior** deben coincidir con el arreglo de **rango inferior**.

Pensemos en el ejemplo anterior, en lugar de agregar un escalar a una matriz (2,3), agregue un vector de dimensión (3) a una matriz de dimensiones (2,3). *Sin especificar la difusión, esta operación no es válida.* Para solicitar correctamente la suma matriz-vector, especifique que la dimensión de difusión sea (1), lo que significa que la dimensión del vector coincide con la dimensión 1 de la matriz. En 2D, si la dimensión 0 se trata como filas y la dimensión 1 como columnas, esto significa que cada elemento del vector se convierte en una columna de un tamaño que coincide con el número de filas de la matriz:

```
|7 8 9| ==> |7 8 9|
            |7 8 9|
```

Como ejemplo más complejo, piense en agregar un vector de 3 elementos (dimensión (3)) a una matriz de 3x3 (dimensiones (3,3)). Hay dos formas en que se puede ejecutar la difusión para este ejemplo:

(1) Se puede utilizar una dimensión de difusión de 1. Cada elemento del vector se convierte en una columna y el vector se duplica para cada fila de la matriz.

```
|7 8 9| ==> |7 8 9|
            |7 8 9|
            |7 8 9|
```

(2) Se puede utilizar una dimensión de difusión de 0. Cada elemento del vector se convierte en una fila y el vector se duplica para cada columna de la matriz.

```
 |7| ==> |7 7 7|
 |8|     |8 8 8|
 |9|     |9 9 9|
```

> Nota: Si se agrega una matriz de 2x3 a un vector de 3 elementos, una dimensión de difusión de 0 no es válida.

Las dimensiones de difusión pueden ser una tupla que describe cómo una forma de rango inferior se transmite a una forma de rango superior. Por ejemplo, para un cuboide de 2x3x4 y una matriz de 3x4, una tupla de difusión (1,2) significa hacer coincidir la matriz con las dimensiones 1 y 2 del cuboide.

Este tipo de difusión se usa en las operaciones binarias en `XlaBuilder`, si se ofrece el argumento `broadcast_dimensions`. Por ejemplo, consulte [XlaBuilder::Add](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.cc). En el código fuente de XLA, este tipo de difusión a veces se denomina difusión "InDim".

### Definición formal

El atributo de difusión permite hacer coincidir un arreglo de rango inferior con un arreglo de rango superior, al especificar qué dimensiones del arreglo de rango superior deben coincidir. Por ejemplo, para un arreglo con dimensiones MxNxPxQ, un vector con dimensión T se puede hacer coincidir de la siguiente manera:

```
          MxNxPxQ

dim 3:          T
dim 2:        T
dim 1:      T
dim 0:    T
```

En cada caso, T tiene que ser igual a la dimensión coincidente del arreglo de rango superior. Luego, los valores del vector se difunden desde la dimensión coincidente a todas las demás dimensiones.

Para hacer coincidir una matriz TxV con el arreglo MxNxPxQ, se utilizan un par de dimensiones de difusión:

```
          MxNxPxQ
dim 2,3:      T V
dim 1,2:    T V
dim 0,3:  T     V
etc...
```

El orden de las dimensiones en la tupla de difusión tiene que ser el orden en el que se espera que las dimensiones del arreglo de rango inferior coincidan con las dimensiones del arreglo de rango superior. El primer elemento de la tupla dice qué dimensión del arreglo de rango superior debe coincidir con la dimensión 0 del arreglo de rango inferior. El segundo elemento para la dimensión 1, y así sucesivamente. El orden de las dimensiones de difusión debe ser estrictamente creciente. Por ejemplo, en el caso anterior no se puede hacer coincidir V con N y T con P; Tampoco se puede hacer coincidir V con P y N.

## Cómo difundir arreglos de rango similar con dimensiones degeneradas

Un problema de difusión relacionado es la difusión de dos arreglos que tienen el mismo rango, pero diferentes tamaños de dimensión. De manera similar a las reglas de Numpy, esto solo es posible cuando los arreglos son *compatibles*. Dos arreglos se consideran compatibles cuando todas sus dimensiones son compatibles. Dos dimensiones son compatibles si se aplica lo siguiente:

- son iguales o
- uno de ellos es 1 (una dimensión "degenerada")

Cuando se encuentran dos arreglos compatibles, la forma resultante tiene el máximo entre las dos entradas en cada índice de dimensión.

Ejemplos:

1. (2,1) y (2,3) se difunden a (2,3).
2. (1,2,5) y (7,2,5) se difunden a (7,2,5)
3. (7,2,5) y (7,1,5) se difunden a (7,2,5)
4. (7,2,5) y (7,2,6) no son compatibles y no se pueden difundir.

Se presenta un caso especial, que también se admite, en el que cada uno de los arreglos de entrada tiene una dimensión degenerada en un índice diferente. En este caso, el resultado es una "operación externa": (2,1) y (1,3) se difunden a (2,3). Para obtener más ejemplos, consulte la [documentación de Numpy sobre difusión](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Composición de la difusión

La difusión de un arreglo de rango inferior a un arreglo de rango superior **y** la difusión con dimensiones degeneradas se pueden ejecutar en la misma operación binaria. Por ejemplo, un vector de tamaño 4 y una matriz de tamaño 1x2 se pueden sumar usando el valor de dimensiones de difusión de (0):

```
|1 2 3 4| + [5 6]    // [5 6] is a 1x2 matrix, not a vector.
```

Primero, el vector se difunde hasta el rango 2 (matriz) a través de las dimensiones de difusión. El valor único (0) en las dimensiones difundidas indica que la dimensión cero del vector coincide con la dimensión cero de la matriz. Esto produce una matriz de tamaño 4xM donde el valor M se elige para que coincida con el tamaño de dimensión correspondiente en el arreglo de 1x2. Por tanto, se produce una matriz de 4x2:

```
|1 1| + [5 6]
|2 2|
|3 3|
|4 4|
```

Luego, la "difusión de dimensiones degeneradas" transmite la dimensión cero de la matriz de 1x2 para que coincida con el tamaño de dimensión correspondiente del lado derecho:

```
|1 1| + |5 6|     |6  7|
|2 2| + |5 6|  =  |7  8|
|3 3| + |5 6|     |8  9|
|4 4| + |5 6|     |9 10|
```

Un ejemplo más complicado es una matriz de tamaño 1x2 agregada a un arreglo de tamaño 4x3x1 utilizando dimensiones de difusión de (1, 2). Primero, la matriz 1x2 se difunde hasta el rango 3 con ayuda de las dimensiones de difusión para producir un arreglo Mx1x2 intermedio donde el tamaño de dimensión M está determinado por el tamaño del operando más grande (el arreglo 4x3x1) que produce un arreglo 4x1x2 intermedio. La M está en la dimensión 0 (dimensión más a la izquierda) porque las dimensiones 1 y 2 están asignadas a las dimensiones de la matriz original de 1x2, ya que la dimensión de transmisión es (1, 2). Este arreglo intermedio se puede agregar a la matriz 4x3x1 mediante la difusión de dimensiones degeneradas para producir un resultado de arreglo 4x3x2.

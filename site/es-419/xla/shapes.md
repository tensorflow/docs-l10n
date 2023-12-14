# Formas y diseño

El protocolo `Shape` de XLA ([xla_data.proto](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto)) describe el rango, el tamaño y el tipo de datos de un arreglo de N dimensiones (se abrevia *arreglo*).

## Terminología, anotación y convenciones

- El rango de un arreglo es igual al número de dimensiones. El *verdadero rango* de un arreglo es el número de dimensiones que tienen un tamaño mayor que 1.

- Las dimensiones están numeradas de `0` a `N-1` en el caso de los arreglos de `N` dimensiones. Para facilitar la lectura, los números de dimensión son etiquetas arbitrarias. El orden de estos números de dimensión no implica un orden menor o mayor en particular en el diseño de la forma. El diseño está determinado por el protocolo `Layout`.

- Por convención, las dimensiones se enumeran en orden creciente de número de dimensión. Por ejemplo, para un arreglo tridimensional de tamaño `[A x B x C]`, la dimensión 0 tiene el tamaño `A`, la dimensión 1 tiene el tamaño `B` y la dimensión 2 tiene el tamaño `C`.

    Algunas utilidades de XLA también admiten indexación negativa, de manera similar a Python; la dimensión -1 es la última dimensión (equivalente a `N-1` para un arreglo de `N` dimensiones). Por ejemplo, para el arreglo tridimensional que se describió anteriormente, la dimensión -1 tiene tamaño `C`, la dimensión -2 tiene tamaño `B` y así sucesivamente.

- Los arreglos de dos, tres y cuatro dimensiones suelen tener letras específicas asociadas con las dimensiones. Por ejemplo, para un arreglo de 2D:

    - dimensión 0: `y`
    - dimensión 1: `x`

    Para un arreglo de 3D:

    - dimensión 0: `z`
    - dimensión 1: `y`
    - dimensión 2: `x`

    Para un arreglo de 4D:

    - dimensión 0: `p`
    - dimensión 1: `z`
    - dimensión 2: `y`
    - dimensión 3: `x`

- Las funciones en la API de XLA que toman dimensiones lo hacen en orden creciente de número de dimensión. Esto se corresponde con el orden que se usa al pasar dimensiones como `initializer_list`; p. ej

    `ShapeUtil::MakeShape(F32, {A, B, C, D})`

    Creará una forma cuyo arreglo de tamaño de dimensión conste de la secuencia `[A, B, C, D]`.

## Diseño

El protocolo `Layout` describe cómo se representa un arreglo en la memoria. El protocolo `Layout` incluye los siguientes campos:

```
message Layout {
  repeated int64 minor_to_major = 1;
  repeated int64 padded_dimensions = 2;
  optional PaddingValue padding_value = 3;
}
```

### Orden de dimensiones de menor a mayor

El único campo obligatorio es `minor_to_major`. Este campo describe el orden de menor a mayor de las dimensiones dentro de una forma. Los valores en `minor_to_major` son un orden de las dimensiones del arreglo (`0` a `N-1` para un arreglo de `N` dimensiones) siendo el primer valor la dimensión más menor hasta el último valor, que es la dimensión más importante. La dimensión más pequeña es la dimensión que cambia más rápidamente al recorrer los elementos del arreglo dispuestos en la memoria lineal.

Por ejemplo, piense en el siguiente arreglo 2D de tamaño `[2 x 3]`:

```
a b c
d e f
```

Aquí, la dimensión `0` es el tamaño 2 y la dimensión `1` es el tamaño 3. Si el campo `minor_to_major` en el diseño es `[0, 1]`, entonces la dimensión `0` es la dimensión más secundaria y la dimensión `1` es la dimensión más importante. Esto corresponde al siguiente diseño en la memoria lineal:

```
a d b e c f
```

Este orden de dimensiones de menor a mayor de `0` a `N-1` es similar a *la columna-mayor* (en el rango 2). Suponiendo un orden monótono de dimensiones, otro nombre que podemos usar para referirnos a este diseño en el código es simplemente "dim 0 es menor".

Por otro lado, si el campo `minor_to_major` en el diseño es `[1, 0]`, entonces el diseño en la memoria lineal es el siguiente:

```
a b c d e f
```

Un orden de dimensión de menor a mayor de `N-1` a `0` para un arreglo de `N` dimensiones es similar a una *fila-mayor* (en el rango 2). Suponiendo un orden monótono de dimensiones, otro nombre que podemos usar para referirnos a este diseño en el código es simplemente "dim 0 es mayor".

#### Orden predeterminado de menor a mayor

El diseño predeterminado para las formas recién creadas es "el orden de las dimensiones es de mayor a menor" (similar a fila-mayor en el rango 2).

### Amortiguado

El amortiguado se define en los campos opcionales `padded_dimensions` y `padding_value`. El campo `padded_dimensions` describe los tamaños (anchos) con los que se amortigua cada dimensión. Si está presente, la cantidad de elementos en `padded_dimensions` debe ser igual al rango de la forma.

Por ejemplo, a partir del arreglo `[2 x 3]` definido anteriormente, si `padded_dimensions` es `[3, 5]`, entonces la dimensión 0 se amortigua hasta un ancho de 3 y la dimensión 1 se amortigua hasta un ancho de 5. El diseño en la memoria lineal (asumiendo un valor de amortiguado de 0 y un diseño de columna-mayor) es el siguiente:

```
a d 0 b e 0 c f 0 0 0 0 0 0 0
```

Esto es equivalente al diseño del siguiente arreglo con el mismo orden de dimensiones de menor a mayor:

```
a b c 0 0
d e f 0 0
0 0 0 0 0
```

### Indexación en arreglos

La clase `IndexUtil` en [index_util.h](https://www.tensorflow.org/code/tensorflow/compiler/xla/index_util.h) ofrece utilidades para convertir entre índices multidimensionales e índices lineales a partir de una forma y un diseño. Los índices multidimensionales incluyen un índice `int64` para cada dimensión. Los índices lineales son un valor `int64` único que se indexa en el búfer que contiene el arreglo. Consulte `shape_util.h` y `layout_util.h` en el mismo directorio para conocer las utilidades que simplifican la creación y manipulación de formas y diseños.

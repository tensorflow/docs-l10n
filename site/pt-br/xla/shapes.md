# Formatos e Layout

O proto XLA `Shape` ([xla_data.proto](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto)) descreve o posto, o tamanho e o tipo de dados de um array N-dimensional (*array* em resumo).

## Terminologia, notação e convenções

- O posto de um array é igual ao número de dimensões. O *posto verdadeiro* de um array é o número de dimensões que possuem tamanho maior que 1.

- As dimensões são numeradas de `0` a `N-1` para um array `N` dimensional. Os números das dimensões são rótulos arbitrários por conveniência. A ordem desses números de dimensão não implica uma ordem menor/maior específica no layout do formato. O layout é determinado pelo protótipo `Layout`.

- Por convenção, as dimensões são listadas em ordem crescente de número de dimensão. Por exemplo, para um array tridimensional de tamanho `[A x B x C]`, a dimensão 0 tem tamanho `A`, a dimensão 1 tem tamanho `B` e a dimensão 2 tem tamanho `C`.

    Alguns utilitários no XLA também suportam indexação negativa, de forma semelhante ao Python; dimensão -1 é a última dimensão (equivalente a `N-1` para um array `N` dimensional). Por exemplo, para o array tridimensional descrito acima, a dimensão -1 tem tamanho `C`, a dimensão -2 tem tamanho `B` e assim por diante.

- Arrays bidimensionais, tridimensionais e tetradimensionais geralmente têm letras específicas associadas às dimensões. Por exemplo, para um array 2D:

    - dimensão 0: `y`
    - dimensão 1: `x`

    Para um array 3D:

    - dimensão 0: `z`
    - dimensão 1: `y`
    - dimensão 2: `x`

    Para um array 4D:

    - dimensão 0: `p`
    - dimensão 1: `z`
    - dimensão 2: `y`
    - dimensão 3: `x`

- As funções na API XLA que recebem dimensões fazem isso em ordem crescente do número da dimensão. Isto corresponde à ordem usada ao passar dimensões como um `initializer_list`; por exemplo

    `ShapeUtil::MakeShape(F32, {A, B, C, D})`

    Irá criar um formato cujo array de dimensão que consiste na sequência `[A, B, C, D]`.

## Layout

O proto `Layout` descreve como um array é representado na memória. O proto `Layout` inclui os seguintes campos:

```
message Layout {
  repeated int64 minor_to_major = 1;
  repeated int64 padded_dimensions = 2;
  optional PaddingValue padding_value = 3;
}
```

### Ordenação de dimensões menor para maior

O único campo obrigatório é `minor_to_major` . Este campo descreve a ordem menor para maior das dimensões dentro de um formato. Os valores em `minor_to_major` são uma ordenação das dimensões do array ( `0` a `N-1` para um array `N` dimensional), com o primeiro valor sendo a dimensão menor até o último valor, que é a dimensão maior. A dimensão menor é a dimensão que muda mais rapidamente ao percorrer os elementos do array dispostos na memória linear.

Por exemplo, considere o seguinte array 2D de tamanho `[2 x 3]`:

```
a b c
d e f
```

Aqui a dimensão `0` é o tamanho 2 e a dimensão `1` é o tamanho 3. Se o campo `minor_to_major` no layout for `[0, 1]`, então a dimensão `0` é a dimensão menor e a dimensão `1` é a dimensão maior. Isto corresponde ao seguinte layout na memória linear:

```
a d b e c f
```

Esta ordem de dimensão menor para maior de `0` a `N-1` é semelhante à *coluna maior* (no posto 2). Assumindo uma ordenação monotônica de dimensões, outro nome que podemos usar para nos referir a esse layout no código é simplesmente “dim 0 é menor”.

Por outro lado, se o campo `minor_to_major` no layout for `[1, 0]` então o layout na memória linear é:

```
a b c d e f
```

Uma ordem de dimensão menor para maior de `N-1` até `0` para um array `N` dimensional é semelhante ao *row-major* (no posto 2). Assumindo uma ordenação monotônica de dimensões, outro nome que podemos usar para nos referir a esse layout no código é simplesmente “dim 0 é maior”.

#### Ordenação menor para maior padrão

O layout padrão para Shapes recém-criados é "a ordem das dimensões é maior para menor" (similar ao row-major nanoposto 2).

### Preenchimento

O preenchimento (padding) é definido nos campos opcionais `padded_dimensions` e `padding_value`. O campo `padded_dimensions` descreve os tamanhos (larguras) para os quais cada dimensão é preenchida. Se presente, o número de elementos em `padded_dimensions` deve ser igual ao posto do formato.

Por exemplo, dado o array `[2 x 3]` definido acima, se `padded_dimensions` for `[3, 5]`, então a dimensão 0 será preenchida com uma largura de 3 e a dimensão 1 será preenchida com uma largura 5. O layout na memória linear (assumindo um valor de preenchimento de 0 e layout de coluna principal) é:

```
a d 0 b e 0 c f 0 0 0 0 0 0 0
```

Isto é equivalente ao layout do seguinte array com a mesma ordem de dimensão menor para maior:

```
a b c 0 0
d e f 0 0
0 0 0 0 0
```

### Indexação em arrays

A classe `IndexUtil` em [index_util.h](https://www.tensorflow.org/code/tensorflow/compiler/xla/index_util.h) fornece utilitários para conversão entre índices multidimensionais e índices lineares, dada um formato e um layout. Os índices multidimensionais incluem um índice `int64` para cada dimensão. Os índices lineares são um único valor `int64` que indexa no buffer que contém o array. Veja `shape_util.h` e `layout_util.h` no mesmo diretório para utilitários que simplificam a criação e manipulação de formatos e layouts.

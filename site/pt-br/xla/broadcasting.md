# Semântica do broadcasting

Este documento descreve como funciona a semântica de broadcasting no XLA.

## O que é broadcasting?

Broadcasting é o processo de fazer com que arrays com formatos diferentes tenham formatos compatíveis para operações aritméticas. A terminologia é emprestada do Numpy [Broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

O broadcasting pode ser necessário para operações entre arrays multidimensionais de diferentes postos ou entre arrays multidimensionais com formatos diferentes, mas compatíveis. Considere a adição `X+v` onde `X` é uma matriz (um array de posto 2) e `v` é um vetor (um array de posto 1). Para realizar a adição elemento a elemento, o XLA precisa fazer o "broadcasting" do vetor `v` para o mesmo posto da matriz `X`, replicando `v` um determinado número de vezes. O comprimento do vetor deve corresponder a pelo menos uma das dimensões da matriz.

Por exemplo:

```
|1 2 3| + |7 8 9|
|4 5 6|
```

As dimensões da matriz são (2,3), as do vetor são (3). O vetor é transmitido replicando-o nas linhas para obter:

```
|1 2 3| + |7 8 9| = |8  10 12|
|4 5 6|   |7 8 9|   |11 13 15|
```

No Numpy, isso é chamado de [broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Princípios

A linguagem XLA é tão estrita e explícita quanto possível, evitando recursos implícitos e “mágicos”. Esses recursos podem deixar algumas computações um pouco mais fáceis de definir, ao custo de mais suposições incorporadas ao código do usuário que serão difíceis de alterar no longo prazo. Se necessário, recursos implícitos e mágicos podem ser adicionados em wrappers no nível do cliente.

No que diz respeito ao broadcasting, são necessárias especificações explícitas de broadcasting sobre operações entre arrays de diferentes postos. Isso difere do Numpy, que infere a especificação sempre que possível.

## Fazendo o broadcasting de um array de posto inferior para um array de posto superior

*Escalares* sempre podem ser convertidos em arrays via broadcasting sem uma especificação explícita das dimensões de broadcasting. Uma operação binária elemento a elemento entre um escalar e um array significa aplicar a operação com o escalar para cada elemento do array. Por exemplo, adicionar um escalar a um array significa produzir um array, onde cada elemento seja uma soma do escalar com o elemento correspondente do array de entrada.

```
|1 2 3| + 7 = |8  9  10|
|4 5 6|       |11 12 13|
```

A maioria das necessidades de broadcasting pode ser capturada usando uma tupla de dimensões numa operação binária. Quando as entradas para a operação têm postos diferentes, esta tupla de broadcasting especifica quais dimensões no array **de posto mais alto** devem corresponder ao array **de posto mais baixo**.

Considere o exemplo anterior, em vez de adicionar um escalar a uma matriz (2,3), adicione um vetor de dimensão (3) a uma matriz de dimensões (2,3). *Sem especificar o broadcasting, esta operação seria inválida.* Para solicitar corretamente a adição de matriz-vetor, especifique a dimensão de broadcasting como (1), o que significa que a dimensão do vetor corresponde à dimensão 1 da matriz. Em 2D, se a dimensão 0 for tratada como linhas e a dimensão 1 como colunas, isso significa que cada elemento do vetor se torna uma coluna de tamanho correspondente ao número de linhas na matriz:

```
|7 8 9| ==> |7 8 9|
            |7 8 9|
```

Como um exemplo mais complexo, considere adicionar um vetor de 3 elementos (dimensão (3)) a uma matriz 3x3 (dimensões (3,3)). Existem duas maneiras pelas quais o broadcasting pode acontecer neste exemplo:

(1) Uma dimensão de broadcasting de 1 pode ser usada. Cada elemento do vetor torna-se uma coluna e o vetor é duplicado para cada linha da matriz.

```
|7 8 9| ==> |7 8 9|
            |7 8 9|
            |7 8 9|
```

(2) Uma dimensão de broadcasting de 0 pode ser usada. Cada elemento do vetor torna-se uma linha e o vetor é repetido para cada coluna da matriz.

```
 |7| ==> |7 7 7|
 |8|     |8 8 8|
 |9|     |9 9 9|
```

> Nota: ao adicionar uma matriz 2x3 a um vetor de 3 elementos, uma dimensão de broadcasting 0 é inválida.

As dimensões de broadcasting podem ser uma tupla que descreve como um formato de posto menor é convertido para um formato de posto maior usando broadcasting. Por exemplo, dado um cubóide 2x3x4 e uma matriz 3x4, uma tupla de broadcasting (1,2) significa combinar a matriz com as dimensões 1 e 2 do cubóide.

Este tipo de broadcasting é usado nas operações binárias em `XlaBuilder`, se o argumento `broadcast_dimensions` for fornecido. Por exemplo, veja [XlaBuilder::Add](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.cc). No código-fonte do XLA, esse tipo de broadcasting às vezes é chamado de broadcasting "InDim".

### Definição formal

O atributo de broadcasting permite corresponder um array de posto inferior com um array de posto superior, especificando quais dimensões do array de posto mais alto devem ser correspondidas. Por exemplo, para um array com dimensões MxNxPxQ, um vetor com dimensão T pode ser correspondido da seguinte forma:

```
          MxNxPxQ

dim 3:          T
dim 2:        T
dim 1:      T
dim 0:    T
```

Em cada caso, T deve ser igual à dimensão correspondente do array de posto mais alto. Os valores do vetor são então convertidos via broadcasting da dimensão correspondente para todas as outras dimensões.

Para corresponder uma matriz TxV com o array MxNxPxQ, um par de dimensões de broadcasting é usado:

```
          MxNxPxQ
dim 2,3:      T V
dim 1,2:    T V
dim 0,3:  T     V
etc...
```

A ordem das dimensões na tupla de broadcasting deve ser a ordem na qual se espera que as dimensões do array de posto inferior correspondam às dimensões do array de posto superior. O primeiro elemento na tupla indica qual dimensão no array de posto mais alto deve corresponder à dimensão 0 no array de posto mais baixo. O segundo elemento para a dimensão 1 e assim por diante. A ordem das dimensões de broadcasting deve ser estritamente crescente. Por exemplo, no exemplo anterior é ilegal combinar V com N e T com P; também é ilegal combinar V com P e N.

## Fazendo o broadcasting de arrays de posto similar com dimensões degeneradas

Um problema de broadcasting relacionado é o broadcasting de dois arrays que possuem o mesmo posto, mas tamanhos de dimensão diferentes. Da mesma forma que as regras do Numpy, isto só é possível quando os arrays são *compatíveis*. Dois arrays são compatíveis quando todas as suas dimensões são compatíveis. Duas dimensões são compatíveis se:

- Forem iguais, ou
- Uma delas for 1 (uma dimensão "degenerada")

Quando dois arrays compatíveis são encontrados, o formato do resultado tem o máximo entre as duas entradas em cada índice de dimensão.

Exemplos:

1. (2,1) e (2,3) convertidos via broadcasting para (2,3).
2. (1,2,5) e (7,2,5) convertidos via broadcasting para (7,2,5)
3. (7,2,5) e (7,1,5) convertidos via broadcasting para (7,2,5)
4. (7,2,5) e (7,2,6) são incompatíveis e não podem ser convertidos via broadcasting.

Surge um caso especial, que também é suportado, onde cada um dos arrays de entrada tem uma dimensão degenerada em um índice diferente. Neste caso, o resultado é uma “operação externa”: (2,1) e (1,3) convertidos via broadcasting para (2,3). Para mais exemplos, veja a [documentação do Numpy sobre broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Broadcasting composto

O broadcasting de um array de posto inferior para um array de posto superior **e** o broadcasting usando dimensões degeneradas podem ser realizados na mesma operação binária. Por exemplo, um vetor de tamanho 4 e uma matriz de tamanho 1x2 podem ser somados usando o valor das dimensões de broadcasting de (0):

```
|1 2 3 4| + [5 6]    // [5 6] is a 1x2 matrix, not a vector.
```

Primeiro, o vetor é convertido via broadcasting até o posto 2 (matriz) usando as dimensões de broadcasting. O valor único (0) nas dimensões de broadcasting indica que a dimensão zero do vetor corresponde à dimensão zero da matriz. Isto produz uma matriz de tamanho 4xM onde o valor M é escolhido para corresponder ao tamanho da dimensão correspondente no array 1x2. Portanto, uma matriz 4x2 é produzida:

```
|1 1| + [5 6]
|2 2|
|3 3|
|4 4|
```

Em seguida, o "broadcasting de dimensão degenerada" converte a dimensão zero da matriz 1x2 para corresponder ao tamanho da dimensão correspondente do lado direito:

```
|1 1| + |5 6|     |6  7|
|2 2| + |5 6|  =  |7  8|
|3 3| + |5 6|     |8  9|
|4 4| + |5 6|     |9 10|
```

Um exemplo mais complicado é uma matriz de tamanho 1x2 adicionada a um array de tamanho 4x3x1 usando dimensões de broadcasting de (1, 2). Primeiro, a matriz 1x2 é convertida até o posto 3 usando as dimensões de broadcasting para produzir um array intermediário Mx1x2 onde o tamanho da dimensão M é determinado pelo tamanho do operando maior (a matriz 4x3x1) produzindo um array intermediário 4x1x2. O M está na dimensão 0 (dimensão mais à esquerda) porque as dimensões 1 e 2 são mapeadas para as dimensões da matriz 1x2 original já que as dimensões de broadcasting são (1, 2). Este array intermediário pode ser adicionado à matriz 4x3x1 usando broadcasting de dimensões degeneradas para produzir como resultado um array 4x3x2.

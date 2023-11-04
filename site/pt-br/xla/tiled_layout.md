# Layout com tiles

Atenção: o layout com tiles é *um pré-lançamento* este documento descreve como ele deve funcionar. Erros poderãi ser ignorados silenciosamente.

<p align="center">   <img src="images/xla_array_layout_figure1.png"> Figura 1</p>

A Figura 1 mostra como um array F32[3,5] é disposto na memória com tiles (ladrilhos) nas dimensões 2x2. Um formato com este layout é escrito como F32[3,5]{1,0:T(2,2)}, onde 1,0 se refere à ordem física das dimensões (campo minor_to_major em Layout) enquanto (2,2) depois dos dois pontos indica o tiling das dimensões físicas por um tile de dimensões 2x2.

Intuitivamente, os tiles são dispostos para cobrir o formato e, em seguida, dentro de cada tile, os elementos são dispostos sem usar tiling, como no exemplo acima, onde a parte direita do exemplo mostra o layout na memória, incluindo os elementos de preenchimento (padding) que são adicionados para ter tiles 2x2 completos, mesmo que os limites do array original não sejam iguais.

Os elementos extras no preenchimento não precisam conter nenhum valor específico.

## Fórmulas de índice linear para o tiling, dados um formato e um tile

Sem tiling, um elemento e=(e <sub>n</sub> , e <sub>n-1</sub>, ... , e <sub>1</sub> ) num array com limites de array d=(d <sub>n</sub> , d <sub>n-1</sub>, ... , d <sub>1</sub>) (d1 é a menor dimensão) é disposta na ordem de maior para menor na posição:

   linear_index(e, d) <br> = linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>), (d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) <br> = e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> + e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

Para simplificar a notação neste documento, assumimos que um tile tem o mesmo número de dimensões que o array. Na implementação de tiling do XLA, isto é generalizado para tiles com menos dimensões, deixando, inicialmente, as dimensões maiores inalteradas e aplicando o tiling apenas às dimensões menores, de modo que o tiling especificado mencione um sufixo das dimensões físicas do formato no qual se está aplicando o tiling.

Quando o tiling de dimensões (t <sub>n</sub>, t <sub>n-1</sub>, ... , t <sub>1</sub>) é usado, um elemento no array com índices (e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>) é mapeado para esta posição no layout final:

   linear_index_with_tile(e, d, t) <br> = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (aritmética é elemento por elemento, (a,b) é concatenação) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> + linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

Pode-se considerar o layout como tendo duas partes: (⌊e <sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e <sub>1</sub> /t<sub>1</sub>⌋), que corresponde a um índice de tiles num array de tiles de tamanho (⌈d <sub>n</sub>/t<sub>n</sub> ⌉, ... , ⌈d <sub>1</sub>/t<sub>1</sub> ⌉), e (e <sub>n</sub>mod t<sub>n</sub> , ... , e <sub>1</sub> mod t <sub>1</sub> ), que corresponde a um índice dentro do tile. A função ceil aparece em ⌈d <sub>i</sub>/t<sub>i</sub> ⌉ porque se os blocos ultrapassarem os limites do array maior, o preenchimento será inserido como na Figura 1. Tanto os blocos quanto os elementos dentro dos blocos são dispostos recursivamente sem tiling.

Para o exemplo da Figura 1, o elemento (2,3) tem o índice do tile (1,1) e índice dentro do tile (0,1), para um vetor de coordenadas combinado de (1, 1, 0, 1). Os índices dos tiles têm limites (2, 3) e o próprio tile é (2, 2) para um vetor combinado de (2, 3, 2, 2). Assim, o índice linear com tile para o elemento com índice (2, 3) na forma lógica é

   linear_index_with_tile((2,3), (3,5), (2,2)) <br> = linear_index((1,1,0,1), (2,3,2,2)) <br> = linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) <br> = (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) <br> = 17.

# Tiling como pad-reshape-transpose

O layout baseado em tiling funciona da seguinte maneira: (d<sub>n</sub>, d<sub>n-1</sub>, ... , d1) (d1 é a dimensão menor). Quando é disposto com um tiling de dimensões (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>) (t<sub>1</sub> é a dimensão menor), esse tiling pode ser descrito em termos de pad-reshape-transpose da seguinte forma

1. O array é preenchido para (⌈d<sub>n</sub>/t<sub>n</sub>⌉∙t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉∙t<sub>1</sub>).
2. Cada dimensão i é dividida em (⌈d<sub>i</sub>/ti⌉, t<sub>i</sub>), ou seja, o array é reformatado para<br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, t<sub>n</sub>, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>1</sub>). <br>Não há nenhuma mudança de layout físico nesta alteração por si só, então esta aleração é um bitcast. Se não estivermos pensando explicitamente em um tile, essa remodelação poderia expressar qualquer formato com o mesmo número de elementos que o formato com preenchimento - o exemplo aqui é sobre como expressar um tile dessa maneira.
3. Uma transposição (transpose) acontece movendo t <sub>n</sub>, ... , t <sub>1</sub> para as dimensões menores, mantendo sua ordem relativa, de modo que a maioria das ordens de dimensões do maior para o menor se torne (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, ... , t<sub>1</sub>).

O formato final tem o prefixo<br> <br> (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉), que descreve o número de tiles em cada dimensão. Um elemento na matriz (e <sub>n</sub>, ... , e <sub>1</sub>) é mapeado para este elemento no formato final: (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>0</sub>/t<sub>0</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>). É fácil perceber que o índice linear do elemento segue a fórmula acima conforme o esperado.

# Tiles repetidos

O tiling do XLA torna-se ainda mais flexível ao aplicá-lo de forma repetida.

<p align="center">   <img src="images/xla_array_layout_figure2.png"> Figura 2</p>

A Figura 2 mostra como um array de tamanho 4x8 é dividido em dois níveis de tiling (primeiro 2x4 e depois 2x1). Representamos esse tiling repetido como (2,4)(2,1). Cada cor indica um tile 2x4 e cada caixa de borda vermelha é um tile 2x1. Os números indicam o índice linear na memória desse elemento no formato do tiling. Este formato corresponde ao formato usado para BF16 na TPU, exceto que o tile inicial é maior, ou seja, o tiling é (8,128)(2,1), onde o objetivo do segundo tiling por 2x1 é coletar dois valores de 16 bits para formar um valor de 32 bits de forma que se alinhe com a arquitetura de uma TPU.

Observe que um segundo tile ou tile posterior pode se referir às dimensões menores dentro do tile, que apenas reorganiza os dados dentro dele, como neste exemplo com (8,128)(2,1), mas também pode se referir às dimensões principais entre tiles obtidas do tiling anterior.

# Combinando dimensões usando tiles

O tiling do XLA também suporta a combinação de dimensões. Por exemplo, ele pode combinar dimensões em F32[2,7,8,11,10]{4,3,2,1,0} para F32[112,110]{1,0} antes de fazer o tiling com (2,3 ). O tile usado é (∗,∗,2,∗,3). Aqui, um asterisco em um tile implica pegar essa dimensão e combiná-la com a próxima dimensão menor. Múltiplas dimensões adjacentes podem ser agrupadas numa dimensão. Uma dimensão agrupada é representada por um valor de tile de -1 naquela dimensão do tile, que de outra forma não seria válido num tile como tamanho de dimensão.

Mais precisamente, se a dimensão i do formato for eliminada por meio de um asterisco no tile, então, antes da definição anterior de tiling ser aplicada, essa dimensão será removida tanto do formato que está sendo disposto usando tiling, quanto do vetor do tile, e o que era a dimensão i-1 do formato tem seu limite de array aumentado de d<sub>i-1</sub> para d<sub>i</sub>d<sub>i-1</sub>. Esse passo é repetido para cada asterisco no vetor de tiles.

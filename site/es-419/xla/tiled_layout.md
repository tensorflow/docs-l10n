# Diseño en mosaico

Atención: El diseño en mosaico es *una versión preliminar* y aquí se describe cómo debe funcionar. No es necesario que se señalen los errores.

<p align="center">   <img src="images/xla_array_layout_figure1.png">   Figura 1</p>

En la Figura 1 se muestra cómo se distribuye un arreglo F32[3,5] en la memoria con mosaicos de 2x2. Una forma con este diseño se escribe como F32[3,5]{1,0:T(2,2)}, donde 1,0 se relaciona con el orden físico de las dimensiones (campo menor_a_mayor en Diseño) mientras que (2,2) después de los dos puntos indica el mosaico de las dimensiones físicas mediante un mosaico de 2x2.

De manera intuitiva, los mosaicos se colocan para cubrir la forma y luego, dentro de cada mosaico, los elementos se colocan sin mosaicos, como en el ejemplo anterior, donde la parte derecha del ejemplo muestra el diseño en la memoria, incluidos los elementos de amortiguado blanco que se agregan para tener mosaicos completos de 2x2 a pesar de que los límites del arreglo original no sean uniformes.

No es necesario que los elementos adicionales en el amortiguado contengan ningún valor en particular.

## Fórmulas de índice lineal para mosaicos a partir de una forma y un mosaico

Sin mosaico, un elemento e=(e <sub>n</sub>, e <sub>n-1</sub>, ..., e <sub>1</sub>) en un arreglo con límites de arreglo d=(d <sub>n</sub>, d <sub>n-1</sub>, ..., d <sub>1</sub>) (d1 es la dimensión menor) se presenta en orden de mayor a menor en la posición:

linear_index(e, d) <br> = linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>), (d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) <br> = e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> + e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

Para simplificar la notación en este documento, asumimos que un mosaico tiene el mismo número de dimensiones que el arreglo. En la implementación de mosaicos de XLA, esto se generaliza a mosaicos con menos dimensiones al dejar las dimensiones iniciales más importantes sin cambios y aplicar el mosaico solo a las dimensiones más pequeñas, de modo que el mosaico que se especifica mencione un sufijo de las dimensiones físicas de la forma que se está construyendo con mosaicos.

Cuando se utiliza el mosaico de tamaño (t <sub>n</sub>, t <sub>n-1</sub>, ..., t <sub>1</sub>), un elemento en el arreglo con índices (e <sub>n</sub>, e <sub>n-1</sub>, ..., e <sub>1</sub>) se asigna a esta posición en el diseño final:

linear_index_with_tile(e, d, t) <br> = linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t))     (la aritmética es elemental, (a,b) es concatenación) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>)) <br> = linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> + linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

Se puede considerar que el diseño tiene dos partes: (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), que corresponde a un índice de mosaico en un arreglo de mosaicos de tamaño (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉), y (e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), que corresponde a un índice de mosaico dentro de un umbral. La función ceil aparece en ⌈d<sub>i</sub>/t<sub>i</sub>⌉ porque si los mosaicos superan los límites del arreglo más grande, se inserta un amortiguado en la Figura 1. Tanto los mosaicos como los elementos dentro de los mosaicos se colocan recursivamente sin mosaico.

Para el ejemplo de la Figura 1, el elemento (2,3) tiene un índice de mosaico (1,1) y un índice dentro del mosaico (0,1), para un vector de coordenadas combinado de (1, 1, 0, 1). Los índices de los mosaicos tienen límites (2, 3) y el mosaico en sí es (2, 2) para un vector combinado de (2, 3, 2, 2). El índice lineal con mosaico para el elemento con índice (2, 3) en la forma lógica es entonces de la siguiente manera

linear_index_with_tile((2,3), (3,5), (2,2)) <br> = linear_index((1,1,0,1), (2,3,2,2)) <br> = linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) <br> = (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) <br> = 17.

# Cómo aplicar mosaicos mediante pad-reshape-transpose

El diseño basado en mosaicos funciona de la siguiente manera: <br> Considere una serie de dimensiones (d <sub>n</sub>, d <sub>n-1</sub>, ..., d1) (d1 es la dimensión menor). Cuando se presenta con mosaicos de tamaño (t <sub>n</sub>, t <sub>n-1</sub>, ..., t <sub>1</sub>) (t <sub>1</sub> es la dimensión menor), ese mosaico se puede describir en términos de pad-reshape-transpose (amortiguado-cambio de forma-transposición) en la siguiente forma.

1. El arreglo se amortigua a (⌈d <sub>n</sub> /t <sub>n</sub> ⌉∙t <sub>n</sub>, ... , ⌈d <sub>1</sub> /t <sub>1</sub> ⌉∙t <sub>1</sub>).
2. Cada dimensión i se divide en (⌈d <sub>i</sub> /ti⌉, t <sub>i</sub> ), es decir, se cambia la forma del arreglo a <br> (⌈d <sub>norte</sub> /t <sub>norte</sub> ⌉, t <sub>norte</sub>, ... , ⌈d <sub>1</sub> /t <sub>1</sub> ⌉, t <sub>1</sub>). <br>Este cambio de forma por sí solo no supone ningún cambio en el diseño físico, por lo que se considera que esta reforma es un bitcast. Si no estamos pensando explícitamente en un mosaico, esta reforma podría expresar cualquier forma con el mismo número de elementos que la forma amortiguada; el ejemplo aquí es de cómo expresar un mosaico de esta manera.
3. Una transposición se produce al mover t <sub>n</sub>, ... , t <sub>1</sub> a las dimensiones menores sin alterar su orden relativo, de modo que el orden de las dimensiones de mayor a menor se convierte en <br> (⌈d <sub>norte</sub> /t <sub>norte</sub> ⌉, ... , ⌈d <sub>1</sub> /t <sub>1</sub> ⌉, t <sub>norte</sub>, ... , t <sub>1</sub>).

La forma final tiene el prefijo <br> (⌈d <sub>n</sub> /t <sub>n</sub> ⌉, ... , ⌈d <sub>1</sub> /t <sub>1</sub> ⌉), que describe el número de mosaicos en cada dimensión. Un elemento del arreglo (e <sub>n</sub>, ..., e <sub>1</sub>) se asigna a este elemento en la forma final: <br> (⌊e <sub>n</sub> /t <sub>n</sub> ⌋, ... , ⌊e <sub>0</sub> /t <sub>0</sub> ⌋, e <sub>n</sub> mod t <sub>n</sub>, ... , e <sub>1</sub> mod t <sub>1</sub>). Es fácil ver que el índice lineal del elemento sigue la fórmula anterior como se esperaba.

# Mosaico repetido

El mosaico de XLA se vuelve aún más flexible si se aplica de forma repetida.

<p align="center">   <img src="images/xla_array_layout_figure2.png">   Figura 2</p>

En la Figura 2 se muestra cómo un arreglo de tamaño 4x8 se divide en dos niveles de mosaico (primero 2x4 y luego 2x1). Representamos este mosaico repetido como (2,4)(2,1). Cada color indica un mosaico de 2x4 y cada cuadro de borde rojo es un mosaico de 2x1. Los números indican el índice lineal en la memoria de ese elemento en el formato en mosaico. Este formato coincide con el formato que se usa para BF16 en TPU, excepto que el mosaico inicial es más grande, es decir, el mosaico es (8,128)(2,1), donde el propósito del segundo mosaico por 2x1 es reunir dos valores de 16 bits para formar un valor de 32 bits de una manera que se alinee con la arquitectura de una TPU.

Tenga en cuenta que un segundo mosaico o un mosaico posterior puede hacer referencia a las dimensiones menores dentro del mosaico, que simplemente reorganiza los datos dentro del mosaico, como en este ejemplo con (8,128)(2,1), pero también puede referirse a las dimensiones mayores transversales del mosaico anterior.

# Cómo combinar dimensiones con mosaicos

El mosaico de XLA también admite la combinación de dimensiones. Por ejemplo, puede combinar dimensiones en F32[2,7,8,11,10]{4,3,2,1,0} en F32[112,110]{1,0} primero antes de colocarlo en mosaico con (2,3 ). El mosaico utilizado es (∗,∗,2,∗,3). Aquí un asterisco en un mosaico implica tomar esa dimensión y combinarla con la siguiente dimensión menor. Se pueden incluir varias dimensiones adyacentes en una sola dimensión. Una dimensión subsumida se representa con un valor de mosaico de -1 en esa dimensión del mosaico, que de lo contrario no es válido en un mosaico como tamaño de dimensión.

Más precisamente, si la dimensión i de la forma se elimina mediante un asterisco en el mosaico, antes de aplicar la definición anterior de mosaico, esa dimensión se elimina tanto de la forma que se está mosaico como del vector del mosaico, y lo que era la dimensión i-1. de la forma tiene su límite de arreglo aumentado de d <sub>i-1</sub> a d <sub>i</sub> d <sub>i-1</sub>. Este paso se repite para cada asterisco en el vector de mosaico.

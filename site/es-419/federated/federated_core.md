# Federated Core

Este documento presenta la capa central de TFF que sirve de base para el [Aprendizaje Federado](federated_learning.md) y posibles futuros algoritmos federados que no sean de aprendizaje.

Si desea acceder a una introducción sencilla a Federated Core, lea los siguientes tutoriales, ya que allí se presentan algunos de los conceptos fundamentales a través de ejemplos y se demuestra paso a paso la construcción de un sencillo algoritmo federado de cálculo de promedios.

- [Algoritmos federados personalizados, parte 1: introducción a Federated Core](tutorials/custom_federated_algorithms_1.ipynb).

- [Algoritmos federados personalizados, parte 2: implementación del promediado federado](tutorials/custom_federated_algorithms_2.ipynb).

Además, le recomendamos que se familiarice con el [Aprendizaje Federado](federated_learning.md) y los tutoriales relacionados con la [clasificación de imágenes](tutorials/federated_learning_for_image_classification.ipynb) y la [generación de textos](tutorials/federated_learning_for_text_generation.ipynb), ya que los usos de la API Federated Core (API FC) para el aprendizaje federado constituyen un contexto importante para algunas de las decisiones que tomamos al diseñar esta capa.

## Descripción general

### Objetivos, usos previstos y alcance

Federated Core (FC) se define como un entorno de programación que permite realizar cálculos distribuidos, es decir, cálculos en los que intervienen varios ordenadores (teléfonos móviles, tabletas, dispositivos integrados, computadoras de escritorio, sensores, servidores de bases de datos, etc.), cada uno de los cuales puede procesar datos no triviales a nivel local y comunicarse a través de la red para coordinar su trabajo.

El término *distribuido* es muy genérico y TFF no se aplica a todos los tipos posibles de algoritmos distribuidos que existen, por lo que preferimos utilizar el término menos genérico de *cálculo federado* para describir los tipos de algoritmos que se pueden expresar en este marco.

Aunque definir el término *cálculo federado* de manera totalmente formal escapa al alcance de este documento, piense en los tipos de algoritmos que podría ver expresados en pseudocódigo en una [divulgación de investigación](https://arxiv.org/pdf/1602.05629.pdf) que describa un nuevo algoritmo de aprendizaje distribuido.

En pocas palabras, el objetivo de FC es permitir una representación suficientemente compacta, con un nivel de abstracción similar al del pseudocódigo, de la lógica de un programa que *no* sea pseudocódigo, sino que se pueda ejecutar en diversos entornos.

La característica clave que define los tipos de algoritmos para los que se diseñó FC es que las acciones de los participantes en el sistema se describen de forma colectiva. Así, tendemos a hablar de *cada dispositivo* que transforma los datos a nivel local, y de los dispositivos que coordinan el trabajo mediante un coordinador centralizado que *difunde*, *recopila* o *agrega* sus resultados.

Si bien es cierto que el diseño de TFF pretende ir más allá de las simples arquitecturas *cliente-servidor*, la noción de procesamiento colectivo es fundamental. Esto se debe a los orígenes de TFF en el aprendizaje federado, una tecnología que originalmente se diseñó para admitir cálculos sobre datos potencialmente sensibles que permanecen bajo el control de los dispositivos cliente, y que por motivos de privacidad no pueden descargarse simplemente a una ubicación centralizada. Si bien cada cliente de estos sistemas contribuye con datos y capacidad de procesamiento para que el sistema calcule un resultado (uno que, por lo general, esperamos que sea valioso para todos los participantes), también nos esforzamos por preservar la privacidad y el anonimato de cada cliente.

Por lo tanto, mientras que la mayoría de los marcos para el cálculo distribuido pretenden expresar el procesamiento desde la perspectiva de los participantes individuales, es decir, a nivel de los intercambios individuales de mensajes punto a punto y la interdependencia de las transiciones de estado locales de los participantes con los mensajes entrantes y salientes, Federated Core de TFF fue diseñado para describir el comportamiento del sistema desde la perspectiva *global* de todo el sistema (de forma similar a, por ejemplo, [MapReduce](https://research.google/pubs/pub62/)).

En consecuencia, mientras que los marcos distribuidos para fines generales pueden ofrecer operaciones como *enviar* y *recibir* en forma de bloques de construcción, FC facilita bloques de construcción como `tff.federated_sum`, `tff.federated_reduce` o `tff.federated_broadcast` que encapsulan protocolos distribuidos sencillos.

## Lenguaje

### Interfaz de Python

TFF aplica un lenguaje interno para representar cálculos federados, cuya sintaxis se define mediante la representación serializable en [computation.proto](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). Sin embargo, por lo general, los usuarios de la API FC no necesitan interactuar directamente con este lenguaje. En su lugar, se ofrece una API de Python (el espacio de nombres `tff`) que lo envuelve como una forma de definir los cálculos.

En concreto, TFF proporciona decoradores de funciones de Python como `tff.federated_computation` que rastrean los cuerpos de las funciones decoradas y producen representaciones serializadas de la lógica de cálculo federado en el lenguaje de TFF. Una función decorada con `tff.federated_computation` actúa como operador de dicha representación serializada y puede insertarla como bloque funcional en el cuerpo de otro cálculo o ejecutarla bajo demanda cuando se invoca.

Este es solo un ejemplo; encontrará más ejemplos en los tutoriales sobre [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb).

```python
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)
```

A los lectores familiarizados con TensorFlow no eager este enfoque les parecerá similar a escribir código de Python que use funciones como `tf.add` o `tf.reduce_sum` en una sección de código de Python que defina un gráfico de TensorFlow. Aunque el código se exprese técnicamente en Python, su propósito es construir una representación serializable de un gráfico `tf.Graph` subyacente, y es el gráfico, no el código de Python, el que se ejecuta internamente mediante el tiempo de ejecución de TensorFlow. Del mismo modo, se puede pensar en `tff.federated_mean` como la inserción de una *operación federada* en un cálculo federado representado por `get_average_temperature`.

Uno de los motivos por los que FC define un lenguaje tiene que ver con el hecho de que, como ya se ha señalado, los cálculos federados especifican comportamientos colectivos distribuidos y, como tales, su lógica no es local. Por ejemplo, TFF facilita operadores cuyas entradas y salidas pueden existir en distintos lugares de la red.

Para ello se necesita un lenguaje y un sistema de tipos que capten la noción de capacidad de distribución.

### Sistema de tipos

Federated Core ofrece las siguientes categorías de tipos. Al describir estos tipos, señalamos los constructores de tipos e introducimos una notación compacta, ya que es una forma práctica de describir tipos de cálculos y operadores.

En primer lugar, aquí están las categorías de tipos que son conceptualmente similares a las que podemos encontrar en los lenguajes convencionales existentes:

- **Tipos de tensor** (`tff.TensorType`). Al igual que en TensorFlow, tienen `dtype` y `shape`. La única diferencia es que los objetos de este tipo no se limitan a instancias de `tf.Tensor` en Python que representan salidas de operaciones de TensorFlow en un gráfico de TensorFlow, sino que también pueden incluir unidades de datos que pueden producirse, por ejemplo, como salida de un protocolo de agregación distribuida. Por lo tanto, el tipo de tensor de TFF es simplemente una versión abstracta de una representación física concreta de dicho tipo en Python o TensorFlow.

    Los `TensorTypes` de TFF pueden tener un tratamiento más estricto (estático) de las formas en comparación con TensorFlow. Por ejemplo, el typesystem de TFF trata un tensor con clasificación desconocida como asignable *desde* cualquier otro tensor del mismo `dtype`, pero no asignable *a* un tensor con clasificación fija. Este tratamiento evita ciertos errores en el tiempo de ejecución (por ejemplo, intentar modificar la forma de un tensor de clasificación desconocida para que tenga una forma con un número incorrecto de elementos), a costa de una mayor rigurosidad en los cálculos que TFF acepta como válidos.

    La anotación compacta para los tipos de tensor es `dtype` o `dtype[shape]`. Por ejemplo, `int32` e `int32[10]` son los tipos de enteros y vectores de enteros, respectivamente.

- **Tipos de secuencia** (`tff.SequenceType`). Se trata del equivalente abstracto de TFF al concepto concreto de TensorFlow de `tf.data.Dataset`s. Los elementos de las secuencias se pueden emplear de forma secuencial y pueden incluir tipos complejos.

    La representación compacta de los tipos de secuencia es `T*`, donde `T` corresponde al tipo de elementos. por ejemplo, `int32*` representa una secuencia de números enteros.

- **Tipos de tupla nombrada** (`tff.StructType`). Es la manera que tiene TFF de construir tuplas y estructuras similares a diccionarios que tienen un número predefinido de *elementos* con tipos específicos, nombrados y sin nombrar. Cabe destacar que el concepto de tupla nombrada de TFF engloba el equivalente abstracto de las tuplas de argumentos de Python, es decir, colecciones de elementos de los que algunos, pero no todos, son nombrados y otros son posicionales.

    La anotación compacta para tuplas nombradas es `<n_1=T_1, ..., n_k=T_k>`, donde `n_k` son nombres de elementos opcionales y `T_k` son tipos de elementos. Por ejemplo, `<int32,int32>` es una anotación compacta para un par de enteros sin nombrar, y `<X=float32,Y=float32>` es una anotación compacta para un par de flotantes nombrados `X` e `Y` que pueden representar un punto en un plano. Las tuplas pueden anidarse y mezclarse con otros tipos, por ejemplo, `<X=float32,Y=float32>*` sería una anotación compacta para una secuencia de puntos.

- **Tipos de función** (`tff.FunctionType`). TFF es un marco de programación funcional, en el que las funciones se tratan como [valores de primera clase](https://en.wikipedia.org/wiki/First-class_citizen). Las funciones tienen como máximo un argumento y exactamente un resultado.

    La anotación compacta para las funciones es `(T -> U)`, donde `T` es el tipo de un argumento y `U` es el tipo del resultado o `( -> U)` si no tiene argumento (aunque las funciones sin argumento son un concepto degenerado que existe principalmente en Python). Por ejemplo, `(int32* -> int32)` es una anotación para un tipo de funciones que reducen una secuencia de enteros a un único valor entero.

Los siguientes tipos se centran en el aspecto de los sistemas distribuidos de los cálculos del TFF. Dado que estos conceptos son en cierto modo exclusivos de TFF, le recomendamos que consulte el tutorial de [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) para acceder a comentarios y ejemplos adicionales.

- **Tipo de colocación**. Este tipo aún no se expuso en la API pública excepto en forma de 2 literales `tff.SERVER` y `tff.CLIENTS` que puede considerar constantes de este tipo. Sin embargo, se usa internamente y se introducirá en las próximas versiones de la API pública. La representación compacta de este tipo es `placement`.

    Una *colocación* representa un conjunto de participantes en el sistema que desempeñan una función determinada. La versión inicial está orientada a los cálculos cliente-servidor, en los que hay dos grupos de participantes: *clientes* y un *servidor* (se puede pensar en este último como un grupo único). No obstante, en arquitecturas más elaboradas, podría haber otros roles, como agregadores intermedios en un sistema multinivel, que podrían ejecutar distintos tipos de agregación, o utilizar distintos tipos de compresión/descompresión de datos con respecto a los utilizados por el servidor o los clientes.

    El objetivo principal de definir la noción de colocaciones es como base para definir *tipos federados*.

- **Tipos federados** (`tff.FederatedType`). Un valor de tipo federado es aquel que está alojado por un grupo de participantes del sistema definidos por una colocación específica (como `tff.SERVER` o `tff.CLIENTS`). Un tipo federado se define por el valor de *colocación* (por lo tanto, se trata de un [tipo dependiente](https://en.wikipedia.org/wiki/Dependent_type)), el tipo de *miembros constituyentes* (qué tipo de contenido aloja localmente cada uno de los participantes) y el bit adicional `all_equal` que especifica si todos los participantes alojan localmente el mismo elemento.

    La anotación compacta para valores de tipo federado que incluyen elementos (miembros constituyentes) de tipo `T`, cada uno alojado en un grupo (colocación) `G` es `T@G` o `{T}@G` en función de si el bit `all_equal` está activado o desactivado, respectivamente.

    Por ejemplo:

    - `<int>@CLIENTS` representa un *valor federado* que consiste en un conjunto de números enteros potencialmente distintos, uno por dispositivo cliente. Tenga en cuenta que estamos hablando de un único *valor federado* que engloba varios elementos de datos que aparecen en varias ubicaciones de la red. Una forma de verlo es como una especie de tensor con una dimensión de "red", aunque esta analogía no es perfecta porque TFF no permite el [acceso aleatorio](https://en.wikipedia.org/wiki/Random_access) a los miembros constituyentes de un valor federado.

    - `{<X=float32,Y=float32>*}@CLIENTS` representa un *conjunto de datos federado*, un valor que consiste de múltiples secuencias de coordenadas `XY`, una secuencia por dispositivo cliente.

    - `<weights=float32[10,5],bias=float32[5]>@SERVER` representa una tupla nombrada de tensores de peso y sesgo en el servidor. Como eliminamos las llaves, esto indica que el bit `all_equal` está activado, es decir, que solo hay una tupla (independientemente de cuántas réplicas del servidor pueda haber en un clúster que aloje este valor).

### Bloques funcionales

El lenguaje de Federated Core es una forma de [cálculo lambda](https://en.wikipedia.org/wiki/Lambda_calculus), con algunos elementos adicionales.

Este lenguaje aporta las siguientes abstracciones de programación expuestas actualmente en la API pública:

- Cálculos de **TensorFlow** (`tff.tf_computation`). Se trata de secciones del código de TensorFlow envueltas como componentes reutilizables en TFF que usan el decorador `tff.tf_computation`. Siempre tienen tipos funcionales y, a diferencia de las funciones en TensorFlow, admiten parámetros estructurados o devuelven resultados estructurados de un tipo de secuencia.

    A continuación, se muestra un ejemplo, un cálculo de tipo de TF `(int32* -> int)` que usa el operador `tf.data.Dataset.reduce` para calcular una suma de enteros:

    ```python
    @tff.tf_computation(tff.SequenceType(tf.int32))
    def add_up_integers(x):
      return x.reduce(np.int32(0), lambda x, y: x + y)
    ```

- Operadores **intrínsecos** o *federados* ( `tff.federated_...` ). Esta es una biblioteca de funciones como `tff.federated_sum` o `tff.federated_broadcast` que constituyen gran parte de la API de FC, que, en su mayoría, representan operadores de comunicación distribuida para su uso con TFF.

    Nos referimos a estos operadores como *intrínsecos* porque, al igual que las [funciones intrínsecas](https://en.wikipedia.org/wiki/Intrinsic_function), son un conjunto abierto y extensible de operadores que TFF entiende y compila en código de bajo nivel.

    La mayoría de estos operadores tienen parámetros y resultados de tipos federados, y la mayoría son plantillas que se pueden aplicar a varios tipos de datos.

    Por ejemplo, `tff.federated_broadcast` puede interpretarse como un operador de plantilla de un tipo funcional `T@SERVER -> T@CLIENTS`.

- **Expresiones lambda** (`tff.federated_computation`). Una expresión lambda en TFF es el equivalente de un `lambda` o `def` en Python; consta del nombre del parámetro y de un cuerpo (expresión) que contiene referencias a este parámetro.

    En código de Python, se pueden crear decorando funciones de Python con `tff.federated_computation` y definiendo un argumento.

    Este es un ejemplo de una expresión lambda que ya mencionamos anteriormente:

    ```python
    @tff.federated_computation(tff.type_at_clients(tf.float32))
    def get_average_temperature(sensor_readings):
      return tff.federated_mean(sensor_readings)
    ```

- **Literales de colocación**. Por ahora, solo `tff.SERVER` y `tff.CLIENTS` permiten definir cálculos cliente-servidor sencillos.

- **Invocaciones de funciones** (`__call__`). Cualquier cosa que tenga un tipo funcional se puede invocar mediante la sintaxis estándar de Python `__call__`. La invocación es una expresión, cuyo tipo es el mismo que el tipo del resultado de la función que se invoca.

    Por ejemplo:

    - `add_up_integers(x)` representa una invocación del cálculo de TensorFlow definido anteriormente sobre un argumento `x`. Esta es una expresión de tipo `int32`.

    - `tff.federated_mean(sensor_readings)` representa una invocación del operador de promediado federado en `sensor_readings`. Esta es una expresión de tipo `float32@SERVER` (si partimos del contexto del ejemplo anterior).

- Formación de **tuplas** y **selección** de sus elementos. Las expresiones de Python de la forma `[x, y]`, `x[y]` o `x.y` que figuran en los cuerpos de las funciones decoradas con `tff.federated_computation`.

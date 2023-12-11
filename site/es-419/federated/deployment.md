# Implementación

Además de definir los cálculos, TFF proporciona herramientas para ejecutarlos. Aunque el objetivo principal son las simulaciones, las interfaces y herramientas que ofrecemos son más generales. En este documento se describen las opciones de implementación en distintos tipos de plataformas.

Nota: Este documento todavía está en fase de elaboración.

## Descripción general

Hay dos modos principales de implementación de los cálculos TFF:

- **Backends nativos**. Nos referiremos a un backend como *nativo* cuando sea capaz de interpretar la estructura sintáctica de las computaciones de TFF tal y como se define en [`computation.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/computation.proto). Un backend nativo no necesariamente tiene que ser compatible con todas las construcciones o intrínsecos del lenguaje. Los backends nativos deben implementar una de las interfaces de *ejecutor* de TFF estándar, como [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor) para su uso con código Python, o la versión independiente del lenguaje que se define en [`executor.proto`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/proto/v0/executor.proto) y se expone como un punto de conexión gRPC.

    Los backends nativos compatibles con las interfaces antes mencionadas se pueden utilizar de forma interactiva en lugar del tiempo de ejecución de referencia predeterminado, por ejemplo, para ejecutar blocs de notas o scripts de experimentos. La mayoría de los backends nativos funcionarán en *modo interpretado*, es decir, procesarán la definición de la computación a medida que se define y la ejecutarán de forma incremental, pero esto no siempre es así. Un backend nativo también puede *transformar* (*compilar*, o compilar con JIT) una parte de la computación para mejorar el rendimiento, o para simplificar su estructura. Un ejemplo de este uso común sería reducir el conjunto de operadores federados que aparecen en una computación, de modo que las partes del backend posteriores a la transformación no tengan que estar expuestas al conjunto completo.

- **Backends no nativos**. Los backends no nativos, a diferencia de los nativos, no pueden interpretar directamente la estructura computacional de TFF y requieren que se convierta en una *representación de destino* diferente comprendida por el backend. Un ejemplo notable de este tipo de backend sería un clúster Hadoop o una plataforma similar para canalizaciones de datos estáticos. Para que un cálculo se implemente en dicho backend, primero debe *transformarse* (o *compilarse*). En función de la configuración, esto se puede hacer de forma transparente para el usuario (es decir, un backend no nativo podría incluirse en una interfaz de ejecutor estándar como [`tff.framework.Executor`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/Executor) que realiza transformaciones a nivel interno), o puede exponerse como una herramienta que permite al usuario convertir manualmente un cálculo, o un conjunto de cálculos, en la representación de destino adecuada entendida por la clase particular de backends. El código que admite tipos específicos de backends no nativos se puede encontrar en el espacio de nombres [`tff.backends`](https://www.tensorflow.org/federated/api_docs/python/tff/backends). Al momento de escribir este artículo, el único tipo de backends no nativos compatible es una clase de sistemas capaces de ejecutar MapReduce de una sola ronda.

## Backends nativos

Pronto le daremos más detalles.

## Backends no nativos

### MapReduce

Pronto le daremos más detalles.

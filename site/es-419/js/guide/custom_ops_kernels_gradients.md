# Escritura de operaciones, núcleos y gradientes personalizados en TensorFlow.js

## Descripción general

En esta guía se describen los mecanismos para definir operaciones (ops), núcleos (kernels) y gradientes personalizados en TensorFlow.js. La guía está prevista para aportar una descripción general de los principales conceptos y punteros que hay que codificar, para demostrar dichos conceptos en acción.

### A quién está dirigida esta guía

Esta es una guía con conceptos bastante avanzados, que trata sobre algunos aspectos internos de TensorFlow.js. Puede resultar particularmente útil para los siguientes grupos de personas:

- Los usuarios avanzados de TensorFlow.js interesados en personalizar el comportamiento de varias operaciones matemáticas (p. ej., los investigadores que sobrescriben implementaciones con gradientes o usuarios que necesitan emparchar funcionalidades que faltan en la biblioteca).
- Usuarios que crean bibliotecas que extienden a TensorFlow.js (p. ej., una biblioteca de álgebra lineal general creada sobre los rudimentos de TensorFlow.js o un nuevo backend de TensorFlow.js).
- Los usuarios interesados por contribuir con operaciones nuevas a tensorflow.js. Quienes quieren acceder a una visión general de cómo funcionan estos mecanismos.

Esta **no** es una guía sobre el uso general de TensorFlow.js, ya que aquí se describen mecanismos de implementación interna. No hace falta entender estos mecanismos para usar TensorFlow.js

Para aprovechar esta guía al máximo, debe estar familiarizado con la lectura de código fuente de TensorFlow.js, o al menos tener buena predisposición para este tipo de lecturas.

## Terminología

Será de gran utilidad describir algunos términos de esta guía con anticipación:

**Operaciones (Ops)**: una operación matemática en uno o más tensores que produce uno o más tensores como salida. Las operaciones son código de "alto nivel" y pueden usar otras operaciones para definir su lógica.

**Núcleo (<em>kernel</em>)**: es una implementación específica de una operación ligada a capacidades de hardware/plataforma también específicas. Los núcleos son de "bajo nivel"" y tienen backend específico. Algunas operaciones tienen un mapeo uno a uno desde la operación al núcleo, mientras que otras usan varios núcleos.

**Gradiente** **/ GradFunc**: la definición de "modo hacia atrás" de una **operación o núcleo** que calcula la derivada de esa función con respecto a alguna entrada. Los gradientes son código de "alto nivel" (no backend específico) y pueden llamar a otras operaciones o núcleos.

**Registro de núcleo (<em>Kernel Registry</em>)**: un mapa desde una tupla **(nombre de núcleo, nombre de backend)** a una implementación de un núcleo.

**Registro de gradiente**: un mapa desde un **nombre de un núcleo hasta la implementación de un gradiente**.

## Organización de código

Las [operaciones](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/ops) y los [gradientes](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients) se definen en [tfjs-core](https://github.com/tensorflow/tfjs/tree/master/tfjs-core).

Los núcleos son específicos de cada backend y se definen en sus respectivas carpetas de backend (p. ej., [tfjs-backend-cpu](https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-cpu/src/kernels)).

Las operaciones, los núcleos y los gradientes personalizados no necesitan ser definidos dentro de los paquetes. Pero, por lo general, usan símbolos similares para su implementación.

## Implementación de operaciones personalizadas

Una forma de pensar en una operación personalizada es como una función de JavaScript que devuelve alguna salida de tensor, por lo común, con tensores como entrada.

- Algunas operaciones se pueden definir por completo en términos de otras operaciones existentes y deberían importar y llamar a esas funciones directamente. [Haga clic aquí para ver un ejemplo](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/moving_average.ts).
- La implementación de una operación también puede enviar a núcleos específicos de backend. Se hace mediante `Engine.runKernel` y se describirá más detalladamente en la sección “Implementación de núcleos personalizados”. [Haga clic aquí para ver un ejemplo](https://github.com/tensorflow/tfjs/blob/1bec37b9364df6164a4a0ad64c29e0859382f0b4/tfjs-core/src/ops/sqrt.ts).

## Implementación de núcleos personalizados

Las implementaciones de núcleos específicas de un backend permiten optimizar la implementación de la lógica para una operación dada. Las operaciones invocan a los núcleos llamando a [`tf.engine().runKernel()`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/engine.ts?q=runKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F). Las implementaciones de núcleos están definidas por cuatro cosas:

- El nombre del núcleo.
- El backend en el que se implementa el núcleo.
- Entradas: argumentos tensores para la función núcleo.
- Atributos: argumentos que no son tensores para la función de núcleo.

Este es un ejemplo de [implementación de un núcleo](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu/src/kernels/Square.ts). Las convenciones utilizadas para la implementación son específicas con respecto al backend y se entienden mejor si observamos cada implementación y documentación de backend en particular.

Por lo general, los núcleos operan a un nivel más bajo que los tensores y, a diferencia de ellos, leen y escriben directamente en la memoria que, con el tiempo, terminará encapsulada en tensores por tfjs-core.

Una vez que el núcleo se implementa, se puede registrar con TensorFlow.js usando la [función `registerKernel`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerKernel&ss=tensorflow%2Ftfjs:tfjs-core%2F) de tfjs-core. Se puede registrar un núcleo por cada backend en el que se desee que trabaje (el núcleo). Una vez registrado, el núcleo se puede invocar con `tf.engine().runKernel(...)` y TensorFlow.js se ocupará de enviarlo para la implementación en el backend que esté activo en ese momento.

## Implementación de gradientes personalizados

Los gradientes, por lo general, se definen por un núcleo dado (identificado por el mismo nombre de núcleo usado en una llamada a `tf.engine().runKernel(...)`). Esto es lo que hace posible que tfjs-core use un registro para buscar definiciones de gradiente, para cualquier núcleo en el tiempo de ejecución.

La implementación de gradientes personalizados es útil para lo siguiente:

- Agregar una definición de gradiente que puede no estar presente en la biblioteca.
- Sobrescribir una definición de gradiente para personalizar su cálculo para un núcleo dado.

Puede ver ejemplos de [implementaciones de gradientes aquí](https://github.com/tensorflow/tfjs/tree/master/tfjs-core/src/gradients).

Una vez que haya implementado un gradiente para una llamada dada, podrá registrarlo con TensorFlow.js usando la [función `registerGradient`](https://cs.opensource.google/tensorflow/tfjs/+/master:tfjs-core/src/kernel_registry.ts?q=registerGradient&ss=tensorflow%2Ftfjs:tfjs-core%2F) de tfjs-core.

Otra manera de implementar gradientes personalizados que omite el registro de gradiente y, por lo tanto, permite el cálculo de gradientes para funciones arbitrarias en formas también arbitrarias es usar [tf.customGrad](https://js.tensorflow.org/api/latest/#customGrad)

[En este enlace hay un ejemplo de una operación dentro de una biblioteca](https://github.com/tensorflow/tfjs/blob/f111dc03a87ab7664688011812beba4691bae455/tfjs-core/src/ops/losses/softmax_cross_entropy.ts#L64) con customGrad

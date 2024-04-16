# Diseño de TensorFlow Quantum

TensorFlow Quantum (TFQ) está diseñado para abordar los problemas del aprendizaje automático cuántico de la era NISQ. Incorpora los conceptos primitivos de la computación cuántica, como la construcción de circuitos cuánticos, en el ecosistema de TensorFlow. Los modelos y las operaciones que se construyen con TensorFlow usan esos elementos primitivos para crear sistemas híbridos cuántico-clásicos potentes.

Con TFQ, los investigadores pueden construir un grafo de TensorFlow utilizando un conjunto de datos cuántico, un modelo cuántico y parámetros de control clásicos. Todo esto se representa como tensores en un único grafo computacional. El resultado de las mediciones cuánticas, que derivan en eventos probabilísticos clásicos, se obtiene mediante operaciones de TensorFlow. El entrenamiento se lleva a cabo con la API [Keras](https://www.tensorflow.org/guide/keras/overview) estándar. Gracias al módulo `tfq.datasets` los investigadores pueden experimentar con conjuntos de datos cuánticos nuevos e interesantes.

## Cirq

<a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> es un marco de programación cuántico de Google. Ofrece todas las operaciones básicas (como los bits cuánticos, las puertas, los circuitos y las mediciones) para crear, modificar e invocar los circuitos cuánticos en computadoras cuánticas o en una computadora cuántica simulada. TensorFlow Quantum usa estos Cirq primitivos para extender TensorFlow para el cálculo de lotes, la creación del modelo y el cálculo del gradiente. Para lograr efectividad con TensorFlow Quantum, conviene ser efectivos con Cirq.

## Valores primitivos en TensorFlow Quantum

En TensorFlow Quantum se implementan los componentes necesarios para integrar TensorFlow con el hardware de computación cuántica. Con este fin, TFQ presenta dos tipos de datos primitivos:

- El *circuito cuántico*: representa a circuitos cuánticos definidos con <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a>  (`cirq.Circuit`) dentro de TensorFlow. Crea lotes de circuitos de diferentes tamaños, similar a los lotes de diferentes puntos de datos con valores reales.
- La *suma de Pauli*: representa a las combinaciones lineales de productos tensores de operadores de Pauli definidos en Cirq (`cirq.PauliSum`). Como los circuitos, crean lotes de operadores de distintos tamaños.

### Operaciones fundamentales

Por medio de la utilización de datos primitivos de circuitos cuánticos dentro de un `tf.Tensor`, TensorFlow Quantum implementa operaciones que procesan esos circuitos y producen salidas significativas.

Las operaciones de TensorFlow se escriben en C++ optimizado. De estas operaciones se obtienen muestras para circuitos, se calculan los valores esperados y se emite el estado producido por el circuito dado. Escribir operaciones flexibles y con buen desempeño tiene sus dificultades:

1. Los circuitos no tienen todos el mismo tamaño. En los circuitos simulados, no se pueden crear operaciones estáticas (como `tf.matmul` o `tf.add`) y después sustituir con números diferentes los circuitos de tamaños diferentes. Estas operaciones deben permitir tamaños dinámicos que el grafo de cálculo de TensorFlow, con tamaños estadísticos, no admite.
2. Los datos cuánticos pueden inducir diferentes estructuras de circuitos a la vez. Este es otro de los motivos por el que conviene admitir tamaños dinámicos en las operaciones de TFQ. Los datos cuánticos pueden representar un cambio estructural en el estado cuántico subyacente que se represente con modificaciones al circuito original. Como los puntos de datos nuevos entran y salen en el tiempo de ejecución, el grafo de cálculo de TensorFlow no se puede modificar después de haberse construido, por lo tanto para este tipo de estructuras variables se necesita ayuda.
3. Los `cirq.Circuits` se asemejan al cálculo de grafos en que son una serie de operaciones y en que algunos pueden contener símbolos o <em>placeholders</em>. Es importante hacer que tengan la mayor compatibilidad posible con TensorFlow.

Por su desempeño, Eigen (la biblioteca C++ que se usa en muchas de las operaciones de TensorFlow) no está bien adaptada para la simulación de circuitos cuánticos. En cambio, los simuladores del circuito que se usan en el <a href="https://ai.googleblog.com/2019/10/quantum-supremacy-using-programmable.html" class="external">experimento clásico, más allá de lo cuántico</a> también se usan como verificadores y se extienden como las bases de las operaciones de TFQ (todo escrito con instrucciones AVX2 y SSE). Se crearon operaciones con firmas funcionales idénticas que se usan con computadoras cuánticas físicas. El cambio entre una computadora cuántica simulada y una física es muy sencillo, solamente hay que cambiar una línea de código. Estas operaciones se ubican en <a href="https://github.com/tensorflow/quantum/blob/master/tensorflow_quantum/core/ops/circuit_execution_ops.py" class="external"><code>circuit_execution_ops.py</code></a>.

### Capas

Las capas de TensorFlow Quantum les ofrecen muestreos, esperanzas y cálculos de estado a los desarrolladores con la interfaz `tf.keras.layers.Layer`. Conviene crear una capa del circuito para los parámetros de control clásicos o para las operaciones de lectura. Además, podemos crear una capa con un alto grado de complejidad para fortalecer el circuito del lote, el valor del parámetro de control del lote y para realizar las operaciones de lectura del lote. Para ver un ejemplo, consulte `tfq.layers.Sample`.

### Diferenciadores

A diferencia de lo que sucede con muchas operaciones en TensorFlow, los observables de los circuitos cuánticos no tienen fórmulas para los gradientes que son relativamente fáciles de calcular. El motivo es que una computadora clásica solamente puede leer muestras de circuitos que se ejecutan en computadoras cuánticas.

Para resolver este problema, el módulo `tfq.differentiators` ofrece varias técnicas de diferenciación estándar. Los usuarios también pueden definir su propio método para calcular los gradientes, tanto en el "mundo real" de los cálculos esperados basados en muestras como en el mundo exacto analítico. Los métodos como las diferencias finitas, por lo general, son los más rápidos (<em>tiempo de reloj de pared</em>) en un entorno analítico/exacto. Mientras que los métodos más lentos (<em>tiempo de reloj de pared</em>) y más prácticos, como el <a href="https://arxiv.org/abs/1811.11184" class="external">cambio de parámetros</a> u otros <a href="https://arxiv.org/abs/1901.05374" class="external">métodos estocásticos</a>, por lo general, son más efectivos. Se instancia un `tfq.differentiators.Differentiator` y se adjunta a una operación existente con `generate_differentiable_op` o bien, se pasa al constructor de `tfq.layers.Expectation` o `tfq.layers.SampledExpectation`. Para implementar un diferenciador personalizado, usamos la herencia de la clase `tfq.differentiators.Differentiator`. Para definir una operación de gradiente para el muestreo o el cálculo del vector de estado, usamos `tf.custom_gradient`.

### Conjuntos de datos

A medida que la computación cuántica avance, surgirán más combinaciones de modelos y datos cuánticos que harán que la comparación estructurada se vuelva más complicada. El módulo `tfq.datasets` se usa como fuente de datos para tareas de aprendizaje automático cuántico. Garantiza las comparaciones estructuradas para el modelo y el desempeño.

Se espera que con grandes aportes comunitarios, el módulo `tfq.datasets` crezca hasta permitir investigaciones más transparentes y reproducibles. Los problemas debidamente seleccionados y organizados en control cuántico, simulación fermiónica, transiciones de clasificación de fase cercana, sensado cuántico, etc. son todos excelentes opciones para agregar a los `tfq.datasets`. Para proponer un conjunto de datos nuevo abra un <a href="https://github.com/tensorflow/quantum/issues">incidente de GitHub</a>.

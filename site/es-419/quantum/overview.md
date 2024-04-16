# TensorFlow Quantum

TensorFlow Quantum (TFQ) es un marco de trabajo de Python para el [aprendizaje automático cuántico](concepts.md). Como marco de aplicación, TFQ permite que los investigadores de algoritmos cuánticos y los de aplicaciones de aprendizaje automático aprovechen los marcos de trabajo de computación cuántica de Google, todo desde dentro de TensorFlow.

TensorFlow Quantum se centra en los *datos cuánticos* y en la creación de *modelos híbridos cuántico-clásicos*. Aporta herramientas para intercalar algoritmos cuánticos y diseño lógico en <a href="https://github.com/quantumlib/Cirq" class="external">Cirq</a> con TensorFlow. Se requiere de un conocimiento básico de la computación cuántica para usar TensorFlow Quantum de manera eficaz.

Para comenzar con TensorFlow Quantum, consulte la [guía de instalación](install.md) y lea algunos de los [tutoriales del bloc de notas](./tutorials/hello_many_worlds.ipynb) ejecutables.

## Diseño

En TensorFlow Quantum se implementan los componentes necesarios para integrar TensorFlow con el hardware de computación cuántica. Con este fin, TensorFlow Quantum presenta dos tipos de datos primitivos:

- El *circuito cuántico*: representa a circuitos cuánticos definidos con Cirq. dentro de TensorFlow. Crea lotes de circuitos de diferentes tamaños, similar a los lotes de diferentes puntos de datos con valores reales.
- La *suma de Pauli*: representa a las combinaciones lineales de productos tensores de operadores de Pauli definidos en Cirq. Como los circuitos, crean lotes de operadores de distintos tamaños.

Mediante la utilización de estos datos primitivos para representar los circuitos cuánticos, TensorFlow Quantum proporciona las siguientes operaciones:

- La muestra de las distribuciones de salida de los lotes de los circuitos.
- El cálculo del valor esperado de los lotes de sumas de Pauli sobre lotes de circuitos. TFQ implementa el cálculo de gradiente compatible con la propagación hacia atrás.
- La simulación de los lotes de los circuitos y los estados. Si bien es cierto que la inspección de todas las amplitudes de estados cuánticos hecha directamente a través de un circuito cuántico es ineficiente a escala en el mundo real, la simulación de estados puede ayudar a que los investigadores entiendan cómo un circuito cuántico mapea (los estados) a un nivel de precisión próximo a la exactitud.

Lea más acerca de la implementación de TensorFlow Quantum en la [guía de diseño](design.md).

## Informe de problemas

Informe errores o solicitudes de características mediante el <a href="https://github.com/tensorflow/quantum/issues" class="external">rastreador de problemas de TensorFlow Quantum</a>.

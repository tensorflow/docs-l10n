# Arquitectura de XLA

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;"> <img style="width:50%" src="./images/xlalogo.png">
</div>

## ¿Por qué compilamos XLA?

Nos planteamos varios objetivos para que XLA funcionara con TensorFlow:

- *Mejorar la velocidad de ejecución.* Compile subgrafos para reducir el tiempo de ejecución de operaciones de corta duración y así eliminar la sobrecarga del tiempo de ejecución de TensorFlow, fusionar operaciones canalizadas para reducir la sobrecarga de memoria y especializarse en formas tensoriales conocidas para permitir una propagación constante más potente.

- *Mejorar el uso de la memoria.* Analice y programe el uso de la memoria, eliminando en principio muchos búferes de almacenamiento intermedio.

- *Reducir la dependencia de operaciones personalizadas.* Elimine la necesidad de ejecutar muchas operaciones personalizadas y mejore el rendimiento de las operaciones de bajo nivel fusionadas automáticamente para igualar el rendimiento de las operaciones personalizadas que se fusionaron manualmente.

- *Reducir la huella móvil.* Elimine el tiempo de ejecución de TensorFlow mediante la compilación anticipada del subgrafo y la emisión de un par de archivo objeto/encabezado que se pueda vincular directamente a otra aplicación. Los resultados pueden reducir la huella de la inferencia móvil en varios niveles de magnitud.

- *Mejorar la portabilidad.* Haga que sea relativamente fácil escribir un nuevo backend para un nuevo hardware, momento en el cual una gran parte de los programas de TensorFlow se ejecutarán sin modificaciones en ese hardware. Esto contrasta con el enfoque de especializar operaciones monolíticas individuales para nuevo hardware, que requiere que los programas de TensorFlow se reescriban para hacer uso de esas operaciones.

## ¿Cómo funciona XLA?

El idioma de entrada a XLA se llama "HLO IR", o simplemente HLO (operaciones de alto nivel, por sus siglas en inglés). La semántica de HLO se describe en la página [Semántica de operaciones](./operation_semantics.md). Lo más conveniente es pensar en HLO como un [IR de compilador](https://en.wikipedia.org/wiki/Intermediate_representation).

XLA toma gráficos ("cálculos") definidos en HLO y los compila en instrucciones de máquina para varias arquitecturas. XLA es modular en el sentido de que es fácil insertar un backend alternativo para [apuntar a alguna arquitectura de nuevo hardware](./developing_new_backend.md). El backend de la CPU para x64 y ARM64, así como el backend de la GPU NVIDIA, se encuentran en el árbol de fuentes de TensorFlow.

El siguiente diagrama muestra el proceso de compilación en XLA:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">   <img src="./images/how-does-xla-work.png">
</div>

XLA viene con varias optimizaciones y pases de análisis que son independientes del objetivo, como [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination), fusión de operaciones independiente del objetivo y análisis de búfer para asignar memoria de tiempo de ejecución para el cálculo.

Después del paso independiente del objetivo, XLA envía el cálculo de HLO a un backend. El backend puede ejecutar más optimizaciones a nivel de HLO, esta vez teniendo en cuenta la información y las necesidades específicas del objetivo. Por ejemplo, el backend de GPU de XLA puede fusionar operaciones que sean específicamente útiles para el modelo de programación de GPU y determinar cómo dividir el cálculo en flujos. En esta etapa, los servidores también pueden hacer coincidir patrones de ciertas operaciones o sus combinaciones con llamadas a librerías optimizadas.

El siguiente paso es la generación de código específico para el objetivo. Los backends de CPU y GPU que se incluyen con XLA usan [LLVM](http://llvm.org) para IR de bajo nivel, optimización y generación de código. Estos backends emiten la IR de LLVM necesaria para representar el cálculo de HLO de XLA de manera eficiente y luego invocan una LLVM para emitir código nativo desde esta IR de LLVM.

El backend de GPU actualmente admite GPU NVIDIA a través del backend LLVM NVPTX; el backend de la CPU admite múltiples ISA de CPU.

# Prácticas recomendadas sobre el rendimiento

Los dispositivos móviles e integrados tienen recursos computacionales limitados, por lo que es importante mantener la eficiencia de los recursos de su aplicación. Hemos recopilado una lista de las mejores prácticas y estrategias que puede usar para mejorar el rendimiento de su modelo TensorFlow Lite.

## Elija el mejor modelo para la tarea

Dependiendo de la tarea, tendrá que hacer un balance entre la complejidad y el tamaño del modelo. Si su tarea requiere una alta precisión, entonces es posible que necesite un modelo grande y complejo. Para tareas que requieren menos precisión, es mejor usar un modelo más pequeño porque no sólo utilizan menos espacio en disco y memoria, sino que también suelen ser más rápidos y más eficientes energéticamente. Por ejemplo, los siguientes gráficos muestran contrapartidas de precisión y latencia para algunos modelos comunes de clasificación de imágenes.

![Gráfico del tamaño del modelo contra la precisión](../images/performance/model_size_vs_accuracy.png "Model Size vs Accuracy")

![Gráfico de precisión contra la latencia](../images/performance/accuracy_vs_latency.png "Accuracy vs Latency")

Un ejemplo de modelos optimizados para dispositivos móviles son [MobileNets](https://arxiv.org/abs/1704.04861), que están optimizados para aplicaciones de visión móvil. [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) enumera varios otros modelos que han sido optimizados específicamente para dispositivos móviles e integrados.

Puede volver a entrenar los modelos listados en su propio conjunto de datos usando el aprendizaje por transferencia. Revise los tutoriales de aprendizaje por transferencia usando [Model Maker](../models/modify/model_maker/) de TensorFlow Lite.

## Haga un perfil de su modelo

Una vez que haya seleccionado un modelo candidato adecuado para su tarea, es una buena práctica perfilar y comparar su modelo. La [herramienta de evaluación del rendimiento TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) tiene un perfilador incorporado que muestra estadísticas de perfilado por operador. Esto puede ayudar a comprender los cuellos de botella en el rendimiento y qué operadores dominan el tiempo de cálculo.

También puede usar el [trazado de TensorFlow Lite](measurement#trace_tensorflow_lite_internals_in_android) para perfilar el modelo en su aplicación Android, usando el trazado estándar del sistema Android, y para visualizar las invocaciones del operario por tiempo con herramientas de perfilado basadas en GUI.

## Haga un perfil y optimize los operarios en el grafo

Si un operador en particular aparece con frecuencia en el modelo y, basándose en la creación de perfiles, usted descubre que es el operador que consume la mayor cantidad de tiempo, considere optimizar ese operador. Este escenario debería ser poco frecuente, ya que TensorFlow Lite dispone de versiones optimizadas para la mayoría de los operadores. Sin embargo, es posible que pueda escribir una versión más rápida de un op personalizado si conoce las restricciones en las que se ejecuta el operador. Revise la [guía de operadores personalizados](../guide/ops_custom).

## Optimice el modelo

La optimización de modelos persigue crear modelos más pequeños que, en general, sean más rápidos y más eficientes energéticamente, de modo que puedan implementarse en dispositivos móviles. TensorFlow Lite admite múltiples técnicas de optimización, como la cuantización.

Vea la [documentación sobre optimización de modelos](model_optimization) para más detalles.

## Ajuste el número de hilos

TensorFlow Lite admite kernels multihilo para muchos operadores. Puede aumentar el número de hilos y acelerar la ejecución de los operarios. Sin embargo, aumentar el número de hilos hará que su modelo use más recursos y potencia.

Para algunas aplicaciones, la latencia puede ser más importante que la eficiencia energética. Puede aumentar el número de hilos configurando el número de [hilos](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) del intérprete. La ejecución multihilo, sin embargo, tiene el costo de una mayor variabilidad del rendimiento en función de lo que se ejecute simultáneamente. Este es particularmente el caso de las apps móviles. Por ejemplo, las pruebas aisladas pueden mostrar un aumento de velocidad del doble frente a la ejecución con un único subproceso, pero, si otra app se está ejecutando al mismo tiempo, puede resultar en un rendimiento peor que la ejecución con un único subproceso.

## Elimine las copias redundantes

Si su aplicación no está cuidadosamente diseñada, puede haber copias redundantes en el momento de suministrar la entrada y leer la salida del modelo. Asegúrese de eliminar las copias redundantes. Si usa las API de alto nivel, como Java, asegúrese de revisar cuidadosamente la documentación para conocer las advertencias sobre el rendimiento. Por ejemplo, la API de Java es mucho más rápida si se usan `ByteBuffers` como [entradas](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175).

## Haga un perfil de su aplicación con herramientas específicas de la plataforma

Las herramientas específicas de la plataforma como [Android profiler](https://developer.android.com/studio/profile/android-profiler) y [Instruments](https://help.apple.com/instruments/mac/current/) aportan una gran cantidad de información de perfilado que puede usarse para depurar su app. A veces, el fallo de rendimiento puede no estar en el modelo, sino en partes del código de la aplicación que interactúan con el modelo. Asegúrese de familiarizarse con las herramientas de perfilado específicas de su plataforma y con las buenas prácticas para la misma.

## Evalúe si su modelo le es útil usar los aceleradores de hardware disponibles en el dispositivo

TensorFlow Lite ha añadido nuevas formas de acelerar modelos con hardware más rápido como GPUs, DSPs y aceleradores neuronales. Normalmente, estos aceleradores se exponen a través de [submódulos delegados](delegates) que se encargan de partes de la ejecución del intérprete. TensorFlow Lite puede usar delegados por:

- Usando la [API de redes neuronales](https://developer.android.com/ndk/guides/neuralnetworks/) de Android. Puede utilizar estos backends aceleradores de hardware para mejorar la velocidad y la eficiencia de su modelo. Para habilitar la API de redes neuronales, revise la guía del [Delegado de NNAPI](https://www.tensorflow.org/lite/android/delegates/nnapi).
- El delegado GPU está disponible en Android e iOS, usando OpenGL/OpenCL y Metal, respectivamente. Para probarlos, consulte el [tutorial](gpu) y la [documentación sobre el delegado de GPU](gpu_advanced).
- El delegado Hexagon está disponible en Android. Aprovecha el DSP Qualcomm Hexagon si está disponible en el dispositivo. Consulte el tutorial del [delegado de Hexagon](https://www.tensorflow.org/lite/android/delegates/hexagon) para saber más.
- Es posible crear su propio delegado si tiene acceso a hardware no estándar. Consulte [Delegados de TensorFlow Lite](delegates) para saber más.

Tenga en cuenta que algunos aceleradores funcionan mejor para distintos tipos de modelos. Algunos delegados sólo admiten modelos float o modelos optimizados de una forma específica. Es importante hacer [benchmark](measurement) de cada delegado para ver si es una buena elección para su aplicación. Por ejemplo, si tiene un modelo muy pequeño, puede que no valga la pena delegar el modelo ni en la API NN ni en la GPU. En cambio, los aceleradores son una gran elección para los modelos grandes que tienen una alta intensidad aritmética.

## Si necesita más ayuda

El equipo de TensorFlow estará encantado de ayudarle a diagnosticar y solucionar los problemas de rendimiento específicos que pueda tener. Registre un incidente en [GitHub](https://github.com/tensorflow/tensorflow/issues) con los detalles del problema.

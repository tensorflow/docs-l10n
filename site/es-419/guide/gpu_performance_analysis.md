# Optimice el rendimiento de la GPU de TensorFlow con TensorFlow Profiler

## Descripción general

Esta guía le mostrará cómo usar TensorFlow Profiler con TensorBoard para obtener información y el máximo rendimiento de sus GPU, y para depurar cuando una o más de sus GPU no se estén utilizando lo suficiente.

Si es la primera vez que usa Profiler:

- Empiece por consultar el bloc de notas [TensorFlow Profiler: rendimiento del modelo de perfil](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) con un ejemplo de Keras y [TensorBoard](https://www.tensorflow.org/tensorboard).
- Obtenga información sobre diferentes herramientas de perfilado y métodos disponibles para optimizar el rendimiento de TensorFlow en el host (CPU) con la guía [Use Profiler para optimizar el rendimiento de TensorFlow](https://www.tensorflow.org/guide/profiler#profiler_tools).

Tenga en cuenta que descargar computaciones a la GPU no siempre es conveniente, en especial cuando se trata de modelos pequeños. Puede haber sobrecarga por los siguientes motivos:

- La transferencia de datos entre el host (CPU) y el dispositivo (GPU); y
- la latencia que se produce cuando el host ejecuta los núcleos de la GPU.

### Flujo de trabajo de optimización del rendimiento

En esta guía se describe cómo depurar los problemas de rendimiento partiendo de una sola GPU para pasar posteriormente a un host con múltiples GPU.

Se recomienda depurar los problemas de rendimiento en el siguiente orden:

1. Optimizar y depurar el rendimiento en una GPU:
    1. Verifique si la canalización de entrada es un cuello de botella.
    2. Depure el rendimiento de una GPU.
    3. Habilite la precisión mixta (con `fp16` (float16)) y, opcionalmente, habilite [XLA](https://www.tensorflow.org/xla).
2. Optimizar y depurar el rendimiento en un único host con múltiples GPU.

Por ejemplo, si está usando una [estrategia de distribución](https://www.tensorflow.org/guide/distributed_training) de TensorFlow para entrenar un modelo en un único host con múltiples GPU y nota que la utilización de la GPU no es óptima, primero debe optimizar y depurar el rendimiento para una GPU antes de depurar el sistema con múltiples GPU.

Como punto de partida para obtener código de alto rendimiento en las GPU, en esta guía se asume que ya sabe usar `tf.function`. Las APi `Model.compile` y `Model.fit` de Keras usarán `tf.function` a nivel interno de forma automática. cuando escriba un bucle de entrenamiento personalizado con `tf.GradientTape`, consulte la guía [Mejor rendimiento con tf.function](https://www.tensorflow.org/guide/function) para aprender a habilitar `tf.function`.

En las siguientes secciones se analizan planteamientos sugeridos para cada uno de los escenarios mencionados anteriormente con el fin de ayudarle a identificar y corregir cuellos de botella en el rendimiento.

## 1. Optimice el rendimiento en una GPU

En un escenario ideal, el programa debería tener una alta utilización de la GPU, una comunicación mínima entre la CPU (el host) y la GPU (el dispositivo) y nada de sobrecarga en la canalización de entrada.

El primer paso de análisis del rendimiento consiste en obtener el perfil de un modelo que se ejecute con una GPU.

La [página de descripción general](https://www.tensorflow.org/guide/profiler#overview_page) de Profiler de TensorBoard, que muestra una vista desde arriba del rendimiento de su modelo durante la ejecución de un perfil, puede darle una idea de cuánto le falta a su programa para alcanzar ese escenario ideal.

![Página de descripción general de TensorFlow Profiler](images/gpu_perf_analysis/overview_page.png "The overview page of the TensorFlow Profiler")

Los números clave a los que debe prestarle atención en la página de descripción general son los siguientes:

1. Qué proporción del tiempo del paso corresponde a la ejecución real del dispositivo
2. Comparativa del porcentaje de operaciones colocadas en el dispositivo y en el host
3. Cuántos núcleos usa `fp16`

Conseguir un rendimiento óptimo supone maximizar estos números en los tres casos. Para comprender en profundidad el programa, deberá familiarizarse con el [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) de Profiler de TensorBoard. Las siguientes secciones muestran algunos patrones comunes del visor de seguimiento a las que debería prestar atención a la hora de diagnosticar cuellos de botella en el rendimiento.

A continuación, verá una imagen de la vista del seguimiento de un modelo que se ejecuta en una GPU. Desde las secciones *TensorFlow Name Scope* y *TensorFlow Ops*, puede identificar distintas partes del modelo, como el paso hacia adelante, la función de pérdida, el paso hacia atrás/el cálculo de gradiente, y la actualización del peso del optimizador. También puede tener las operaciones que se ejecutan en cada GPU junto a cada *Stream* (flujo), que hace referencia a los flujos CUDA. Cada flujo se usa para tareas específicas. En este seguimiento, *Stream#118* se usa para iniciar núcleos de cómputo y copias de dispositivo a dispositivo. *Stream#119* se utiliza para la copia de host a dispositivo y *Stream#120* para copia de dispositivo a host.

El seguimiento que se muestra a continuación presenta características comunes de un modelo de alto rendimiento.

![imagen](images/gpu_perf_analysis/traceview_ideal.png "An example TensorFlow Profiler trace view")

Por ejemplo, la línea de tiempo de cómputo de la GPU (*Stream#118*) parece "ocupada" con muy pocas brechas. Hay copias mínimas de host a dispositivo (*Stream #119*) y de dispositivo a host (*Stream #120*), así como brechas mínimas entre pasos. Cuando ejecuta Profiler para su programa, quizás no pueda identificar estas características ideales en su vista de seguimiento. El resto de esta guía aborda escenarios comunes y cómo solucionarlos.

### 1. Depure la canalización de entrada

El primer paso en la depuración del rendimiento de la GPU consiste en determinar si el programa está ligado a la entrada. La forma más sencilla de averiguarlo es utilizar el [analizador de canalización de entrada](https://www.tensorflow.org/guide/profiler#input_pipeline_analyzer) de Profiler, en TensorBoard, que ofrece una descripción general del tiempo que se emplea en la canalización de entrada.

![imagen](images/gpu_perf_analysis/input_pipeline_analyzer.png "TensorFlow Profiler Input-Analyzer")

Puede tomar las siguientes medidas potenciales si su canalización de entrada influye significativamente en el tiempo del paso:

- Puede utilizar la [guía](https://www.tensorflow.org/guide/data_performance_analysis) específica de `tf.data` para aprender a depurar su canalización de entrada.
- Otro método rápido para comprobar si la canalización de entrada es el cuello de botella consiste en utilizar datos de entrada generados aleatoriamente que no necesiten ningún preprocesamiento. [Este es un ejemplo](https://github.com/tensorflow/models/blob/4a5770827edf1c3974274ba3e4169d0e5ba7478a/official/vision/image_classification/resnet/resnet_runnable.py#L50-L57) del uso de esta técnica para un modelo ResNet. Si la canalización de entrada es óptima, debería experimentar un rendimiento similar con datos reales y con datos aleatorios o sintéticos generados. La única sobrecarga en el caso de los datos sintéticos corresponderá a la copia de los datos de entrada, que también puede preprocesarse y optimizarse.

Además, consulte las [mejores prácticas para optimizar la canalización de datos de entrada](https://www.tensorflow.org/guide/profiler#optimize_the_input_data_pipeline).

### 2. Depure el rendimiento de una GPU

Hay varios factores que pueden contribuir a una baja utilización de la GPU. Estos son algunos de los ejemplos más comunes que se observan en el [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) y sus posibles soluciones.

#### 1. Analice las brechas entre pasos

Una observación común cuando un programa no se ejecuta de forma óptima son las brechas entre los pasos de entrenamiento. En la imagen de la vista de seguimiento que aparece a continuación, hay una gran brecha entre los pasos 8 y 9, lo que sugiere que la GPU está inactiva durante ese tiempo.

![imagen](images/gpu_perf_analysis/traceview_step_gaps.png "TensorFlow Profile trace view showing gaps between steps")

Si su visor de seguimiento muestra grandes brechas entre los pasos, esto podría indicar que su programa está ligado a la entrada. En ese caso, debería consultare la sección anterior para aprender a depurar su canalización de entrada, en caso de que aún no lo haya hecho.

No obstante, incluso con una canalización de entrada optimizada, la contención de subprocesos de la CPU puede generar desfases entre el final de un paso y el inicio de otro. `tf.data` utiliza subprocesos en segundo plano para paralelizar el procesamiento de canalizaciones. Estos subprocesos pueden interferir con la actividad del host de la GPU que se lleva a cabo al principio de cada paso, como la copia de datos o la programación de operaciones de la GPU.

Si nota grandes brechas en el extremo del host, que programa estas operaciones en la GPU, puede establecer la variable de entorno `TF_GPU_THREAD_MODE=gpu_private`. Esto garantiza que los núcleos de la GPU se inicien desde sus propios subprocesos dedicados y no se agreguen a la cola de trabajo de `tf.data`.

Las brechas entre pasos también pueden deberse a cálculos métricos, retrollamadas de Keras u operaciones fuera de `tf.function` que se ejecutan en el host. Estas operaciones no tienen un rendimiento tan bueno como las operaciones que se ejecutan dentro de un gráfico de TensorFlow. Además, algunas de estas operaciones se ejecutan en la CPU y copian tensores desde y hacia la GPU.

Si, tras optimizar la canalización de entrada, sigue notando brechas entre pasos en el visor de seguimiento, debería revisar el código del modelo entre pasos y comprobar si al deshabilitar retrollamadas o métricas se mejora el rendimiento. El visor de seguimiento también muestra algunos detalles sobre estas operaciones (tanto del extremo del dispositivo como del host). La recomendación en este escenario consiste en amortizar la sobrecarga de estas operaciones ejecutándolas después de un número fijo de pasos en lugar de hacerlo en cada paso. Cuando se usa el método `Model.compile` en la API de `tf.keras`, establecer la marca `steps_per_execution` hace esto automáticamente. Para bucles de entrenamiento personalizados, use `tf.while_loop`.

#### 2. Consiga una mejor utilización del dispositivo

##### 1. Pequeños núcleos de GPU y retrasos en el inicio del núcleo de host

El host pone en cola los núcleos que se ejecutarán en la GPU, pero existe una latencia (de unos 20-40 μs) antes de que los núcleos se ejecuten realmente en la GPU. En un escenario ideal, el host pone en cola suficientes núcleos en la GPU para que esta pase la mayor parte del tiempo ejecutándolos, en lugar de esperar a que el host ponga en cola más núcleos.

La [página de descripción general](https://www.tensorflow.org/guide/profiler#overview_page) del Profiler en TensorBoard muestra cuánto tiempo estuvo inactiva la GPU esperando a que el host iniciara los núcleos. En la imagen que se muestra a continuación, la GPU está inactiva durante aproximadamente el 10 % del tiempo del paso esperando a que se inicien los núcleos.

![imagen](images/gpu_perf_analysis/performance_summary.png "Summary of performance from TensorFlow Profile")

El [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) para este mismo programa muestra pequeñas brechas entre núcleos mientras el host está ocupado iniciando núcleos en la GPU.

![imagen](images/gpu_perf_analysis/traceview_kernel_gaps.png "TensorFlow Profile trace view demonstrating gaps between kernels")

Al iniciar muchas operaciones pequeñas en la GPU (como una suma escalar, por ejemplo), es posible que el host no pueda seguir el ritmo de la GPU. La herramienta [TensorFlow Stats](https://www.tensorflow.org/guide/profiler#tensorflow_stats) en TensorBoard para el mismo perfil muestra 126 224 operaciones Mul que demoran 2,77 segundos. Por tanto, cada núcleo demora unos 21,9 μs, lo que es muy poco (prácticamente el mismo tiempo que la latencia de lanzamiento) y puede provocar retrasos en el inicio del núcleo del host.

![imagen](images/gpu_perf_analysis/tensorflow_stats_page.png "TensorFlow Profile stats page")

Si su [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) muestra muchas brechas pequeñas entre operaciones en la GPU como en la imagen de arriba, puede hacer lo siguiente:

- Concatene pequeños tensores y use operaciones vectorizadas o use un tamaño de lote más grande para que cada núcleo iniciado trabaje más, lo que mantendrá la GPU ocupada durante más tiempo.
- Asegúrese de estar usando `tf.function` para crear gráficos de TensorFlow, de modo que no esté ejecutando operaciones en modo eager puro. Si está utilizando `Model.fit` (en lugar de un bucle de entrenamiento personalizado con `tf.GradientTape`), entonces `tf.keras.Model.compile` hará esto por usted de forma automática.
- Use XLA para fusionar núcleos con `tf.function(jit_compile=True)` o agrupación automática. Para obtener más información, consulte la sección [Habilite la precisión mixta y XLA](#3._enable_mixed_precision_and_xla) a continuación para aprender cómo habilitar XLA para mejorar el rendimiento. Esta característica puede generar una alta utilización del dispositivo.

##### 2. Colocación de operaciones de TensorFlow

La [página de descripción general](https://www.tensorflow.org/guide/profiler#overview_page) de Profiler nos muestra de forma comparativa el porcentaje de operaciones colocadas en el host y en el dispositivo (también podemos verificar la colocación de operaciones específicas si miramos el [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer)). Como en la siguiente imagen, es conveniente que el porcentaje de operaciones en el host sea muy pequeño en comparación con el del dispositivo.

![imagen](images/gpu_perf_analysis/opp_placement.png "TF Op Placement")

Lo ideal es que la mayoría de las operaciones intensivas de cómputo se coloquen en la GPU.

Para averiguar a qué dispositivos se asignan las operaciones y los tensores de su modelo, establezca `tf.debugging.set_log_device_placement(True)` como primera instrucción de su programa.

Tenga en cuenta que, en algunos casos, aunque especifique que una operación debe colocarse en un dispositivo concreto, su implementación puede anular esta condición (por ejemplo:`tf.unique`). Incluso para el entrenamiento con una sola GPU, especificar una estrategia de distribución, como `tf.distribute.OneDeviceStrategy`, puede generar una colocación más determinista de las operaciones en el

Uno de los motivos por los que la mayoría de las operaciones se colocan en la GPU es para evitar un exceso de copias de memoria entre el host y el dispositivo (se espera que se realicen copias de memoria para los datos de entrada/salida del modelo entre el host y el dispositivo). Un ejemplo de copia excesiva se muestra en la vista de seguimiento a continuación en los flujos de GPU *#167*, *#168* y *#169*.

![imagen](images/gpu_perf_analysis/traceview_excessive_copy.png "TensorFlow Profile trace view demonstrating excessive H2D/D2H copies")

En ocasiones, estas copias pueden perjudicar el rendimiento si bloquean la ejecución de los núcleos de la GPU. Las operaciones de copia de memoria en el [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) tienen más información sobre las operaciones que son la fuente de estos tensores copiados, pero puede que no siempre sea fácil asociar una copia de memoria con una operación. En estos casos, es de gran ayuda mirar las operaciones cercanas para comprobar si la copia de memoria se produce en la misma ubicación en cada paso.

#### 3. Núcleos más eficientes en las GPU

Una vez que se alcanza una utilización aceptable de la GPU del programa, el siguiente paso consiste en tratar de aumentar la eficiencia de los núcleos de la GPU mediante el uso de Tensor Cores o la fusión de operaciones.

##### 1. Uso de Tensor Cores

Las GPU modernas de NVIDIA® cuentan con [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/) especializados que pueden mejorar considerablemente el rendimiento de los núcleos elegibles.

Puedes utilizar las [estadísticas de núcleos GPU](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) de TensorBoard para comprobar qué núcleos GPU son aptos para Tensor Core y qué núcleos están utilizando Tensor Cores. Activar `fp16` (consulte la sección Habilite la precisión mixta a continuación) es una forma de hacer que los núcleos de Multiplicación Matricial General (GEMM) de tu programa (operaciones matmul) utilice Tensor Core. Los núcleos de GPU usan Tensor Cores eficientemente cuando la precisión es fp16 y las dimensiones del tensor de entrada/salida se pueden dividir por 8 o 16 (para `int8`).

Nota: Con cuDNN v7.6.3 y versiones posteriores, las dimensiones de convolución se completarán automáticamente cuando sea necesario para aprovechar los Tensor Cores.

Si desea obtener más recomendaciones detalladas sobre cómo hacer que los núcleos sean eficientes para las GPU, consulte la guía de [rendimiento del aprendizaje profundo de NVIDIA®](https://docs.nvidia.com/deeplearning/performance/index.html#perf-guidelines).

##### 2. Fusión de operaciones

Use `tf.function(jit_compile=True)` para formar núcleos más grandes a partir de la fusión de operaciones más pequeñas, lo que se traduce en un aumento significativo del rendimiento. Para obtener más información, consulte la guía [XLA](https://www.tensorflow.org/xla).

### 3. Habilite la precisión mixta y XLA

Después de seguir los pasos mencionados anteriormente, habilitar la precisión mixta y XLA son dos medidas opcionales que puede adoptar para mejorar aún más el rendimiento. Se recomienda habilitarlos uno por uno y comprobar que el rendimiento sea el esperado.

#### 1. Habilitar la precisión mixta

La guía [Presición mixta](https://www.tensorflow.org/guide/keras/mixed_precision) de TensorFlow muestra cómo habilitar la precisión `fp16` en las GPU. Habilitar [AMP](https://developer.nvidia.com/automatic-mixed-precision) en las GPu de NVIDIA® para usar Tensor Cores y obtener hasta 3 veces más velocidad si se compara con el uso de la precisión `fp32` (float32) en Volta y arquitecturas de GPU más nuevas.

Asegúrese de que las dimensiones de la matriz/tensor cumplan con los requisitos para llamar a los núcleos que usan Tensor Cores. Los núcleos GPU utilizan los Tensor Cores eficientemente cuando la precisión es fp16 y las dimensiones de entrada/salida son divisibles por 8 o 16 (para int8).

Tenga en cuenta que con cuDNN v7.6.3 y versiones posteriores, las dimensiones de convolución se completarán automáticamente cuando sea necesario para aprovechar los Tensor Cores.

Para maximizar los beneficios de rendimiento de la precisión `fp16`, aplique las prácticas recomendadas que se indican a continuación.

##### 1. Uso de núcleos fp16 óptimos

Con `fp16` habilitada, los núcleos de multiplicación matricial (GEMM) de su programa deberían usar la versión de `fp16` correspondiente que use Tensor Cores. Sin embargo, en algunos casos, esto no sucede y no se experimenta la aceleración esperada al habilitar `fp16`, ya que el programa retoma la implementación ineficiente.

![imagen](images/gpu_perf_analysis/gpu_kernels.png "TensorFlow Profile GPU Kernel Stats page")

La página de estadísticas del [núcleo de la GPU](https://www.tensorflow.org/guide/profiler#gpu_kernel_stats) muestra qué operaciones son compatibles con Tensor Core y qué núcleos están utilizando realmente el Tensor Core eficiente. La guía de [NVIDIA® sobre rendimiento del aprendizaje profundo ](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores) contiene sugerencias adicionales sobre cómo aprovechar Tensor Cores. Además, los beneficios del uso de `fp16` también se verán en los núcleos que anteriormente estaban atados a la memoria, ya que ahora las operaciones tardarán la mitad de tiempo.

##### 2. Diferencias entre el escalado de pérdidas dinámico y el estático

El escalado de pérdidas es necesario cuando se utiliza `fp16` para evitar el bajo flujo debido a la baja precisión. Hay dios tipos de escalado de pérdida, dinámico y estático, y ambos se explican detalladamente en la [guía de Precisión mixta](https://www.tensorflow.org/guide/keras/mixed_precision). Puede usar la política `mixed_float16` para habilitar automáticamente el escalado de pérdidas en el optimizador Keras.

Nota: La API de precisión mixta Keras evalúa de forma predeterminada las operaciones softmax independientes (operaciones que no forman parte de una función de pérdida de Keras) como `fp16`, lo que puede generar problemas numéricos y una convergencia deficiente. Para obtener un rendimiento óptimo, convierta estas operaciones a `fp32`.

Al tratar de optimizar el rendimiento, es importante recordar que el escalado dinámico de pérdidas puede introducir operaciones condicionales adicionales que se ejecutan en el host y dar lugar a brechas entre los pasos que se podrán ver en el visor de seguimiento. Por otro lado, el escalado estático de pérdidas no presenta esas sobrecargas y puede ser una mejor opción en términos de rendimiento, con el detalle de que deberá especificar el valor de escala de pérdidas estática correcto.

#### 2. Habilitar XLA con tf.function(jit_compile=True) o agrupación automática

Como paso final para obtener el mejor rendimiento con una sola GPU, puede probar habilitando XLA, que fusionará las operaciones y tendrá como resultado una mejor utilización del dispositivo y un menor consumo de memoria. Si desea obtener información sobre cómo habilitar XLA en su programa mediante el uso de `tf.function(jit_compile=True)` o la agrupación automática, consulte la guía [XLA](https://www.tensorflow.org/xla).

Puede establecer el nivel de JIT global en `-1` (desactivado), `1`, o `2`. Un nivel más alto será más agresivo y podría reducir el paralelismo y consumir más memoria. Establezca el valor en `1` si tiene limitaciones de memoria. Tenga en cuenta que XLA no tiene un buen rendimiento en modelos con formas de tensor de entrada variables, ya que el compilador XLA debería continuar compilando núcleos cada vez que encontrara nuevas formas.

## 2. Optimice el rendimiento en un único host con múltiples GPU

La API `tf.distribute.MirroredStrategy` se puede usar para escalar el entrenamiento de un modelo de una GPU a múltiples GPU en un único host. (Para obtener más información sobre cómo llevar a cabo un entrenamiento distribuido con TensorFlow, consulte las guías [Aprendizaje distribuido con TensorFlow](https://www.tensorflow.org/guide/distributed_training), [Use una GPU](https://www.tensorflow.org/guide/gpu) y [Use TPU](https://www.tensorflow.org/guide/tpu) y el tutorial [Entrenamiento distribuido con Keras](https://www.tensorflow.org/tutorials/distribute/keras)).

Si bien la transición de una GPU a múltiples GPU debería ser escalable desde el primer momento, a veces surgen problemas de rendimiento.

Cuando se pasa de entrenar con una sola GPU a hacerlo con múltiples GPU en el mismo host, lo ideal es experimentar el escalado de rendimiento solo con la sobrecarga adicional de la comunicación de gradientes y el aumento del uso de los subprocesos del host. Como consecuencia de esta sobrecarga, la velocidad no se multiplicará exactamente por 2 si, por ejemplo, se pasa de 1 a 2 GPU.

La vista de seguimiento que se muestra a continuación ilustra la sobrecarga de comunicación adicional que se produce al entrenar en múltiples GPU. Se produce cierto grado de sobrecarga al concatenar los gradientes, comunicarlos entre las réplicas y dividirlos antes de actualizar los pesos.

![imagen](images/gpu_perf_analysis/traceview_multi_gpu.png "TensorFlow Profile trace view for single host multi GPU scenario")

La siguiente lista de comprobación le ayudará a obtener un mejor rendimiento cuando optimice el rendimiento en un escenario de múltiples GPU:

1. Trate de maximizar el tamaño del lote, lo que dará lugar a una mayor utilización del dispositivo y amortizará los costes de comunicación a través de múltiples GPU. Usar el [perfilador de memoria](https://www.tensorflow.org/guide/profiler#memory_profile_summary) ayuda a tener una idea de cuánto le falta al programa para alcanzar el pico de utilización de memoria. Tenga en cuenta que, si bien un tamaño de lote más grande puede afectar la convergencia, esto generalmente se compensa con los beneficios del rendimiento.
2. Al pasar de una sola GPU a varias, un mismo host ahora tiene que procesar muchos más datos de entrada. Así que, después del paso (1), se recomienda volver a comprobar el rendimiento de la canalización de entrada y asegurarse de que no se trate de un cuello de botella.
3. Compruebe la línea de tiempo de la GPU en la vista de seguimiento de su programa para ver si hay alguna llamada AllReduce innecesaria, ya que esto se traduce en una sincronización en todos los dispositivos. En la vista de seguimiento que se muestra más arriba, el proceso AllReduce se ejecuta a través del núcleo [NCCL](https://developer.nvidia.com/nccl) y hay una única llamada NCCL en cada GPU para los gradientes de cada paso.
4. Compruebe si hay operaciones de copia D2H, H2D y D2D innecesarias que puedan minimizarse.
5. Compruebe el tiempo del paso para asegurarse de que cada réplica esté haciendo el mismo trabajo. Por ejemplo, puede ocurrir que una GPU (normalmente, la `GPU0`) esté sobresuscrita porque el host, por error, le asigne más trabajo.
6. Por último, compruebe el paso de entrenamiento en todas las GPU de la vista de seguimiento para ver si hay operaciones que se ejecutan secuencialmente. Esto suele ocurrir cuando un programa incluye dependencias de control de una GPU a otra. En el pasado, la depuración del rendimiento en este supuesto se resolvía caso por caso. Si observa este comportamiento en su programa, [presente una incidencia en GitHub](https://github.com/tensorflow/tensorflow/issues/new/choose) con imágenes de su vista de seguimiento.

### 1. Optimice el gradiente AllReduce

Cuando se entrena con una estrategia sincrónica, cada dispositivo recibe una parte de los datos de entrada.

Después de procesar los pases hacia delante y hacia atrás a través del modelo, hay que agregar y reducir los gradientes calculados en cada dispositivo. Este *gradiente AllReduce* se lleva a cabo después del cálculo del gradiente en cada dispositivo y antes de que el optimizador actualice los pesos del modelo.

Cada GPU concatena primero los gradientes a través de las capas del modelo, los comunica a través de las GPU mediante `tf.distribute.CrossDeviceOps` (`tf.distribute.NcclAllReduce` es el valor predeterminado) y, a continuación, devuelve los gradientes tras la reducción por capa.

El optimizador usará estos gradientes reducidos para actualizar los pesos del modelo. Lo ideal es que este proceso se realice al mismo tiempo en todas las GPU para evitar sobrecargas.

El tiempo de AllReduce debería ser aproximadamente el mismo que el siguiente:

```
(number of parameters * 4bytes)/ (communication bandwidth)
```

Este cálculo es útil como una comprobación rápida para entender si el rendimiento que se obtiene al ejecutar un trabajo de entrenamiento distribuido es el esperado, o si es necesario continuar depurando el rendimiento. Puede obtener el número de parámetros de su modelo en `Model.summary`.

Tenga en cuenta que cada parámetro del modelo tiene un tamaño de 4 bytes ya que TensorFlow utiliza `fp32` (float32) para comunicar gradientes. Incluso si se habilitó `fp16`, NCCL AllReduce usa parámetros `fp32`.

Para obtener los beneficios del escalado, el tiempo del paso tiene que ser mucho mayor en comparación con estas sobrecargas. Una forma de conseguirlo es utilizar un tamaño de lote más grande, ya que el tamaño de lote afecta al tiempo del paso, pero no a la sobrecarga de comunicación.

### 2. Contención de subprocesos en el host de la GPU

Cuando se ejecutan múltiples GPU, el trabajo de la CPU es mantener ocupados todos los dispositivos al iniciar los núcleos de la GPU de forma eficiente en todos los dispositivos.

Sin embargo, cuando hay muchas operaciones independientes que la CPU puede programar en una GPU, la CPU puede decidir si utiliza muchos de sus subprocesos de host para mantener ocupada una GPU y, a continuación, iniciar los núcleos en otra GPU en un orden no determinista. Esto puede generar un sesgo o un escalado negativo, lo que podría afectar negativamente al rendimiento.

El [visor de seguimiento](https://www.tensorflow.org/guide/profiler#trace_viewer) a continuación muestra la sobrecarga cuando la CPU escalona los inicios del núcleo de la GPU de forma ineficiente, ya que la`GPU1` está inactiva y luego comienza a ejecutar operaciones después de que la `GPU2` se haya iniciado.

![imagen](images/gpu_perf_analysis/traceview_gpu_idle.png "TensorFlow Profile device trace view demonstrating inefficient kernel launch")

La vista de seguimiento del host muestra que el host está iniciando núcleos en la `GPU2` antes de iniciarlos en la `GPU1` (Tenga en cuenta que las operaciones `tf_Compute*` a continuación no indican subprocesos de CPU).

![imagen](images/gpu_perf_analysis/traceview_host_contention.png "TensorFlow Profile host trace view demonstrating inefficient kernel launch")

Si experimenta este tipo de escalonamiento de los núcleos de la GPU en la vista de seguimiento de su programa, le recomendamos que tome las siguientes medidas:

- Establezca la variable de entorno de TensorFlow `TF_GPU_THREAD_MODE` en `gpu_private`. Esta variable de entorno le indicará al host que mantenga privados los subprocesos para una GPU.
- Por defecto,`TF_GPU_THREAD_MODE=gpu_private` establece el número de subprocesos en 2, lo que, en la mayoría de los casos, suele ser suficiente. No obstante, ese número puede cambiarse al configurar la variable de entorno de TensorFlow `TF_GPU_THREAD_COUNT` en el número de subprocesos deseado.

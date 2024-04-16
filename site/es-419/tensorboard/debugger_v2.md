# Depuración de problemas numéricos en programas de TensorFlow mediante el depurador TensorBoard V2

> *NOTA*: tf.debugging.experimental.enable_dump_debug_info() es una API experimental y puede estar sujeta a cambios de última hora en el futuro.

A veces, durante un programa TensorFlow pueden producirse eventos catastróficos que afectan a [NaN](https://en.wikipedia.org/wiki/NaN)s, y que paralizan los procesos de entrenamiento del modelo. La causa raíz de tales eventos con frecuencia es poco clara, especialmente en el caso de modelos de tamaño y complejidad no triviales. Para facilitar la depuración de este tipo de errores en los modelos, TensorBoard 2.3+ (junto con TensorFlow 2.3+) proporciona un tablero especializado llamado Debugger V2. Aquí demostramos cómo utilizar esta herramienta trabajando mediante un error real que implica NaNs en una red neuronal escrita en TensorFlow.

Las técnicas ilustradas en este tutorial son aplicables a otros tipos de actividades de depuración, como la inspección de formas de tensores en tiempo de ejecución en programas complejos. Este tutorial se centra en los NaN debido a su relativamente alta frecuencia de aparición.

## Observando el error

El código fuente del programa TF2 que depuraremos está [disponible en GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/debug/examples/v2/debug_mnist_v2.py). El programa de ejemplo también está empaquetado en el paquete pip de tensorflow (versión 2.3+) y puede invocarse mediante:

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2
```

Este programa TF2 crea un perceptrón multicapa (MLP) y lo entrena para reconocer imágenes [MNIST](https://en.wikipedia.org/wiki/MNIST_database). En este ejemplo se utiliza a propósito la API de bajo nivel de TF2 para definir las construcciones personalizadas de las capas, la función de pérdida y el bucle de entrenamiento, porque la probabilidad de que se produzcan errores NaN es mayor cuando utilizamos esta API más flexible pero más propensa a errores que cuando utilizamos las API de alto nivel más fáciles de usar pero ligeramente menos flexibles, como [tf.keras](https://www.tensorflow.org/guide/keras).

El programa imprime una precisión de prueba después de cada paso de entrenamiento. Podemos ver en la consola que la precisión de la prueba se estanca en un nivel cercano a la probabilidad (~0.1) después del primer paso. Desde luego, no es así como se espera que se comporte el entrenamiento del modelo: esperamos que la precisión se acerque gradualmente a 1.0 (100%) a medida que aumenta el paso.

```
Accuracy at step 0: 0.216
Accuracy at step 1: 0.098
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
...
```

Una hipótesis fundamentada es que este problema está causado por una inestabilidad numérica, como NaN o infinito. Sin embargo, ¿cómo confirmamos que este es realmente el caso y cómo encontramos la operación (op) de TensorFlow responsable de generar la inestabilidad numérica? Para responder a estas preguntas, procedamos a instrumentalizar el programa que presenta errores con el Depurador V2.

## Instrumentalización del código de TensorFlow con el Depurador V2

[`tf.debugging.experimental.enable_dump_debug_info()`](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info) es el punto de entrada de la API del depurador V2. Instrumenta un programa TF2 con una sola línea de código. Por ejemplo, si añade la siguiente línea cerca del principio del programa, la información de depuración se escribirá en el directorio de registro (logdir) en /tmp/tfdbg2_logdir. La información de depuración cubre varios aspectos del tiempo de ejecución de TensorFlow. En TF2, incluye la historia completa de la ejecución eager, la construcción de gráficos realizada por [@tf.function](https://www.tensorflow.org/api_docs/python/tf/function), la ejecución de los gráficos, los valores de tensor generados por los eventos de ejecución, así como la localización del código (Python stack traces) de esos eventos. La riqueza de la información de depuración permite a los usuarios acotar errores poco claros.

```py
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
```

El argumento `tensor_debug_mode` controla la información que el Depurador V2 extrae de cada tensor eager o en el gráfico. "FULL_HEALTH" es un modo que captura la siguiente información sobre cada tensor de tipo flotante (por ejemplo, el comúnmente visto float32 y el menos común [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) dtype):

- DType
- Rank
- Número total de elementos
- Desglose de los elementos de tipo flotante en las siguientes categorías: finito negativo (`-`), cero (`0`), finito positivo (`+`), infinito negativo (`-∞`), infinito positivo (`+∞`) y `NaN`.

El modo "FULL_HEALTH" es adecuado para depurar errores que implican NaN e infinito. Consulte a continuación que otros `tensor_debug_mode` son compatibles.

El argumento `circular_buffer_size` controla el número de eventos de tensor que se guardan en el logdir. De forma predeterminada es 1000, lo que hace que sólo se guarden en el disco los últimos 1000 tensores antes del final del programa TF2 instrumentado. Este comportamiento predeterminado reduce la sobrecarga del depurador sacrificando la integridad de los datos de depuración. Si se prefiere la integridad, como en este caso, podemos desactivar el búfer circular al establecer el argumento en un valor negativo (por ejemplo, en este caso es -1).

El ejemplo debug_mnist_v2 llama a `enable_dump_debug_info()` al pasarle banderas de línea de comandos. Para volver a ejecutar nuestro programa TF2 problemático con esta instrumentación de depuración activada, haga lo siguiente:

```sh
python -m tensorflow.python.debug.examples.v2.debug_mnist_v2 \
    --dump_dir /tmp/tfdbg2_logdir --dump_tensor_debug_mode FULL_HEALTH
```

## Iniciando la GUI del Depurador V2 en el TensorBoard

La ejecución del programa con la instrumentación del depurador crea un logdir en /tmp/tfdbg2_logdir. Podemos iniciar el TensorBoard y dirigirlo al logdir con:

```sh
tensorboard --logdir /tmp/tfdbg2_logdir
```

En el navegador web, navegue hasta la página de TensorBoard en http://localhost:6006. El complemento "Depurador V2" estará inactivo de forma predeterminada, así que selecciónelo en el menú "Complementos inactivos" situado en la parte superior derecha. Una vez seleccionado, debería tener el siguiente aspecto:

![Debugger V2 full view screenshot](https://gitlocalize.com/repo/4592/es/site/en-snapshot/tensorboard/images/debugger_v2_1_full_view.png)

## Uso de la GUI del depurador V2 para encontrar la causa raíz de los NaN

La GUI del Depurador V2 en el TensorBoard está organizada en seis secciones:

- **Alertas**: Esta sección superior izquierda contiene una lista de eventos de "alerta" detectados por el depurador en los datos de depuración del programa TensorFlow instrumentado. Cada alerta indica una anomalía determinada que merece atención. En nuestro caso, esta sección destaca 499 eventos NaN/∞ con un llamativo color rosa-rojo. Esto confirma nuestra sospecha de que el modelo no aprende debido a la presencia de NaN y/o infinitos en sus valores de tensor internos. Profundizaremos en estas alertas muy pronto.
- **Línea de tiempo de ejecución de Python**: Esta es la mitad superior de la sección superior-promedio. Presenta el historial completo de la ejecución eager de ops y gráficos. Cada casilla de la línea de tiempo está marcada por la letra inicial del nombre de la op o del gráfico (por ejemplo, "T" para la op "TensorSliceDataset", "m" para el "modelo" `tf.function`). Podemos navegar por esta línea de tiempo utilizando los botones de navegación y la barra de desplazamiento situada encima de la línea de tiempo.
- **Ejecución de gráficos** : Situada en la esquina superior derecha de la interfaz gráfica de usuario, esta sección será fundamental para nuestra tarea de depuración. Contiene un historial de todos los tensores de tipo flotante computados dentro de gráficos (es decir, compilados por `@tfunción`).
- **Estructura del gráfico** (mitad inferior de la sección superior-promedio), **Código fuente** (sección inferior-izquierda), y **Rastreo de la pila** (sección inferior-derecha) están inicialmente vacías. Su contenido se rellenará cuando interactuemos con la GUI. Estas tres secciones también desempeñarán papeles importantes en nuestra tarea de depuración.

Después de habernos orientado en la organización de la interfaz del usuario, sigamos los siguientes pasos para llegar al fondo de por qué aparecieron los NaN. En primer lugar, haga clic en la alerta **NaN/∞** de la sección Alertas. Esto desplazará automáticamente la lista de 600 tensores del gráfico en la sección Ejecución del gráfico y se centrará en el #88, que es un tensor llamado `Log:0` generado por una operación `Log` (logaritmo natural). Un destacado color rosa-rojo resalta un elemento -∞ entre los 1000 elementos del tensor 2D float32. Este es el primer tensor en el historial de ejecución del programa TF2 que contenía algún NaN o infinito: los tensores calculados antes de él no contienen NaN o ∞; muchos (de hecho, la mayoría) de los tensores calculados después contienen NaN. Podemos confirmarlo desplazándonos hacia arriba y hacia abajo por la lista de ejecución del gráfico. Esta observación proporciona una fuerte pista de que el op `Log` es la fuente de la inestabilidad numérica en este programa TF2.

![Debugger V2: Nan / Infinity alerts and graph execution list](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/debugger_v2_2_nan_inf_alerts.png?raw=true)

¿Por qué este op `Log` arroja un -∞? Responder a esa pregunta requiere examinar la entrada a la op. Al hacer clic en el nombre del tensor (`Log:0`) aparecerá una visualización sencilla pero informativa de la vecindad de la op `Log` en su gráfico de TensorFlow en la sección Estructura del gráfico. Observe la dirección ascendente y descendente del flujo de información. El op en sí se muestra en negritas en el centro. Inmediatamente por encima de ella, podemos ver que una op Marcador de posición proporciona la única entrada a la op `Log`. ¿Dónde está el tensor generado por este `probs` marcador de posición en la lista de ejecución del gráfico? Utilizando el color de fondo amarillo como ayuda visual, podemos ver que el tensor `probs:0` se encuentra tres filas por encima del tensor `Log:0`, es decir, en la fila 85.

![Debugger V2: Graph structure view and tracing to input tensor](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/debugger_v2_3_graph_input.png?raw=true)

Una mirada más atenta al desglose numérico del tensor `probs:0` en la fila 85 revela por qué su consumidor `Log:0` produce un -∞: Entre los 1,000 elementos de `probs:0`, un elemento tiene el valor 0. ¡El -∞ es el resultado de calcular el logaritmo natural de 0! Si podemos asegurarnos de alguna manera de que el op `Log` sólo se exponga a entradas positivas, podremos evitar que se produzca el NaN/∞. Esto puede lograrse al aplicar el recorte (por ejemplo, utilizando [`tf.clip_by_value()`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)) en el tensor del marcador de posición `probs`.

Nos estamos acercando a la solución del error, pero aún no hemos terminado. Para aplicar la corrección, necesitamos saber en qué parte del código fuente de Python se originaron la op `Log` y su entrada de marcador de posición. El depurador V2 proporciona un soporte de primera clase para rastrear los ops del gráfico y los eventos de ejecución hasta su origen. Cuando pulsamos el tensor `Log:0` en Ejecuciones de gráfico, la sección Rastreo de la pila se rellenó con el rastreo de la pila original de la creación del op `Log`. El seguimiento de la pila es algo extenso porque incluye muchos marcos del código interno de TensorFlow (por ejemplo, gen_math_ops.py y dumping_callback.py), que podemos ignorar con seguridad para la mayoría de las tareas de depuración. El marco de interés es la línea 216 de debug_mnist_v2.py (es decir, el archivo Python que realmente estamos tratando de depurar). Al hacer clic en "Línea 216" aparece una vista de la línea de código correspondiente en la sección del Código fuente.

![Debugger V2: Source code and stack trace](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/debugger_v2_4_source_code.png?raw=true)

Esto nos lleva finalmente al código fuente que creó la problemática `Log` op a partir de su `probs` de entrada. Esta es nuestra función de pérdida de entropía cruzada categórica personalizada decorada con `@tf.function` y por lo tanto convertida en un gráfico de TensorFlow. El marcador de posición op `probs` corresponde al primer argumento de entrada de la función de pérdida. El op `Log` se crea con la llamada a la API tf.math.log().

La solución de recorte de valor para este error tendrá un aspecto similar:

```py
  diff = -(labels *
           tf.math.log(tf.clip_by_value(probs), 1e-6, 1.))
```

Resolverá la inestabilidad numérica en este programa TF2 y hará que el MLP se entrene correctamente. Otro enfoque posible para solucionar la inestabilidad numérica es utilizar [`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy).

Con esto concluye nuestro viaje desde la observación de un error en el modelo TF2 hasta la elaboración de un cambio de código que soluciona el error, con la ayuda de la herramienta Depurador V2, que proporciona una visibilidad completa del historial de ejecución eager y del gráfico del programa TF2 instrumentado, incluidos los resúmenes numéricos de los valores de los tensores y la asociación entre los ops, los tensores y su código fuente original.

## Compatibilidad con el hardware del Depurador V2

El Depurador V2 es compatible con el hardware de entrenamiento habitual, incluidas la CPU y la GPU. También es compatible el entrenamiento multi-GPU con [tf.distributed.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy). El soporte para [TPU](https://www.tensorflow.org/guide/tpu) está aún en una fase inicial y requiere llamar a

```py
tf.config.set_soft_device_placement(True)
```

antes de llamar a `enable_dump_debug_info()`. También puede tener otras limitaciones en TPU. Si se encuentra con problemas al utilizar el Depurador V2, por favor informe de los errores en nuestra [página de problemas de GitHub](https://github.com/tensorflow/tensorboard/issues).

## Compatibilidad API del Depurador V2

El Depurador V2 se implementa en un nivel relativamente bajo de la pila de software de TensorFlow, y por lo tanto es compatible con [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras), [tf.data](https://www.tensorflow.org/guide/data), y otras API construidas sobre los niveles inferiores de TensorFlow. El Depurador V2 también es retrocompatible con TF1, aunque la Línea de tiempo de ejecución eager estará vacía para los logdirs de depuración generados por los programas TF1.

## Consejos de uso de la API

Una pregunta frecuente sobre esta API de depuración es en qué parte del código de TensorFlow se debe insertar la llamada a `enable_dump_debug_info()`. Normalmente, la API debe llamarse lo antes posible en su programa TF2, preferiblemente después de las líneas de importación de Python y antes de que comience la construcción y ejecución del gráfico. Esto asegurará una cobertura completa de todas las operaciones y gráficos que alimentan su modelo y su entrenamiento.

Los modos tensor_debug_modes compatibles actualmente son: `NO_TENSOR`, `CURT_HEALTH`, `CONCISE_HEALTH`, `FULL_HEALTH`, y `SHAPE`. Varían en la cantidad de información extraída de cada tensor y en la sobrecarga de rendimiento para el programa depurado. Consulte la [sección args](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info) de la documentación de `enable_dump_debug_info()`.

## Sobrecarga de rendimiento

La API de depuración introduce una sobrecarga de rendimiento en el programa TensorFlow instrumentado. La sobrecarga varía según `tensor_debug_mode`, el tipo de hardware y la naturaleza del programa de TensorFlow instrumentado. Como punto de referencia, en una GPU, el modo `NO_TENSOR` aumenta la sobrecarga en un 15% durante el entrenamiento de un [modelo transformador](https://github.com/tensorflow/models/tree/master/official/legacy/transformer) con un tamaño de lote de 64. El porcentaje de sobrecarga para otros modos tensor_debug_modes es mayor: aproximadamente un 50% para los modos `CURT_HEALTH`, `CONCISE_HEALTH`, `FULL_HEALTH` y `SHAPE`. En las CPU, la sobrecarga es ligeramente inferior. En las TPU, la sobrecarga es actualmente superior.

## Relación con otras API de depuración de TensorFlow

Tenga en cuenta que TensorFlow ofrece otras herramientas y API para la depuración. Puede explorar dichas API dentro del espacio de nombres [`tf.debugging.*`](https://www.tensorflow.org/api_docs/python/tf/debugging) en la página de documentación de las API. Entre estas API, la más utilizada es [`tf.print()`](https://www.tensorflow.org/api_docs/python/tf/print). ¿Cuándo se debe utilizar el Depurador V2 y cuándo se debe utilizar `tf.print()` en su lugar? `tf.print()` es conveniente en caso de que:

1. sepamos exactamente qué tensores imprimir,
2. sabemos exactamente en qué parte del código fuente insertar esas sentencias `tf.print()`,
3. el número de estos tensores no es demasiado grande.

Para otros casos (por ejemplo, el examen de muchos valores del tensor, el examen de los valores del tensor generados por el código interno de TensorFlow, y la búsqueda del origen de la inestabilidad numérica como mostramos anteriormente), el Depurador V2 proporciona una forma más rápida de depuración. Además, el Depurador V2 proporciona un enfoque unificado para inspeccionar tensores ávidos y gráficos. Además, proporciona información sobre la estructura del gráfico y las ubicaciones del código, que están más allá de la capacidad de `tf.print()`.

Otra API que puede utilizarse para depurar problemas relacionados con ∞ y NaN es [`tf.debugging.enable_check_numerics()`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics). A diferencia de `enable_dump_debug_info()`, `enable_check_numerics()` no guarda la información de depuración en el disco. En cambio, se limita a supervisar ∞ y NaN durante el tiempo de ejecución de TensorFlow y los errores con la ubicación del código de origen tan pronto como cualquier op genera tales valores numéricos malos. Tiene una menor sobrecarga de rendimiento en comparación con `enable_dump_debug_info()`, pero no permite un seguimiento completo de la historia de ejecución del programa y no viene con una interfaz gráfica de usuario como el Depurador V2.

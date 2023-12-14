# Preguntas frecuentes

Si no encuentra aquí una respuesta a su pregunta, consulte nuestra documentación detallada sobre el tema o reporte un [problema de GitHub](https://github.com/tensorflow/tensorflow/issues).

## Conversión de modelos

#### ¿Qué formatos son compatibles para la conversión de TensorFlow a TensorFlow Lite?

Los formatos admitidos se enumeran [aquí](../models/convert/index#python_api)

#### ¿Por qué algunas operaciones no se implementan en TensorFlow Lite?

Para mantener la ligereza de TFLite, sólo ciertos operadores TF (enumerados en la [lista de permitidos](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/op_select_allowlist.md)) están soportados en TFLite.

#### ¿Por qué no se convierte mi modelo?

Dado que el número de operaciones de TensorFlow Lite es menor que el de TensorFlow, es posible que algunos modelos no puedan realizar la conversión. [Aquí](../models/convert/index#conversion-errors) se enumeran algunos errores comunes.

Para problemas de conversión no relacionados con operaciones faltantes u ops de flujo de control, busque en nuestros [problemas de GitHub](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+) o reporte uno [nuevo](https://github.com/tensorflow/tensorflow/issues).

#### ¿Cómo pruebo que un modelo TensorFlow Lite se comporta igual que el modelo TensorFlow original?

La mejor forma de comprobarlo es comparar las salidas de los modelos TensorFlow y TensorFlow Lite para las mismas entradas (datos de prueba o entradas aleatorias) como se muestra [aquí](inference#load-and-run-a-model-in-python).

#### ¿Cómo determino las entradas/salidas para el búfer de protocolo GraphDef?

La forma más sencilla de inspeccionar un grafo de un archivo `.pb` es usar [Netron](https://github.com/lutzroeder/netron), un visor de código abierto para modelos de aprendizaje automático.

Si Netron no puede abrir el grafo, puede probar con la herramienta [summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs).

Si la herramienta summarize_graph arroja un error, puede visualizar el GraphDef con [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) y buscar las entradas y salidas en el grafo. Para visualizar un archivo `.pb`, use el script [`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py) como el que se muestra a continuación:

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### ¿Cómo puedo inspeccionar un archivo `.tflite`?

[Netron](https://github.com/lutzroeder/netron) es la forma más sencilla de visualizar un modelo TensorFlow Lite.

Si Netron no puede abrir su modelo TensorFlow Lite, puede probar con el script [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) de nuestro repositorio.

Si está usando TF 2.5 o una versión posterior,

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

De lo contrario, puede ejecutar este script con Bazel

- [Clone el repositorio TensorFlow](https://www.tensorflow.org/install/source)
- Ejecute el script `visualize.py` con bazel:

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## Optimización

#### ¿Cómo puedo reducir el tamaño de mi modelo TensorFlow Lite convertido?

La [cuantización postentrenamiento](../performance/post_training_quantization) puede usarse durante la conversión a TensorFlow Lite para reducir el tamaño del modelo. La cuantización postentrenamiento cuantiza las ponderaciones a 8 bits de precisión de punto flotante y las decuantiza durante el runtime para realizar cálculos de punto flotante. Sin embargo, tenga en cuenta que esto podría tener algunas implicaciones de precisión.

Si es posible volver a entrenar el modelo, considere [el entrenamiento consciente de la cuantización](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize). Sin embargo, tenga en cuenta que el entrenamiento consciente de la cuantización sólo está disponible para un subconjunto de arquitecturas de redes neuronales convolucionales.

Para conocer mejor los distintos métodos de optimización, consulte [Optimización de modelos](../performance/model_optimization).

#### ¿Cómo puedo optimizar el rendimiento de TensorFlow Lite para mi tarea de aprendizaje automático?

El proceso de alto nivel para optimizar el rendimiento de TensorFlow Lite es más o menos así:

- *Asegúrese de que dispone del modelo adecuado para la tarea.* Para la clasificación de imágenes, consulte el [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification).
- *Ajuste el número de hilos.* Muchos operadores de TensorFlow Lite admiten kernels multihilo. Puede usar `SetNumThreads()` en la [API en C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/interpreter_builder.h#L110) para hacerlo. Sin embargo, aumentar los hilos resulta en un rendimiento variable en función del entorno.
- *Use aceleradores de hardware.* TensorFlow Lite admite la aceleración de modelos para hardware específico usando delegados. Consulte nuestra guía sobre [Delegados](../performance/delegates) para más información sobre qué aceleradores son admitidos y cómo usarlos con su modelo en el dispositivo.
- *(Avanzado) Modelo de perfil.* La [herramienta de benchmarking](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) de Tensorflow Lite tiene un perfilador incorporado que puede mostrar estadísticas por operador. Si sabe cómo puede optimizar el rendimiento de un operador para su plataforma específica, puede implementar un [operador personalizado](ops_custom).

Para una conversación más profunda sobre cómo optimizar el rendimiento, eche un vistazo a [Prácticas recomendadas](../performance/best_practices).

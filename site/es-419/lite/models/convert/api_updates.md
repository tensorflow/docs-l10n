# Actualizaciones de la API <a name="api_updates"></a>

Esta página proporciona información sobre las actualizaciones realizadas en la [API de Python](index.md) de `tf.lite.TFLiteConverter` en TensorFlow 2.x.

Nota: Si alguno de los cambios le preocupa, levante un [GitHub issue](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

- TensorFlow 2.3

    - Admite el tipo de entrada/salida entero (antes, sólo flotante) para modelos cuantificados enteros usando los nuevos atributos `inference_input_type` e `inference_output_type`. Consulte este [ejemplo de uso](../../performance/post_training_quantization.md#integer_only).
    - Admite la conversión y el redimensionamiento de modelos con dimensiones dinámicas.
    - Se ha añadido un nuevo modo de cuantización experimental con activaciones de 16 bits y ponderaciones de 8 bits.

- TensorFlow 2.2

    - De forma predeterminada, aprovecha la conversión basada en [MLIR](https://mlir.llvm.org/), la tecnología de compilación de vanguardia de Google para el aprendizaje automático. Esto permite la conversión de nuevas clases de modelos, incluidos Mask R-CNN, Mobile BERT, etc. y admite modelos con flujo de control funcional.

- TensorFlow 2.0 vs TensorFlow 1.x

    - Se cambió el nombre del atributo `target_ops` a `target_spec.supported_ops`.
    - Se eliminaron los siguientes atributos:
        - *cuantización*: `inference_type`, `quantized_input_stats`, `post_training_quantize`, `default_ranges_stats`, `reorder_across_fake_quant`, `change_concat_input_ranges`, `get_input_arrays()`. En su lugar, [el entrenamiento consciente de la cuantización](https://www.tensorflow.org/model_optimization/guide/quantization/training) se soporta a través de la API `tf.keras` y [la cuantización posterior al entrenamiento](../../performance/post_training_quantization.md) usa menos atributos.
        - *visualización*: `output_format`, `dump_graphviz_dir`, `dump_graphviz_video`. En su lugar, el enfoque recomendado para visualizar un modelo TensorFlow Lite es usar [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py).
        - *grafos congelados*: `drop_control_dependency`, ya que los grafos congelados no están soportados en TensorFlow 2.x.
    - Se eliminaron otras API de conversores como `tf.lite.toco_convert` y `tf.lite.TocoConverter`.
    - Se eliminaron otras API relacionadas, como `tf.lite.OpHint` y `tf.lite.constants` (los tipos `tf.lite.constants.*` se han mapeado a los tipos de datos TensorFlow `tf.*`, para reducir la duplicación)

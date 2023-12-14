# Compatibilidad de los operadores TensorFlow Lite y TensorFlow

Los operadores de aprendizaje automático (ML) que use en su modelo pueden afectar al proceso de conversión de un modelo TensorFlow al formato TensorFlow Lite. El convertidor TensorFlow Lite soporta un número limitado de operaciones TensorFlow usadas en modelos de inferencia comunes, lo que significa que no todos los modelos son directamente convertibles. La herramienta del convertidor le permite incluir operadores adicionales, pero convertir un modelo de esta forma también requiere que modifique el entorno runtime de TensorFlow Lite que utiliza para ejecutar su modelo, lo que puede limitar su capacidad para usar opciones de implementación estándar en tiempo de ejecución, como [servicios de Google Play](../android/play_services).

El conversor de TensorFlow Lite está diseñado para analizar la estructura del modelo y aplicar optimizaciones con el fin de hacerlo compatible con los operadores admitidos directamente. Por ejemplo, según cuáles sean los operadores ML de su modelo, el conversor puede [eliminar o fusionar](../models/convert/operation_fusion) esos operadores para mapearlos a sus homólogos de TensorFlow Lite.

Incluso para las operaciones admitidas, a veces se esperan patrones de uso específicos, por motivos de rendimiento. La mejor manera de entender cómo generar un modelo TensorFlow que pueda usarse con TensorFlow Lite es considerar detenidamente cómo se convierten y optimizan las operaciones, junto con las limitaciones impuestas por este proceso.

## Operadores admitidos

Los operadores incorporados en TensorFlow Lite son un subconjunto de los operadores que forman parte de la librería central de TensorFlow. Su modelo TensorFlow también puede incluir operadores personalizados en forma de operadores compuestos o nuevos operadores definidos por usted. El siguiente diagrama muestra las relaciones entre estos operadores.

![Operadores de TensorFlow](../images/convert/tf_operators_relationships.png)

De esta gama de operadores de modelos ML, existen 3 tipos de modelos compatibles con el proceso de conversión:

1. Modelos con sólo el operador incorporado TensorFlow Lite. (**Recomendado**)
2. Modelos con los operadores incorporados y operadores centrales selectos de TensorFlow.
3. Modelos con los operadores incorporados, los operadores básicos de TensorFlow y/o operadores personalizados.

Si su modelo sólo contiene operaciones soportadas de forma nativa por TensorFlow Lite, no necesita ningún indicador adicional para convertirlo. Esta es la ruta recomendada porque este tipo de modelo se convertirá sin problemas y es más sencillo de optimizar y ejecutar usando el runtime predeterminado de TensorFlow Lite. También tiene más opciones de implementación para su modelo, como [servicios de Google Play](../android/play_services). Puede empezar con la [Guía del conversor a TensorFlow Lite](../models/convert/convert_models). Consulte la página [Operadores de TensorFlow Lite](https://www.tensorflow.org/mlir/tfl_ops) para obtener una lista de los operadores incorporados.

Si necesita incluir operaciones TensorFlow selectas de la librería central, debe especificarlo en la conversión y asegurarse de que su runtime incluye esas operaciones. Consulte el tema [Operadores TensorFlow selectos](ops_select.md) para ver los pasos detallados.

Siempre que sea posible, evite la última opción de incluir operadores personalizados en su modelo convertido. Los [operadores personalizados](https://www.tensorflow.org/guide/create_op) son operadores creados combinando múltiples operadores primitivos del núcleo de TensorFlow o definiendo uno completamente nuevo. Cuando se convierten operadores personalizados, pueden aumentar el tamaño del modelo global al incurrir en dependencias fuera de la librería incorporada de TensorFlow Lite. Los ops personalizados, si no se han creado específicamente para su implementación en dispositivos móviles o de otro tipo, pueden resultar en un peor rendimiento cuando se implementan en dispositivos con recursos limitados en comparación con un entorno de servidor. Por último, al igual que para la inclusión de determinados operadores centrales de TensorFlow, los operadores personalizados requieren que [modifique el entorno runtime del modelo](ops_custom#create_and_register_the_operator), lo que le impide aprovechar los servicios estándares de runtime, como los [servicios de Google Play](../android/play_services).

## Tipos admitidos

La mayoría de las operaciones de TensorFlow Lite tienen como objetivo la inferencia tanto en punto flotante (`float32`) como cuantizada (`uint8`, `int8`), pero muchas ops aún no lo hacen para otros tipos como `tf.float16` y cadenas.

Aparte de usar una versión diferente de las operaciones, la otra diferencia entre los modelos de punto flotante y cuantizado es la forma en que se convierten. La conversión cuantizada requiere información de rango dinámico para los tensores. Esto requiere una "cuantización falsa" durante el entrenamiento del modelo, obtener la información del rango mediante un conjunto de datos de calibración o realizar una estimación del rango "sobre la marcha". Para más detalles, véase [cuantización](../performance/model_optimization.md).

## Conversiones sencillas, plegado y fusión de constantes

Varias operaciones de TensorFlow pueden ser procesadas por TensorFlow Lite aunque no tengan un equivalente directo. Este es el caso de las operaciones que pueden ser simplemente eliminadas del grafo (`tf.identity`), sustituidas por tensores (`tf.placeholder`), o fusionadas en operaciones más complejas (`tf.nn.bias_add`). Incluso algunas operaciones admitidas pueden eliminarse a veces mediante uno de estos procesos.

Aquí tiene una lista no exhaustiva de las operaciones de TensorFlow que suelen eliminarse del grafo:

- `tf.add`
- `tf.debugging.check_numerics`
- `tf.constant`
- `tf.div`
- `tf.divide`
- `tf.fake_quant_with_min_max_args`
- `tf.fake_quant_with_min_max_vars`
- `tf.identity`
- `tf.maximum`
- `tf.minimum`
- `tf.multiply`
- `tf.no_op`
- `tf.placeholder`
- `tf.placeholder_with_default`
- `tf.realdiv`
- `tf.reduce_max`
- `tf.reduce_min`
- `tf.reduce_sum`
- `tf.rsqrt`
- `tf.shape`
- `tf.sqrt`
- `tf.square`
- `tf.subtract`
- `tf.tile`
- `tf.nn.batch_norm_with_global_normalization`
- `tf.nn.bias_add`
- `tf.nn.fused_batch_norm`
- `tf.nn.relu`
- `tf.nn.relu6`

Nota: Muchas de esas operaciones no tienen equivalentes en TensorFlow Lite, y el modelo correspondiente no será convertible si no pueden eliminarse o fusionarse.

## Operaciones experimentales

Las siguientes operaciones de TensorFlow Lite están presentes, pero no están preparadas para modelos personalizados:

- `CALL`
- `CONCAT_EMBEDDINGS`
- `CUSTOM`
- `EMBEDDING_LOOKUP_SPARSE`
- `HASHTABLE_LOOKUP`
- `LSH_PROJECTION`
- `SKIP_GRAM`
- `SVDF`

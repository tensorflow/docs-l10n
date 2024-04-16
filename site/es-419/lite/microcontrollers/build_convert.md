# Generar y convertir modelos

Los microcontroladores tienen RAM y almacenamiento limitados, lo que impone restricciones a los tamaños de los modelos de aprendizaje automático. Además, TensorFlow Lite para microcontroladores admite actualmente un subconjunto limitado de operaciones, por lo que no todas las arquitecturas de modelos son posibles.

Este documento explica el proceso de conversión de un modelo TensorFlow para que se ejecute en microcontroladores. También esboza las operaciones admitidas y ofrece algunas orientaciones sobre el diseño y el entrenamiento de un modelo para que quepa en una memoria limitada.

Consulte el siguiente Colab, que forma parte del ejemplo *Hello World*, para ver un ejemplo ejecutable de principio a fin de la generación y conversión de un modelo:

<a class="button button-primary" href="https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb">train_hello_world_model.ipynb</a>

## Conversión de modelos

Para convertir un modelo TensorFlow entrenado para que funcione en microcontroladores, debe usar la [API de Python de conversión a TensorFlow Lite](https://www.tensorflow.org/lite/models/convert/). Esto convertirá el modelo en un [`FlatBuffer`](https://google.github.io/flatbuffers/), reduciendo el tamaño del modelo, y lo modificará para usar las operaciones de TensorFlow Lite.

Para obtener el tamaño de modelo más pequeño posible, debería considerar usar [cuantización despues del entrenamiento](https://www.tensorflow.org/lite/performance/post_training_quantization).

### Convertir a un arreglo en C

Muchas plataformas de microcontroladores no tienen soporte nativo para sistemas de archivos. La forma más sencilla de usar un modelo de su programa es incluirlo como arreglo en C y compilarlo en su programa.

El siguiente comando unix generará un archivo fuente en C que contiene el modelo TensorFlow Lite como un arreglo de `char`:

```bash
xxd -i converted_model.tflite > model_data.cc
```

La salida tendrá un aspecto similar al siguiente:

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

Una vez generado el archivo, puede incluirlo en su programa. Es importante cambiar la declaración del arreglo a `const` para una mejor eficiencia de memoria en plataformas embebidas.

Consulte [`evaluate_test.cc`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/evaluate_test.cc) en el ejemplo *Hello World* para saber cómo incluir y usar un modelo en su programa.

## Arquitectura y entrenamiento de modelos

Al diseñar un modelo para usarlo en microcontroladores, es importante tener en cuenta el tamaño del modelo, la carga de trabajo y las operaciones que se usan.

### Tamaño del modelo

Un modelo debe ser lo suficientemente pequeño como para caber en la memoria de su dispositivo objetivo junto con el resto de su programa, tanto como binario como en runtime.

Para crear un modelo más pequeño, puede usar menos capas y más pequeñas en su arquitectura. Sin embargo, los modelos pequeños tienen más probabilidades de quedar insuficientemente ajustados. Esto significa que, para muchos problemas, tiene sentido intentar usar el modelo más grande que quepa en la memoria. Sin embargo, usar modelos más grandes también supondrá una mayor carga de trabajo para el procesador.

Nota: El núcleo del runtime de TensorFlow Lite para microcontroladores cabe en 16 KB en un Cortex M3.

### Carga de trabajo

El tamaño y la complejidad del modelo influyen en la carga de trabajo. Los modelos grandes y complejos pueden resultar en un ciclo de trabajo más alto, lo que significa que el procesador de su dispositivo pasa más tiempo trabajando y menos tiempo inactivo. Esto aumentará el consumo de energía y la producción de calor, lo que podría ser un problema dependiendo de su aplicación.

### Apoyo a la operación

TensorFlow Lite para microcontroladores admite actualmente un subconjunto limitado de operaciones TensorFlow, lo que repercute en las arquitecturas modelo que es posible ejecutar. Estamos trabajando para ampliar el soporte de operaciones, tanto en términos de implementaciones de referencia como de optimizaciones para arquitecturas específicas.

Las operaciones soportadas pueden verse en el archivo [`micro_mutable_ops_resolver.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h).

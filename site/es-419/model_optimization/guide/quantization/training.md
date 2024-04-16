# Entrenamiento con reconocimiento de la cuantización

<sub>Actualizado por TensorFlow Model Optimization</sub>

Hay dos formas de cuantización: cuantización posentrenamiento y entrenamiento con reconocimiento de la cuantización. Comience con la [cuantización posentrenamiento](post_training.md), ya que es más fácil de usar, aunque el entrenamiento con reconocimiento de la cuantización suele ser mejor para la precisión del modelo.

Esta página proporciona una descripción general sobre el entrenamiento con reconocimiento de la cuantización para ayudarlo a determinar cómo se adapta a su caso de uso.

- Para profundizar en un ejemplo de principio a fin, consulte el [ejemplo de entrenamiento con reconocimiento de cuantización](training_example.ipynb).
- Para encontrar rápidamente las API que necesita para su caso de uso, consulte la [guía completa de entrenamiento con reconocimiento de la cuantización](training_comprehensive_guide.ipynb).

## Descripción general

El entrenamiento con reconocimiento de la cuantización emula la cuantización del tiempo de inferencia y crea un modelo que las herramientas posteriores usarán para producir modelos realmente cuantizados. Los modelos cuantizados usan una precisión más baja (por ejemplo, 8 bits en lugar de 32 bits flotantes), lo que genera beneficios durante la implementación.

### Implementar con cuantización

La cuantización aporta mejoras mediante la compresión del modelo y la reducción de la latencia. Con los valores predeterminados de la API, el tamaño del modelo se reduce 4 veces y normalmente vemos mejoras de entre 1.5 y 4 veces en la latencia de la CPU en los backends probados. Con el tiempo, se pueden ver mejoras de la latencia en aceleradores de aprendizaje automático compatibles, como [EdgeTPU](https://coral.ai/docs/edgetpu/benchmarks/) y NNAPI.

La técnica se usa en producción en casos de uso de voz, visión, texto y traducción. El código actualmente admite un [subconjunto de estos modelos](#general-support-matrix).

### Experimente con la cuantización y el hardware asociado

Los usuarios pueden configurar los parámetros de cuantización (por ejemplo, número de bits) y, hasta cierto punto, los algoritmos subyacentes. Tenga en cuenta que con estos cambios de los valores predeterminados de la API, actualmente no existe ninguna ruta compatible para la implementación en un backend. Por ejemplo, la conversión a TFLite y las implementaciones del núcleo solo admiten la cuantización de 8 bits.

Las API específicas de esta configuración son experimentales y no están sujetas a compatibilidad con versiones anteriores.

### Compatibilidad con las API

Los usuarios pueden aplicar la cuantización con las siguientes API:

- Compilación de modelos: `tf.keras` solo con modelos secuenciales y funcionales.
- Versiones de TensorFlow: TF 2.x para tf-nightly.
    - No se admite `tf.compat.v1` con un paquete de TF 2.X.
- Modo de ejecución de TensorFlow: ejecución eager

Tenemos planeado agregar compatibilidad en las siguientes áreas:

<!-- TODO(tfmot): file Github issues. -->

- Compilación de modelos: aclarar cómo los modelos subclasificados tienen compatibilidad limitada o nula
- Entrenamiento distribuido: `tf.distribute`

### Matriz de compatibilidad general

Hay compatibilidad en las siguientes áreas:

- Cobertura del modelo: modelos que usan [capas incluidas en la lista de permitidos](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py), BatchNormalization cuando le sigue a las capas Conv2D y DepthwiseConv2D y, en casos limitados, `Concat`.
    <!-- TODO(tfmot): add more details and ensure they are all correct. -->
- Aceleración de hardware: nuestros valores predeterminados de la API son compatibles con la aceleración en backends EdgeTPU, NNAPI y TFLite, entre otros. Vea la advertencia en la hoja de ruta.
- Implementación con cuantización: actualmente solo se admite la cuantización por eje para capas convolucionales, no la cuantización por tensor.

Tenemos planeado agregar compatibilidad en las siguientes áreas:

<!-- TODO(tfmot): file Github issue. Update as more functionality is added prior
to launch. -->

- Cobertura del modelo: se amplió para incluir RNN/LSTM y compatibilidad general de Concat.
- Aceleración de hardware: asegúrese de que el convertidor TFLite pueda producir modelos enteros. Consulte [este artículo](https://github.com/tensorflow/tensorflow/issues/38285) para obtener más detalles.
- Experimente con casos de uso de cuantización:
    - Experimente con algoritmos de cuantización que abarquen capas de Keras o requieran el paso de entrenamiento.
    - Estabilice las API.

## Resultados

### Clasificación de imágenes con herramientas

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-bit Quantized Accuracy </th>
    </tr>
    <tr>
      <td>MobilenetV1 224</td>
      <td>71.03%</td>
      <td>71.06%</td>
    </tr>
    <tr>
      <td>Resnet v1 50</td>
      <td>76.3%</td>
      <td>76.1%</td>
    </tr>
    <tr>
      <td>MobilenetV2 224</td>
      <td>70.77%</td>
      <td>70.01%</td>
    </tr>
 </table>
</figure>

Los modelos se probaron en Imagenet y se evaluaron tanto en TensorFlow como en TFLite.

### Clasificación de imágenes por técnica

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Non-quantized Top-1 Accuracy </th>
      <th>8-Bit Quantized Accuracy </th>
    <tr>
      <td>Nasnet-Mobile</td>
      <td>74%</td>
      <td>73%</td>
    </tr>
    <tr>
      <td>Resnet-v2 50</td>
      <td>75.6%</td>
      <td>75%</td>
    </tr>
 </table>
</figure>

Los modelos se probaron en Imagenet y se evaluaron tanto en TensorFlow como en TFLite.

## Ejemplos

Además del [ejemplo de entrenamiento con reconocimiento de cuantización](training_example.ipynb), consulte los siguientes ejemplos:

- Modelo de red neuronal convolucional (CNN) en la tarea de clasificación de dígitos escritos a mano de MNIST con cuantización: [código](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_functional_test.py)

Para obtener información sobre algo similar, consulte el [artículo](https://arxiv.org/abs/1712.05877) *Cuantización y entrenamiento de redes neuronales para una inferencia aritmética entera eficiente*. Este artículo presenta algunos conceptos que usa esta herramienta. La implementación no es exactamente la misma y se usan más conceptos en esta herramienta (por ejemplo, cuantización por eje).

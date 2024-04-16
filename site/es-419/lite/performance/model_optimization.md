# Optimización de modelos

Los dispositivos Edge suelen tener una memoria o una potencia de cálculo limitadas. Pueden aplicarse diversas optimizaciones a los modelos para que puedan ejecutarse dentro de estas limitaciones. Además, algunas optimizaciones permiten usar hardware especializado para acelerar la inferencia.

TensorFlow Lite y el [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) ofrecen herramientas para minimizar la complejidad de la optimización de la inferencia.

Es recomendable que considere la optimización de modelos durante el proceso de desarrollo de su aplicación. Este documento describe algunas de las mejores prácticas para optimizar los modelos TensorFlow para su implementación en hardware EDGE.

## Por qué deben optimizarse los modelos

Hay varias formas principales en las que la optimización de modelos puede ayudar en el desarrollo de aplicaciones.

### Reducción de tamaño

Algunas formas de optimización pueden usarse para reducir el tamaño de un modelo. Los modelos más pequeños tienen los siguientes beneficios:

- **Menor tamaño de almacenamiento:** Los modelos más pequeños ocupan menos espacio de almacenamiento en los dispositivos de sus usuarios. Por ejemplo, una app para Android que use un modelo más pequeño ocupará menos espacio de almacenamiento en el dispositivo móvil de un usuario.
- **Menor tamaño de descarga:** Los modelos más pequeños requieren menos tiempo y ancho de banda para descargarse en los dispositivos de los usuarios.
- **Menos uso de memoria:** Los modelos más pequeños utilizan menos memoria RAM cuando se ejecutan, lo que libera memoria para que la usen otras partes de su aplicación y puede traducirse en un mejor rendimiento y estabilidad.

La cuantización puede reducir el tamaño de un modelo en todos estos casos, potencialmente a expensas de cierta precisión. La poda y la agrupación pueden reducir el tamaño de un modelo para su descarga haciéndolo más fácilmente compresible.

### Reducción de la latencia

La *latencia* es la cantidad de tiempo que se tarda en ejecutar una única inferencia con un modelo dado. Algunas formas de optimización pueden reducir la cantidad de cálculo necesaria para ejecutar la inferencia usando un modelo, lo que resulta en una menor latencia. La latencia también puede tener un impacto en el consumo de energía.

Actualmente, la cuantización puede usarse para reducir la latencia simplificando los cálculos que se producen durante la inferencia, potencialmente a expensas de cierta precisión.

### Compatibilidad del acelerador

Algunos aceleradores de hardware, como el [Edge TPU](https://cloud.google.com/edge-tpu/), pueden ejecutar la inferencia extremadamente rápido con modelos que hayan sido correctamente optimizados.

Por lo general, este tipo de dispositivos requieren que los modelos se cuantifiquen de una manera específica. Consulte la documentación de cada acelerador de hardware para saber más sobre sus requisitos.

## Contrapartidas

Las optimizaciones pueden dar lugar a cambios en la precisión del modelo, que deben tenerse en cuenta durante el proceso de desarrollo de la aplicación.

Los cambios en la precisión dependen del modelo individual que se esté optimizando y son difíciles de predecir con antelación. Por lo general, los modelos que se optimizan para el tamaño o la latencia perderán una pequeña cantidad de precisión. Dependiendo de su aplicación, esto puede afectar o no a la experiencia de sus usuarios. En raras ocasiones, algunos modelos pueden ganar algo de precisión como resultado del proceso de optimización.

## Tipos de optimización

TensorFlow Lite admite actualmente la optimización mediante la cuantización, la poda y la agrupación.

Éstas forman parte del [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization), que ofrece recursos para técnicas de optimización de modelos compatibles con TensorFlow Lite.

### Cuantización

La [cuantización](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) funciona reduciendo la precisión de los números usados para representar los parámetros de un modelo, que de forma predeterminada son números de punto flotante de 32 bits. Esto resulta en un modelo de menor tamaño y un cálculo más rápido.

Los siguientes tipos de cuantización están disponibles en TensorFlow Lite:

Técnica | Requisitos de datos | Reducción de tamaño | Precisión | Hardware compatible
--- | --- | --- | --- | ---
[Cuantización de float16 posentrenamiento](post_training_float16_quant.ipynb) | Sin datos | Hasta 50% | Pérdida de precisión insignificante | CPU, GPU
[Cuantización del rango dinámico postentrenamiento](post_training_quant.ipynb) | Sin datos | Hasta 75% | La menor pérdida de precisión | CPU, GPU (Android)
[Cuantización entera postentrenamiento](post_training_integer_quant.ipynb) | Muestra representativa sin etiquetar | Hasta 75% | Pequeña pérdida de precisión | CPU, GPU (Android), EdgeTPU, Hexagon DSP
[Entrenamiento consciente de la cuantización](http://www.tensorflow.org/model_optimization/guide/quantization/training) | Datos de entrenamiento etiquetados | Hasta 75% | La menor pérdida de precisión | CPU, GPU (Android), EdgeTPU, Hexagon DSP

El siguiente árbol de decisión le ayuda a seleccionar los esquemas de cuantización que puede querer usar para su modelo, basándose simplemente en el tamaño y la precisión esperados del modelo.

![árbol de decisión de cuantización](images/quantization_decision_tree.png)

A continuación se muestran los resultados de latencia y precisión para la cuantización postentrenamiento y el entrenamiento consciente de la cuantización en algunos modelos. Todas las cifras de latencia se han medido en dispositivos Pixel 2 usando una CPU de núcleo grande. A medida que mejore el conjunto de herramientas, también lo harán los números aquí:

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> Benefits of model quantization for select CNN models
  </figcaption>
</figure>

### Cuantización entera con activaciones int16 y ponderaciones int8

[Cuantificación con activaciones int16](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) es un esquema de cuantización entero completo con activaciones en int16 y ponderaciones en int8. Este modo puede mejorar la precisión del modelo cuantizado en comparación con el esquema de cuantización entera completa con activaciones y ponderaciones en int8 manteniendo un tamaño de modelo similar. Se recomienda cuando las activaciones son sensibles a la cuantización.

<i>NOTA:</i> Actualmente sólo se dispone en TFLite de implementaciones no optimizadas del kernel de referencia para este esquema de cuantización, por lo que de forma predeterminada el rendimiento será lento en comparación con los kernels int8. Actualmente se puede acceder a todas las ventajas de este modo a través de hardware especializado, o software personalizado.

A continuación, se muestran los resultados de precisión de algunos modelos que son útiles en este modo.

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Accuracy metric type </th>
      <th>Accuracy (float32 activations) </th>
      <th>Accuracy (int8 activations) </th>
      <th>Accuracy (int16 activations) </th>
    </tr> <tr><td>Wav2letter</td><td>WER</td><td>6.7%</td><td>7.7%</td>
      <td>7.2%</td></tr>
    <tr><td>DeepSpeech 0.5.1 (unrolled)</td><td>CER</td><td>6.13%</td><td>43.67%</td>
      <td>6.52%</td></tr>
    <tr><td>YoloV3</td><td>mAP(IOU=0.5)</td><td>0.577</td><td>0.563</td>
      <td>0.574</td></tr>
    <tr><td>MobileNetV1</td><td>Top-1 Accuracy</td><td>0.7062</td><td>0.694</td>
      <td>0.6936</td></tr>
    <tr><td>MobileNetV2</td><td>Top-1 Accuracy</td><td>0.718</td><td>0.7126</td>
      <td>0.7137</td></tr>
    <tr><td>MobileBert</td><td>F1(Exact match)</td><td>88.81(81.23)</td><td>2.08(0)</td>
      <td>88.73(81.15)</td></tr>
 </table>
  <figcaption>
    <b>Table 2</b> Benefits of model quantization with int16 activations
  </figcaption>
</figure>

### Poda

La [poda](https://www.tensorflow.org/model_optimization/guide/pruning) funciona eliminando parámetros dentro de un modelo que sólo tienen un impacto menor en sus predicciones. Los modelos podados tienen el mismo tamaño en disco y la misma latencia runtime, pero pueden comprimirse con mayor eficacia. Esto hace que la poda sea una técnica útil para reducir el tamaño de descarga de los modelos.

En el futuro, TensorFlow Lite ofrecerá una reducción de latencia para los modelos podados.

### Agrupación

[La agrupación](https://www.tensorflow.org/model_optimization/guide/clustering) funciona agrupando las ponderaciones de cada capa de un modelo en un número predefinido de clusters y compartiendo después los valores del centroide de las ponderaciones pertenecientes a cada cluster individual. Esto reduce el número de valores de ponderación únicos en un modelo, reduciendo así su complejidad.

Como resultado, los modelos agrupados pueden comprimirse con mayor eficacia, lo que ofrece beneficios de implementación similares a la poda.

## Flujo de trabajo de desarrollo

Como punto de partida, revise si los modelos de [modelos alojados](../guide/hosted_models.md) pueden funcionar para su aplicación. Si no es así, recomendamos a los usuarios que empiecen con la [herramienta de cuantización posterior al entrenamiento](post_training_quantization.md), ya que es ampliamente aplicable y no requiere datos de entrenamiento.

Para los casos en los que no se cumplan los objetivos de precisión y latencia, o sea importante la compatibilidad con aceleradores de hardware, la mejor opción es el [entrenamiento basado en la cuantización](https://www.tensorflow.org/model_optimization/guide/quantization/training){:.external}. Consulte técnicas de optimización adicionales en [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization).

Si desea reducir aún más el tamaño de su modelo, puede intentar [la poda](#pruning) y/o [el agrupamiento](#clustering) antes de cuantizar sus modelos.

# Optimización colaborativa

<sub>Actualizado por Arm ML Tooling</sub>

En este documento, se proporciona una descripción general de las API experimentales para combinar varias técnicas la optimización de modelos de aprendizaje automático para su implementación.

## Descripción general

La optimización colaborativa es un proceso global que abarca varias técnicas para producir un modelo que, en el momento de la implementación, muestre el mejor equilibrio entre las características del destino, como la velocidad de inferencia, el tamaño del modelo y la precisión.

La idea de las optimizaciones colaborativas es aprovechar técnicas individuales aplicándolas una tras otra para lograr el efecto de optimización acumulado. Se pueden realizar varias combinaciones de las siguientes optimizaciones:

- [Eliminación de pesos](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)

- [Agrupación de pesos](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

- Cuantización

    - [Cuantización posentrenamiento](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
    - [Entrenamiento con reconocimiento de la cuantización](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html) (QAT)

El problema que surge al intentar encadenar estas técnicas es que la aplicación de una normalmente destruye los resultados de la técnica anterior, y se arruina el beneficio general de aplicarlas todas simultáneamente. Por ejemplo, la agrupación no preserva la dispersión que introduce la API de poda. Para resolver este problema, introducimos las siguientes técnicas experimentales de optimización colaborativa:

- [Agrupación que preserva la dispersión](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)
- [Entrenamiento con reconocimiento de la cuantización que preserva la dispersión](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example) (PQAT)
- [Entrenamiento con reconocimiento de la cuantización que preserva los grupos](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example) (CQAT)
- [Entrenamiento con reconocimiento de la cuantización que preserva la dispersión y los grupos](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example)

Estos proporcionan varias rutas de implementación que podrían usarse para comprimir un modelo de aprendizaje automático y aprovechar la aceleración del hardware en el momento de la inferencia. En el siguiente diagrama se muestran varias rutas de implementación que se pueden explorar en la búsqueda del modelo con las características de implementación deseadas, donde los nodos hoja son modelos listos para la implementación, lo que significa que están parcial o totalmente cuantizados y en formato tflite. El relleno verde indica los pasos en los que se requiere reentrenamiento/ajuste y un borde rojo discontinuo resalta los pasos de optimización colaborativa. La técnica que se usa para obtener un modelo en un determinado nodo se indica en la etiqueta correspondiente.

![collaborative optimization](images/collaborative_optimization.png "collaborative optimization")

En la figura anterior se omite la ruta de implementación directa de solo cuantización (posentrenamiento o QAT).

La idea es conseguir el modelo totalmente optimizado en el tercer nivel del árbol de implementación anterior; sin embargo, cualquiera de los otros niveles de optimización podría resultar satisfactorio y lograr el equilibrio que se requiere entre latencia de inferencia y precisión, en cuyo caso no se necesita ninguna optimización adicional. El proceso de entrenamiento recomendado sería recorrer de forma iterativa los niveles del árbol de implementación aplicables al escenario de implementación de destino y ver si el modelo cumple con los requisitos de latencia de inferencia. De lo contrario, use la técnica de optimización colaborativa correspondiente para comprimir aún más el modelo y repita hasta que el modelo esté completamente optimizado (podado, agrupado y cuantizado), si es necesario.

En la siguiente figura se muestran los gráficos de densidad del núcleo de peso de muestra que pasa por el proceso de optimización colaborativa.

![collaborative optimization density plot](images/collaborative_optimization_dist.png "collaborative optimization density plot")

El resultado es un modelo de implementación cuantizado con una cantidad reducida de valores únicos, así como una cantidad significativa de pesos dispersos, según la dispersión de destino especificada durante el entrenamiento. Además de las importantes ventajas de la compresión de modelos, el soporte de hardware específico puede aprovechar estos modelos dispersos y agrupados para reducir significativamente la latencia de inferencia.

## Resultados

A continuación se muestran algunos resultados de precisión y compresión que obtuvimos al experimentar con rutas de optimización colaborativa PQAT y CQAT.

### Entrenamiento con reconocimiento de la cuantización que preserva la dispersión (PQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Pruned Model (50% sparsity)</th><th>QAT Model</th><th>PQAT Model</th></tr>
 <tr><td>DS-CNN-L</td><td>FP32 Top1 Accuracy</td><td><b>95.23%</b></td><td>94.80%</td><td>(Fake INT8) 94.721%</td><td>(Fake INT8) 94.128%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.48%</td><td><b>93.80%</b></td><td>94.72%</td><td><b>94.13%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>528,128 → 434,879 (17.66%)</td><td>528,128 → 334,154 (36.73%)</td><td>512,224 → 403,261 (21.27%)</td><td>512,032 → 303,997 (40.63%)</td></tr>
 <tr><td>Mobilenet_v1-224</td><td>FP32 Top 1 Accuracy</td><td><b>70.99%</b></td><td>70.11%</td><td>(Fake INT8) 70.67%</td><td>(Fake INT8) 70.29%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.37%</td><td><b>67.82%</b></td><td>70.67%</td><td><b>70.29%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,520 → 3,880,331 (16.83%)</td><td>4,665,520 → 2,939,734 (37.00%)</td><td>4,569,416 → 3,808,781 (16.65%)</td><td>4,569,416 → 2,869,600 (37.20%)</td></tr>
</table>
</figure>

### Entrenamiento con reconocimiento de la cuantización que preserva los grupos (CQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Clustered Model</th><th>QAT Model</th><th>CQAT Model</th></tr>
 <tr><td>Mobilenet_v1 on CIFAR-10</td><td>FP32 Top1 Accuracy</td><td><b>94.88%</b></td><td>94.48%</td><td>(Fake INT8) 94.80%</td><td>(Fake INT8) 94.60%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.65%</td><td><b>94.41%</b></td><td>94.77%</td><td><b>94.52%</b></td></tr>
 <tr><td> </td><td>Size</td><td>3.00 MB</td><td>2.00 MB</td><td>2.84 MB</td><td>1.94 MB</td></tr>
 <tr><td>Mobilenet_v1 on ImageNet</td><td>FP32 Top 1 Accuracy</td><td><b>71.07%</b></td><td>65.30%</td><td>(Fake INT8) 70.39%</td><td>(Fake INT8) 65.35%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.34%</td><td><b>60.60%</b></td><td>70.35%</td><td><b>65.42%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,568 → 3,886,277 (16.7%)</td><td>4,665,568 → 3,035,752 (34.9%)</td><td>4,569,416 → 3,804,871 (16.7%)</td><td>4,569,472 → 2,912,655 (36.25%)</td></tr>
</table>
</figure>

### Resultados de CQAT y PCQAT para modelos agrupados por canal

Los resultados a continuación se obtienen con la técnica de [agrupación por canal](https://www.tensorflow.org/model_optimization/guide/clustering). Demuestran que si las capas convolucionales del modelo se agrupan por canal, la precisión del modelo es mayor. Si su modelo tiene muchas capas convolucionales, le recomendamos agruparlo por canal. La relación de compresión sigue siendo la misma, pero la precisión del modelo será mayor. La canalización de optimización del modelo está 'agrupada -&gt; QAT de preservación del agrupamiento-&gt; cuantización posentrenamiento, int8' en nuestros experimentos.

<figure>
<table  class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Clustered -> CQAT, int8 quantized</th><th>Clustered per channel -> CQAT, int8 quantized</th>
 <tr><td>DS-CNN-L</td><td>95.949%</td><td> 96.44%</td></tr>
 <tr><td>MobileNet-V2</td><td>71.538%</td><td>72.638%</td></tr>
 <tr><td>MobileNet-V2 (pruned)</td><td>71.45%</td><td>71.901%</td></tr>
</table>
</figure>

## Ejemplos

Para ver ejemplos de principio a fin de las técnicas de optimización colaborativa que se describen aquí, consulte los cuadernos de ejemplo de [CQAT](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example), [PQAT](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example), [agrupación que preserva la dispersión](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example) y [PCQAT](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example).

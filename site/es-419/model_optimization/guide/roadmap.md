**Actualizado: junio del 2021**

El kit de herramientas de optimización de modelos (MOT, por sus siglas en inglés) de TensorFlow se ha usado ampliamente para convertir/optimizar modelos de TensorFlow a modelos de TensorFlow Lite con un tamaño más pequeño, mejor rendimiento y precisión aceptable para ejecutarlos en dispositivos móviles e IoT. Ahora estamos trabajando para ampliar las técnicas y herramientas de MOT más allá de TensorFlow Lite para admitir también TensorFlow SavedModel.

A continuación se describe una visión general de alto nivel de nuestra hoja de ruta. Debe tener en cuenta que esta hoja de ruta puede cambiar en cualquier momento y que el orden que figura a continuación no refleja ningún tipo de prioridad. Le recomendamos encarecidamente que deje su comentario sobre la hoja de ruta en el [grupo de discusión](https://groups.google.com/a/tensorflow.org/g/tflite).

## Cuantización

#### TensorFlow Lite

- Cuantización selectiva postentrenamiento para excluir ciertas capas de la cuantización.
- Depurador de cuantización para inspeccionar las pérdidas por error de cuantización por capa.
- Aplicación de un entrenamiento con reconocimiento de la cuantización en una mayor cobertura de modelos, por ejemplo, TensorFlow Model Garden.
- Mejoras en la calidad y el rendimiento de la cuantización de rango dinámico posentrenamiento.

#### TensorFlow

- Cuantización posentrenamiento (rango dinámico bf16 * int8).
- Entrenamiento con reconocimiento de la cuantización (solo peso con cuantización falsa bf16 * int8).
- Cuantización selectiva postentrenamiento para excluir ciertas capas de la cuantización.
- Depurador de cuantización para inspeccionar las pérdidas por error de cuantización por capa.

## Dispersión

#### TensorFlow Lite

- Soporte de ejecución de modelos dispersos para más modelos.
- Creación con reconocimiento del destino para Dispersión.
- Amplíe el conjunto de operaciones dispersas con núcleos x86 de alto rendimiento.

#### TensorFlow

- Soporte de dispersión en TensorFlow.

## Técnicas de compresión en cascada

- Cuantización + Compresión tensorial + Dispersión: demuestra las 3 técnicas ejecutándose al mismo tiempo.

## Compresión

- API de compresión tensorial para ayudar a los desarrolladores de algoritmos de compresión a implementar su propio algoritmo de compresión de modelos (por ejemplo, agrupación de pesos), incluso proporcionar una forma estándar de prueba/comparación.

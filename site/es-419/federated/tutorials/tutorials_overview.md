# Tutoriales de TensorFlow Federados

Estos tutoriales [de colab](https://colab.research.google.com/) le servirán de guía a través de los principales conceptos y de las API de TFF mediante ejemplos prácticos. Puede encontrar la documentación de referencia en las [guías de TFF](../get_started.md).

Nota: actualmente, TFF requiere Python 3.9 o posterior, pero los tiempos de ejecución alojados de [Google Colaboratory](https://research.google.com/colaboratory/) usan Python 3.7, por lo que para ejecutar estos cuadernos se debe usar un [tiempo de ejecución local personalizado](https://research.google.com/colaboratory/local-runtimes.html).

**Empezar a aprender con el aprendizaje federado**

- En [Aprendizaje federado para la clasificación de imágenes](federated_learning_for_image_classification.ipynb) se presentan las partes clave de la API de aprendizaje federado (FL) y se demuestra cómo usar TFF para simular el aprendizaje federado en datos federados similares a MNIST.
- En [Aprendizaje federado para la generación de texto](federated_learning_for_text_generation.ipynb) se demuestra además cómo usar la API FL de TFF para refinar un modelo serializado preentrenado para una tarea de modelado de lenguaje.
- En [Ajustes de las agregaciones recomendadas para el aprendizaje](tuning_recommended_aggregators.ipynb) se muestra cómo se pueden combinar los cálculos básicos de FL en `tff.learning` con rutinas de agregación especializadas que ofrecen solidez, privacidad diferencial, compresión y más.
- En [Reconstrucción federada para la factorización de matrices](federated_reconstruction_for_matrix_factorization.ipynb) se introduce un aprendizaje federado parcialmente local, donde algunos parámetros del cliente nunca se agregan en el servidor. En este tutorial se demuestra como usar la API de aprendizaje federado para entrenar un modelo de factorización de matrices parcialmente local.

**Empezar a aprender con el análisis federado**

- En [Pesos pesados privados](private_heavy_hitters.ipynb) se muestra cómo usar `tff.analytics.heavy_hitters` para crear un cálculo analítico federado para descubrir pesos pesados ​​privados.

**Escribir cálculos federados personalizados**

- En [Crear su propio algoritmo de aprendizaje federado](building_your_own_federated_learning_algorithm.ipynb) se muestra cómo usar las API principales de TFF para implementar algoritmos de aprendizaje federado, con el promedio federado como ejemplo.
- En [Componer algoritmos de aprendizaje](composing_learning_algorithms.ipynb) se muestra cómo usar la API de aprendizaje de TFF para implementar nuevos algoritmos de aprendizaje federado de forma fácil, especialmente las variantes de promedio federado.
- En [El algoritmo federado personalizado con optimizadores de TFF](custom_federated_algorithm_with_tff_optimizers.ipynb) se muestra cómo usar `tff.learning.optimizers` para crear un proceso iterativo personalizado para el promedio federado.
- En [Algoritmos federados personalizados, Parte 1: Introducción al núcleo federado](custom_federated_algorithms_1.ipynb) y [Parte 2: Implementar el promedio federado](custom_federated_algorithms_2.ipynb) se presentan los conceptos e interfaces clave que ofrece la API del núcleo federado (FC API).
- En [Implementar agregaciones personalizadas](custom_aggregators.ipynb) se explican los principios de diseño detrás del módulo `tff.aggregators` y las mejores prácticas para implementar agregaciones personalizadas de valores desde el cliente al servidor.

**Mejores prácticas de simulación**

- En [Simulación de TFF con aceleradores (GPU)](simulations_with_accelerators.ipynb) se muestra cómo se puede usar el tiempo de ejecución de alto rendimiento de TFF con los GPU.

- En [Trabajar con ClientData](working_with_client_data.ipynb) se brindan las prácticas recomendadas para integrar los conjuntos de datos de simulación que se basan ​​en [ClientData](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/ClientData) de TFF en los cálculos de TFF.

**Tutoriales intermedios y avanzados**

- En [Generación de ruido aleatorio](random_noise_generation.ipynb) se señalan algunas sutilezas del uso de la aleatoriedad en cálculos descentralizados, se proponen mejores prácticas y se recomiendan patrones.

- En [Enviar datos diferentes a ciertos clientes con tff.federated_select](federated_select.ipynb) se presenta el operador `tff.federated_select` y se da un ejemplo simple de un algoritmo federado personalizado que envía datos diferentes a clientes diferentes.

- En [Aprendizaje federado de modelos grandes eficiente para el cliente a través de federated_select y la agregación dispersa](sparse_federated_learning.ipynb) se muestra cómo se puede usar TFF para entrenar un modelo muy grande donde cada dispositivo de cliente solo descarga y actualiza una pequeña parte del modelo, con `tff.federated_select` y agregación dispersa.

- En [TFF para la investigación de aprendizaje federado: compresión de modelos y actualizaciones](tff_for_federated_learning_research_compression.ipynb) se demuestra cómo se pueden usar las agregaciones personalizadas basadas en la [API tensor_encoding](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) en TFF.

- En [Aprendizaje federado con privacidad diferencial en TFF](federated_learning_with_differential_privacy.ipynb) se demuestra cómo usar TFF para entrenar modelos con privacidad diferencial a nivel de usuario.

- En [Soporte para JAX en TFF](../tutorials/jax_support.ipynb) se muestra cómo se pueden usar los cálculos [JAX](https://github.com/google/jax) en TFF, lo que demuestra cómo está diseñado TFF para poder interoperar con otros marcos frontend y backend de ML.

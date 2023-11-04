# Keras: La API de nivel superior para TensorFlow

Keras es la API de nivel superior de la plataforma de TensorFlow. Proporciona una interfaz accesible, altamente productiva para resolver problemas de aprendizaje automático (ML, por sus siglas en inglés), con un enfoque en el aprendizaje profundo moderno. Keras cubre cada paso del flujo de trabajo del aprendizaje automático, desde el procesamiento de datos hasta los ajustes de los hiperparámetros para la implementación. Se desarrolló con la idea de habilitar una experimentación rápida.

Con Keras, se tiene acceso total a las funcionalidades de escalabilidad y multiplataforma de TensorFlow. Puede ejecutar Keras en un TPU Pod o en unidades de asignación grandes de los GPU. Puede exportar los modelos de Keras para ejecutarlos en un explorador o en dispositivos móviles. También puede usar los modelos de Keras a través de la API web.

Keras está diseñado para reducir la carga cognitiva y cumple con los siguientes objetivos:

- Ofrece interfaces simples y consistentes.
- Minimiza la cantidad de acciones que se requieren para los casos de uso común.
- Proporciona mensajes de error claros que requieren acción.
- Sigue el principio de revelación progresiva de la complejidad: es fácil empezar a aprenderlo, y puede completar flujos de trabajo avanzados mientras aprende sobre la marcha.
- Ayuda a que la escritura del código sea concisa y que se pueda leer.

## Quién debería usar Keras

La respuesta rápida es que todas las personas que usan TensorFlow deberían usar las API de Keras de forma predeterminada. No importa si es ingeniero, investigador o un practicante de ML, debe empezar a aprender Keras.

Hay algunos casos de uso (por ejemplo, construir herramientas aparte de TensorFlow o desarrollar una plataforma propia de alto rendimiento) que requieren las [Core API de TensorFlow](https://www.tensorflow.org/guide/core).  Pero si no necesita las [aplicaciones de Core API](https://www.tensorflow.org/guide/core#core_api_applications) para su caso de uso, quizás prefiera usar Keras.

## Componentes de la API de Keras

Las estructuras de datos centrales de Keras son [capas](https://keras.io/api/layers/) y [modelos](https://keras.io/api/models/). Una capa es una transformación simple de entrada/salida y un modelo es un gráfico acíclico (DAG ) de capas.

### Capas

La clase `tf.keras.layers.Layer` es la abstracción fundamental de Keras. Una `Layer` encapsula un estado (pesos) y algunos cálculos (definidos en el método `tf.keras.layers.Layer.call`).

Los pesos que crean las capas pueden ser entrenables o no entrenables. Las capas se pueden componer de forma recursiva: si se asigna una instancia de capa como un atributo de otra capa, la capa exterior empezará a seguir los pesos que cree la capa interior.

También puede usar las capas para manipular las tareas de preprocesamiento de datos como la normalización y la vectorización de texto. Las capas de preprocesamiento pueden incluirse directamente en un modelo, ya sea durante o después del entrenamiento, lo que hace que el modelo sea portátil.

### Modelos

Un modelo es un objeto que agrupa las capas y que puede entrenarse con datos.

El tipo de modelo más simple es el modelo [`Sequential`](https://www.tensorflow.org/guide/keras/sequential_model), que es una pila lineal de capas. Para arquitecturas más complejas, puede usar la [API funcional de Keras](https://www.tensorflow.org/guide/keras/functional_api), que le permite construir gráficos arbitrarios de capas o puede [usar la subclasificación para escribir modelos desde cero](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing).

La clase `tf.keras.Model` tiene métodos de entrenamiento y de evaluación integrados:

- `tf.keras.Model.fit`: Entrena al modelo para una cantidad de épocas fija.
- `tf.keras.Model.predict`: Genera predicciones de salida para las muestras de entrada.
- `tf.keras.Model.evaluate`: Devuelve los valores de pérdida y de métrica para el modelo; se configura mediante el método `tf.keras.Model.compile`.

Estos métodos le dan acceso a las siguientes funciones integradas:

- [Retrollamadas](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks). Puede aprovechar las retrollamadas para la interrupción temprana, para guardar puntos de verificación del modelo y para monitorear el [TensorBoard](https://www.tensorflow.org/tensorboard). También puede [implementar retrollamadas personalizadas](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks).
- [ Entrenamiento distribuido ](https://www.tensorflow.org/guide/keras/distributed_training).  Puede ampliar fácilmente su entrenamiento en varios GPU, TPU o dispositivos.
- Fusión de pasos. Con el argumento `steps_per_execution` en `tf.keras.Model.compile`, puede procesar varios lotes en una sola llamada de `tf.function`, que puede mejorar de forma notable el uso del dispositivo en los TPU.

Para ver una descripción general en detalle sobre cómo usar `fit`, vea la [guía de entrenamiento y evaluación ](https://www.tensorflow.org/guide/keras/training_with_built_in_methods). Para aprender cómo personalizar el entrenamiento y los bucles de evaluación integrados, consulte [Personalizar lo que ocurre en `fit()`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).

### Otras API y herramientas

Keras proporciona muchas otras API y herramientas para el aprendizaje automático, entre ellas se incluyen:

- [Optimizadores](https://keras.io/api/optimizers/)
- [Métricas](https://keras.io/api/metrics/)
- [Pérdidas](https://keras.io/api/losses/)
- [ Utilidades de carga de datos](https://keras.io/api/data_loading/)

Para ver una lista completa de las API disponibles, consulte la [referencia de API de Keras](https://keras.io/api/). Para obtener más información sobre los proyectos y las iniciativas de Keras, consulte [El ecosistema de Keras](https://keras.io/getting_started/ecosystem/).

## Próximos pasos

Para empezar a usar Keras con TensorFlow, échele un vistazo a los siguientes temas:

- [El modelo Sequential](https://www.tensorflow.org/guide/keras/sequential_model)
- [La API funcional](https://www.tensorflow.org/guide/keras/functional)
- [Entrenamiento y evaluación con los métodos integrados](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [Crear nuevas capas y modelos mediante la subclasificación](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [Serialización y guardado](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [Trabajar con capas de preprocesamiento](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [Personalizar lo que ocurre en fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [Escribir un bucle de entrenamiento desde cero](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [Trabajar con los RNN](https://www.tensorflow.org/guide/keras/rnn)
- [Entender el enmascaramiento y el espaciado](https://www.tensorflow.org/guide/keras/masking_and_padding)
- [Escribir sus propias retrollamadas](https://www.tensorflow.org/guide/keras/custom_callback)
- [Transferir el aprendizaje y los ajustes](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Varios GPU y entrenamiento distribuido](https://www.tensorflow.org/guide/keras/distributed_training)

Para obtener más información sobre Keras, consulte los siguientes temas en [keras.io](http://keras.io):

- [Acerca de Keras](https://keras.io/about/)
- [Introducción a Keras para ingenieros](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [Introducción a Keras para investigadores](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Referencia de la API de Keras](https://keras.io/api/)
- [El ecosistema de Keras](https://keras.io/getting_started/ecosystem/)

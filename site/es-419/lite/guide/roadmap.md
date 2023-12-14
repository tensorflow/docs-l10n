# Hoja de ruta de TensorFlow Lite

**Actualizado: Mayo de 2021**

A continuación se incluye una visión general de alto nivel de nuestra hoja de ruta. Debe tener en cuenta que esta hoja de ruta puede cambiar en cualquier momento y que el orden que figura a continuación no refleja ningún tipo de prioridad.

Dividimos nuestra hoja de ruta en cuatro segmentos clave: usabilidad, rendimiento, optimización y portabilidad. Le animamos encarecidamente a que analice nuestra hoja de ruta y nos dé su retroalimentación en el [grupo de conversación TensorFlow Lite](https://groups.google.com/a/tensorflow.org/g/tflite).

## Usabilidad

- **Cobertura de ops ampliada**
    - Añadir ops específicas basadas en la retroalimentación de los usuarios.
    - Añadir conjuntos de ops dirigidos a dominios y áreas específicos, incluyendo ops aleatorias, ops de la capa Keras base, tablas hash, ops de entrenamiento seleccionadas.
- **Más herramientas de asistencia**
    - Dar anotaciones de grafos TensorFlow y herramientas de compatibilidad para validar la compatibilidad de TFLite y de los aceleradores de hardware durante el entrenamiento y después de la conversión.
    - Permitir la focalización y optimización para aceleradores específicos durante la conversión.
- **Capacitación en el dispositivo**
    - Admitir entrenamiento en el dispositivo para la personalización y el aprendizaje por transferencia, incluido un Colab que demuestre el uso de principio a fin.
    - Admitir tipos de variables/recursos (tanto para la inferencia como para el entrenamiento)
    - Admita la conversión y ejecución de grafos con múltiples puntos de entrada de funciones (o firmas).
- **Integración mejorada de Android Studio**
    - Arrastrar y soltar los modelos TFLite en Android Studio para generar interfaces de modelo.
    - Mejorar el soporte de perfilado de Android Studio, incluido el perfilado de memoria.
- **Model Maker**
    - Admitir tareas más novedosas, como la detección de objetos, la recomendación y la clasificación de audio, abarcando una amplia recopilación de uso común.
    - Admitir más conjuntos de datos para facilitar el aprendizaje por transferencia.
- **Librería de tareas**
    - Admitir más tipos de modelos (por ejemplo, audio, PNL) con capacidades asociadas de preprocesamiento y postprocesamiento.
    - Actualizar más ejemplos de referencia con las API de tareas.
    - Admitir la aceleración inmediata para todas las tareas.
- **Más modelos y ejemplos de SOTA**
    - Añadir más ejemplos (por ejemplo, de audio, PNL, relacionados con datos estructurales) para demostrar el uso del modelo, así como nuevas funciones y API, que abarquen distintas plataformas.
    - Cree modelos de backbone compartibles para dispositivos con el fin de reducir los costes de entrenamiento e implementación.
- **Implementación sin fisuras en múltiples plataformas**
    - Ejecutar modelos TensorFlow Lite en la web.
- **Soporte multiplataforma mejorado**
    - Ampliar y mejorar las API para Java en Android, Swift en iOS, Python en RPi.
    - Mejorar el soporte de CMake (por ejemplo, mayor soporte de aceleradores).
- **Mejor soporte de frontend**
    - Mejorar la compatibilidad con varios frontends de autoría, incluyendo Keras, tf.numpy.

## Rendimiento

- **Mejores herramientas**
    - Panel de control público para dar seguimiento a las mejoras de rendimiento con cada versión.
    - Herramientas para comprender mejor la compatibilidad de los grafos con los aceleradores objetivo.
- **Rendimiento mejorado de la CPU**
    - XNNPack habilitado de forma predeterminada para una inferencia en coma flotante más rápida.
    - Compatible con precisión media (float16) de extremo a extremo con kernels optimizados.
- **Soporte actualizado de NNAPI**
    - Compatibilidad total con las funciones, ops y tipos de NNAPI de la versión más reciente de Android.
- **Optimizaciones de la GPU**
    - Se ha mejorado el tiempo de inicio gracias a la compatibilidad con la serialización de delegados.
    - Interoperabilidad del búfer de hardware para la inferencia de copia cero.
    - Mayor disponibilidad de la aceleración en el dispositivo.
    - Mejor cobertura de op.

## Optimización

- **Cuantización**

    - Cuantización selectiva postentrenamiento para excluir ciertas capas de la cuantización.
    - Depurador de cuantización para inspeccionar las pérdidas por error de cuantización en cada capa.
    - Aplicar un entrenamiento que tenga en cuenta la cuantización en una mayor cobertura de modelos, por ejemplo, TensorFlow Model Garden.
    - Mejoras en la calidad y el rendimiento de la cuantización de rango dinámico posterior al entrenamiento.
    - API de compresión tensorial para permitir algoritmos de compresión como SVD.

- **Poda / dispersión**

    - Combinar API configurables en tiempo de entrenamiento (poda + entrenamiento consciente de la cuantización).
    - Aumentar la aplicación de la dispersión en los modelos TF Model Garden.
    - Soporte de ejecución de modelos dispersos en TensorFlow Lite.

## Portabilidad

- **Soporte de microcontroladores**
    - Añadir soporte para una serie de casos de uso de la arquitectura MCU de 32 bits para la clasificación de voz e imágenes.
    - Frontend de audio: Preprocesamiento de audio en el grafo y soporte de aceleración
    - Código de muestra y modelos para datos de visión y audio.

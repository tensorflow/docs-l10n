# Inferencia en TensorFlow Lite con metadatos

Inferenciar [modelos con metadatos](../models/convert/metadata.md) puede ser tan fácil como unas pocas líneas de código. Los metadatos de TensorFlow Lite contienen una rica descripción de lo que hace el modelo y de cómo usarlo. Puede empoderar a los generadores de código para que generen automáticamente el código de inferencia por usted, como al usar la función de [Android Studio ML Binding](codegen.md#mlbinding) o el [Generador de código Android de TensorFlow Lite](codegen.md#codegen). También puede usarlo para configurar su canalización de inferencia personalizada.

## Herramientas y librerías

TensorFlow Lite proporciona variedades de herramientas y librerías para servir a diferentes niveles de requisitos de implementación, como se indica a continuación:

### Generar la interfaz del modelo con los generadores de código de Android

Hay dos maneras de generar automáticamente el código contenedor Android necesario para el modelo TensorFlow Lite con metadatos:

1. [Android Studio ML Model Binding](codegen.md#mlbinding) es una herramienta disponible en Android Studio para importar modelos TensorFlow Lite a través de una interfaz gráfica. Android Studio establecerá automáticamente los ajustes para el proyecto y generará clases contenedoras basadas en los metadatos del modelo.

2. [Generador de código TensorFlow Lite](codegen.md#codegen) es un ejecutable que genera automáticamente la interfaz del modelo basándose en los metadatos. Actualmente soporta Android con Java. El código contenedor elimina la necesidad de interactuar directamente con `ByteBuffer`. En su lugar, los desarrolladores pueden interactuar con el modelo TensorFlow Lite con objetos tipados como `Bitmap` y `Rect`. Los usuarios de Android Studio también pueden acceder a la función codegen a través de [Android Studio ML Binding](codegen.md#mlbinding).

### Aprovechar las API listas para usar con la librería de tareas TensorFlow Lite

La [librería de tareas TensorFlow Lite](task_library/overview.md) ofrece interfaces de modelo optimizadas y listas para usar para tareas populares de aprendizaje automático, como clasificación de imágenes, preguntas y respuestas, etc. Las interfaces de modelo están diseñadas específicamente para cada tarea con el fin de lograr el mejor rendimiento y usabilidad. La librería de tareas funciona en varias plataformas y es compatible con Java, C++ y Swift.

### Generar canalizaciones de inferencia personalizadas con la librería de soporte TensorFlow Lite

La [librería de soporte TensorFlow Lite](lite_support.md) es una librería multiplataforma que ayuda a personalizar la interfaz del modelo y a generar canalizaciones de inferencia. Contiene variedades de métodos de utilización y estructuras de datos para realizar el pre/post procesamiento y la conversión de datos. También está diseñada para adaptarse al comportamiento de los módulos TensorFlow, como TF.Image y TF.Text, garantizando la consistencia desde el entrenamiento hasta la inferencia.

## Explorar modelos preentrenados con metadatos

Explore los [modelos alojados en TensorFlow Lite](https://www.tensorflow.org/lite/guide/hosted_models) y [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) para descargar modelos preentrenados con metadatos tanto para tareas de visión como de texto. Vea también diferentes opciones de [visualización de los metadatos](../models/convert/metadata.md#visualize-the-metadata).

## Repositorio en GitHub de soporte para TensorFlow Lite

Visite el [repositorio en GitHub de Soporte de TensorFlow Lite](https://github.com/tensorflow/tflite-support) para ver más ejemplos y código fuente. Háganos saber su retroalimentación creando una [nueva incidencia en GitHub](https://github.com/tensorflow/tflite-support/issues/new).

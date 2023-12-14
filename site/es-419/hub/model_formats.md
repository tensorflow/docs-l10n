# Formatos de modelo

[tfhub.dev](https://tfhub.dev) aloja los siguientes formatos de modelo: TF2 SavedModel, formato TF1 Hub, TF.js y TFLite. En esta página encontrará una descripción general de cada formato de modelo.

El contenido que se publica en tfhub.dev se puede reflejar automáticamente en otros hubs de modelos, siempre que se siga un formato específico y lo permitan nuestros Términos (https://tfhub.dev/terms). Consulte [nuestra documentación de publicación](publish.md) para obtener más detalles y [nuestra documentación de contribución](contribute_a_model.md) si desea optar por no participar en la creación de reflejos.

## Formatos de TensorFlow

[tfhub.dev](https://tfhub.dev) aloja modelos de TensorFlow en el formato TF2 SavedModel y en el formato TF1 Hub. Recomendamos usar modelos en el formato TF2 SavedModel estandarizado en lugar del formato obsoleto TF1 Hub cuando sea posible.

### SavedModel

TF2 SavedModel es el formato recomendado para compartir modelos de TensorFlow. Puede obtener más información sobre el formato SavedModel en la guía [SavedModel de TensorFlow](https://www.tensorflow.org/guide/saved_model).

Puede explorar SavedModels en tfhub.dev con el filtro de versión TF2 en la [página de exploración de tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) o mediante [este enlace](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2).

Puede usar SavedModels desde tfhub.dev sin depender de la biblioteca `tensorflow_hub`, ya que este formato es parte del núcleo de TensorFlow.

Obtenga más información sobre SavedModels en TF Hub:

- [Usar TF2 SavedModel](tf2_saved_model.md)
- [Exportar un TF2 SavedModel](exporting_tf2_saved_model.md)
- [Compatibilidad TF1/TF2 de TF2 SavedModels](model_compatibility.md)

### Formato TF1 Hub

El formato TF1 Hub es un formato de serialización personalizado que usa la biblioteca TF Hub. El formato TF1 Hub es similar al formato SavedModel de TensorFlow 1 en un nivel sintáctico (los mismos nombres de archivo y mensajes de protocolo) pero es semánticamente diferente para permitir la reutilización, composición y reentrenamiento del módulo (por ejemplo, almacenamiento diferente de inicializadores de recursos, convenciones de etiquetado diferentes para los metagráficos). La forma más fácil de diferenciarlos en el disco es la presencia o ausencia del archivo `tfhub_module.pb`.

Puede explorar los modelos en el formato TF1 Hub en tfhub.dev con el filtro de versión TF1 en la [página de exploración de tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) o mediante [este enlace](https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf1).

Obtenga más información sobre los modelos en formato TF1 Hub en TF Hub:

- [Usar modelos de formato TF1 Hub](tf1_hub_module.md)
- [Exportar un modelo en formato TF1 Hub](exporting_hub_format.md)
- [Compatibilidad TF1/TF2 del formato TF1 Hub](model_compatibility.md)

## Formato TFLite

El formato TFLite se usa para la inferencia en el dispositivo. Puede obtener más información en la [documentación de TFLite](https://www.tensorflow.org/lite).

Puede explorar los modelos TF Lite en tfhub.dev con el filtro de formato de modelo TF Lite en la [página de exploración de tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) o mediante [este enlace](https://tfhub.dev/lite).

## Formato TFJS

El formato TF.js se usa para el aprendizaje automático en el navegador. Puede obtener más información en la [documentación de TF.js](https://www.tensorflow.org/js).

Puede explorar los modelos TF.js en tfhub.dev con el filtro de formato de modelo TF.js en la [página de exploración de tfhub.dev](https://tfhub.dev/s?subtype=module,placeholder) o mediante [este enlace](https://tfhub.dev/js).

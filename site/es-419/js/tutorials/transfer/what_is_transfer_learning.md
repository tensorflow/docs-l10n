# Qué es el aprendizaje por transferencia

Los modelos de aprendizaje profundo tienen millones de parámetros (pesos). Entrenarlos desde cero requiere una gran cantidad de datos de recursos calculados. El aprendizaje por transferencia es una técnica que acorta mucho el proceso al tomar una parte de un modelo que ya ha sido entrenado en una tarea relacionada y reutilizarla en un modelo nuevo.

Por ejemplo, en el siguiente tutorial de esta sección le mostraremos cómo crear un reconocedor propio de imágenes que aprovecha un modelo ya entrenado para reconocer 1000 tipos diferentes de objetos dentro de las imágenes. Lo que hacemos es adaptar el conocimiento existente en el modelo previamente entrenado para detectar las propias clases de imágenes con muchos menos datos de entrenamiento que los requeridos para el modelo original.

Esto resulta útil para el desarrollo rápido de modelos nuevos, así como también para la personalización de modelos en entornos con recursos limitados como los navegadores o los dispositivos móviles.

Con gran frecuencia, mientras hacemos el aprendizaje por transferencia, no ajustamos los pesos del modelo original. En cambio, quitamos la capa final y entrenamos un modelo nuevo (casi siempre bastante poco profundo) sobre la salida del modelo truncado. Esta es la técnica que verá demostrada en los tutoriales de esta sección.

- [Creación de un clasificador de imágenes basado en aprendizaje por transferencia](image_classification)
- [Creación de un reconocedor de audio basado en aprendizaje por transferencia](audio_recognizer)

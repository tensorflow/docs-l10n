# Preguntas y respuestas sobre el BERT

Usar un modelo TensorFlow Lite para responder a preguntas basadas en el contenido de un pasaje dado.

Nota: (1) Para integrar un modelo existente, pruebe la [Librería de tareas de TensorFlow Lite](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer). (2) Para personalizar un modelo, pruebe [Model Maker de TensorFlow Lite](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

## Empecemos

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Si es nuevo en TensorFlow Lite y trabaja con Android o iOS, le recomendamos explorar las siguientes aplicaciones de ejemplo que pueden ayudarle a empezar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/android">Ejemplo de Android</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios">Ejemplo iOS</a>

Si utiliza una plataforma que no sea Android/iOS, o si ya está familiarizado con las [ APIs de TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), puede descargar nuestro modelo de preguntas y respuestas para principiantes.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Descargue el modelo inicial y el vocabulario</a>

Para más información sobre los metadatos y los campos asociados (por ejemplo, `vocab.txt`) consulte <a href="https://www.tensorflow.org/lite/models/convert/metadata#read_the_metadata_from_models">Lectura de los metadatos de los modelos</a>.

## Cómo funciona

El modelo puede usarse para generar un sistema capaz de responder a las preguntas de los usuarios en lenguaje natural. Se creó usando un modelo BERT preentrenado y afinado en el conjunto de datos SQuAD 1.1.

[BERT](https://github.com/google-research/bert), o Representaciones codificadoras bidireccionales a partir de transformadores, es un método de entrenamiento previo de representaciones lingüísticas que obtiene resultados de vanguardia en un amplio arreglo de tareas de Procesamiento del Lenguaje Natural.

Esta app usa una versión comprimida de BERT, MobileBERT, que funciona 4 veces más rápido y tiene un tamaño de modelo 4 veces menor.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), o Conjunto de datos de respuesta a preguntas de Stanford, es un conjunto de datos de comprensión lectora formado por artículos de Wikipedia y un conjunto de pares pregunta-respuesta para cada artículo.

El modelo toma un pasaje y una pregunta como entrada y, a continuación, devuelve un segmento del pasaje que muy probablemente responda a la pregunta. Requiere un preprocesamiento semicomplejo que incluye pasos de tokenización y posprocesamiento que se describen en el [documento del BERT](https://arxiv.org/abs/1810.04805) y se implementan en la app de muestra.

## Puntos de referencia del rendimiento

Los números de referencia del rendimiento se generan con la herramienta [descrita aquí](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nombre del modelo</th>
      <th>Tamaño del modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite">Mobile Bert</a>
</td>
    <td rowspan="3">       100.5 Mb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>123 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>74 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
    <td>257 ms**</td>
  </tr>
</table>

* 4 hilos usados.

** 2 hilos usados en el iPhone para obtener el mejor resultado de rendimiento.

## Salida de ejemplo

### Pasaje (Entrada)

> Google LLC es una empresa tecnológica multinacional estadounidense especializada en servicios y productos relacionados con Internet, que incluyen tecnologías de publicidad en línea, motor de búsqueda, computación en la nube, software y hardware. Está considerada una de las cuatro grandes empresas tecnológicas, junto con Amazon, Apple y Facebook.
>
> Google fue fundada en septiembre de 1998 por Larry Page y Sergey Brin mientras eran estudiantes de doctorado en la Universidad de Stanford, en California. Juntos poseen alrededor del 14 % de sus acciones y controlan el 56 % del poder de voto de los accionistas a través de acciones con derecho a voto. Constituyeron Google como empresa privada californiana el 4 de septiembre de 1998, en California. Posteriormente, Google se reincorporó en Delaware el 22 de octubre de 2002. El 19 de agosto de 2004 tuvo lugar una oferta pública inicial (IPO) y Google se trasladó a su sede en Mountain View, California, apodada Googleplex. En agosto de 2015, Google anunció sus planes de reorganizar sus diversos intereses como un conglomerado llamado Alphabet Inc. Google es la principal filial de Alphabet y seguirá siendo la empresa paraguas de los intereses de Alphabet en Internet. Sundar Pichai fue nombrado Director General de Google, en sustitución de Larry Page, que pasó a ser Director General de Alphabet.

### Pregunta (Entrada)

> ¿Quién es el CEO de Google?

### Respuesta (Salida)

> Sundar Pichai

## Más información sobre el BERT

- Artículo académico: [BERT: Entrenamiento previo de transformadores bidireccionales profundos para la comprensión del lenguaje](https://arxiv.org/abs/1810.04805)
- [Implementación de código abierto de BERT](https://github.com/google-research/bert)

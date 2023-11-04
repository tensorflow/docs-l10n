# Aprendizaje federado

## Descripción general

Este documento presenta interfaces que facilitan las tareas de aprendizaje federado, como el entrenamiento federado o la evaluación con modelos de aprendizaje automático existentes implementados en TensorFlow. Al diseñar estas interfaces, nuestro objetivo principal era posibilitar la experimentación con el aprendizaje federado sin necesidad de conocer cómo funciona a nivel interno y evaluar los algoritmos de aprendizaje federado implementados en una variedad de modelos y datos existentes. Además, invitamos a los usuarios a contribuir a la plataforma. TFF se diseñó teniendo en cuenta la extensibilidad y la capacidad de composición, y agradecemos las contribuciones; ¡estamos ansiosos por ver sus ideas!

Las interfaces que ofrece esta capa constan de estas tres partes clave:

- **Modelos**. Clases y funciones ayudantes que permiten envolver los modelos existentes para usarlos con TFF. Envolver un modelo puede ser tan simple como llamar a una única función de envoltorio (por ejemplo, `tff.learning.models.from_keras_model`) o definir una subclase de la interfaz `tff.learning.models.VariableModel` para obtener una personalización completa.

- **Generadores de cálculos federados**. Funciones ayudantes que construyen cálculos federados para entrenamiento o evaluación, a partir de sus modelos existentes.

- **Conjuntos de datos**. Colecciones predefinidas de datos que se pueden descargar y a las que se puede acceder en Python para utilizarlas en la simulación de escenarios de aprendizaje federado. Aunque el aprendizaje federado se diseñó para usarse con datos descentralizados que no pueden descargarse simplemente en una ubicación centralizada, en las fases de investigación y desarrollo a menudo es conveniente realizar experimentos iniciales con datos que puedan descargarse y manipularse a nivel local, especialmente en el caso de desarrolladores que no estén familiarizados con este enfoque.

Estas interfaces se definen principalmente en el espacio de nombres `tff.learning`, excepto para los conjuntos de datos de investigación y otras capacidades relacionadas con la simulación que se agrupan en `tff.simulation`. Esta capa se implementa a través de interfaces de bajo nivel disponibles en [Federated Core (FC)](federated_core.md), que también facilita un entorno de ejecución.

Antes de continuar, le recomendamos que primero se tome un momento para consultar los tutoriales sobre [clasificación de imágenes](tutorials/federated_learning_for_image_classification.ipynb) y [generación de textos](tutorials/federated_learning_for_text_generation.ipynb), ya que allí se explican la mayoría de los conceptos que aquí se describen mediante ejemplos concretos. Si desea obtener más información sobre el funcionamiento de TFF, puede consultar el tutorial sobre [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) como introducción a las interfaces de bajo nivel que utilizamos para expresar la lógica de los cálculos federados y estudiar la implementación existente de las interfaces `tff.learning`.

## Modelos

### Supuestos arquitectónicos

#### Serialización

TFF pretende admitir una variedad de escenarios de aprendizaje distribuido en los que el código del modelo de aprendizaje automático que escriba pueda ejecutarse en un gran número de clientes heterogéneos con distintas capacidades. Mientras que, en un extremo del espectro, en algunas aplicaciones esos clientes podrían ser potentes servidores de bases de datos, muchos usos importantes que nuestra plataforma pretende ofrecer implican dispositivos móviles e integrados con recursos limitados. No podemos asumir que estos dispositivos serán capaces de alojar tiempos de ejecución de Python; lo único que podemos asumir en este momento es que serán capaces de alojar un tiempo de ejecución local de TensorFlow. Por lo tanto, un supuesto arquitectónico fundamental que hacemos en TFF es que el código de su modelo debe poder serializarse como un gráfico de TensorFlow.

Usted puede (y debe) seguir desarrollando su código de TF según las prácticas recomendadas más recientes, como el uso del modo eager. Sin embargo, el código final debe ser serializable (por ejemplo, puede ser envuelto como una `tf.function` para un código en modo eager). Esto asegura que cualquier estado o flujo de control de Python necesario en el momento de la ejecución se pueda serializar (posiblemente con la ayuda de [Autograph](https://www.tensorflow.org/guide/autograph)).

Por el momento, TensorFlow no es totalmente compatible con la serialización y la deserialización de TensorFlow en modo eager. Por ello, la serialización en TFF actualmente se ajusta al patrón de TF 1.0, donde todo el código debe ser construido dentro de un `tf.Graph` que TFF controla. Esto significa que actualmente TFF no puede usar un modelo ya construido; en lugar de eso, la lógica de definición del modelo se empaqueta en una función sin argumento que devuelve un `tff.learning.models.VariableModel`. A continuación, TFF llama a esta función para garantizar que todos los componentes del modelo se serialicen. Además, al ser un entorno fuertemente tipado, TFF solicitará algunos *metadatos* adicionales, como la especificación del tipo de entrada del modelo.

#### Agregación

Recomendamos enfáticamente que la mayoría de los usuarios construyan modelos con Keras, consulte la sección [Convertidores para Keras](#converters-for-keras) que figura más adelante. Estos envoltorios se encargan de la agregación de las actualizaciones del modelo, así como de cualquier métrica definida para el modelo de forma automática. Sin embargo, sigue siendo útil entender cómo se gestiona la agregación para un `tff.learning.models.VariableModel` general.

En el aprendizaje federado siempre hay al menos dos capas de agregación; la agregación local en el dispositivo y la agregación entre dispositivos (o federada):

- **Agregación local**. Este nivel de agregación se refiere a la agregación a través de múltiples lotes de ejemplos que pertenecen a un cliente individual. Se aplica tanto a los parámetros del modelo (variables), que continúan evolucionando secuencialmente a medida que el modelo se entrena a nivel local, como a las estadísticas que se calculan (como la pérdida media, la precisión y otras métricas), que el modelo volverá a actualizar localmente a medida que itera sobre el flujo de datos local de cada cliente individual.

    La agregación a este nivel es responsabilidad del código de su modelo, y se consigue mediante construcciones estándar de TensorFlow.

    La estructura general del procesamiento es la siguiente:

    - En primer lugar, el modelo construye `tf.Variable`s para almacenar agregados, como el número de lotes o el número de ejemplos procesados, la suma de pérdidas por lote o por ejemplo, etc.

    - TFF invoca el método `forward_pass` varias veces en su `Model`, secuencialmente sobre lotes posteriores de datos del cliente, lo que le permite actualizar las variables que contienen varios agregados como efecto secundario.

    - Por último, TFF invoca el método `report_local_unfinalized_metrics` en su Model para permitir que este compile todas las estadísticas de resumen recopiladas en un conjunto compacto de métricas que el cliente pueda exportar. Aquí es donde el código de su modelo puede, por ejemplo, dividir la suma de pérdidas por el número de ejemplos procesados para exportar la pérdida media, etc.

- **Agregación federada**. Este nivel de agregación se refiere a la agregación a través de múltiples clientes (dispositivos) en el sistema. Como ya dijimos, se aplica tanto a los parámetros del modelo (variables), que se promedian entre los clientes, como a las métricas que el modelo exporta como resultado de la agregación local.

    La agregación a este nivel es responsabilidad de TFF. Sin embargo, como creador de modelos, podrá controlar este proceso (más adelante encontrará más información al respecto).

    La estructura general del procesamiento es la siguiente:

    - Un servidor distribuye el modelo inicial y los parámetros necesarios para el entrenamiento a un subconjunto de clientes que participarán en una ronda de entrenamiento o evaluación.

    - En cada cliente, de forma independiente y en paralelo, se invoca reiteradamente el código de su modelo sobre un flujo de lotes de datos locales para producir un nuevo conjunto de parámetros del modelo (cuando se entrena), y un nuevo conjunto de métricas locales, como se ha descrito anteriormente (esto se denomina agregación local).

    - TFF ejecuta un protocolo de agregación distribuida para acumular y agregar los parámetros del modelo y las métricas exportadas localmente en todo el sistema. Esta lógica se expresa de manera declarativa mediante el lenguaje *de cálculo federado* propio de TFF (no en TensorFlow). Consulte el tutorial sobre [algoritmos personalizados](tutorials/custom_federated_algorithms_1.ipynb) para obtener más información sobre la API de agregación.

### Interfaces abstractas

Esta interfaz básica de *constructor* + *metadatos* se representa mediante la interfaz `tff.learning.models.VariableModel`, de la siguiente manera:

- El constructor, `forward_pass`, y los métodos `report_local_unfinalized_metrics` deben generar las variables del modelo, el paso hacia adelante, y las estadísticas que se desean reportar, según corresponda. El TensorFlow construido por esos métodos debe ser serializable, como se comentó anteriormente.

- La propiedad `input_spec`, así como las 3 propiedades que devuelven subconjuntos de sus variables locales, no entrenables y entrenables representan los metadatos. TFF utiliza esta información para determinar cómo conectar partes de su modelo a los algoritmos de optimización federados y para definir firmas de tipo internas que ayuden a verificar la exactitud del sistema construido (de modo que no se pueda crear una instancia de su modelo a partir de datos que no coincidan con los que el modelo está diseñado para procesar).

Además, la interfaz abstracta `tff.learning.models.VariableModel` expone una propiedad `metric_finalizers` que toma los valores no finalizados de una métrica (devueltos por `report_local_unfinalized_metrics()`) y devuelve los valores finalizados de la métrica. Los métodos `metric_finalizers` y `report_local_unfinalized_metrics()` se usan conjuntamente para crear un agregador de métricas entre clientes cuando se definen los procesos de entrenamiento federados o los cálculos de evaluación. Por ejemplo, un simple agregador `tff.learning.metrics.sum_then_finalize` sumará primero los valores métricos no finalizados de los clientes y, a continuación, llamará a las funciones finalizadoras en el servidor.

Puede encontrar ejemplos de cómo definir su propio `tff.learning.models.VariableModel` personalizado en la segunda parte de nuestro tutorial de [clasificación de imágenes](tutorials/federated_learning_for_image_classification.ipynb), así como en los modelos de ejemplo que usamos para las pruebas en [`model_examples.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/models/model_examples.py).

### Convertidores para Keras

Casi toda la información solicitada por TFF se puede derivar con una llamada a las interfaces `tf.keras`, por lo que, si tiene un modelo Keras, puede confiar en `tff.learning.models.from_keras_model` para construir un `tff.learning.models.VariableModel`.

Tenga en cuenta que TFF todavía quiere que usted proporcione un constructor, una *función modelo* sin argumentos como la siguiente:

```python
def model_fn():
  keras_model = ...
  return tff.learning.models.from_keras_model(keras_model, sample_batch, loss=...)
```

Además del modelo en sí, se suministra un lote de datos de muestra que TFF utiliza para determinar el tipo y la forma de la entrada del modelo. Esto garantiza que TFF pueda crear una instancia adecuada del modelo para los datos que realmente estarán presentes en los dispositivos cliente (ya que suponemos que estos datos no están generalmente disponibles en el momento en que se construye el TensorFlow que se va a serializar).

El uso de envoltorios de Keras se ilustra en nuestros tutoriales de [clasificación de imágenes](tutorials/federated_learning_for_image_classification.ipynb) y [generación de textos](tutorials/federated_learning_for_text_generation.ipynb).

## Generadores de cálculos federados

El paquete `tff.learning` proporciona varios generadores de `tff.Computation`s que llevan a cabo tareas relacionadas con el aprendizaje; esperamos que este conjunto de cálculos se amplíe en el futuro.

### Supuestos arquitectónicos

#### Ejecución

La ejecución de un cálculo federado consta de dos fases distintas.

- **Compilación**: en primer lugar, TFF *compila* algoritmos de aprendizaje federados en una representación serializada abstracta de todo el cálculo distribuido. Aquí es cuando se produce la serialización de TensorFlow, pero pueden ocurrir otras transformaciones para permitir una ejecución más eficiente. Nos referimos a la representación serializada que el compilador emite en forma de *cálculo federado*.

- **Ejecución:** TFF ofrece métodos de *ejecución* para estos cálculos. Por ahora, la ejecución solo es posible mediante una simulación local (por ejemplo, en un bloc de notas que use datos descentralizados simulados).

Un cálculo federado generado por la API de aprendizaje federado de TFF, como un algoritmo de entrenamiento que se vale del [promediado de modelo federado](https://arxiv.org/abs/1602.05629), o de una evaluación federada, incluye una serie de elementos, entre los que destacan los siguientes:

- Una forma serializada del código de su modelo, así como código TensorFlow adicional construido por el marco de Aprendizaje Federado para impulsar el bucle de entrenamiento/evaluación de su modelo (como la construcción de optimizadores, la aplicación de actualizaciones del modelo, la iteración sobre `tf.data.Dataset`s, y el cálculo de métricas, y la aplicación de la actualización agregada en el servidor, por nombrar algunos).

- Una especificación declarativa de la comunicación entre los *clientes* y un *servidor* (por lo general varias formas de *agregación* entre los dispositivos cliente, y la *difusión* desde el servidor a todos los clientes), y cómo esta comunicación distribuida se intercala con la ejecución cliente-local o servidor-local del código de TensorFlow.

Los *cálculos federados* que se representan en esta forma serializada se expresan en un lenguaje interno independiente de la plataforma y distinto de Python, pero para utilizar la API de Aprendizaje Federado, no tendrá que preocuparse por los detalles de esta representación. Los cálculos se representan en su código Python como objetos de tipo `tff.Computation`, que en su mayoría puede tratar como `callable`s opacos de Python.

En los tutoriales, invocará esos cálculos federados como si fueran funciones normales de Python, para ejecutarlos localmente. Sin embargo, TFF está diseñado para expresar cálculos federados de una manera agnóstica a la mayoría de los aspectos del entorno de ejecución, por lo que potencialmente puede ser implementado en, por ejemplo, grupos de dispositivos con `Android`, o en clústeres en un centro de datos. Como ya mencionamos, la principal consecuencia de esto son los importantes supuestos sobre la [serialización](#serialization). En particular, cuando se invoca uno de los métodos `build_...` descritos a continuación, el cálculo se serializa completamente.

#### Cómo modelar el estado

TFF es un entorno de programación funcional, pero muchos procesos de interés en el aprendizaje federado tienen estado. Por ejemplo, un bucle de entrenamiento que implique múltiples rondas de promediado de modelos federados es un ejemplo de lo que podríamos clasificar como *proceso con estado*. En este proceso, el estado que evoluciona de ronda en ronda incluye el conjunto de parámetros del modelo que se están entrenando y, posiblemente, el estado adicional asociado con el optimizador (por ejemplo, un vector de impulso).

Como TFF es funcional, los procesos con estado se modelan en TFF como cálculos que aceptan el estado actual como entrada y luego ofrecen el estado actualizado como salida. Para definir correctamente un proceso con estado, también es necesario especificar de dónde procede el estado inicial (de lo contrario, no podemos iniciar el proceso). Esto se incluye en la definición de la clase ayudante `tff.templates.IterativeProcess`, con las 2 propiedades `initialize` y `next` que corresponden a la inicialización y la iteración, respectivamente.

### Generadores disponibles

Por el momento, TFF ofrece varias funciones de construcción que generan cálculos federados para el entrenamiento y la evaluación federados. Destacamos dos ejemplos:

- `tff.learning.algorithms.build_weighted_fed_avg`, que toma como entrada una *función modelo* y un *optimizador de clientes*, y devuelve un `tff.learning.templates.LearningProcess` con estado (que genera subclases de `tff.templates.IterativeProcess`).

- `tff.learning.build_federated_evaluation` toma una *función modelo* y devuelve un único cálculo federado para la evaluación federada de modelos, ya que la evaluación no consta de estados.

## Conjuntos de datos

### Supuestos arquitectónicos

#### Selección de clientes

En un escenario de aprendizaje federado típico, tenemos una gran *población* de cientos de millones de dispositivos cliente, de los cuales solo una pequeña porción puede estar activa y disponible para el entrenamiento en un momento dado (por ejemplo, esto puede limitarse a los clientes que estén conectados a una fuente de alimentación, que no estén en una red medida y que, por otra parte, estén inactivos). Por lo general, el conjunto de clientes disponibles para participar en el entrenamiento o la evaluación escapa al control del desarrollador. Además, como no resulta práctico coordinar a millones de clientes, una ronda típica de entrenamiento o evaluación incluirá solo a una fracción de los clientes disponibles, que podrán [extraerse como muestra de forma aleatoria](https://arxiv.org/pdf/1902.01046.pdf).

La consecuencia clave de esto es que los cálculos federados, por su diseño, se expresan de una manera que es independiente del conjunto exacto de participantes; todo el procesamiento se expresa como operaciones agregadas en un grupo abstracto de *clientes* anónimos, y ese grupo puede variar de una ronda de entrenamiento a otra. La vinculación real del cálculo a los participantes concretos, y por tanto a los datos concretos que introducen en el cálculo, se modela así fuera del propio cálculo.

Para simular una implementación realista de su código de aprendizaje federado, generalmente tendrá que crear un bucle de entrenamiento similar a este:

```python
trainer = tff.learning.algorithms.build_weighted_fed_avg(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  result = trainer.next(state, data_for_this_round)
  state = result.state
```

Para facilitar esto, cuando se usa TFF en simulaciones, los datos federados se aceptan como `list`as de Python, con un elemento por dispositivo cliente que participa para representar el `tf.data.Dataset` local de ese dispositivo.

### Interfaces abstractas

Para estandarizar el tratamiento de los conjuntos de datos federados simulados, TFF ofrece una interfaz abstracta `tff.simulation.datasets.ClientData`, que permite enumerar el conjunto de clientes y construir un `tf.data.Dataset` que contenga los datos de un cliente específico. Estos `tf.data.Dataset`s pueden introducirse directamente como entrada en los cálculos federados generados en modo eager.

Cabe señalar que la capacidad de acceder a las identidades de los clientes es una característica que solo proporcionan los conjuntos de datos para su uso en simulaciones, donde puede ser necesaria la capacidad de entrenar con datos de subconjuntos específicos de clientes (por ejemplo, para simular la disponibilidad diurna de diferentes tipos de clientes). Los cálculos compilados y el tiempo de ejecución subyacente *no* conllevan ninguna noción de identidad del cliente. Una vez que se han seleccionado los datos de un subconjunto específico de clientes como entrada, por ejemplo, en una llamada a `tff.templates.IterativeProcess.next`, las identidades de los clientes ya no aparecen en ella.

### Conjuntos de datos disponibles

Se ha dedicado el espacio de nombres `tff.simulation.datasets` a los conjuntos de datos que implementan la interfaz `tff.simulation.datasets.ClientData` para su uso en simulaciones, y se ha sembrado con conjuntos de datos compatibles con los tutoriales de [clasificación de imágenes](tutorials/federated_learning_for_image_classification.ipynb) y [generación de textos](tutorials/federated_learning_for_text_generation.ipynb). Nos gustaría que contribuyera con sus propios conjuntos de datos a la plataforma.

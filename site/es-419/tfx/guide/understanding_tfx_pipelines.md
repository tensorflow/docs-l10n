# Explicación de las canalizaciones de TFX

MLOps es la aplicación de prácticas de DevOps para ayudar a automatizar, administrar y auditar los flujos de trabajo de aprendizaje automático (ML). Los flujos de trabajo de ML incluyen pasos para ejecutar las siguientes acciones:

- Preparar, analizar y transformar datos.
- Entrenar y evaluar un modelo.
- Implementar modelos entrenados en producción.
- Hacer un seguimiento de los artefactos de ML y comprender sus dependencias.

Es posible que gestionar estos pasos ad hoc resulte difícil y lleve mucho tiempo.

TFX facilita la implementación de MLOps al proporcionar un conjunto de herramientas que lo ayuda a organizar su proceso de aprendizaje automático en varios orquestadores, como: Apache Airflow, Apache Beam y Kubeflow Pipelines. Al implementar su flujo de trabajo como una canalización de TFX, puede:

- Automatizar su proceso de aprendizaje automático, lo que le permite volver a entrenar, evaluar e implementar su modelo con regularidad.
- Usar recursos informáticos distribuidos para procesar grandes conjuntos de datos y cargas de trabajo.
- Aumentar la velocidad de la experimentación al ejecutar una canalización con diferentes conjuntos de hiperparámetros.

Esta guía describe los conceptos básicos necesarios para comprender las canalizaciones de TFX.

## Artefacto

Las salidas de los pasos en una canalización de TFX se denominan **artefactos**. Los pasos posteriores de su flujo de trabajo pueden utilizar estos artefactos como entradas. De esta manera, TFX le permite transferir datos entre pasos del flujo de trabajo.

Por ejemplo, el componente estándar `ExampleGen` emite ejemplos serializados, que componentes como el componente estándar `StatisticsGen` usan como entradas.

Los artefactos deben estar fuertemente tipados con un **tipo d eartefacto** registrado en el almacén de [ML Metadata](mlmd.md). Obtenga más información sobre los [conceptos utilizados en ML Metadata](mlmd.md#concepts).

Los tipos de artefactos tienen un nombre y definen un esquema de sus propiedades. Los nombres de los tipos de artefactos deben ser únicos en su almacén de ML Metadata. TFX proporciona varios [tipos de artefactos estándar](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py) {: .external } que describen tipos de datos y tipos de valores complejos, como: cadena, entero y flotante. Puede [reutilizar estos tipos de artefactos](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py) {: .external } o definir tipos de artefactos personalizados que se deriven de [`Artifact`](https://github.com/tensorflow/tfx/blob/master/tfx/types/artifact.py) {: .external }.

## Parámetro

Los parámetros son entradas a las canalizaciones que se conocen antes de que se ejecute la canalización. Los parámetros le permiten cambiar el comportamiento de una canalización, o parte de una canalización, mediante configuración en lugar de código.

Por ejemplo, puede usar parámetros para ejecutar una canalización con diferentes conjuntos de hiperparámetros sin cambiar el código de la canalización.

El uso de parámetros le permite aumentar la velocidad de la experimentación al facilitar la ejecución de una canalización con diferentes conjuntos de parámetros.

Obtenga más información sobre la [clase RuntimeParameter](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/data_types.py) {: .external }.

## Componente

Un **componente** es una implementación de una tarea de ML que se puede usar como paso en una canalización de TFX. Los componentes incluyen lo siguiente:

- Una especificación del componente, que define los artefactos de entrada y salida del componente, y los parámetros necesarios para el componente.
- Un ejecutor, que implementa el código para ejecutar un paso en su flujo de trabajo de aprendizaje automático, como la ingesta y la transformación de datos o el entrenamiento y evaluación de un modelo.
- Una interfaz de componente, que empaqueta la especificación del componente y el ejecutor para su uso en una canalización.

TFX ofrece varios [componentes estándar](index.md#tfx-standard-components) que se puede usar en las canalizaciones. Si estos componentes no se adaptan a sus necesidades, puede compilar componentes personalizados. [Obtenga más información sobre los componentes personalizados](understanding_custom_components.md).

## Canalización

Una canalización de TFX es una implementación portátil de un flujo de trabajo de aprendizaje automático que se puede ejecutar en varios orquestadores, como: Apache Airflow, Apache Beam y Kubeflow Pipelines. Una canalización se compone de instancias de componentes y parámetros de entrada.

Las instancias de componentes producen artefactos como salidas y normalmente dependen de los artefactos producidos por instancias de componentes ascendentes como entradas. La secuencia de ejecución de las instancias de componentes se determina mediante la creación de un grafo acíclico dirigido de las dependencias de los artefactos.

Por ejemplo, piense en una canalización que haga lo siguiente:

- Ingiera datos directamente desde un sistema propietario a través de un componente personalizado.
- Calcule estadísticas para los datos de entrenamiento a partir del componente estándar StatisticsGen.
- Cree un esquema de datos a partir del componente estándar SchemaGen.
- Use el componente estándar ExampleGen para comprobar los datos de entrenamiento en busca de anomalías.
- Use el componente estándar Transform para ejecutar ingeniería de características en el conjunto de datos.
- Entrene un modelo con ayuda del componente estándar Trainer.
- Evalúe el modelo entrenado con el componente Evaluator.
- Si el modelo pasa su evaluación, la canalización usa un componente personalizado para poner en cola el modelo entrenado en un sistema de implementación propietario.

![](images/tfx_pipeline_graph.svg)

Para determinar la secuencia de ejecución de las instancias de los componentes, TFX analiza las dependencias de los artefactos.

- El componente de ingesta de datos no tiene dependencias de artefactos, por lo que puede ser el primer nodo del grafo.
- StatisticsGen depende de los *ejemplos* producidos por la ingesta de datos, por lo que debe ejecutarse después de la ingesta de datos.
- SchemaGen depende de las *estadísticas* creadas por StatisticsGen, por lo que debe ejecutarse después de StatisticsGen.
- ExampleValidator depende de las *estadísticas* creadas por StatisticsGen y del *esquema* creado por SchemaGen, por lo que debe ejecutarse después de StatisticsGen y SchemaGen.
- Transform depende de los *ejemplos* producidos por la ingesta de datos y el *esquema* creado por SchemaGen, por lo que debe ejecutarse después de la ingesta de datos y SchemaGen.
- Trainer depende de los *ejemplos* producidos por la ingesta de datos, el *esquema* creado por SchemaGen y el *modelo guardado* producido por Transform. Trainer solo se puede ejecutar después de la ingesta de datos, SchemaGen y Transform.
- Evaluator depende de los *ejemplos* producidos por la ingesta de datos y el *modelo guardado* producido por Trainer, por lo que debe ejecutarse después de la ingesta de datos y Trainer.
- El implementador personalizado depende del *modelo guardado* producido por Trainer y los *resultados del análisis* creados por Evaluator, por lo que el implementador debe ejecutarse después de Trainer y Evaluator.

Según este análisis, un orquestador ejecuta lo siguiente:

- Las instancias de los componentes de ingesta de datos, StatisticsGen y SchemaGen de forma secuencial.
- Los componentes ExampleValidator y Transform se pueden ejecutar en paralelo ya que comparten dependencias de artefactos de entrada y no dependen de la salida de cada uno.
- Una vez que se haya ejecutado el componente Transform, las instancias del componente Trainer, Evaluator y del implementador personalizado se ejecutan secuencialmente.

Obtenga más información sobre [cómo compilar una canalización de TFX](build_tfx_pipeline.md).

## Plantilla de canalización de TFX

Las plantillas de canalización de TFX facilitan los primeros pasos del desarrollo de canalizaciones, ya que ofrecen una canalización prediseñada que puede personalizar según su caso de uso.

Obtenga más información sobre [cómo personalizar una plantilla de canalización de TFX](build_tfx_pipeline.md#pipeline-templates).

## Ejecución de canalizaciones

Una ejecución es una ejecución única de una canalización.

## Orquestador

Un orchestrator es un sistema donde puede ejecutar ejecuciones de canalizaciones. TFX admite orquestadores como: [Apache Airflow](airflow.md), [Apache Beam](beam.md) y [Kubeflow Pipelines](kubeflow.md). TFX también usa el término *DagRunner* para referirse a una implementación que admite un orquestador.

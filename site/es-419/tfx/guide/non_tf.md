# Cómo usar otros marcos de ML en TFX

Como plataforma, TFX es neutral en lo que respecta a los marcos de trabajo y se puede usar con otros marcos de trabajo de ML, por ejemplo, JAX, scikit-learn.

Para los desarrolladores de modelos, esto significa que no tienen la necesidad de reescribir el código de su modelo implementado en otro marco de aprendizaje automático, sino que pueden reutilizar la mayor parte del código de entrenamiento tal como está en TFX y aprovechar otras capacidades de TFX y del resto del ecosistema que ofrece TensorFlow.

El SDK de canalizaciones de TFX y la mayoría de los módulos de TFX, por ejemplo, el orquestador de canalizaciones, no dependen directamente de TensorFlow, pero hay algunos aspectos que están orientados hacia TensorFlow, como los formatos de datos. Teniendo en cuenta las necesidades de un marco de modelado en particular, se puede utilizar una canalización de TFX para entrenar modelos en cualquier otro marco de aprendizaje automático que se base en Python. Esto incluye Scikit-learn, XGBoost y PyTorch, entre otros. A continuación, se incluyen algunas de las consideraciones que hay que tener en cuenta a la hora de usar los componentes estándar de TFX con otros marcos de trabajo:

- **ExampleGen** genera un [tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) en archivos TFRecord. Es una representación genérica para datos de entrenamiento y los componentes posteriores usan [TFXIO](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md) para leerlos como Arrow/RecordBatch en la memoria, que a su vez se pueden convertir a `tf.dataset`, `Tensors` u otros formatos. Se están considerando formatos de carga útil o de archivo distintos de tf.train.Example/TFRecord, pero para los usuarios de TFXIO debería ser una caja negra.
- **Transform** se puede usar para generar ejemplos de entrenamiento transformados sin importar qué marco se use para el entrenamiento, pero si el formato del modelo no es `saved_model`, los usuarios no podrán insertar el gráfico de transformación en el modelo. En ese caso, la predicción del modelo debe tomar características transformadas en lugar de características sin procesar, y los usuarios pueden ejecutar la transformación como un paso de preprocesamiento antes de llamar a la predicción del modelo al servir.
- **Trainer** es compatible con [GenericTraining](https://www.tensorflow.org/tfx/guide/trainer#generic_trainer) para que los usuarios puedan entrenar sus modelos con cualquier marco de ML.
- De forma predeterminada, **Evaluator** solo es compatible con `saved_model`, pero los usuarios pueden proporcionar un UDF que genere predicciones para la evaluación del modelo.

Para entrenar un modelo en un marco no basado en Python se debe aislar un componente de entrenamiento personalizado en un contenedor Docker, como parte de una canalización que se ejecuta en un entorno en contenedores como Kubernetes.

## JAX

[JAX](https://github.com/google/jax) es Autograd y XLA, unidos para la investigación del aprendizaje automático de alto rendimiento. [Flax](https://github.com/google/flax) es una biblioteca y un ecosistema de redes neuronales para JAX, diseñado para brindar flexibilidad.

Con [jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf), podemos convertir modelos JAX/Flax entrenados al formato `saved_model`, que se puede usar sin problemas en TFX con entrenamiento genérico y evaluación de modelos. Para obtener más información, consulte este [ejemplo](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_flax_experimental.py).

## scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) es una biblioteca de aprendizaje automático para el lenguaje de programación Python. Disponemos de un [ejemplo](https://github.com/tensorflow/tfx-addons/tree/main/examples/sklearn_penguins) e2e con entrenamiento y evaluación personalizados en TFX-Addons.

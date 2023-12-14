# Fairness Indicators

Fairness Indicators se diseñó para ayudar a los equipos a evaluar y mejorar modelos en términos de equidad en colaboración con el conjunto de herramientas más amplio de Tensorflow. En la actualidad, muchos de nuestros productos usan la herramienta a nivel interno y ahora está disponible en BETA para que la prueben en sus propios casos de uso.

![Panel de Fairness Indicators](images/fairnessIndicators.png)

## ¿Qué es Fairness Indicators?

Fairness Indicators es una biblioteca que permite calcular sin problemas métricas de equidad comúnmente identificadas para clasificadores binarios y multiclase. Muchas herramientas existentes diseñadas para evaluar la equidad no funcionan bien en conjuntos de datos y modelos a gran escala. En Google, consideramos que es importante que las herramientas funcionen en sistemas con miles de millones de usuarios. Fairness Indicators le permitirá evaluar casos de uso de cualquier tamaño.

En particular, Fairness Indicators incluye las siguientes tareas:

- Evaluar la distribución de conjuntos de datos.
- Evaluar el rendimiento del modelo, dividido en grupos definidos de usuarios
    - Los intervalos de confianza y las evaluaciones en varios umbrales le permiten confiar en sus resultados.
- Profundizar en segmentos individuales para explorar las causas raíz y las oportunidades de mejora

Este [estudio de caso](https://developers.google.com/machine-learning/practica/fairness-indicators), que incluye [videos](https://www.youtube.com/watch?v=pHT-ImFXPQo) y ejercicios de programación, demuestra cómo puede usar Fairness Indicators en uno de sus propios productos para evaluar los problemas de equidad a lo largo del tiempo.

[](http://www.youtube.com/watch?v=pHT-ImFXPQo)

La descarga del paquete pip incluye esto:

- **[Tensorflow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started)**
- **[Tensorflow Model Analysis (TFMA)](https://www.tensorflow.org/tfx/model_analysis/get_started)**
    - **Fairness Indicators**
- **[La herramienta What-If (WIT)](https://www.tensorflow.org/tensorboard/what_if_tool)**

## Cómo usar Fariness Indicators con modelos de Tensorflow

### Datos

Para ejecutar Fariness Indicators con TFMA, asegúrese de que el conjunto de datos de evaluación esté etiquetado para las características que desea segmentar. Si no tiene las características de segmento exactas para sus problemas de equidad, trate de encontrar un conjunto de evaluación que las tenga, o piense qué características proxy dentro de su conjunto de características pueden resaltar las disparidades en los resultados. Para obtener más aydua, consulte [aquí](https://tensorflow.org/responsible_ai/fairness_indicators/guide/guidance).

### Modelo

Puede usar la clase Estimator de Tensorflow para construir su modelo. La compatibilidad con los modelos de Keras pronto estará disponible en TFMA. Si desea ejecutar TFMA en un modelo de Keras, consulte la sección "TFMA independiente del modelo" a continuación.

Una vez que su Estimator esté entrenado, deberá exportar un modelo guardado para fines de evaluación. Para obtener más información, consulte la [guía sobre TFMA](/tfx/model_analysis/get_started).

### Cómo configurar segmentos

A continuación, defina los segmentos que le gustaría evaluar:

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur color’])
]
```

Si desea evaluar cortes interseccionales (por ejemplo, tanto el color del pelaje como la altura), puede establecer lo siguiente:

```python
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur_color’, ‘height’])
]`
```

### Cálculo de métricas de equidad

Agregue una retrollamada de Fairness Indicators a la lista `metrics_callback`. En la retrollamada, puede definir una lista de umbrales en los que se evaluará el modelo.

```python
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = \
    [tfma.post_export_metrics.fairness_indicators(thresholds=[0.1, 0.3,
     0.5, 0.7, 0.9])]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

Antes de ejecutar la configuración, determine si desea habilitar o no el cálculo de intervalos de confianza. Los intervalos de confianza se calculan mediante arranque de Poisson y requieren un nuevo cálculo a partir de 20 muestras.

```python
compute_confidence_intervals = True
```

Ejecute la canalización de evaluación de TFMA:

```python
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

### Representación de Fairness Indicators

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result=eval_result)
```

![Fairness Indicators](images/fairnessIndicators.png)

Consejos para el uso de Fairness Indicators:

- **Seleccione las métricas que desee mostrar** marcando las casillas en el lado izquierdo. Los gráficos individuales para cada una de las métricas aparecerán en el widget, en orden.
- **Cambie el segmento de línea base**, la primera barra del gráfico, con ayuda del selector desplegable. Los deltas se calcularán con este valor de referencia.
- **Elija umbrales** desde el selector desplegable. Puede ver varios umbrales en el mismo gráfico. Los umbrales seleccionados aparecerán en negrita y podrá hacer clic en un umbral en negrita para anular su selección.
- **Pase el cursor sobre una barra** para ver las métricas de ese segmento.
- **Identifique las disparidades con la línea base** con la columna "Diff w. baseline" (Dif. con línea base), que identifica la diferencia porcentual entre el segmento actual y la línea base.
- **Explore los puntos de datos de un segmento en profundidad** con la [herramienta What-If](https://pair-code.github.io/what-if-tool/). Vea un ejemplo [aquí](https://github.com/tensorflow/fairness-indicators/).

#### Representación de Fairness Indicators para múltiples modelos

También se puede usar Fairness Indicators para comparar modelos. En lugar de pasar un único eval_result, pase un objeto multi_eval_results, que es un diccionario que asigna dos nombres de modelos a objetos eval_result.

```python
from tensorflow_model_analysis.addons.fairness.view import widget_view

eval_result1 = tfma.load_eval_result(...)
eval_result2 = tfma.load_eval_result(...)
multi_eval_results = {"MyFirstModel": eval_result1, "MySecondModel": eval_result2}

widget_view.render_fairness_indicator(multi_eval_results=multi_eval_results)
```

![Fairness Indicators - Comparación de modelos](images/fairnessIndicatorsCompare.png)

La comparación de modelos se puede usar junto con la comparación de umbrales. Por ejemplo, puede comparar dos modelos en dos conjuntos de umbrales para encontrar la mejor combinación para sus métricas de equidad.

## Cómo usar Fariness Indicators con modelos que no sean de Tensorflow

Para prestar un mejor servicio a los clientes que tienen diferentes modelos y flujos de trabajo, desarrollamos una biblioteca de evaluación que es independiente del modelo que se esté evaluando.

Cualquiera que quiera evaluar su sistema de aprendizaje automático puede usarla, especialmente si tiene modelos que no estén basados ​​en TensorFlow. Con SDK Apache Beam de Python, puede crear un binario de evaluación TFMA independiente y luego ejecutarlo para analizar su modelo.

### Datos

Este paso consiste en proporcionar el conjunto de datos sobre el que desea ejecutar las evaluaciones. Debe estar en formato [tf.Proto](https://www.tensorflow.org/tutorials/load_data/tfrecord) con etiquetas, predicciones y otras características que quizás desee segmentar.

```python
tf.Example {
    features {
        feature {
          key: "fur_color" value { bytes_list { value: "gray" } }
        }
        feature {
          key: "height" value { bytes_list { value: "tall" } }
        }
        feature {
          key: "prediction" value { float_list { value: 0.9 } }
        }
        feature {
          key: "label" value { float_list { value: 1.0 } }
        }
    }
}
```

### Modelo

En lugar de especificar un modelo, puede crear una configuración de evaluación y un extractor independientes del modelo para parsear y proporcionar los datos que TFMA necesita para calcular las métricas. La especificación [ModelAgnosticConfig](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_predict.py) define las características, predicciones y etiquetas que se usarán a partir de los ejemplos de entrada.

Para esto, cree un mapa de características con claves que representen todas las características, incluidas etiquetas y claves de predicción, y valores que representen el tipo de datos de la característica.

```python
feature_map[label_key] = tf.FixedLenFeature([], tf.float32, default_value=[0])
```

Cree una configuración independiente del modelo a partir de claves de etiqueta, claves de predicción y mapa de características.

```python
model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
    label_keys=list(ground_truth_labels),
    prediction_keys=list(predition_labels),
    feature_spec=feature_map)
```

### Configure un extractor independiente del modelo

[Extractor](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/model_agnostic_eval/model_agnostic_extractor.py) se usa para extraer las características, etiquetas y predicciones de la entrada con una configuración independiente del modelo. Y si desea segmentar sus datos, también debe definir la [especificación de clave de segmento](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/slicer), que contiene información sobre las columnas que desea segmentar.

```python
model_agnostic_extractors = [
    model_agnostic_extractor.ModelAgnosticExtractor(
        model_agnostic_config=model_agnostic_config, desired_batch_size=3),
    slice_key_extractor.SliceKeyExtractor([
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(columns=[‘height’]),
    ])
]
```

### Calcule las métricas de equidad

Como parte de [EvalSharedModel](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/types/EvalSharedModel), puede proporcionar todas las métricas según las cuales desea que se evalúe su modelo. Las métricas se proporcionan en forma de retrollamadas de métricas como las que se definen en [post_export_metrics](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py) o [fairness_indicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py).

```python
metrics_callbacks.append(
    post_export_metrics.fairness_indicators(
        thresholds=[0.5, 0.9],
        target_prediction_keys=[prediction_key],
        labels_key=label_key))
```

También toma una `construct_fn` que se usa para crear un gráfico de tensorflow para ejecutar la evaluación.

```python
eval_shared_model = types.EvalSharedModel(
    add_metrics_callbacks=metrics_callbacks,
    construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
        add_metrics_callbacks=metrics_callbacks,
        fpl_feed_config=model_agnostic_extractor
        .ModelAgnosticGetFPLFeedConfig(model_agnostic_config)))
```

Una vez que esté todo listo, use una de las funciones `ExtractEvaluate` o `ExtractEvaluateAndWriteResults` proporcionadas por [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) para evaluar el modelo.

```python
_ = (
    examples |
    'ExtractEvaluateAndWriteResults' >>
        model_eval_lib.ExtractEvaluateAndWriteResults(
        eval_shared_model=eval_shared_model,
        output_path=output_path,
        extractors=model_agnostic_extractors))

eval_result = tensorflow_model_analysis.load_eval_result(output_path=tfma_eval_result_path)
```

Finalmente, siga las instrucciones de la sección "Representación de Fairness Indicators" para crear una representación de Fairness Indicators.

## Más ejemplos

El [directorio de ejemplos de Fairness Indicators](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/) contiene varios ejemplos:

- [Fairness_Indicators_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Example_Colab.ipynb) ofrece una descripción general de Fairness Indicators en [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma) y cómo usarlos con un conjunto de datos real. Este bloc de notas también analiza [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) y la [herramienta What-If](https://pair-code.github.io/what-if-tool/), dos herramientas para analizar modelos de TensorFlow que vienen con Fairness Indicators.
- [Fairness_Indicators_on_TF_Hub.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_on_TF_Hub_Text_Embeddings.ipynb) demuestra cómo usar Fairness Indicators para comparar modelos entrenados en diferentes [inserciones de texto](https://en.wikipedia.org/wiki/Word_embedding). Este bloc de notas usa inserciones de texto de [TensorFlow Hub](https://www.tensorflow.org/hub), la biblioteca de TensorFlow para publicar, descubrir y reutilizar componentes del modelo.
- [Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) demuestra cómo visualizar Fairness Indicators en TensorBoard.

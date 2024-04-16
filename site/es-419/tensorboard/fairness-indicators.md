# Evaluación de modelos con el panel de control de indicadores de equidad [Beta]

![Fairness Indicators](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/fairness-indicators.png?raw=true)

Los indicadores de equidad para TensorBoard permiten calcular fácilmente métricas de equidad comúnmente identificadas para clasificadores *binarios* y *multiclase*. Con el complemento, puede visualizar las evaluaciones de equidad de sus ejecuciones y comparar fácilmente el rendimiento entre grupos.

En particular, los indicadores de equidad para TensorBoard le permiten evaluar y visualizar el rendimiento de los modelos, divididos en grupos definidos de usuarios. Siéntase seguro de sus resultados con intervalos de confianza y evaluaciones en múltiples umbrales.

Muchas herramientas existentes para evaluar los problemas de equidad no funcionan bien en conjuntos de datos y modelos a gran escala. En Google, es importante para nosotros disponer de herramientas que puedan funcionar en sistemas de miles de millones de usuarios. Los indicadores de equidad le permitirán evaluar casos de uso de cualquier tamaño, en el entorno TensorBoard o en [Colab](https://github.com/tensorflow/fairness-indicators).

## Requisitos

Para instalar los indicadores de equidad para TensorBoard, ejecute:

```
python3 -m virtualenv ~/tensorboard_demo
source ~/tensorboard_demo/bin/activate
pip install --upgrade pip
pip install fairness_indicators
pip install tensorboard-plugin-fairness-indicators
```

## Demostración

Si desea probar los indicadores de equidad en TensorBoard, puede descargar los resultados de la evaluación del análisis de modelos de TensorFlow como muestra (archivos eval_config.json, métricas y gráficas) y una utilidad `demo.py` de la Plataforma Google Cloud, [aquí](https://console.cloud.google.com/storage/browser/tensorboard_plugin_fairness_indicators/) utilizando el siguiente comando.

```
pip install gsutil
gsutil cp -r gs://tensorboard_plugin_fairness_indicators/ .
```

Navegue hasta el directorio que contiene los archivos descargados.

```
cd tensorboard_plugin_fairness_indicators
```

Estos datos de evaluación se basan en el conjunto de datos [Comentarios civiles](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), calculado mediante la biblioteca [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) de Tensorflow Análisis de modelos. También contiene un archivo de datos de resumen de TensorBoard de muestra a modo de referencia.

La utilidad `demo.py` escribe un archivo de datos de resumen TensorBoard, que será leído por TensorBoard para renderizar el panel de control de los indicadores de equidad (Consulte el tutorial [TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/README.md) para obtener más información sobre los archivos de datos de resumen).

Los indicadores se utilizarán con la utilidad `demo.py`:

- `--logdir`: Directorio donde TensorBoard escribirá el resumen
- `--eval_result_output_dir`: Directorio que contiene los resultados de la evaluación realizada por TFMA (descargados en el último paso).

Ejecute la utilidad `demo.py` para escribir los resultados resumidos en el directorio de registro:

`python demo.py --logdir=. --eval_result_output_dir=.`

Ejecute TensorBoard:

Nota: Para esta demostración, ejecute TensorBoard desde el mismo directorio que contiene todos los archivos descargados.

`tensorboard --logdir=.`

Esto iniciará una instancia local. Una vez iniciada la instancia local, aparecerá un enlace en la terminal. Abra el enlace en su navegador para ver el panel de indicadores de imparcialidad.

### Demostración de Colab

[Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb) contiene una demostración de principio a fin para entrenar y evaluar un modelo y visualizar los resultados de la evaluación de equidad en TensorBoard.

## Uso

Para utilizar los indicadores de equidad con sus propios datos y evaluaciones:

1. Entrene un nuevo modelo y evalúe utilizando `tensorflow_model_analysis.run_model_analysis` o `tensorflow_model_analysis.ExtractEvaluateAndWriteResult` API en [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py). Para obtener fragmentos de código sobre cómo hacerlo, consulte el colab de los indicadores de equidad [aquí](https://github.com/tensorflow/fairness-indicators).

2. Escriba el resumen de los indicadores de imparcialidad utilizando la API `tensorboard_plugin_fairness_indicators.summary_v2`.

    ```
    writer = tf.summary.create_file_writer(<logdir>)
    with writer.as_default():
        summary_v2.FairnessIndicators(<eval_result_dir>, step=1)
    writer.close()
    ```

3. Ejecute TensorBoard

    - `tensorboard --logdir=<logdir>`
    - Seleccione el nuevo proceso de evaluación utilizando el menú desplegable situado en la parte izquierda del panel de control para visualizar los resultados.

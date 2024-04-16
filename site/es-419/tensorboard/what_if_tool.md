# Modelar la interpretación con el panel control de las herramientas What-If

> **Advertencia** Esta documentación sólo se aplica a TensorBoard 2.11 y anteriores, ya que la herramienta What-If ya no se mantiene activamente. Consulte la [herramienta de aprendizaje de interpretación (LIT)](https://pair-code.github.io/lit/) que se mantiene activamente con este fin.

![What-If Tool](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/what_if_tool.png?raw=true)

La herramienta What-If (WIT) proporciona una interfaz fácil de usar para ampliar la comprensión de los modelos ML de clasificación y regresión de la caja negra. Con el complemento, puede realizar inferencias sobre un gran conjunto de ejemplos y visualizar inmediatamente los resultados de diversas formas. Además, los ejemplos pueden editarse manual o programáticamente y volver a ejecutarse a través del modelo para ver los resultados de los cambios. Contiene herramientas para investigar el rendimiento y la equidad del modelo sobre subconjuntos de un conjunto de datos.

El propósito de la herramienta es ofrecer a la gente una forma sencilla, intuitiva y potente de explorar e investigar modelos ML entrenados mediante una interfaz visual sin necesidad de utilizar ningún tipo de código.

Se puede acceder a la herramienta por medio de TensorBoard o directamente en un bloc de notas Jupyter o Colab. Para obtener más detalles en profundidad, demostraciones, recorridos e información específica sobre el uso de WIT en modo bloc de notas, consulte el sitio web de la herramienta [What-If](https://pair-code.github.io/what-if-tool).

## Requisitos

Para utilizar WIT en TensorBoard, son necesarias dos cosas:

- El modelo o modelos que desee explorar deben servirse mediante el [Servidor de TensorFlow](https://github.com/tensorflow/serving) utilizando la API de clasificación, regresión o predicción.
- El conjunto de datos a inferir por los modelos debe estar en un archivo TFRecord accesible por el servidor web TensorBoard.

## Uso

Al abrir el panel de la herramienta What-If en TensorBoard, verá una pantalla de configuración en la que deberá proporcionar el host y el puerto del servidor del modelo, el nombre del modelo que será servido, el tipo de modelo y la ruta del archivo TFRecords a cargar. Tras rellenar esta información y hacer clic en "Aceptar", WIT cargará el conjunto de datos y ejecutará la inferencia con el modelo, mostrando los resultados.

Para obtener más información sobre las distintas funciones de WIT y cómo pueden ayudar en la comprensión del modelo y en las investigaciones de equidad, consulte el recorrido en la página web de ls herramienta [What-If](https://pair-code.github.io/what-if-tool).

## Modelo de demostración y conjunto de datos

Si desea probar WIT en TensorBoard con un modelo preentrenado, puede descargar y descomprimir un modelo preentrenado y un conjunto de datos de https://storage.googleapis.com/what-if-tool-resources/uci-census-demo/uci-census-demo.zip. Se trata de un modelo de clasificación binaria que utiliza el conjunto de datos [UCI Census](https://archive.ics.uci.edu/ml/datasets/census+income) para predecir si una persona gana más de 50,000 dólares al año. Este conjunto de datos y esta tarea de predicción se utilizan a menudo en la investigación sobre modelos de aprendizaje automático y equidad.

Establezca la variable de entorno MODEL_PATH en la ubicación del directorio del modelo resultante en su máquina.

Instale los servidores Docker y TensorFlow siguiendo la [documentación oficial](https://www.tensorflow.org/tfx/serving/docker).

Sirva el modelo utilizando docker mediante `docker run -p 8500:8500 --mount type=bind,source=${MODEL_PATH},target=/models/uci_income -e MODEL_NAME=uci_income -t tensorflow/serving`. Tenga en cuenta que puede necesitar ejecutar el comando con `sudo` dependiendo de su configuración docker.

Ahora inicie tensorboard y utilice el menú desplegable del panel de control para navegar hasta la herramienta What-if.

Establezca en la pantalla de configuración la dirección de inferencia como "localhost:8500", el nombre del modelo como "uci_income" y la ruta a los ejemplos como la ruta completa al archivo `adult.tfrecord` descargado, y pulse "Aceptar".

![Setup screen for demo](https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tensorboard/images/what_if_tool_demo_setup.png?raw=true)

Algunas cosas que puede probar con la herramienta What-if en esta demostración son:

- Edite un único punto de datos y vea el cambio resultante en la inferencia.
- Explorar la relación entre las características individuales del conjunto de datos y los resultados de la inferencia del modelo mediante gráficos de dependencia parcial.
- Divida el conjunto de datos en subconjuntos y compare el rendimiento entre ellos.

Si desea conocer en profundidad las características de la herramienta, consulte el [Paseo por la herramienta What-If](https://pair-code.github.io/what-if-tool/walkthrough.html).

Tenga en cuenta que la característica de la verdad fundamental en el conjunto de datos que este modelo está intentando predecir se denomina "Objetivo", por lo que cuando utilice la pestaña "Rendimiento y equidad", "Objetivo" es lo que querrá especificar en el menú desplegable de la característica de la verdad fundamental.

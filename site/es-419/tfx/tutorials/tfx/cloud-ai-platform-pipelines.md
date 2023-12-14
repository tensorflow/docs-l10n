# TFX en Cloud AI Platform Pipelines

## Introducción

Este tutorial está diseñado para presentar tanto [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) como [AIPlatform Pipelines] (https://cloud.google.com/ai-platform/pipelines/docs/introduction) y ayudarlo a aprender a crear sus propias canalizaciones de aprendizaje automático en Google Cloud. Muestra la integración con TFX, AI Platform Pipelines y Kubeflow, así como la interacción con TFX en blocs de notas Jupyter.

Al final de este tutorial, habrá creado y ejecutado una canalización de ML, alojada en Google Cloud. Podrá visualizar los resultados de cada ejecución y ver el linaje de los artefactos creados.

Término clave: Una canalización de TFX es un grafo acíclico dirigido o "DAG". A menudo nos referiremos a las canalizaciones como DAG.

Seguiremos un proceso típico de desarrollo de ML, que comienza por examinar el conjunto de datos y termina con una canalización de trabajo completa. A lo largo del camino, exploraremos las distintas formas de depurar y actualizar una canalización, y mediremos el rendimiento.

Nota: Completar este tutorial puede llevar entre 45 y 60 minutos.

### Conjunto de datos Chicago Taxi

<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->

![Taxi](images/airflow_workshop/taxi.jpg)![Chicago taxi](images/airflow_workshop/chicago.png)

Usaremos el [conjunto de datos Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) publicado por la ciudad de Chicago.

Nota: Este sitio ofrece aplicaciones que usan datos que fueron modificados para su uso desde su fuente original, www.cityofchicago.org, el sitio web oficial de la ciudad de Chicago. La ciudad de Chicago no garantiza el contenido, la exactitud, la puntualidad o la integridad de ninguno de los datos que se proporcionan en este sitio. Los datos proporcionados en este sitio están sujetos a cambios en cualquier momento. Se entiende que los datos proporcionados en este sitio se usan bajo su propia responsabilidad.

[Más información](https://cloud.google.com/bigquery/public-data/chicago-taxi) sobre el conjunto de datos en [Google BigQuery](https://cloud.google.com/bigquery/). Explore el conjunto de datos completo en la [interfaz de usuario de BigQuery](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).

#### Objetivo del modelo: clasificación binaria

¿El cliente dará una propina mayor o menor al 20 %?

## 1. Configure un proyecto de Google Cloud

### 1.a Configure su entorno en Google Cloud

Para comenzar, necesita una cuenta de Google Cloud. Si ya tiene una, omita los pasos hasta [Crear nuevo proyecto](#create_project).

Advertencia: Esta demostración fue diseñada para que no exceda los límites del [Nivel gratuito de Google Cloud](https://cloud.google.com/free). Si ya tiene una cuenta de Google, es posible que haya alcanzado los límites de su nivel gratuito o haya agotado los créditos gratuitos de Google Cloud otorgados a nuevos usuarios. **Si ese es el caso, después de esta demostración se generarán cargos en su cuenta de Google Cloud**.

1. Vaya a la [Consola de Google Cloud](https://console.cloud.google.com/).

2. Acepte los términos y condiciones de Google Cloud

    <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/welcome-popup.png">

3. Si desea comenzar con una cuenta de prueba gratuita, haga clic en [**Probar gratis**](https://console.cloud.google.com/freetrial) (o [**Comenzar gratis**](https://console.cloud.google.com/freetrial)).

    1. Elija su país.

    2. Acepte los términos del servicio.

    3. Ingrese los detalles de facturación.

        No se le cobrará en este momento. Si no tiene otros proyectos de Google Cloud, puede completar este tutorial sin exceder los límites del [Nivel gratuito de Google Cloud](https://cloud.google.com/free), que incluye un máximo de 8 núcleos en ejecución simultánea.

Nota: En este punto, puede elegir convertirse en un usuario pago en lugar de depender de la prueba gratuita. Dado que este tutorial se mantiene dentro de los límites del nivel gratuito, no se le cobrará si este es su único proyecto y se mantiene dentro de esos límites. Para obtener más información al respecto, consulte [Calculadora de precios de Google Cloud](https://cloud.google.com/products/calculator/) y [Nivel gratuito de Google Cloud Platform](https://cloud.google.com/free).

### 1.b Cree un nuevo proyecto.<a name="create_project"></a>

Nota: En este tutorial se asume que desea trabajar en esta demostración en un nuevo proyecto. Puede, si lo desea, trabajar en un proyecto existente.

Nota: Debe tener una tarjeta de crédito verificada registrada antes de crear el proyecto.

1. Desde el [principal panel de Google Cloud](https://console.cloud.google.com/home/dashboard), haga clic en el menú desplegable del proyecto junto al encabezado de **Google Cloud Platform** y seleccione **Nuevo proyecto**.
2. Póngale nombre e ingrese otros detalles de su proyecto.
3. **Una vez que haya creado un proyecto, asegúrese de seleccionarlo en el menú desplegable de proyectos.**

## 2. Configure e implemente AI Platform Pipeline en un nuevo clúster de Kubernetes

Nota: Esto tardará hasta 10 minutos, ya que hay que esperar en varios puntos para que se aprovisionen los recursos.

1. Vaya a la página [Clústeres de AI Platform Pipelines](https://console.cloud.google.com/ai-platform/pipelines).

    En el menú de navegación principal: ≡ &gt; AI Platform &gt; Pipelines

2. Haga clic en **+ Nueva instancia** para crear un nuevo clúster.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-instance.png">

3. En la página de descripción general de **Kubeflow Pipelines**, haga clic en **Configurar**.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/configure.png">

4. Haga clic en "Habilitar" para habilitar la API de Kubernetes Engine

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/enable_api.png">

    Nota: Es posible que tenga que esperar varios minutos antes de continuar, mientras se habilitan las API de Kubernetes Engine.

5. En la página **Implementar Kubeflow Pipelines**:

    1. Seleccione una [zona](https://cloud.google.com/compute/docs/regions-zones) (o "región") para su clúster. La red y la subred se pueden configurar, pero para los fines de este tutorial las dejaremos con sus valores predeterminados.

    2. **IMPORTANTE** Marque la casilla *Permitir acceso a las siguientes API de la nube*. (Esto es necesario para que este clúster acceda a las otras partes de su proyecto. Si omite este paso, solucionarlo más tarde es un poco complicado).

        <img style="width: 50%;" src="images/cloud-ai-platform-pipelines/check-the-box.png">

    3. Haga clic en **Crear nuevo clúster** y espere varios minutos hasta que se haya creado el clúster. Esto tomará unos pocos minutos. Cuando se complete, verá un mensaje como este:

        > Cluster "cluster-1" successfully created in zone "us-central1-a".

    4. Seleccione un espacio de nombres y un nombre de instancia (se pueden usar los valores predeterminados). Para los fines de este tutorial, no marque *executor.emissary* ni *wantedstorage.enabled*.

    5. Haga clic en **Implementar** y espere unos momentos hasta que se haya implementado la canalización. Al implementar Kubeflow Pipelines, acepte los Términos del servicio.

## 3. Configure la instancia de Cloud AI Platform Notebook

1. Vaya a la página de [Vertex AI Workbench](https://console.cloud.google.com/vertex-ai/workbench). La primera vez que ejecute Workbench, deberá habilitar la API de Notebooks.

    En el menú de navegación principal: ≡ -&gt; Vertex AI -&gt; Workbench

2. Si se le solicita, habilite la API de Compute Engine.

3. Cree una **Nuevo bloc de notas** con TensorFlow Enterprise 2.7 (o superior) instalado.

    <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/new-notebook.png">

    Nuevo bloc de notas -&gt; TensorFlow Enterprise 2.7 -&gt; Sin GPU

    Seleccione una región y zona, y asigne un nombre a la instancia del bloc de notas.

    Si desea permanecer dentro de los límites del Nivel gratuito, quizás deba cambiar la configuración predeterminada aquí para reducir la cantidad de vCPU disponibles para esta instancia de 4 a 2:

    1. Seleccione **Opciones avanzadas** en la parte inferior del formulario **Nuevo bloc de notas**.

    2. En **Configuración de la máquina,** tal vez quiera seleccionar una configuración con 1 o 2 vCPU si necesita permanecer en el nivel gratuito.

        <img style="width: 65%;" src="images/cloud-ai-platform-pipelines/two-cpus.png">

    3. Espere a que se cree el nuevo bloc de notas y luego haga clic en **Habilitar API de Notebooks.**

Nota: Es posible que experimente un rendimiento lento en su bloc de notas si usa 1 o 2 vCPU en lugar del valor predeterminado o superior. Esto no debería dificultar seriamente la finalización de este tutorial. Si desea utilizar la configuración predeterminada, [actualice su cuenta](https://cloud.google.com/free/docs/gcp-free-tier#to_upgrade_your_account) a al menos 12 vCPU. Esto acumulará cargos. Consulte [Precios de Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/pricing/) para obtener más detalles sobre los precios, incluida una [calculadora de precios](https://cloud.google.com/products/calculator) e información sobre el [Nivel gratuito de Google Cloud](https://cloud.google.com/free).

## 4. Inicie el bloc de notas de introducción

1. Vaya a la página [**Clústeres de AI Platform Pipelines**] (https://console.cloud.google.com/ai-platform/pipelines).

    En el menú de navegación principal: ≡ &gt; AI Platform &gt; Pipelines

2. En la línea del clúster que se usa en este tutorial, haga clic en **Abrir panel de canalizaciones**.

    <img src="images/cloud-ai-platform-pipelines/open-dashboard.png">

3. En la página **Introducción**, haga clic en **Abrir un bloc de notas de Cloud AI Platform en Google Cloud**.

    <img src="images/cloud-ai-platform-pipelines/open-template.png">

4. Seleccione la instancia de Notebook que se está usando para este tutorial, **Continuar** y luego **Confirmar**.

    ![seleccionar-bloc](images/cloud-ai-platform-pipelines/select-notebook.png)

## 5. Continúe trabajando en el bloc de notas

Importante: El resto de este tutorial debe completarse en el bloc de notas JupyterLab que abrió en el paso anterior. Las instrucciones y explicaciones están disponibles aquí como referencia.

### Instale

El bloc de notas de introducción comienza por instalar [TFX](https://www.tensorflow.org/tfx) y [Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/) en una VM en la que se ejecuta Jupyter Lab.

Luego verifica qué versión de TFX está instalada, realiza una importación y configura e imprime el ID del proyecto:

![comprobar la versión de Python e importar](images/cloud-ai-platform-pipelines/check-version-nb-cell.png)

### Conéctese con sus servicios de Google Cloud

La configuración de la canalización necesita el ID de su proyecto, que puede obtener a través del bloc de notas y configurar como una variable del entorno.

```python
# Read GCP project id from env.
shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
GCP_PROJECT_ID=shell_output[0]
print("GCP project ID:" + GCP_PROJECT_ID)
```

Ahora configure el punto de conexión de su clúster de KFP.

Esto se puede encontrar en la URL del panel de Pipelines. Vaya al panel de Kubeflow Pipeline y observe la URL. El punto de conexión es todo lo que está en la URL *, comienza con * `https://` *y se extiende hasta* `googleusercontent.com` inclusive.

```python
ENDPOINT='' # Enter YOUR ENDPOINT here.
```

Luego, el bloc de notas establece un nombre único para la imagen de Docker personalizada:

```python
# Docker image name for the pipeline image
CUSTOM_TFX_IMAGE='gcr.io/' + GCP_PROJECT_ID + '/tfx-pipeline'
```

## 6. Copie una plantilla en el directorio de su proyecto

Edite la siguiente celda del bloc de notas para establecer un nombre para su canalización. En este tutorial usaremos `my_pipeline`.

```python
PIPELINE_NAME="my_pipeline"
PROJECT_DIR=os.path.join(os.path.expanduser("~"),"imported",PIPELINE_NAME)
```

Luego, el bloc de notas usa la CLI de `tfx` para copiar la plantilla de canalización. En este tutorial se usa el conjunto de datos Chicago Taxi para ejecutar una clasificación binaria, por lo que la plantilla configura el modelo para `taxi`:

```python
!tfx template copy \
  --pipeline-name={PIPELINE_NAME} \
  --destination-path={PROJECT_DIR} \
  --model=taxi
```

Luego, el bloc de notas cambia su contexto CWD al directorio del proyecto:

```
%cd {PROJECT_DIR}
```

### Explore los archivos de canalización

En el lado izquierdo del bloc de notas de Cloud AI Platform, debería ver un explorador de archivos. Debería haber un directorio con el nombre de su canalización (`my_pipeline`). Ábralo y mire los archivos. (También podrá abrirlos y editarlos desde el entorno del bloc de notas).

```
# You can also list the files from the shell
! ls
```

El comando `tfx template copy` anterior creó una estructura básica de archivos que compilan una canalización. Estos incluyen códigos fuente de Python, datos de muestra y bloc de notas Jupyter. Estos están destinados a este ejemplo en particular. Para sus propias canalizaciones, estos serían los archivos de soporte que requiere su canalización.

Esta es una breve descripción de los archivos de Python.

- `pipeline`: este directorio contiene la definición de la canalización.
    - `configs.py`: define constantes comunes para los ejecutores de la canalización
    - `pipeline.py`: define los componentes de TFX y una canalización
- `models`: este directorio contiene definiciones de modelos de ML.
    - `features.py` `features_test.py`: define características para el modelo
    - `preprocessing.py` / `preprocessing_test.py`: define trabajos de preprocesamiento con ayuda de `tf::Transform`
    - `estimator`: este directorio contiene un modelo basado en Estimator.
        - `constants.py`: define las constantes del modelo
        - `model.py` / `model_test.py`: define el modelo DNN utilizando el estimador de TF
    - `keras`: este directorio contiene un modelo basado en Keras.
        - `constants.py`: define las constantes del modelo
        - `model.py` / `model_test.py`: define el modelo DNN usando Keras
- `beam_runner.py` / `kubeflow_runner.py`: define ejecutores para cada motor de orquestación

## 7. Ejecute su primera canalización de TFX en Kubeflow

El bloc de notas ejecutará la canalización mediante el comando de CLI de `tfx run`.

### Cómo conectarse al almacenamiento

La ejecución de canalizaciones crea artefactos que deben almacenarse en [ML-Metadata](https://github.com/google/ml-metadata). Los artefactos se refieren a cargas útiles, que son archivos que deben almacenarse en un sistema de archivos o almacenamiento en bloque. Para este tutorial, usaremos GCS para almacenar nuestras cargas de metadatos, usando el depósito que se creó automáticamente durante la configuración. Su nombre será `<your-project-id>-kubeflowpipelines-default`.

### Cómo crear la canalización

El bloc de notas cargará nuestros datos de muestra en el depósito de GCS para que podamos usarlos más adelante en nuestra canalización.

```python
!gsutil cp data/data.csv gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/tfx-template/data/taxi/data.csv
```

Luego, el bloc de notas usa el comando `tfx pipeline create` para crear la canalización.

```python
!tfx pipeline create  \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT} \
--build-image
```

Mientras se crea una canalización, se genera un `Dockerfile` para compilar una imagen de Docker. Recuerde que debe agregar estos archivos a su sistema de control de código fuente (por ejemplo, git) junto con otros archivos fuente.

### Cómo ejecutar la canalización

Luego, el cuaderno utiliza el comando `tfx run create` para iniciar una ejecución de su canalización. También verá esta ejecución en la lista de Experimentos en el panel de Kubeflow Pipelines.

```python
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

Puede ver su canalización desde el panel de Kubeflow Pipelines.

Nota: Si la ejecución de su canalización falla, puede ver registros detallados en el Panel de KFP. Una de las principales fuentes de error son los problemas relacionados con los permisos. Asegúrese de que su clúster de KFP tenga permisos para acceder a las API de Google Cloud. Esto se puede configurar [al momento de crear un clúster de KFP en GCP](https://cloud.google.com/ai-platform/pipelines/docs/setting-up) o bien puede consultar el [documento de solución de problemas en GCP](https://cloud.google.com/ai-platform/pipelines/docs/troubleshooting).

## 8. Valide sus datos

La primera tarea en cualquier proyecto de ciencia de datos o aprendizaje automático es comprender y limpiar los datos.

- Analice los tipos de datos para cada característica
- Busque anomalías y valores faltantes.
- Comprenda las distribuciones de cada característica.

### Componentes

![Componentes de datos](images/airflow_workshop/examplegen1.png)![Componentes de datos](images/airflow_workshop/examplegen2.png)

- [ExempleGen](https://www.tensorflow.org/tfx/guide/examplegen) ingiere y divide el conjunto de datos de entrada.
- [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) calcula estadísticas para el conjunto de datos.
- [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) SchemaGen examina las estadísticas y crea un esquema de datos.
- [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) busca anomalías y valores faltantes en el conjunto de datos.

### En el editor de archivos de Jupyter lab:

En `pipeline` / `pipeline.py`, descomente las líneas que agregan estos componentes a su canalización:

```python
# components.append(statistics_gen)
# components.append(schema_gen)
# components.append(example_validator)
```

(`ExampleGen` ya estaba habilitado al momento de copiar los archivos de la plantilla).

### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Compruebe la canalización

Para Kubeflow Orchestrator, visite el panel de KFP y busque resultados de canalización en la página para la ejecución de su canalización. Haga clic en la pestaña "Experimentos" a la izquierda y en "Todas las ejecuciones" en la página Experimentos. Debería poder encontrar la ejecución con el nombre de su canalización.

### Ejemplo más avanzado

El ejemplo que se presenta aquí en realidad solo pretende servirle de puntapié inicial. Para ver un ejemplo más avanzado, consulte el apartado [Colab de TensorFlow Data Validation](https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi).

Para obtener más información sobre el uso de TFDV para explorar y validar un conjunto de datos, [consulte los ejemplos en tensorflow.org](https://www.tensorflow.org/tfx/data_validation).

## 9. Ingeniería de características

Puede aumentar la calidad predictiva de sus datos o reducir la dimensionalidad con la ingeniería de características.

- Cruces de características
- Vocabularios
- Incorporaciones
- PCA
- Codificación categórica

Uno de los beneficios de usar TFX es que escribirá su código de transformación una vez y las transformaciones resultantes serán coherentes entre el entrenamiento y el servicio.

### Componentes

![Transform](images/airflow_workshop/transform.png)

- [Transform](https://www.tensorflow.org/tfx/guide/transform) realiza ingeniería de características en el conjunto de datos.

### En el editor de archivos de Jupyter lab:

En `pipeline` / `pipeline.py`, busque y descomente la línea que agrega [Transform](https://www.tensorflow.org/tfx/guide/transform) a la canalización.

```python
# components.append(transform)
```

### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique las salidas de la canalización

Para Kubeflow Orchestrator, visite el panel de KFP y busque resultados de canalización en la página para la ejecución de su canalización. Haga clic en la pestaña "Experimentos" a la izquierda y en "Todas las ejecuciones" en la página Experimentos. Debería poder encontrar la ejecución con el nombre de su canalización.

### Ejemplo más avanzado

El ejemplo que se presenta aquí en realidad solo pretende servirle de puntapié inicial. Para ver un ejemplo más avanzado, consulte el apartado [Colab de TensorFlow Transform](https://www.tensorflow.org/tfx/tutorials/transform/census).

## 10. Entrenamiento

Entrene un modelo de TensorFlow con sus datos depurados y transformados.

- Incluya las transformaciones del paso anterior para que se apliquen de manera coherente.
- Guarde los resultados como SavedModel para producción.
- Visualice y explore el proceso de entrenamiento con TensorBoard
- Guarde también un EvalSavedModel para analizar el rendimiento del modelo.

### Componentes

- [Trainer](https://www.tensorflow.org/tfx/guide/trainer) entrena un modelo de TensorFlow.

### En el editor de archivos de Jupyter lab:

En `pipeline` / `pipeline.py`, busque y descomente el elemento que agrega Trainer a la canalización:

```python
# components.append(trainer)
```

### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique las salidas de la canalización

Para Kubeflow Orchestrator, visite el panel de KFP y busque resultados de canalización en la página para la ejecución de su canalización. Haga clic en la pestaña "Experimentos" a la izquierda y en "Todas las ejecuciones" en la página Experimentos. Debería poder encontrar la ejecución con el nombre de su canalización.

### Ejemplo más avanzado

El ejemplo que se presenta aquí en realidad solo pretende servirle de puntapié inicial. Para ver un ejemplo más avanzado, consulte el [Tutorial de TensorFlow](https://www.tensorflow.org/tensorboard/get_started).

## 11. Cómo analizar el rendimiento del modelo

Comprender mucho más que las métricas de nivel superior.

- Los usuarios experimentan el rendimiento del modelo solo para sus consultas.
- Un rendimiento deficiente en segmentos de datos puede ocultarse mediante métricas de nivel superior.
- La equidad del modelo es importante.
- A menudo, los subconjuntos clave de usuarios o datos son muy importantes y pueden ser pequeños.
    - Rendimiento en condiciones críticas pero inusuales
    - Rendimiento para audiencias clave como influencers
- Si está reemplazando un modelo que está actualmente en producción, primero asegúrese de que el nuevo sea mejor.

### Componentes

- [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) realiza un análisis profundo de los resultados del entrenamiento.

### En el editor de archivos de Jupyter lab:

En `pipeline` / `pipeline.py`, busque y descomente la línea que agrega Evaluator a la canalización:

```python
components.append(evaluator)
```

### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
! tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

! tfx run create --pipeline-name "{PIPELINE_NAME}"
```

### Verifique las salidas de la canalización

Para Kubeflow Orchestrator, visite el panel de KFP y busque resultados de canalización en la página para la ejecución de su canalización. Haga clic en la pestaña "Experimentos" a la izquierda y en "Todas las ejecuciones" en la página Experimentos. Debería poder encontrar la ejecución con el nombre de su canalización.

## 12. Servicio del modelo

Si el nuevo modelo está listo, perfecto.

- Pusher implementa SavedModels en ubicaciones conocidas

Los objetivos de implementación reciben nuevos modelos de ubicaciones conocidas

- <a>TensorFlow Serving</a>
- TensorFlow Lite
- TensorFlow JS
- TensorFlow Hub

### Componentes

- [Pusher](https://www.tensorflow.org/tfx/guide/pusher) implementa el modelo en una infraestructura de servicio.

### En el editor de archivos de Jupyter lab:

En `pipeline` / `pipeline.py`, busque y descomente la línea que agrega Pusher a la canalización:

```python
# components.append(pusher)
```

### Verifique las salidas de la canalización

Para Kubeflow Orchestrator, visite el panel de KFP y busque resultados de canalización en la página para la ejecución de su canalización. Haga clic en la pestaña "Experimentos" a la izquierda y en "Todas las ejecuciones" en la página Experimentos. Debería poder encontrar la ejecución con el nombre de su canalización.

### Objetivos de implementación disponibles

Ya ha entrenado y validado su modelo, y ya está listo para producción. Ahora puede implementar su modelo en cualquiera de los objetivos de implementación de TensorFlow, incluidos los siguientes:

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), para servir su modelo en un servidor o granja de servidores y procesar solicitudes de inferencia REST o gRPC.
- [TensorFlow Lite](https://www.tensorflow.org/lite), para incluir su modelo en una aplicación móvil nativa de Android o iOS, o en una aplicación Raspberry Pi, IoT o microcontrolador.
- [TensorFlow.js](https://www.tensorflow.org/js), para ejecutar su modelo en un navegador web o aplicación Node.JS.

## Ejemplos más avanzados

El ejemplo presentado anteriormente en realidad solo pretende ayudarle a empezar. A continuación, se muestran algunos ejemplos de integración con otros servicios de Cloud.

### Consideraciones sobre los recursos de Kubeflow Pipelines

En función de los requisitos de su carga de trabajo, la configuración predeterminada para su implementación de Kubeflow Pipelines puede satisfacer o no sus necesidades. Puede personalizar sus configuraciones de recursos mediante la aplicación de `pipeline_operator_funcs` en su llamada a `KubeflowDagRunnerConfig`.

`pipeline_operator_funcs` es una lista de elementos `OpFunc`, que transforma todas las instancias `ContainerOp` generadas en la especificación de canalización de KFP que se compila a partir de `KubeflowDagRunner`.

Por ejemplo, para configurar la memoria podemos usar [`set_memory_request`](https://github.com/kubeflow/pipelines/blob/646f2fa18f857d782117a078d626006ca7bde06d/sdk/python/kfp/dsl/_container_op.py#L249) para declarar la cantidad de memoria necesaria. Una forma típica de hacerlo es crear un envoltorio para `set_memory_request` y usarlo para agregarlo a la lista de `OpFunc` de la canalización:

```python
def request_more_memory():
  def _set_memory_spec(container_op):
    container_op.set_memory_request('32G')
  return _set_memory_spec

# Then use this opfunc in KubeflowDagRunner
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
pipeline_op_funcs.append(request_more_memory())
config = KubeflowDagRunnerConfig(
    pipeline_operator_funcs=pipeline_op_funcs,
    ...
)
kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)
```

Funciones de configuración de recursos similares incluyen:

- `set_memory_limit`
- `set_cpu_request`
- `set_cpu_limit`
- `set_gpu_limit`

### Pruebe `BigQueryExampleGen`

[BigQuery](https://cloud.google.com/bigquery) es un almacén de datos en la nube sin servidor, altamente escalable y rentable. BigQuery sirve como fuente de ejemplos de entrenamiento en TFX. En este paso, agregaremos `BigQueryExampleGen` a la canalización.

#### En el editor de archivos de Jupyter lab:

**Haga doble clic para abrir `pipeline.py`**. Comente `CsvExampleGen` y descomente la línea que crea una instancia de `BigQueryExampleGen`. También se debe descomentar el argumento `query` de la función `create_pipeline`.

Tenemos que especificar qué proyecto de GCP se usará para BigQuery, y para hacer esto hay que configurar `--project` en `beam_pipeline_args` cuando se crea una canalización.

**Haga doble clic para abrir `configs.py`**. Descomente la definición de `BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS` y `BIG_QUERY_QUERY`. Debe reemplazar la identificación del proyecto y el valor de la región en este archivo con los valores correctos para su proyecto de GCP.

> **Nota: DEBE configurar el ID y la región de su proyecto de GCP en el archivo `configs.py` antes de continuar.**

**Cambie el directorio un nivel hacia arriba.** Haga clic en el nombre del directorio encima de la lista de archivos. El nombre del directorio es el nombre de la canalización, que es `my_pipeline` si no cambió el nombre de la canalización.

**Haga doble clic para abrir `kubeflow_runner.py`**. Descomente dos argumentos, `query` y `beam_pipeline_args`, para la función `create_pipeline`.

Ahora la canalización está lista para usar BigQuery como fuente de ejemplo. Actualice la canalización como antes y cree una nueva ejecución como lo hicimos en los pasos 5 y 6.

#### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

### Pruebe Dataflow

Varios [componentes de TFX usan Apache Beam](https://www.tensorflow.org/tfx/guide/beam) para implementar canalizaciones de datos paralelas, lo que significa que puede distribuir cargas de trabajo de procesamiento de datos mediante [Google Cloud Dataflow](https://cloud.google.com/dataflow/). En este paso, configuraremos el orquestador de Kubeflow para que use Dataflow como backend de procesamiento de datos para Apache Beam.

> **Nota:** Si la API de Dataflow aún no está habilitada, puede habilitarla a través de la consola o desde la CLI con ayuda de este comando (por ejemplo, en Cloud Shell):

```bash
# Select your project:
gcloud config set project YOUR_PROJECT_ID

# Get a list of services that you can enable in your project:
gcloud services list --available | grep Dataflow

# If you don't see dataflow.googleapis.com listed, that means you haven't been
# granted access to enable the Dataflow API.  See your account adminstrator.

# Enable the Dataflow service:

gcloud services enable dataflow.googleapis.com
```

> **Nota:** La velocidad de ejecución puede estar limitada por la cuota predeterminada de [Google Compute Engine (GCE)](https://cloud.google.com/compute). Recomendamos establecer una cuota suficiente para aproximadamente 250 máquinas virtuales de Dataflow: **250 CPU, 250 direcciones IP y 62500 GB de disco persistente**. Para obtener más detalles, consulte la documentación sobre [Cuotas de GCE](https://cloud.google.com/compute/quotas) y [Cuotas de Dataflow](https://cloud.google.com/dataflow/quotas). Si está bloqueado por la cuota de direcciones IP, el uso de un [`worker_type`](https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options) más grande reducirá la cantidad de IP necesarias.

**Haga doble clic en `pipeline` para cambiar de directorio y haga doble clic para abrir `configs.py`**. Descomente la definición de `GOOGLE_CLOUD_REGION` y `DATAFLOW_BEAM_PIPELINE_ARGS`.

**Cambie el directorio un nivel hacia arriba.** Haga clic en el nombre del directorio encima de la lista de archivos. El nombre del directorio es el nombre de la canalización, que es `my_pipeline` si no lo cambió.

**Haga doble clic para abrir `kubeflow_runner.py`**. Descomente `beam_pipeline_args`. (También asegúrese de comentar `beam_pipeline_args` actuales que agregó en el Paso 7).

#### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

Puedes encontrar sus trabajos de Dataflow en [Dataflow en la consola de Cloud](http://console.cloud.google.com/dataflow).

### Pruebe entrenamiento y predicción de Cloud AI Platform con KFP

TFX interopera con varios servicios administrados de GCP, como [Cloud AI Platform for Training y Prediction](https://cloud.google.com/ai-platform/). Puede configurar su componente `Trainer` para que use Cloud AI Platform Training, un servicio administrado para entrenar modelos de ML. Además, cuando su modelo esté construido y listo para ser servido, puede *insertarlo* en Cloud AI Platform Prediction para su servicio. En este paso, configuraremos nuestro componente `Trainer` y `Pusher` para utilizar los servicios de Cloud AI Platform.

Antes de editar archivos, es posible que primero tenga que habilitar la *API de Training y Prediction de AI Platform*.

**Haga doble clic en `pipeline` para cambiar de directorio y haga doble clic para abrir `configs.py`**. Descomente la definición de `GOOGLE_CLOUD_REGION`, `GCP_AI_PLATFORM_TRAINING_ARGS` y `GCP_AI_PLATFORM_SERVING_ARGS`. Usaremos nuestra imagen de contenedor personalizada para entrenar un modelo en Cloud AI Platform Training, por lo que debemos configurar `masterConfig.imageUri` en `GCP_AI_PLATFORM_TRAINING_ARGS` con el mismo valor que `CUSTOM_TFX_IMAGE` arriba.

**Cambie el directorio un nivel hacia arriba y haga doble clic para abrir `kubeflow_runner.py`**. Descomente `ai_platform_training_args` y `ai_platform_serving_args`.

> Nota: Si recibe un error de permisos en el paso de Entrenamiento, es posible que deba proporcionar permisos de Visor de objetos de almacenamiento a la cuenta de servicio Cloud Machine Learning Engine (AI Platform Prediction y Training). Hay más información disponible en la [documentación del Registro de contenedores](https://cloud.google.com/container-registry/docs/access-control#grant).

#### Actualice la canalización y vuelva a ejecutarla

```python
# Update the pipeline
!tfx pipeline update \
  --pipeline-path=kubeflow_runner.py \
  --endpoint={ENDPOINT}

!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

Puede encontrar sus trabajos de entrenamiento en [Cloud AI Platform Jobs](https://console.cloud.google.com/ai-platform/jobs). Si su canalización se completó correctamente, puede encontrar su modelo en [Cloud AI Platform Models](https://console.cloud.google.com/ai-platform/models).

## 14. Use sus propios datos

En este tutorial, se creó una canalización para un modelo con ayuda del conjunto de datos Chicago Taxi. Ahora intente poner sus propios datos en la canalización. Sus datos se pueden almacenar en cualquier lugar al que la canalización pueda acceder, incluidos Google Cloud Storage, BigQuery o archivos CSV.

Debe modificar la definición de la canalización para adaptarla a sus datos.

### Si sus datos se almacenan en archivos

1. Modifique `DATA_PATH` en `kubeflow_runner.py`, indicando la ubicación.

### Si tus datos se almacenan en BigQuery

1. Modifique `BIG_QUERY_QUERY` en configs.py a su declaración de consulta.
2. Agregue características en `models` / `features.py`.
3. Modifique `models` / `preprocessing.py` para [transformar los datos de entrada para el entrenamiento](https://www.tensorflow.org/tfx/guide/transform).
4. Modifique `models` / `keras` / `model.py` y `models` / `keras` / `constants.py` para [describir su modelo de ML](https://www.tensorflow.org/tfx/guide/trainer).

### Obtenga más información sobre Trainer

Consulte [la guía de componentes de Trainer](https://www.tensorflow.org/tfx/guide/trainer) para obtener más detalles sobre las canalizaciones de entrenamiento.

## Limpieza

Para limpiar todos los recursos de Google Cloud utilizados en este proyecto, puede [eliminar el proyecto de Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) que se usó para el tutorial.

Alternativamente, puede eliminar recursos individuales visitando cada consola: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)

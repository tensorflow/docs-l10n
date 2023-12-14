## Cómo probar la canalización mediante el uso de ejecutores de códigos auxiliares

### Introducción

**Debe completar el tutorial [template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) hasta el *Paso 6* para poder continuar con este tutorial.**

En este documento se proporcionan instrucciones para probar una canalización de TensorFlow Extended (TFX) utilizando `BaseStubExecuctor`, que genera artefactos falsos a partir de los datos de la prueba dorada. Esto está destinado a que los usuarios reemplacen los ejecutores que no quieran probar para que ahorren tiempo al ejecutar los ejecutores reales. El ejecutor de código auxiliar se proporciona con el paquete de Python para TFX en `tfx.experimental.pipeline_testing.base_stub_executor`.

Este tutorial sirve como una extensión del tutorial `template.ipynb`, por lo que también usará el [conjunto de datos Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) publicado por la ciudad de Chicago. Le recomendamos encarecidamente que intente modificar los componentes antes de utilizar ejecutores de código auxiliar.

### 1. Registre las salidas de la canalización en Google Cloud Storage

Primero debemos registrar las salidas de la canalización para que los ejecutores del código auxiliar puedan copiar los artefactos de las salidas registradas.

Dado que este tutorial supone que ha completado `template.ipynb` hasta el paso 6, se debe haber guardado una ejecución exitosa de la canalización en [MLMD](https://www.tensorflow.org/tfx/guide/mlmd). Se puede acceder a la información de ejecución en MLMD mediante el servidor gRPC.

Abra una Terminal y ejecute los siguientes comandos:

1. Genere un archivo kubeconfig con las credenciales apropiadas: `bash gcloud container clusters get-credentials $cluster_name --zone $compute_zone --project $gcp_project_id{/code0} {code1}$compute_zone` es la región para gcp engine y `$gcp_project_id` es la identificación del proyecto de GCP.

2. Configure el reenvío de puertos para conectarse a MLMD: `bash nohup kubectl port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &` `$namespace{/code1} es el espacio de nombres del clúster y {code2}$port` es cualquier puerto sin usar que se usará para el reenvío de puertos.

3. Clone el repositorio tfx de GitHub. Dentro del directorio tfx, ejecute el siguiente comando:

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

`$output_dir` debe establecerse en una ruta en Google Cloud Storage donde se registrarán las salidas de la canalización, así que asegúrese de reemplazar `<gcp_project_id>` con la identificación del proyecto de GCP.

`$host{/code0} y {code1}$port` son el nombre de host y el puerto del servidor grpc de metadatos para conectarse a MLMD. `$port` debe configurarse en el número de puerto que se usó para el reenvío de puertos y puede configurar "localhost" como nombre de host.

En el tutorial `template.ipynb`, el nombre de la canalización se establece como "my_pipeline" de forma predeterminada, por lo tanto, configure `pipeline_name="my_pipeline"`. Si modificó el nombre de la canalización al ejecutar el tutorial de la plantilla, debe modificar `--pipeline_name` en consecuencia.

### 2. Habilite los ejecutores de código auxiliar en ejecutores de DAG de Kubeflow

En primer lugar, asegúrese de que la plantilla predefinida se haya copiado en el directorio del proyecto con el comando de CLI `tfx template copy`. Se deben editar los dos archivos siguientes en los archivos fuente copiados.

1. Cree un archivo llamado `stub_component_launcher.py` en el directorio donde se encuentra kubeflow_dag_runner.py y agregue el siguiente contenido.

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    NOTA: Este iniciador de componentes auxiliares no se puede definir en `kubeflow_dag_runner.py` porque la clase de iniciador se importa mediante la ruta del módulo.

2. Configure los identificadores de componentes para que formen una lista con los identificadores de componentes que se van a probar (en otras palabras, los ejecutores de otros componentes se reemplazan por BaseStubExecutor).

3. Abra `kubeflow_dag_runner.py`. Agregue la siguiente declaración de importación en la parte superior para usar la clase `StubComponentLauncher` que acabamos de agregar.

    ```python
    import stub_component_launcher
    ```

4. En `kubeflow_dag_runner.py`, agregue la clase `StubComponentLauncher` a la `supported_launcher_class` de `KubeflowDagRunnerConfig` para habilitar el inicio de ejecutores de código auxiliar:

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. Actualice y ejecute la canalización con ejecutores de código auxiliar

Actualice la canalización existente con la definición de canalización modificada que contiene ejecutores de código auxiliar.

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

`$endpoint` debe configurarse en el punto conexión de su clúster de KFP.

Ejecute el siguiente comando para crear una nueva ejecución de su canalización actualizada.

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## Limpieza

Use el comando `fg` para acceder al reenvío de puertos en segundo plano y luego presione Ctrl-C para finalizar. Puede eliminar el directorio con salidas de canalización registradas con ayuda de `gsutil -m rm -R $output_dir`.

Para limpiar todos los recursos de Google Cloud utilizados en este proyecto, puede [eliminar el proyecto de Google Cloud](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) que se usó para el tutorial.

Alternativamente, puede limpiar recursos individuales visitando cada consola: - [Google Cloud Storage](https://console.cloud.google.com/storage) - [Google Container Registry](https://console.cloud.google.com/gcr) - [Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)

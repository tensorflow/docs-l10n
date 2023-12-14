# **Tutorial de TFX Airflow**

## Descripción general

## Descripción general

Este tutorial está diseñado para enseñarle a crear sus propias canalizaciones de aprendizaje automático utilizando TensorFlow Extended (TFX) y Apache Airflow como orquestador. Se ejecuta en Vertex AI Workbench y muestra la integración con TFX y TensorBoard, así como la interacción con TFX en un entorno de Jupyter Lab.

### ¿Qué hará?

Aprenderá cómo crear una canalización de ML con TFX.

- Una canalización de TFX es un grafo acíclico dirigido o "DAG". A menudo nos referiremos a las canalizaciones como DAG.
- Las canalizaciones de TFX son apropiadas cuando se desea implementar una aplicación de aprendizaje automático en producción.
- Las canalizaciones de TFX son apropiadas cuando los conjuntos de datos son grandes o pueden llegar a ser grandes.
- Las canalizaciones de TFX son apropiadas cuando la coherencia entre el entrenamiento y el servicio es importante.
- Las canalizaciones de TFX son apropiadas cuando la gestión de versiones para la inferencia es importante
- Google usa canalizaciones de TFX para ML de producción

Consulte la [Guía del usuario de TFX](https://www.tensorflow.org/tfx/guide) para obtener más información.

Seguirá un proceso típico de desarrollo de ML:

- Ingerir, comprender y limpiar nuestros datos
- Ingeniería de características
- Entrenamiento
- Análisis del rendimiento del modelo
- Enjabonar, enjuagar, repetir
- Listo para la producción

## **Apache Airflow para la orquestación de canalizaciones**

Los orquestadores de TFX son responsables de programar los componentes de la canalización de TFX en función de las dependencias definidas por la canalización. TFX fue diseñado para que pueda adaptarse a múltiples entornos y marcos de orquestación. Uno de los orquestadores predeterminados compatibles con TFX es [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow). Esta práctica de laboratorio ilustra el uso de Apache Airflow para la orquestación de canalizaciones de TFX. Apache Airflow es una plataforma útil para crear, programar y monitorear flujos de trabajo mediante programación. TFX usa Airflow para crear flujos de trabajo como grafos acíclicos dirigidos (DAG) de tareas. La completa interfaz de usuario facilita la visualización de canalizaciones que se ejecutan en producción, monitorea el progreso y soluciona problemas cuando es necesario. Los flujos de trabajo de Apache Airflow se definen como código. Esto los hace más mantenibles, versionables, comprobables y colaborativos. Apache Airflow es adecuado para canalizaciones de procesamiento por lotes. Es ligero y fácil de aprender.

En este ejemplo, ejecutaremos una canalización de TFX en una instancia mediante la configuración manual de Airflow.

Los otros orquestadores predeterminados compatibles con TFX son Apache Beam y Kubeflow. [Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) se puede ejecutar en múltiples servidores de procesamiento de datos (Beam Runners). Cloud Dataflow es uno de esos ejecutores de Beam que se puede utilizar para ejecutar canalizaciones de TFX. Apache Beam se puede utilizar para canalizaciones de procesamiento por secuencias y por lotes.<br> [Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow) es una plataforma de aprendizaje automático de código abierto dedicada a hacer que las implementaciones de flujos de trabajo de aprendizaje automático (ML) en Kubernetes sean simples, portátiles y escalables. Kubeflow se puede usar como orquestador para canalizaciones de TFX cuando es necesario implementarlas en clústeres de Kubernetes. Además, también puede usar su propio [orquestador personalizado](https://www.tensorflow.org/tfx/guide/custom_orchestrator) para ejecutar una canalización de TFX.

Hay más información sobre Airflow disponible [aquí](https://airflow.apache.org/).

## **Conjunto de datos de taxis de Chicago**

![taxi.jpg](images/airflow_workshop/taxi.jpg)

![chicago.png](images/airflow_workshop/chicago.png)

Usaremos el [conjunto de datos Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) publicado por la ciudad de Chicago.

Nota: En este tutorial se ofrece una aplicación que usa datos que fueron modificados para su uso desde su fuente original, www.cityofchicago.org, el sitio web oficial de la ciudad de Chicago. La ciudad de Chicago no garantiza el contenido, la exactitud, la puntualidad o la integridad de ninguno de los datos que se proporcionan en este tutorial. Los datos proporcionados en este sitio están sujetos a cambios en cualquier momento. Se entiende que los datos proporcionados en este tutorial se usan bajo su propia responsabilidad.

### Objetivo del modelo: clasificación binaria

¿El cliente dará una propina mayor o menor al 20 %?

## Cómo configurar el proyecto de Google Cloud

**Antes de hacer clic en el botón Iniciar laboratorio,** lea estas instrucciones. Los laboratorios están cronometrados y no se pueden poner en pausa. El cronómetro, que comienza cuando se hace clic en **Iniciar laboratorio**, muestra durante cuánto tiempo podrá acceder a los recursos de Google Cloud.

Esta práctica de laboratorio le permite llevar a cabo las actividades de laboratorio usted mismo en un entorno de nube real, no en un entorno de simulación o demostración. Lo hace al proporcionarle credenciales nuevas y temporales que usa para iniciar sesión y acceder a Google Cloud durante la duración de la práctica de laboratorio.

**Qué necesita** Para completar esta práctica de laboratorio, necesita lo siguiente:

- Acceso a un navegador de Internet estándar (se recomienda el navegador Chrome).
- Tiempo para completar el laboratorio.

**Nota:** Si ya tiene su propia cuenta o proyecto personal de Google Cloud, no lo use para esta práctica de laboratorio.

**Nota:** Si está usando un dispositivo con sistema operativo Chrome, abra una ventana de incógnito para ejecutar esta práctica de laboratorio.

**Cómo iniciar su laboratorio e iniciar sesión en Google Cloud Console** 1. Haga clic en el botón **Iniciar laboratorio**. Si necesita pagar por el laboratorio, se abre una ventana emergente para que seleccione su método de pago. A la izquierda hay un panel con las credenciales temporales que debe usar para esta práctica de laboratorio.

![qwiksetup1.png](images/airflow_workshop/qwiksetup1.png)

1. Copie el nombre de usuario y luego haga clic en **Abrir Google Console**. El laboratorio activa los recursos y luego abre otra pestaña que muestra la página **Inicio de sesión**.

![qwiksetup2.png](images/airflow_workshop/qwiksetup2.png)

***Consejo:*** Abra las pestañas en ventanas separadas, una al lado de la otra.

![qwiksetup3.png](images/airflow_workshop/qwiksetup3.png)

1. En la página **Inicio de sesión**, pegue el nombre de usuario que copió del panel izquierdo. Luego copie y pegue la contraseña.

***Importante:*** Debe usar las credenciales del panel izquierdo. No utilice sus credenciales de Google Cloud Training. Si tiene su propia cuenta de Google Cloud, no la use para esta práctica de laboratorio (para no incurrir en cargos).

1. Haga clic en las siguientes páginas:
2. Acepte los términos y condiciones.

- No agregue opciones de recuperación ni de autenticación de dos factores (porque es una cuenta temporal).

- No se registre en pruebas gratuitas.

Después de unos segundos, Cloud Console se abre en esta pestaña.

**Nota:** Puede ver el menú con una lista de productos y servicios de Google Cloud al hacer clic en el **menú de navegación** en la parte superior izquierda.

![qwiksetup4.png](images/airflow_workshop/qwiksetup4.png)

### Cómo activar Cloud Shell

Cloud Shell es una máquina virtual cargada de herramientas de desarrollo. Ofrece un directorio de inicio persistente de 5 GB y se ejecuta en Google Cloud. Cloud Shell ofrece acceso a la línea de comandos a los recursos de Google Cloud.

En Cloud Console, en la barra de herramientas superior derecha, haga clic en el botón **Activar Cloud Shell**.

![qwiksetup5.png](images/airflow_workshop/qwiksetup5.png)

Haga clic en **Continuar**.

![qwiksetup6.png](images/airflow_workshop/qwiksetup6.png)

El aprovisionamiento y la conexión al entorno demoran unos instantes. Cuando se conecte, ya estará autenticado y el proyecto estará configurado con su *PROJECT_ID*. Por ejemplo:

![qwiksetup7.png](images/airflow_workshop/qwiksetup7.png)

`gcloud` es la herramienta de línea de comandos para Google Cloud. Viene previamente instalada en Cloud Shell y admite la función de completado por tabulación.

Puede enumerar el nombre de la cuenta activa con este comando:

```
gcloud auth list
```

(Salida)

> ACTIVE: * ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net To set the active account, run: $ gcloud config set account `ACCOUNT`

Puede enumerar el ID del proyecto con este comando: `gcloud config list project` (Salida)

> [core] project = &lt;project_ID&gt;

(Salida de ejemplo)

> [core] project = qwiklabs-gcp-44776a13dea667a6

Para obtener la documentación completa de gcloud, consulte la [descripción general de la herramienta de línea de comandos de gcloud](https://cloud.google.com/sdk/gcloud).

## Cómo habilitar los servicios de Google Cloud

1. En Cloud Shell, puede usar gcloud para habilitar los servicios usados ​​en el laboratorio. `gcloud services enable notebooks.googleapis.com`

## Cómo implementar una instancia de bloc de notas de Vertex

1. Haga clic en **Menú de navegación** y navegue hasta **Vertex AI**, luego hasta **Workbench**.

![vertex-ai-workbench.png](images/airflow_workshop/vertex-ai-workbench.png)

1. En la página Instancias de bloc de notas, haga clic en **Nuevo bloc de notas**.

2. En el menú Personalizar instancia, seleccione **TensorFlow Enterprise** y elija la versión de **TensorFlow Enterprise 2.x (con LTS)** &gt; **Sin GPU**.

![vertex-notebook-create-2.png](images/airflow_workshop/vertex-notebook-create-2.png)

1. En el cuadro de diálogo **Nueva instancia del bloc de notas**, haga clic en el ícono de lápiz para **Editar** propiedades de la instancia.

2. En **Nombre de instancia**, ingrese un nombre para su instancia.

3. Para **Región**, seleccione `us-east1` y para **Zona**, seleccione una zona dentro de la región seleccionada.

4. Desplácese hacia abajo hasta Configuración de la máquina y seleccione **e2-standard-2** para Tipo de máquina.

5. Deje los campos restantes con sus valores predeterminados y haga clic en **Crear**.

Después de unos minutos, la consola de Vertex AI mostrará el nombre de su instancia, seguido de **Abrir Jupyterlab**.

1. Haga clic en **Abrir JupyterLab**. Se abrirá una ventana de JupyterLab en una nueva pestaña.

## Cómo configurar el entorno

### Clonar el repositorio lab

A continuación, debe clonar el repositorio `tfx` en su instancia de JupyterLab. 1. En JupyterLab, haga clic en el icono **Terminal** para abrir una nueva terminal.

{ql-infobox0}<strong>Nota:</strong> Si se le solicita, haga clic en <code>Cancel</code> para la compilación recomendada.{/ql-infobox0}

1. Para clonar el repositorio `tfx` de Github, escriba el siguiente comando y presione **Intro**.

```
git clone https://github.com/tensorflow/tfx.git
```

1. Para confirmar que ha clonado el repositorio, haga doble clic en el directorio `tfx` y confirme que puede ver su contenido.

![repo-directory.png](images/airflow_workshop/repo-directory.png)

### Instalar dependencias de lab

1. Ejecute el siguiente código para ir a la carpeta `tfx/tfx/examples/airflow_workshop/taxi/setup/`, luego ejecute `./setup_demo.sh` para instalar las dependencias del laboratorio:

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

El código anterior, cumple las siguientes funciones:

- Instalar los paquetes necesarios.
- Crear una carpeta de `airflow` en la carpeta de inicio.
- Copiar la carpeta `dags` de `tfx/tfx/examples/airflow_workshop/taxi/setup/` a la `~/airflow/`.
- Copiar el archivo csv de `tfx/tfx/examples/airflow_workshop/taxi/setup/data` a `~/airflow/data`.

![airflow-home.png](images/airflow_workshop/airflow-home.png)

## Cómo configurar el servidor de Airflow

### Cree una regla de firewall para acceder al servidor de Airflow en el navegador

1. Vaya a `https://console.cloud.google.com/networking/firewalls/list` y asegúrese de que el nombre del proyecto esté seleccionado correctamente
2. Haga clic en la opción `CREATE FIREWALL RULE` en la parte superior

![firewall-rule.png](images/airflow_workshop/firewall-rule.png)

En el **cuadro de diálogo Crear un firewall**, siga los pasos que se enumeran a continuación.

1. En **Nombre**, escriba `airflow-tfx`.
2. En **Prioridad**, seleccione `1`.
3. En **Destinos**, seleccione `All instances in the network`.
4. En **Rangos de IPv4 de origen**, seleccione `0.0.0.0/0`
5. En **Protocolos y puertos**, haga clic en `tcp` e ingrese `7000` en el cuadro al lado de `tcp`
6. Haga clic en `Create`.

![create-firewall-dialog.png](images/airflow_workshop/create-firewall-dialog.png)

### Ejecute el servidor de Airflow desde su shell

En la ventana Terminal de Jupyter Lab, cambie al directorio de inicio, ejecute el comando `airflow users create` para crear un usuario administrador para Airflow:

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

Luego, ejecute el `airflow webserver` y el comando `airflow scheduler` para ejecutar el servidor. Elija el puerto `7000` ya que está permitido a través del firewall.

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### Consiga su ip externa

1. En Cloud Shell, use `gcloud` para obtener la IP externa.

```
gcloud compute instances list
```

![gcloud-instance-ip.png](images/airflow_workshop/gcloud-instance-ip.png)

## Cómo ejecutar un DAG/canalización

### En un navegador

Abra un navegador y vaya a http://&lt;external_ip&gt;:7000

- En la página de inicio de sesión, ingrese el nombre de usuario (`admin`) y la contraseña (`admin`) que eligió al ejecutar el comando `airflow users create`.

![airflow-login.png](images/airflow_workshop/airflow-login.png)

Airflow carga los DAG desde archivos fuente de Python. Toma cada archivo y lo ejecuta. Luego carga cualquier objeto DAG de ese archivo. Todos los archivos `.py` que definen objetos DAG aparecerán como canalizaciones en la página de inicio de Airflow.

En este tutorial, Airflow escanea la carpeta `~/airflow/dags/` en busca de objetos DAG.

Si abre `~/airflow/dags/taxi_pipeline.py` y se desplaza hasta el final, podrá ver que se crea y almacena un objeto DAG en una variable llamada `DAG`. Por lo tanto, aparecerá como canalización en la página de inicio de Airflow como se muestra a continuación:

![dag-home-full.png](images/airflow_workshop/dag-home-full.png)

Si hace clic en taxi, será redirigido a la vista de cuadrícula del DAG. Puede hacer clic en la opción `Graph` en la parte superior para obtener la vista de grafo del DAG.

![airflow-dag-graph.png](images/airflow_workshop/airflow-dag-graph.png)

### Cómo activar la canalización de taxi

En la página de inicio puede ver los botones que se pueden usar para interactuar con el DAG.

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)

Debajo del encabezado de **acciones**, haga clic en el botón de **activación** para activar la canalización.

En la página del **DAG** de taxi, use el botón de la derecha para actualizar el estado de grafo del DAG a medida que se ejecuta la canalización. Además, puede habilitar **la actualización automática** para indicarle a Airflow que actualice automáticamente la vista de grafo a medida que cambia el estado.

![dag-button-refresh.png](images/airflow_workshop/dag-button-refresh.png)

También puede usar la [CLI de Airflow](https://airflow.apache.org/cli.html) en la terminal para habilitar y activar sus DAG:

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### Tiempo de espera para que se complete la canalización

Una vez que haya activado su canalización, en la vista del DAG, puede observar el progreso de su canalización mientras se ejecuta. A medida que se ejecuta cada componente, el color del contorno del componente en el grafo DAG cambiará para mostrar su estado. Cuando un componente haya terminado de procesarse, el contorno se mostrará en verde oscuro para indicar que ha finalizado.

![dag-step7.png](images/airflow_workshop/dag-step7.png)

## Explicación de los componentes

Ahora veremos los componentes de este proceso en detalle y analizaremos individualmente los resultados producidos por cada paso del proceso.

1. En JupyterLab, vaya a `~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/`

2. Abra **notebook.ipynb.** ![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)

3. Continúe con la práctica de laboratorio en el bloc de notas y ejecute cada celda haciendo clic en el icono **Ejecutar** (<img src="images/airflow_workshop/f1abc657d9d2845c.png" width="28.00" alt="botón-ejecutar.png">) ubicado en la parte superior de la pantalla. Alternativamente, puede ejecutar el código en una celda con **SHIFT + INTRO**.

Lea la narrativa y asegúrese de comprender lo que sucede en cada celda.

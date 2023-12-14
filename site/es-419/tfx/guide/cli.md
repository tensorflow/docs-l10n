# Cómo usar la interfaz de línea de comandos de TFX

La interfaz de línea de comandos (CLI) de TFX ejecuta una gama completa de acciones de canalización con orquestadores de canalizaciones, como Kubeflow Pipelines y Vertex Pipelines. El orquestador local también se puede usar para desarrollar o depurar más rápido. Apache Beam y Apache Airflow son compatibles como características experimentales. Por ejemplo, puede utilizar la CLI para las siguientes tareas:

- Crear, actualizar y eliminar canalizaciones.
- Ejecutar una canalización y supervisar la ejecución en varios orquestadores.
- Enumerar canalizaciones y ejecuciones de canalizaciones.

Nota: La CLI de TFX actualmente no ofrece garantías de compatibilidad. La interfaz CLI podría cambiar a medida que se publiquen nuevas versiones.

## Acerca de la CLI de TFX

La CLI de TFX se instala como parte del paquete de TFX. Todos los comandos de la CLI siguen la siguiente estructura:

<pre class="devsite-terminal">
tfx &lt;var&gt;command-group&lt;/var&gt; &lt;var&gt;command&lt;/var&gt; &lt;var&gt;flags&lt;/var&gt;
</pre>

Actualmente se admiten las siguientes opciones de <var>grupo de comandos</var>:

- [tfx pipeline](#tfx-pipeline): cree y administre canalizaciones de TFX.
- [tfx run](#tfx-run): cree y administre ejecuciones de canalizaciones de TFX en distintas plataformas de orquestación.
- [tfx template](#tfx-template-experimental): comandos experimentales para enumerar y copiar plantillas de canalización de TFX.

Cada grupo de comandos proporciona un conjunto de <var>comandos</var>. Siga las instrucciones de las secciones [comandos de canalización](#tfx-pipeline), [comandos de ejecución](#tfx-run) y [comandos de plantilla](#tfx-template-experimental) para obtener más información sobre el uso de estos comandos.

Aviso: Actualmente no todos los comandos son compatibles con todos los orquestadores. Dichos comandos mencionan explícitamente los motores compatibles.

Las marcas le permiten pasar argumentos a comandos de CLI. Las palabras de las marcas se separan por un guion (`-`) o un guion bajo (`_`). Por ejemplo, el indicador del nombre de la canalización se puede especificar como `--pipeline-name` o `--pipeline_name`. Este documento especifica marcas con guiones bajos para mayor brevedad. Obtenga más información sobre las [<var>marcas</var> que se usan en la CLI de TFX](#understanding-tfx-cli-flags).

## tfx pipeline

La estructura de los comandos en el grupo de comandos `tfx pipeline` es la siguiente:

<pre class="devsite-terminal">
tfx pipeline &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Use las siguientes secciones para obtener más información sobre los comandos en el grupo de comandos `tfx pipeline`.

### create

Crea una nueva canalización en el orquestador dado.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; \
--build_image --build_base_image=&lt;var&gt;build-base-image&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>La ruta al archivo de configuración de la canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP cuando se utiliza Kubeflow Pipelines.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>


  <dt>--build_image</dt>
  <dd>
    <p>       (Opcional). Cuando el <var>motor</var> es <strong>kubeflow</strong> o <strong>vertex</strong>, TFX crea una imagen de contenedor para su canalización, si se especifica. Se usa `Dockerfile` en el directorio actual y, si no existe, TFX generará una automáticamente.</p>
    <p>       La imagen creada se enviará al registro remoto que se especifica en `KubeflowDagRunnerConfig` o `KubeflowV2DagRunnerConfig`.</p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var>
</dt>
  <dd>
    <p>       (Opcional). Cuando el <var>motor</var> es <strong>kubeflow</strong>, TFX crea una imagen de contenedor para su canalización. La imagen base de compilación especifica la imagen base del contenedor que se usará cuando se compile la imagen del contenedor de la canalización.</p>
  </dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline create --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline create --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline create --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

Para detectar automáticamente el motor desde el entorno del usuario, simplemente evite usar la marca del motor como en el siguiente ejemplo. Si desea obtener más detalles, consulte la sección sobre marcas.

<pre class="devsite-terminal">
tfx pipeline create --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### update

Actualiza una canalización existente en el orquestador determinado.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline update --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --build_image]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>La ruta al archivo de configuración de la canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>
  <dt>--build_image</dt>
  <dd>
    <p>       (Opcional). Cuando el <var>motor</var> es <strong>kubeflow</strong> o <strong>vertex</strong>, TFX crea una imagen de contenedor para su canalización, si se especifica. Se usa `Dockerfile` en el directorio actual.</p>
    <p>       La imagen creada se enviará al registro remoto que se especifica en `KubeflowDagRunnerConfig` o `KubeflowV2DagRunnerConfig`.</p>
  </dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline update --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--build_image
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline update --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline update --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; \
--build_image
</pre>

### compile

Compila el archivo de configuración de la canalización para crear un archivo de flujo de trabajo en Kubeflow y ejecuta las siguientes comprobaciones durante la compilación:

1. Comprueba si la ruta de la canalización es válida.
2. Comprueba si los detalles de la canalización se extraen correctamente del archivo de configuración de la canalización.
3. Comprueba si el DagRunner en la configuración de la canalización coincide con el motor.
4. Comprueba si el archivo de flujo de trabajo se creó correctamente en la ruta del paquete proporcionada (solo para Kubeflow).

Se recomienda usarlo antes de crear o actualizar una canalización.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline compile --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>La ruta al archivo de configuración de la canalización.</dd>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
</dl>

#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline compile --engine=kubeflow --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline compile --engine=local --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline compile --engine=vertex --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt;
</pre>

### delete

Elimina una canalización del orquestador determinado.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline delete --pipeline_path=&lt;var&gt;pipeline-path&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>La ruta al archivo de configuración de la canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline delete --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline delete --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline delete --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

### list

Enumera todas las canalizaciones en el orquestador determinado.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline list [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline list --engine=kubeflow --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx pipeline list --engine=local
</pre>

Vertex:

<pre class="devsite-terminal">
tfx pipeline list --engine=vertex
</pre>

## tfx run

La estructura de los comandos en el grupo de comandos `tfx run` es la siguiente:

<pre class="devsite-terminal">
tfx run &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Utilice las siguientes secciones para obtener más información sobre los comandos en el grupo de comandos `tfx run`.

### create

Crea una nueva instancia de ejecución para una canalización en el orquestador. Para Kubeflow, se utiliza la versión más reciente de la canalización en el clúster.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run create --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>El nombre de la canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>


  <dt>--runtime_parameter=<var>parameter-name</var>=<var>parameter-value</var>
</dt>
  <dd>     (Opcional). Establece un valor de parámetro de tiempo de ejecución. Se puede configurar varias veces para establecer valores de múltiples variables. Solo se puede aplicar a los motores `airflow`, `kubeflow` y `vertex`.</dd>


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
  <dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>


  <dt>--project=<var>GCP-project-id</var>
</dt>
  <dd>     (Obligatorio para Vertex). El ID del proyecto de GCP para la canalización de Vertex.</dd>


  <dt>--region=<var>GCP-region</var>
</dt>
  <dd>     (Obligatorio para Vertex). Nombre de región de GCP como us-central1. Consulte la [documentación de Vertex](https://cloud.google.com/vertex-ai/docs/general/locations) para conocer las regiones disponibles.</dd>





#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run create --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

Local:

<pre class="devsite-terminal">
tfx run create --engine=local --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt;
</pre>

Vertex:

<pre class="devsite-terminal">
tfx run create --engine=vertex --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
  --runtime_parameter=&lt;var&gt;var_name&lt;/var&gt;=&lt;var&gt;var_value&lt;/var&gt; \
  --project=&lt;var&gt;gcp-project-id&lt;/var&gt; --region=&lt;var&gt;gcp-region&lt;/var&gt;
</pre>

### terminate

Detiene una ejecución de una canalización determinada.

** Nota importante: Actualmente solo se admite en Kubeflow.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run terminate --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; --engine=&lt;var&gt;engine&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador único para una ejecución de canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### list

Enumera todas las ejecuciones de una canalización.

** Nota importante: Actualmente no es compatible con Local y Apache Beam.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run list --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>El nombre de la canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run list --engine=kubeflow --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### status

Devuelve el estado actual de una ejecución.

** Nota importante: Actualmente no es compatible con Local y Apache Beam.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run status --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; --run_id=&lt;var&gt;run-id&lt;/var&gt; [--endpoint=&lt;var&gt;endpoint&lt;/var&gt; \
--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt;]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>El nombre de la canalización.</dd>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador único para una ejecución de canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run status --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; --namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

### delete

Elimina una ejecución de una canalización determinada.

** Nota importante: Actualmente solo se admite en Kubeflow

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run delete --run_id=&lt;var&gt;run-id&lt;/var&gt; [--engine=&lt;var&gt;engine&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador único para una ejecución de canalización.</dd>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       (Opcional). Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       (Opcional). El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     (Opcional). El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     (Opcional). Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



#### Ejemplos:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=&lt;var&gt;run-id&lt;/var&gt; --iap_client_id=&lt;var&gt;iap-client-id&lt;/var&gt; \
--namespace=&lt;var&gt;namespace&lt;/var&gt; --endpoint=&lt;var&gt;endpoint&lt;/var&gt;
</pre>

## tfx template [Experimental]

La estructura de los comandos en el grupo de comandos `tfx template` es la siguiente:

<pre class="devsite-terminal">
tfx template &lt;var&gt;command&lt;/var&gt; &lt;var&gt;required-flags&lt;/var&gt; [&lt;var&gt;optional-flags&lt;/var&gt;]
</pre>

Use las siguientes secciones para obtener más información sobre los comandos en el grupo de comandos `tfx template`. La plantilla es una característica experimental y podría sufrir cambios en cualquier momento.

### list

Enumera las plantillas de canalización de TFX disponibles.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template list
</pre>

### copy

Copia una plantilla en el directorio de destino.

Uso:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template copy --model=&lt;var&gt;model&lt;/var&gt; --pipeline_name=&lt;var&gt;pipeline-name&lt;/var&gt; \
--destination_path=&lt;var&gt;destination-path&lt;/var&gt;
</pre>

<dl>
  <dt>--model=<var>model</var>
</dt>
  <dd>El nombre del modelo creado por la plantilla de canalización.</dd>
  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>El nombre de la canalización.</dd>
  <dt>--destination_path=<var>destination-path</var>
</dt>
  <dd>La ruta en la que se copiará la plantilla.</dd>
</dl>

## Explicación de las marcas de la CLI de TFX

### Marcas comunes

<dl>
  <dt>--engine=<var>engine</var>
</dt>
  <dd>
    <p>       El orquestador que se usará para la canalización. El valor del motor debe coincidir con uno de los siguientes valores:</p>
    <ul>
      <li>
<strong>kubeflow</strong>: configura el motor en Kubeflow</li>
      <li>
<strong>local</strong>: establece el motor en el orquestador local</li>
      <li>
<strong>vertex</strong>: establece el motor en Vertex Pipelines</li>
      <li>
<strong>airflow</strong>: (experimental) configura el motor en Apache Airflow</li>
      <li>
<strong>beam</strong>: (experimental) configura el motor en Apache Beam</li>
    </ul>
    <p>       Si el motor no está configurado, se detecta automáticamente en función del entorno.</p>
    <p>       ** Nota importante: El orquestador requerido por DagRunner en el archivo de configuración de la canalización debe coincidir con el motor que se seleccione o se detecte automáticamente. La detección automática del motor se basa en el entorno del usuario. Si Apache Airflow y Kubeflow Pipelines no están instalados, se utiliza el orquestador local de forma predeterminada.</p>
  </dd>
</dl>

  <dt>--pipeline_name=<var>pipeline-name</var>
</dt>
  <dd>El nombre de la canalización.</dd>


  <dt>--pipeline_path=<var>pipeline-path</var>
</dt>
  <dd>La ruta al archivo de configuración de la canalización.</dd>


  <dt>--run_id=<var>run-id</var>
</dt>
  <dd>Identificador único para una ejecución de canalización.</dd>





### Marcas específicas de Kubeflow

<dl>
  <dt>--endpoint=<var>endpoint</var>
</dt>
  <dd>
    <p>       Punto de conexión del servicio de API de Kubeflow Pipelines. El punto de conexión de su servicio de API de Kubeflow Pipelines es el mismo que la URL del panel de Kubeflow Pipelines. El valor de su punto de conexión debería ser similar a lo que sigue:</p>
</dd>
</dl>

```
<pre>https://<var>host-name</var>/pipeline</pre>

<p>
  If you do not know the endpoint for your Kubeflow Pipelines cluster,
  contact you cluster administrator.
</p>

<p>
  If the <code>--endpoint</code> is not specified, the in-cluster service
  DNS name is used as the default value. This name works only if the
  CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
  <a href="https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/"
       class="external">Kubeflow Jupyter notebooks</a> instance.
</p>
```

  


  <dt>--iap_client_id=<var>iap-client-id</var>
</dt>
  <dd>     El ID de cliente para el punto de conexión protegido por IAP.</dd>


  <dt>--namespace=<var>namespace</var>
</dt>
<dd>     Espacio de nombres de Kubernetes para conectarse a la API de Kubeflow Pipelines. Si no se especifica el espacio de nombres, el valor predeterminado es <code>kubeflow</code>.</dd>



## Archivos generados por la CLI de TFX

Cuando se crean y ejecutan canalizaciones, se generan varios archivos para su gestión.

- ${HOME}/tfx/local, beam, airflow, vertex
    - Los metadatos de canalización leídos desde la configuración se almacenan en `${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}`. Esta ubicación se puede personalizar al configurar variables de entorno como `AIRFLOW_HOME` o `KUBEFLOW_HOME`. Este comportamiento podría cambiar en versiones futuras. Este directorio se usa para almacenar información de canalizaciones, incluidos los ID de canalizaciones en el clúster de Kubeflow Pipelines, que se necesita para crear ejecuciones o actualizar canalizaciones.
    - Antes de TFX 0.25, estos archivos se ubicaban en `${HOME}/${ORCHESTRATION_ENGINE}`. En TFX 0.25, los archivos en la ubicación anterior se moverán automáticamente a la nueva ubicación para facilitar la migración.
    - Desde TFX 0.27, kubeflow no crea estos archivos de metadatos en el sistema de archivos local. Sin embargo, consulte a continuación otros archivos que crea kubeflow.
- (Solo Kubeflow) Dockerfile y una imagen de contenedor
    - Kubeflow Pipelines requiere dos tipos de entradas para una canalización. TFX se encarga de generar estos archivos en el directorio actual.
    - Una es una imagen de contenedor que se usará para ejecutar componentes en la canalización. Esta imagen de contenedor se construye cuando se crea o actualiza una canalización para Kubeflow Pipelines con el indicador `--build-image`. Si no existe, la CLI de TFX generará `Dockerfile` y creará y enviará una imagen de contenedor al registro especificado en KubeflowDagRunnerConfig.

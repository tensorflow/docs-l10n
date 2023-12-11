# Optimizar el rendimiento de TensorFlow con Profiler

[TDC]

En esta guía, se enseña cómo usar las herramientas disponibles con TensorFlow Profiler para registrar el rendimiento de sus modelos de TensorFlow. Aprenderá a entender como se desempeñan sus modelos en el host (CPU), el dispositivo (GPU, unidad de procesamiento de gráficos) o en una combinación de ambos.

El perfilamiento ayuda a entender el consumo de recursos del hardware (tiempo y memoria) de varias operaciones (ops) de TensorFlow en su modelo y a resolver los cuellos de botella del rendimiento y, finalmente, a hacer que su modelo se ejecute más rápido.

En esta guía, le enseñaremos paso a paso cómo instalar Profiler, las distintas herramientas disponibles, los diferentes modos en los que Profiler recopila datos de rendimiento y algunas buenas prácticas recomendadas para optimizar el rendimiento del modelo.

 Si quiere perfilar el rendimiento de su modelo en las Cloud TPU, consulte la [guía de Cloud TPU](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile).

## Instalar los prerequisitos de Profiler y GPU

Instale el plugin de Profiler para TensorBoard con pip. Tenga en cuenta que Profiler requiere las últimas versiones de TensorFlow y TensorBoard (&gt;=2.2).

```shell
pip install -U tensorboard_plugin_profile
```

Para perfilar en la GPU, debe:

1. Cumplir con los requisitos de los controladores de GPU de NVIDIA® y el kit de herramientas de CUDA® que se enumeran en [requisitos del software de soporte de GPU de TensorFlow](https://www.tensorflow.org/install/gpu#linux_setup).

2. Asegurarse de que [la interfaz de las herramientas de perfilamiento de NVIDIA® CUDA®](https://developer.nvidia.com/cupti) (CUPTI, por sus siglas en inglés) exista en la ruta de acceso:

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

Si la CUPTI no está en la ruta de acceso, se debe anteponer su directorio de instalación en la variable del entorno `$LD_LIBRARY_PATH` ejecutando:

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

Luego, ejecute el comando `ldconfig` anterior para verificar que se encuentre la biblioteca de CUPTI.

### Resolver problemas de permisos

Cuando se ejecuta el perfilamiento con el kit de herramientas CUDA® en un entorno Docker o en Linux, es posible que tenga problemas relacionados con la falta de permisos de CUPTI  (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). Vea los [Documentos para desarrollador de NVIDIA ](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters){:.external} para obtener más información sobre cómo puede resolver estos problemas en Linux.

Para resolver los permisos de CUPTI en un entorno Docker, ejecute

```shell
docker run option '--privileged=true'
```

<a name="profiler_tools"></a>

## Herramientas de perfilamiento

Puede acceder a Profiler desde la pestaña **Profile** en TensorBoard, que solo aparece cuando se capturan algunos datos del modelo.

Nota: Profiler necesita acceder a internet para cargar las [bibliotecas de Google Chart](https://developers.google.com/chart/interactive/docs/basic_load_libs#basic-library-loading). Es posible que se pierdan algunos gráficos y tablas si se ejecuta TensorBoard totalmente sin conexión en su equipo local, con un firewall corporativo o en un centro de datos.

Profiler tiene una selección de herramientas que ayudan con el análisis de rendimiento:

- La página de descripción general
- El analizador de la canalización de entrada
- Las estadísticas de TensorFlow
- El visor de trazado
- Las estadísticas de Kernel de la GPU
- La herramienta de perfilamiento de memoria
- Visor de Pod

<a name="overview_page"></a>

### Página de descripción general

La página de descripción general proporciona una vista de nivel superior sobre el rendimiento de su modelo durante la ejecución de perfilamiento. La página muestra una descripción general global de su host y de todos los dispositivos, y algunas recomendaciones para mejorar el rendimiento del entrenamiento de su modelo. También puede seleccionar hosts individuales en la lista desplegable Host.

La página de descripción general muestra algunos de los siguientes datos:

![imagen](./images/tf_profiler/overview_page.png)

- **Resumen del rendimiento**: Muestra un resumen de nivel superior del rendimiento de su modelo. El resumen de rendimiento tiene dos partes:

    1. Desglosamiento de la duración de los pasos: Desglosa la duración promedio de los pasos en varias categorías de empleo del tiempo:

        - Compilación: el tiempo que se emplea en compilar los kernels.
        - Entrada: el tiempo que se emplea en leer los datos de entrada.
        - Salida: el tiempo que se emplea para leer los datos de salida.
        - Inicio del kernel: el tiempo que emplea el host para iniciar los kernel.
        - Duración del cálculo del host.
        - Duración de la comunicación de dispositivo a dispositivo.
        - Duración del cálculo en el dispositivo.
        - Todos los demás, incluída la sobrecarga de Python.

    2. Precisiones del cálculo del dispositivo: informa el porcentaje de tiempo del cálculo del dispositivo que usa cálculos de entre 16 y 32 bits.

- **Gráfico de la duración de pasos**: Muestra un gráfico de la duración de los pasos del dispositivo (en milisegundos) durante todo el muestreo de los pasos. Se divide cada paso en varias categorías (con diferentes colores) del empleo del tiempo. El área roja corresponde a la duración del paso en la que los dispositivos estuvieron inactivos esperando los datos de entrada del host. El área verde muestra cuánto tiempo estuvo trabajando el dispositivo.

- **Las 10 mejores operaciones de TensorFlow en el dispositivo (por ejemplo, en la GPU)**: Muestra las ops que se ejecutaron durante más tiempo en el dispositivo.

    Cada fila muestra la duración propia de la op (como el porcentaje de tiempo que tardan todas las operaciones), la duración total, la categoría y el nombre.

- **Entorno de ejecución/strong0}: Muestra un resumen de nivel superior del entorno de ejecución del modelo, esto incluye:**

    - La cantidad de hosts que se usaron.
    - El tipo de dispositivo (GPU/TPU).
    - La cantidad de núcleos del dispositivo.

- **Recomendación para el siguiente paso**: Informa cuando un modelo está vinculado a la entrada y recomienda herramientas que pueden usarse para encontrar y resolver cuellos de botella en el rendimiento del modelo.

<a name="input_pipeline_analyzer"></a>

### Analizador de la canalización de entrada

Cuando un programa de TensorFlow lee datos desde un archivo, comienza desde el principio del gráfico de TensorFlow en una forma canalizada. El proceso de lectura se divide en varias etapas de procesamiento de datos que se conectan en serie, donde la salida de una de las etapas es la entrada de la siguiente. Este sistema de lectura de datos se llama *canalización de entrada*.

Una canalización típica para la lectura de registros de un archivo tiene las siguientes etapas:

1. Lectura del archivo.
2. Procesamiento del archivo (opcional).
3. Transferencia del archivo desde el host al dispositivo.

Una canalización de entrada ineficaz puede hacer que su aplicación funcione muy lenta. Se considera que una aplicación  **está vinculada a la entrada** cuando tarda mucho tiempo en la canalización de entrada. Use las conclusiones que se obtienen del analizador de la canalización de entrada para entender en donde es ineficaz la canalización de entrada.

El administrador de la canalización de entrada le informa inmediatamente si el programa está vinculado a la entrada y le enseña paso a paso el análisis lateral del dispositivo, y del host, para depurar los cuellos de botella del rendimiento en cualquier etapa de la canalización de entrada.

Consulte las guías sobre el rendimiento de la canalización de entrada para ver las mejores prácticas recomendadas para optimizar sus canalizaciones de entrada de datos.

####  Panel de la canalización de entrada

Para abrir el analizador de la canalización de entrada, seleccione **Profile**, luego seleccione **input_pipeline_analyzer** desde el menú desplegable **Tools**.

![imagen](./images/tf_profiler/input_pipeline_analyzer.png)

El panel tiene 3 secciones:

1. **Resumen**: Resume la canalización de entrada en general con información sobre si la aplicación está vinculada a la entrada o no y, si es así, qué tan vinculada.
2. **Análisis del lado del dispositivo**: Muestra los resultados del análisis del dispositivo en detalle, incluso la duración de los pasos y el rango del tiempo empleado en esperar los datos de entrada en los núcleos en cada paso.
3. **Análisis del lado del host**: Muestra un análisis completo del lado del host, incluso un desglosamiento de la duración del procesamiento de entrada en el host.

#### Resumen de la canalización de entrada

El **Resumen** informa si el programa está vinculado a la entrada al presentar el porcentaje de tiempo que el dispositivo emplea en esperar la entrada desde el host. Si se usa una canalización de entrada estándar que se ha instrumentado, la herramienta informa dónde se emplea la mayoría del tiempo de procesamiento de la entrada.

####  Análisis del lado del dispositivo

El análisis del lado del dispositivo proporciona información sobre el tiempo que se emplea en el dispositivo en vez de en el host y cuánto tiempo emplea el dispositivo en esperar los datos de entrada desde el host.

1. **Duración de los pasos trazada con respecto a la cantidad de pasos**: Muestra un gráfico de la duración del paso del dispositivo (en milisegundos) durante todo el muestreo de los pasos. Se divide cada paso en varias categorías (con diferentes colores) que representan el tiempo empleado. El área roja corresponde a la parte del tiempo del paso en el que los dispositivos estuvieron inactivos esperando los datos de entrada desde el host. El área verde muestra cuánto tiempo estuvo trabajando el dispositivo.
2. **Estadísticas de la duración de los pasos**: Informan la desviación estándar promedio y el rango ([mínimo, máximo]) de la duración del paso del dispositivo.

#### Análisis del lado del host

El análisis del lado del host informa el desglosamiento del tiempo de procesamiento de la entrada (el tiempo que se emplea en las operaciones de la API `tf.data`) en el host en diferentes categorías:

- **Lectura de datos desde los archivos a petición**: El tiempo que se emplea en leer los datos desde los archivos sin el almacenamiento en caché, preextracción ni intercalación.
- **Lectura de datos desde los archivos por adelantado**: El tiempo que se emplea en leer los datos desde los archivos, con el almacenamiento en caché, la preextracción y la intercalación.
- **Procesamiento de datos**: El tiempo que se emplea en procesar las ops, así como la descompresión de imágenes.
- **Poner datos en cola para transferirlos al dispositivo**: El tiempo que se emplea en poner los datos en una cola de alimentación antes de transferirlos al dispositivo.

Amplíe **{nbsp}las estadísticas de la op de entrada** para inspeccionar las estadísticas de las ops individuales de entrada y sus categorías desglosadas por el tiempo de ejecución.

![imagen](./images/tf_profiler/input_op_stats.png)

Aparecerá una tabla de datos de origen con cada entrada con la siguiente información:

1. **{nbsp}Operación de entrada**: Muestra el nombre de la operación de TensorFlow de la operación de entrada.
2. **Conteo**: Muestra la cantidad total de las instancias de la ejecución de la operación durante el período de la generación de perfiles.
3. **{nbsp}Tiempo total (en milisegundos)**: Muestra la suma acumulada del tiempo empleado en cada una de las instancias.
4. **% del tiempo total**: Muestra el tiempo total empleado en la operación como una fracción del tiempo total empleado en el procesamiento de entrada.
5. **Tiempo propio total (en milisegundos)**: Muestra la suma acumulada del tiempo propio empleado en cada una de las instancias. el tiempo propio aquí mide el tiempo empleado dentro del cuerpo de la función y se excluye el tiempo que se emplea en la función a la que llama.
6. **% del tiempo propio total**: Muestra el tiempo propio total como una fracción del tiempo total empleado en el procesamiento de entrada.
7. **Categoría**. Muestra la categoría del procesamiento de la operación de entrada.

<a name="tf_stats"></a>

### Estadísticas de TensorFlow

Las estadísticas de TensorFlow muestran el rendimiento de cada operación (op) de tensorflow que se ejecuta en el host o en el dispositivo durante la sesión de generación de perfiles.

![imagen](./images/tf_profiler/tf_stats.png)

La herramienta muestra la información de rendimiento en dos paneles:

- El panel superior muestra hasta cuatro gráficos circulares:

    1. La distribución del tiempo de ejecución propio de cada operación en el host.
    2. La distribución del tiempo de ejecución propio de cada tipo de operación en el host.
    3. La distribución del tiempo de ejecución propio de cada operación en el dispositivo.
    4. La distribución del tiempo de ejecución propio de cada tipo de operación en el dispositivo.

- En el panel inferior se muestra una tabla que informa los datos sobre las operaciones de TensorFlow con una fila por cada operación y una columna por cada tipo de dato (puede ordenar las columnas al hacer clic en el encabezado de la columna). Haga clic en el **botón Exportar como CSV** en la parte derecha del panel superior para exportar los datos de la tabla de un archivo CSV.

    Tenga en cuenta que:

    -  Si alguna operación tiene operaciones secundarias:

        - El tiempo total "acumulado" de una operación incluye el tiempo empleado en las operaciones secundarias.
        - El tiempo "propio" de una operación no incluye el tiempo empleado en las operaciones secundarias.

    - Si se ejecuta una operación en el host:

        - El porcentaje de tiempo propio total en el dispositivo incurrido por la operación será de 0.
        - El porcentaje acumulado del tiempo propio total en el dispositivo hasta la operación y con la operación será de 0.

    - Si se ejecuta una operación en el dispositivo:

        - El porcentaje del tiempo propio total en el host incurrido por la operación será de 0.
        - El porcentaje del tiempo propio total en el host efectuado hasta la operación y con la operación será de 0.

Puede elegir si quiere incluir o excluir el tiempo de inactividad en los gráficos circulares y la tabla.

<a name="trace_viewer"></a>

### Visor de trazado

El visor de trazado muestra una línea del tiempo que presenta:

- Las duraciones de las operaciones que ejecutó su modelo de TensorFlow
- Las partes del sistema (host o dispositivo) que ejecutaron una operación. Por lo general, el host ejecuta operaciones de entrada, preprocesa los datos de entrenamiento y los transfiere al dispositivo, mientras que el dispositivo ejecuta el entrenamiento del modelo en sí

El visor de trazado le permite identificar los problemas de rendimiento que tiene su modelo, y luego actúa para resolverlos. Por ejemplo, en un nivel superior, se puede identificar si la entrada o el modelo de entrenamiento están tardando mucho tiempo. Si se analiza en profundidad, se pueden identificar cuáles operaciones toman más tiempo de ejecución. Tenga en cuenta que el visor del trazado se limita a 1 millón de eventos por dispositivo.

####  Interfaz del visor de trazado

Cuando se abre el visor de trazado, se muestra la ejecución más reciente:

![imagen](./images/tf_profiler/trace_viewer.png)

Esta pantalla contiene los siguientes elementos principales:

1. **Panel de la línea de tiempo**: Muestra las operaciones que el dispositivo y el host ejecutan a lo largo del tiempo.
2. **Panel de detalles**: Muestra la información adicional de las operaciones seleccionadas en el panel de la línea de tiempo.

El panel de la línea de tiempo contiene los siguientes elementos principales:

1. **Barra superior**: Contiene varios controles auxiliares.
2. **Eje de tiempo**: Muestra el tiempo en relación con el inicio del trazado.
3. **Etiquetas de sección y de seguimiento**: Cada sección contiene varios seguimientos, y se puede hacer clic en el triángulo que se encuentra en la parte izquierda para ampliar y colapsar la sección. Hay una sección para cada elemento de procesamiento en el sistema.
4. **Selección de herramientas**: Contiene varias herramientas para interactuar con el visor de trazado, tales como el Zoom, Selección, Mover y Cronometrar. Use la herramienta de cronometrar para marcar un intervalo de tiempo.
5. **Eventos**: Muestran el tiempo durante el cual se ejecuta una operación o la duración de los eventos meta, tales como los pasos de entrenamiento.

##### Secciones y seguimientos

 El visor de trazado contiene las siguiente secciones:

- **Una sección para cada nodo del dispositivo**, con la etiqueta del número de chip del dispositivo y del nodo del dispositivo dentro del chip (por ejemplo, `/device:GPU:0 (pid 0)`). Cada sección del nodo del dispositivo contiene los siguientes seguimientos:
    - **Paso**: Muestra la duración de los pasos de entrenamiento que se estaban ejecutando en el dispositivo
    - **Operaciones de TensorFlow**: Muestra las operaciones que se ejecutaron en el dispositivo
    - **Operaciones de XLA**: Muestra las operaciones (ops) de [XLA](https://www.tensorflow.org/xla/) que se ejecutaron en el dispositivo si se usa XLA como el compilador (cada operación de TensorFlow se traduce en una o más operaciones de XLA. El compilador XLA traduce las operaciones de XLA en un código que se ejecuta en el dispositivo.
- **Una sección para los procesos que se ejecutan en la CPU del equipo del host,** con la etiqueta **" Subprocesos del host"**. La sección contiene un seguimiento para cada subproceso de la CPU. Tenga en cuenta que puede ignorar la información que se muestra junto a las etiquetas de la sección.

##### Eventos

Los eventos en la línea de tiempo se muestran en diferentes colores: los colores en sí no tienen ningún significado específico.

El visor de trazado también puede mostrar los trazados de las llamadas de la función de Python en el programa de TensorFlow. Si usa la API `tf.profiler.experimental.start`, puede habilitar el trazado de Python con la tupla `ProfilerOptions` al iniciar el perfilamiento. Como alternativa, si se usa un modelo de muestra para el perfilamiento, puede seleccionar el nivel de trazado desde las opciones del menú desplegable en el cuadro de diálogo **Capturar perfil**.

![imagen](./images/tf_profiler/python_tracer.png)

<a name="gpu_kernel_stats"></a>

### Estadísticas del Kernel de la GPU

Esta herramienta muestra las estadísticas de rendimiento y la operación que se origina para cada kernel acelerado de la GPU.

![imagen](./images/tf_profiler/gpu_kernel_stats.png)

La herramienta muestra la información en dos paneles:

- En el panel superior, se muestra un gráfico circular que representa los kernel de CUDA que tienen el tiempo total transcurrido más alto.

- En el panel inferior, se muestra una tabla con los siguientes datos para cada par único de kernel y operación:

    - Una clasificación por orden descendiente de la duración total transcurrida de la GPU agrupada en en el par kernel y operación.
    - El nombre del kernel iniciado.
    - La cantidad de registros de la GPU que usa el kernel.
    - El tamaño total de la memoria compartida (estática y dinámica compartida) que se usan en bytes.
    - La dimension del bloque expresada en `blockDim.x, blockDim.y, blockDim.z`.
    - Las dimensiones de la cuadrícula expresadas en `gridDim.x, gridDim.y, gridDim.z`.
    - Si la operación es elegible para usar [Tensor Cores](https://www.nvidia.com/en-gb/data-center/tensor-cores/) o no.
    - Si el kernel tiene instrucciones de Tensor Core.
    - El nombre de la operación que inició el kernel.
    - Las veces que se repite el par kernel y operación.
    - El tiempo total de la GPU transcurrido en microsegundos.
    - El tiempo promedio de la GPU transcurrido en microsegundos.
    - El tiempo mínimo de la GPU transcurrido en microsegundos.
    - El tiempo máximo de al GPU transcurrido en microsegundos.

<a name="memory_profile_tool"></a>

### Herramienta de perfilamiento de la memoria {: id = 'memory_profile_tool'}

La herramienta de **Perfilamiento de la memoria** monitorea el uso de la memoria de su dispositivo durante los intervalos de perfilamiento. Puede usar esta herramienta para lo siguiente:

- Depurar los problemas de memoria insuficiente (OOM) al señalar los usos máximos de memoria y al asignación memoria correspondiente en las operaciones de TensorFlow. También se pueden depurar los problemas de OOM que puedan surgir cuando se ejecuta la inferencia [de varios inquilinos](https://arxiv.org/pdf/1901.06887.pdf).
- Depurar problemas de fragmentación de memoria.

La herramienta de perfilamiento de la memoria muestra los datos en tres secciones:

1. **Resumen del perfilamiento de la memoria**
2. **Grafico de la línea de tiempo de la memoria**
3. **Tabla de desglosamiento de la memoria**

#### Resumen del perfilamiento de memoria

En esta sección, se muestra un resumen de nivel superior del perfilamiento de la memoria de su programa de TensorFlow tal como se muestra a continuación:

&lt;img src="./images/tf_profiler/memory_profile_summary.png" width="400", height="450"&gt;

El resumen del perfilamiento de la memoria tiene seis campos:

1. **ID de memoria**: Menú desplegable donde se enumeran todos los sistemas de memoria del dispositivo disponibles. Seleccione el sistema de memoria que quiera ver desde el menú desplegable.
2. **Nro de asignaciones**: La cantidad de asignaciones de la memoria que se hacen durante el intervalo de perfilamiento.
3. **Nro de desasignaciones**: La cantidad de desasignaciones de la memoria en el intervalo de perfilamiento.
4. **Capacidad de la memoria**: La capacidad total (en GiBs) del sistema de la memoria que seleccione.
5. **Uso del montón máximo**: El uso de memoria máximo (en Gibs) desde el inicio de ejecución del modelo.
6. **Uso máximo de la memoria**: El uso de memoria máximo (en GiBs). Este campo contiene los siguientes subcampos:
    1. La marca de tiempo en la que el uso de memoria máximo ocurrió en el gráfico de la línea de tiempo.
    2. **Reserva de apilamiento**: La cantidad de memoria reservada en el apilamiento (en GiBs).
    3. **Asignación del montón**: La cantidad de memoria que se asigna en el montón (en GiBs).
    4. **Memoria libre**: La cantidad de memoria libre (en GiBs). La capacidad de la memoria es la suma total de la reserva de apilamiento, la asignación del montón y la memoria libre.
    5. **Fragmentación**: El porcentaje de desfragmentación (cuanto más bajo mejor). Se calcula como un porcentaje de `(1 - Size of the largest chunk of free memory / Total free memory)`.

#### Gráfico de la linea de tiempo de la memoria

En esta sección, se muestra un trazado del uso de la memoria (en GiBs) y el porcentaje de fragmentación con respecto al tiempo (en milisegundos).

![imagen](./images/tf_profiler/memory_timeline_graph.png)

El eje X representa la línea del tiempo (en milisegundos) del intervalo de perfilamiento. El eje Y, en la parte izquierda, representa el uso de la memoria (en gibs) y el eje Y, en la parte derecha, representa el porcentaje de fragmentación. En cada punto del tiempo en el eje X, se desglosa la memoria total en 3 categorías: Apilamiento (en rojo), montón (en naranja) y libre (en verde). Pase el cursor sobre la marca de tiempo específica para ver los detalles sobre la asignación/desasignación de los eventos de memoria en ese momento como a continuación:

![imagen](./images/tf_profiler/memory_timeline_graph_popup.png)

La ventana emergente muestra la siguiente información:

- **timestamp(ms)**: La ubicación del evento seleccionado en la línea de tiempo.
- **event**: El tipo de evento (asignación o desasignación).
- **requested_size(GiBs)**: La cantidad de memoria requerida. Será un número negativo para los eventos de desasignación.
- **allocation_size(GiBs)**: La cantidad real de memoria asignada. Será un número negativo para los eventos de desasignación.
- **tf_op**: La operación de TensorFlow que solicita la asignación/desasignación.
- **step_id**: El paso del entrenamiento en el que ocurre el evento.
- **region_type**: El tipo de entidad de los datos para los que se asigna esta memoria. Los valores posibles son `temp` para los temporales, `output` para las activaciones y los gradientes y `persist`/`dynamic` para los pesos y las constantes.
- **data_type**: El tipo de elemento del tensor (por ejemplo, uint8 por el entero no asignado de 8-bit).
- **tensor_shape**: La forma del tensor que se asigna/desasigna.
- **memory_in_use(GiBs)**: La memoria total que está en uso en este momento.

#### Tabla de desglosamiento de la memoria

La tabla muestra las asignaciones activas de memoria en el momento de uso máximo de la memoria en el intervalo de perfilamiento.

![imagen](./images/tf_profiler/memory_breakdown_table.png)

Hay una fila para cada operación de TensorFlow y cada fila tiene las siguientes columnas:

- **Nombre de la operación**: El nombre de la operación de TensorFlow.
- **Tamaño de la asignación (GiBs)**: La cantidad total de memoria asignada a esta operación.
- **Tamaño requerido (GiBs)**: La cantidad total de memoria requerida para esta operación.
- **Repeticiones**: La cantidad de asignaciones para esta operación.
- **Tipo de región**: El tipo de entidad de los datos para los que se asigna esta memoria. Los valores posibles son `temp` para los temporales, `output` para las activaciones y los gradientes y `persist`/`dynamic` para los pesos y las constantes.
- **Tipo de dato**: El tipo de elemento del tensor.
- **Forma**: La forma de los tensores asignados.

Nota: Puede ordenar cualquier columna de la tabla y también filtrar las líneas por el nombre de la operación.

<a name="pod_viewer"></a>

### Visor de Pod

La herramienta de visor de Pod muestra el desglosamiento de los pasos de entrenamiento de todos los trabajadores.

![imagen](./images/tf_profiler/pod_viewer.png)

- El panel superior tiene un deslizador para seleccionar la cantidad de pasos.
- El panel inferior muestra un gráfico de las columnas apiladas. Está es una vista de nivel superior de las categorías desglosadas de la duración del paso, una arriba de la otra. Cada columna apilada representa un trabajador único.
- Cuando pasa el cursor sobre una columna apilada, la tarjeta en la parte izquierda muestra más detalles sobre el desglosamiento del paso.

<a name="tf_data_bottleneck_analysis"></a>

### Análisis del cuello de botella de tf.data

Advertencia: Se trata de una herramienta experimental. Abra [Problema de GitHub](https://github.com/tensorflow/profiler/issues) si cree que el resultado del análisis es incorrecto.

La herramienta de análisis del cuello de botella `tf.data` detecta automáticamente los cuellos de botella en las canalizaciones de entrada `tf.data` en su programa y proporciona recomendaciones para resolverlos. Funciona con cualquier programa con `tf.data` sin importar la plataforma (CPU/GPU/TPU). Sus análisis y las recomendaciones se basan en esta [guía](https://www.tensorflow.org/guide/data_performance_analysis).

Detecta un cuello de botella mediante los siguientes pasos:

1. Busca el host que está más vinculado a la entrada.
2. Busca la ejecución más lenta de una canalización de entrada de `tf.data`.
3. Reconstruye el gráfico de la canalización de entrada desde el trazado de perfilamiento.
4. Busca la ruta de acceso crítica en el gráfico de la canalización de entrada.
5. Identifica la transformación más lenta en la ruta de acceso crítica como un cuello de botella.

La UI se divide en 3 secciones: **El resumen del análisis de rendimiento**, **El resumen de todas las canalizaciones de entrada** y **El gráfico de canalización de entrada**.

#### Resumen del análisis de rendimiento

![imagen](./images/tf_profiler/tf_data_summary.png)

En esta sección, se proporciona el resumen del análisis. Este informa las canalizaciones de entrada de `tf.data` lentas que se detectan en el perfilamiento. En esta sección se también se muestra el host más vinculado a la entrada y su canalización de entrada más lenta con latencia máxima. Lo más importante, identifica qué parte de la canalización de entrada es el cuello de botella y cómo resolverlo. Proporciona la información del cuello de botella con el tipo de elemento de iteración y su nombre largo.

##### Cómo leer el nombre largo del elemento de iteración de tf.data

El formato del nombre largo es `Iterator::<Dataset_1>::...::<Dataset_n>`. En el nombre largo, `<Dataset_n>` coincide con el tipo de elemento de iteración y los otros conjuntos de datos en el nombre largo representan las transformaciones del canal de bajada.

Por ejemplo, observe el siguiente conjunto de datos de la canalización de entrada:

```python
dataset = tf.data.Dataset.range(10).map(lambda x: x).repeat(2).batch(5)
```

Los nombres largos de los elementos de iteración del conjunto de datos anterior serán:

Tipo de elemento de iteración | Nombre largo
:-- | :--
Rango | Iterator::Batch::Repeat::Map::Range
Mapa | Iterator::Batch::Repeat::Map
Repetición | Iterator::Batch::Repeat
Lote | Iterator::Batch

#### Resumen de todas las canalizaciones de entrada

![imagen](./images/tf_profiler/tf_data_all_hosts.png)

En esta sección, se proporciona el resumen de todas las canalizaciones de entrada en todos los host. Cuando se usa la estrategia de distribución, hay una canalización de entrada del host que ejecuta el código `tf.data` del programa y varias canalizaciones de entrada del dispositivo recuperan los datos desde la canalización de entrada del host y los transfieren a los dispositivos.

Para cada canalización de entrada, se muestran las estadísticas de su tiempo de ejecución. Una llamada es considerada lenta si tarda más de 50 μs.

#### Gráfico de la canalización de entrada

![imagen](./images/tf_profiler/tf_data_graph_selector.png)

En esta sección, se muestra el gráfico de la canalización de entrada con la información del tiempo de ejecución. Puede usar "Host" y "Canalización de entrada" para elegir qué host o qué canalización de entrada ver. Las ejecuciones de la canalización de entrada están ordenadas por el tiempo de ejecución en orden descendente que se puede elegir con el menú desplegable de **Rango**.

![imagen](./images/tf_profiler/tf_data_graph.png)

Los nodos en la ruta de acceso crítica tiene bordes en negrita. El nodo del cuello de botella. que es el nodo con el tiempo propio más largo en la ruta de acceso crítica, tiene un borde rojo. Los otros nodos no críticos tienen un borde discontinuo gris.

En cada nodo, **el tiempo de inicio ** indica el tiempo de inicio de la ejecución. El mismo nodo puede ejecutarse varias veces, por ejemplo, si hay una operación de `Batch` en la canalización de entrada. Si se ejecuta varias veces, es el tiempo de inicio de la primera ejecución.

La **duración total** es el tiempo real de la ejecución. Si se ejecuta varias veces es la suma de todos los tiempos reales de todas las ejecuciones.

El **tiempo propio** es el **tiempo total** sin tener en cuenta el tiempo que se superpone con sus nodos secundarios inmediatos.

"Nro de llamadas" es la cantidad de veces que la canalización de entrada se ejecuta.

<a name="collect_performance_data"></a>

## Recopilar datos de rendimiento

El perfilamiento de TensorFlow recopila las actividades del host y los trazados de la GPU de su modelo de TensorFlow. Puede configurar el perfilamiento para recopilar los datos de rendimiento a través del modo de programación o del modo simple.

### Las API de perfilamiento

Puede usar las siguientes API para realizar el perfilamiento.

- El modo de programación con la retrollamada de Keras del TensorBoard (`tf.keras.callbacks.TensorBoard`)

    ```python
    # Profile from batches 10 to 15
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch='10, 15')

    # Train the model and use the TensorBoard Keras callback to collect
    # performance profiling data
    model.fit(train_data,
              steps_per_epoch=20,
              epochs=5,
              callbacks=[tb_callback])
    ```

- El modo de programación con la API de función `tf.profiler`

    ```python
    tf.profiler.experimental.start('logdir')
    # Train the model here
    tf.profiler.experimental.stop()
    ```

- El modo de programación con el gestor de contexto

    ```python
    with tf.profiler.experimental.Profile('logdir'):
        # Train the model here
        pass
    ```

Si se ejecuta el perfilamiento durante mucho tiempo, es posible que se agote toda la memoria. Se recomienda no usar perfilamiento durante más de 10 pasos al mismo tiempo. Evite usar el perfilamiento en los primeros lotes para evitar errores debido a una sobrecarga de inicialización.

<a name="sampling_mode"></a>

- Modo de muestreo: realiza perfilamiento a petición con `tf.profiler.experimental.server.start` para iniciar un servidor gRPC con la ejecución de su modelo de TensorFlow. Luego de iniciar el servidor gRPC y de ejecutar su modelo, puede capturar un perfil a través del botón **Capturar perfil** en el plugin del perfil de TensorBoard. Use el guión en la sección de Instalar el perfilamiento anterior para iniciar una instancia de TensorBoard si aún no se está ejecutando.

    Como ejemplo,

    ```python
    # Start a profiler server before your model runs.
    tf.profiler.experimental.server.start(6009)
    # (Model code goes here).
    #  Send a request to the profiler server to collect a trace of your model.
    tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                          'gs://your_tb_logdir', 2000)
    ```

    Un ejemplo para generar los perfiles de varios trabajadores:

    ```python
    # E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
    # would like to profile for a duration of 2 seconds.
    tf.profiler.experimental.client.trace(
        'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
        'gs://your_tb_logdir',
        2000)
    ```

<a name="capture_dialog"></a>

&lt;img src="./images/tf_profiler/capture_profile.png" width="400", height="450"&gt;

Use el cuadro de diálogo **Capturar perfil** para especificar:

- Una lista de valores separados por coma de los nombres de la URL o de la TPU del servicio de perfilamiento.
- Una duración del perfilamiento.
- El nivel de trazado de la llamada del dispositivo, del host y de la función de Python.
- La cantidad de veces que quiere que el Profiler intente capturar los perfiles si no funciona la primera vez.

### Perfilar de bucles de entrenamiento personalizados

Para perfilar bucles de entrenamiento personalizados en su código de TensorFlow, indiquele al bucle de entrenamiento que marque los límites de los pasos para el Profiler con la API `tf.profiler.experimental.Trace`.

El argumento `name` se utiliza como prefijo para los nombres de los pasos, el argumento de la palabra clave `step_num` se agrega a los nombres de los pasos y el argumento de la palabra clave `_r` hace que Profiler procese este evento de seguimiento como un evento de paso.

Como ejemplo,

```python
for step in range(NUM_STEPS):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_data = next(dataset)
        train_step(train_data)
```

Esto permitirá el análisis de rendimiento basado en pasos del Profiler y hará que los eventos de los pasos aparezcan en el visor de seguimiento.

Asegúrese de incluir el elemento de iteración del conjunto de datos dentro del contexto `tf.profiler.experimental.Trace` para un análisis preciso de la canalización de entrada.

El siguiente fragmento de código es un antipatrón:

Advertencia: Esto dará como resultado un análisis inexacto de la canalización de entrada.

```python
for step, train_data in enumerate(dataset):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        train_step(train_data)
```

### Casos de uso de perfilamiento

El perfilamiento cubre una serie de casos de uso en cuatro ejes diferentes. Algunas de las combinaciones son compatibles actualmente y otras se agregarán en el futuro. Algunos de los casos de uso son:

- *Perfilamiento local vs remoto*: estas son dos formas comunes de configurar su entorno de perfilamiento. En el perfilamiento local, se llama a la API de perfilamiento en la misma máquina que ejecuta su modelo, por ejemplo, una estación de trabajo local con GPU. En el perfilamiento remoto, se llama a la API de perfilamiento en una máquina diferente desde donde se ejecuta su modelo, por ejemplo, en una Cloud TPU.
- *Perfilamiento de varios trabajadores*: puede perfilar varias máquinas cuando utilice las capacidades de entrenamiento distribuido de TensorFlow.
- *Plataforma de hardware*: perfila la CPU, GPU y TPU.

La siguiente tabla proporciona una descripción general rápida de los casos de uso compatibles con TensorFlow mencionados anteriormente:

<a name="profiling_api_table"></a>

| API de perfilamiento | Local | Remoto | Múltiples | Hardware | : : : : trabajadores : Plataformas : | :--------------------- | :-------- | :-------- | :-------- | :-------- | | **TensorBoard Keras | Compatible | No | No | CPU, GPU | : Retrollamada** : : Compatible : Compatible : : | **`tf.profiler.experimental` | Compatible | No | No | CPU, GPU | : iniciar/detener [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental#functions_2)** : : Compatible : Compatible : : | **`tf.profiler.experimental` | Compatible | Compatible | Compatible | CPU, GPU, | : client.trace [API](https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace)** : : : : TPU : | **API del administrador de contexto** | Compatible | No | No | CPU, GPU | : : : compatible : Compatible : :

<a name="performance_best_practices"></a>

## Mejores prácticas para un rendimiento óptimo del modelo

Use las siguientes recomendaciones según corresponda para sus modelos de TensorFlow para lograr un rendimiento óptimo.

En general, realice todas las transformaciones en el dispositivo y asegúrese de utilizar la última versión compatible de bibliotecas como cuDNN e Intel MKL para su plataforma.

### Optimizar la canalización de datos de entrada

Use los datos del [#input_pipeline_analyzer] para optimizar su canalización de entrada de datos. Una canalización de entrada de datos eficiente puede mejorar drásticamente la velocidad de ejecución de su modelo al reducir el tiempo de inactividad del dispositivo. Intente incorporar las mejores prácticas detalladas en la guía [Mejor rendimiento con la API tf.data](https://www.tensorflow.org/guide/data_performance) y lo siguiente para hacer que su canalización de entrada de datos sea más eficiente.

- En general, paralelizar cualquier operación que no necesite ejecutarse secuencialmente puede optimizar significativamente la canalización de entrada de datos.

- En muchos casos, resulta útil cambiar el orden de algunas llamadas o ajustar los argumentos para que funcionen mejor para su modelo. Mientras optimiza la canalización de datos de entrada, compare solo el cargador de datos sin los pasos de entrenamiento y retropropagación para cuantificar el efecto de las optimizaciones de forma independiente.

- Intente ejecutar su modelo con datos sintéticos para verificar si la canalización de entrada es un cuello de botella en el rendimiento.

- Use `tf.data.Dataset.shard` para el entrenamiento con varias GPU. Asegúrese de particionar desde el principio del bucle de entrada para evitar reducciones en el rendimiento. Cuando trabaje con archivos TFRecords, asegúrese de particionar la lista de archivos TFRecords y no el contenido de los TFRecords.

- Puede ejecutar varias operaciones en paralelo al establecer dinámicamente el valor de `num_parallel_calls` con `tf.data.AUTOTUNE`.

- Considere limitar el uso de `tf.data.Dataset.from_generator`, ya que es más lento en comparación con las operaciones puras de TensorFlow.

- Considere limitar el uso de `tf.py_function` ya que no se puede serializar y no se admite su ejecución en TensorFlow distribuido.

- Use `tf.data.Options` para controlar las optimizaciones estáticas en la canalización de entrada.

Lea también la [guía](https://www.tensorflow.org/guide/data_performance_analysis) de análisis de rendimiento de <code>tf.data</code> para obtener más ayuda sobre cómo optimizar su canalización de entrada.

#### Optimizar el aumento de datos

Cuando trabaje con datos de imágenes, se puede mejorar la eficacia del [el aumento de datos](https://www.tensorflow.org/tutorials/images/data_augmentation) al convertirlos en diferentes tipos de datos <b><i>después</i></b> de aplicar las transformaciones espaciales, como voltear, recortar, rotar, etc.

Nota: Algunas operaciones como `tf.image.resize` cambian de forma clara el `dtype` a `fp32`. Asegúrese de normalizar sus datos para que estén entre `0` y `1` si no se hace automáticamente. Omitir este paso podría provocar errores `NaN` si se habilitó [AMP](https://developer.nvidia.com/automatic-mixed-precision).

#### Usar NVIDIA® DALI

En algunos casos, como cuando tiene un sistema con una tasa de uso alta de GPU a CPU, es posible que todas las optimizaciones anteriores no sean suficientes para eliminar los cuellos de botella en el cargador de datos que surgen por las limitaciones de los ciclos de la CPU.

Si está utilizando unas GPU de NVIDIA® para aplicaciones de aprendizaje profundo de audio y visuales en la computadora, considere usar la biblioteca de carga de datos ( [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting%20started.html) ) para acelerar la canalización de datos.

Consulte la documentación de [NVIDIA® DALI: operaciones](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html) para obtener una lista de operaciones DALI compatibles.

### Usar subprocesos y ejecución paralela

Ejecute operaciones en varios subprocesos de la CPU con la API `tf.config.threading` para ejecutarlas más rápido.

TensorFlow establece automáticamente la cantidad de subprocesos en paralelo de forma predeterminada. El grupo de subprocesos disponible para ejecutar operaciones de TensorFlow depende de la cantidad de subprocesos disponibles de la CPU.

Controle la aceleración en paralelo máxima de una sola operación con `tf.config.threading.set_intra_op_parallelism_threads`. Tenga en cuenta que si ejecuta varias operaciones en paralelo, todas compartirán el grupo de subprocesos disponible.

Si tiene operaciones independientes sin bloqueo (operaciones sin una ruta de acceso dirigida entre ellas en el gráfico), use `tf.config.threading.set_inter_op_parallelism_threads` para ejecutarlas simultáneamente con el grupo de subprocesos disponible.

### Varios

Cuando trabaje con modelos más pequeños en las GPU de NVIDIA®, puede configurar `tf.compat.v1.ConfigProto.force_gpu_compatible=True` para forzar que todos los tensores de la CPU se asignen con memoria fija CUDA para potenciar de forma significativa el rendimiento del modelo. Sin embargo, tenga cuidado al utilizar esta opción para modelos desconocidos o muy grandes, ya que esto podría afectar negativamente el rendimiento del host (CPU).

### Mejorar el rendimiento del dispositivo

Siga las mejores prácticas que se detallan aquí y en la [guía de optimización del rendimiento de la GPU](https://www.tensorflow.org/guide/gpu_performance_analysis) para optimizar el rendimiento del modelo de TensorFlow en el dispositivo.

Si está usando unas GPU de NVIDIA, registre la GPU y el uso de la memoria en un archivo CSV con la ejecución del siguiente código:

```shell
nvidia-smi
--query-gpu=utilization.gpu,utilization.memory,memory.total,
memory.free,memory.used --format=csv
```

#### Configurar el diseño de datos

Cuando trabaje con datos que tengan información del canal (como imágenes), optimice el formato del diseño de datos para marcar qué canal se prefiere al final (NHWC, mejor que NCHW).

Los formatos de los datos del último canal mejoran el uso de [Tensor Core](https://www.nvidia.com/en-gb/data-center/tensor-cores/) y proporcionan mejoras de rendimiento significativas, especialmente en los modelos convolucionales cuando se combinan con AMP. Los Tensor Core aún puede operar los diseños de datos de NCHW, pero introduce una sobrecarga adicional debido a las operaciones de transposición automática.

Puede optimizar el diseño de los datos para marcar los diseños NHWC como preferidos al configurar `data_format="channels_last"` para capas como `tf.keras.layers.Conv2D`, `tf.keras.layers.Conv3D` y `tf.keras.layers.RandomRotation`.

Utilice `tf.keras.backend.set_image_data_format` para configurar el formato de diseño de datos predeterminado para la API backend de Keras.

#### Maximizar la memoria caché L2

Cuando trabaje con las GPU de NVIDIA®, ejecute el siguiente fragmento de código antes del ciclo de entrenamiento para maximizar la granularidad de recuperación de L2 a 128 bytes.

```python
import ctypes

_libcudart = ctypes.CDLL('libcudart.so')
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
assert pValue.contents.value == 128
```

#### Configurar el uso de subprocesos de la GPU

El modo de subprocesos de la GPU decide cómo se utilizan los subprocesos de la GPU.

Configure el modo de subprocesos en `gpu_private` para asegurarse de que el preprocesamiento no robe todos los subprocesos de la GPU. Esto reducirá el retraso en el inicio del kernel durante el entrenamiento. También puede configurar la cantidad de subprocesos por GPU. Establezca estos valores con las variables de entorno.

```python
import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'
```

#### Configurar las opciones de memoria de la GPU

En general, aumente el tamaño del lote y escale el modelo para utilizar mejor las GPU y obtener un mayor rendimiento. Tenga en cuenta que aumentar el tamaño del lote cambiará la precisión del modelo, por lo que es necesario escalarlo ajustando los hiperparámetros como la tasa de aprendizaje para alcanzar la precisión deseada.

Además, use `tf.config.experimental.set_memory_growth` para permitir que la memoria de la GPU crezca y evitar que absolutamente toda la memoria disponible se asigne a las operaciones que requieren solo una fracción de la memoria. Esto permite que otros procesos que consumen memoria de la GPU se ejecuten en el mismo dispositivo.

Para obtener más información, consulte la ayuda de [Limitar el crecimiento de la memoria de la GPU](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth) en la guía de GPU para obtener más información.

#### Varios

- Aumente el tamaño del minilote de entrenamiento (cantidad de muestras de entrenamiento usadas por dispositivo en una iteración del ciclo de entrenamiento) a la cantidad máxima que entre sin un error de memoria insuficiente (OOM) en la GPU. Aumentar el tamaño del lote afecta la precisión del modelo, así que asegúrese de escalar el modelo ajustando los hiperparámetros para alcanzar la precisión deseada.

- Deshabilite los informes de errores de OOM durante la asignación de tensores en el código de producción. Establezca `report_tensor_allocations_upon_oom=False` en `tf.compat.v1.RunOptions`.

- Para modelos con capas convolucionales, elimine la adición de sesgo si utiliza la normalización por lotes. La normalización por lotes cambia los valores según su media y esto elimina la necesidad de tener un término de sesgo constante.

- Use TF Stats para averiguar qué la eficacia con la que se ejecutan las operaciones en el dispositivo.

- Use `tf.function` para realizar cálculos y, opcionalmente, habilite el indicador `jit_compile=True` (`tf.function(jit_compile=True`). Para obtener más información, consulte [Usar tf.function de XLA](https://www.tensorflow.org/xla/tutorials/jit_compile).

- Minimice las operaciones de Python del host en los pasos y reduzca las retrollamadas. Calcule las métricas cada pocos pasos en lugar de en cada paso.

- Mantenga las unidades informáticas del dispositivo ocupadas.

- Envíe datos a varios dispositivos en paralelo.

- Considere [el uso de representaciones numéricas de 16 bits](https://www.tensorflow.org/guide/mixed_precision), como `fp16` (el formato de punto flotante de media precisión que especifíca el Instituto de Ingenieros Eléctricos y Electrónicos [IEEE, Institute of Electrical and Electronics Engineers]) o el formato [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) de punto flotante de Brain.

## Recursos adicionales

- El tutorial de [TensorFlow Profiler: perfilr el rendimiento del modelo](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) con Keras y TensorBoard donde puede aplicar los consejos de esta guía.
- La charla sobre [Perfilamiento del rendimiento en TensorFlow 2](https://www.youtube.com/watch?v=pXHAQIhhMhI) de TensorFlow Dev Summit 2020.
- La [demostración de TensorFlow Profiler](https://www.youtube.com/watch?v=e4_4D7uNvf8) de TensorFlow Dev Summit 2020.

## Limitaciones conocidas

### Perfilar varias GPU en TensorFlow 2.2 y TensorFlow 2.3

TensorFlow 2.2 y 2.3 admiten varios perfilamientos de la GPU solo para sistemas de host único; no se admiten varios perfilamientos de la GPU para sistemas de varios hosts. Para perfilar las configuraciones de la GPU de varios trabajadores, se debe perfilar cada trabajador de forma independiente. Desde TensorFlow 2.4, se pueden perfilar varios trabajadores con la API `tf.profiler.experimental.client.trace`.

Se requiere CUDA® Toolkit 10.2 o posteriores para perfilar varias GPU. Ya que TensorFlow 2.2 y 2.3 admiten versiones de CUDA® Toolkit solo hasta 10.1, debe crear enlaces simbólicos a `libcudart.so.10.1` y `libcupti.so.10.1`:

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```

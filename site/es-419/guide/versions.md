# Compatibilidad de versiones en TensorFlow

Este documento es para aquellos usuarios que necesitan contar con compatibilidad hacia atrás con elementos anteriores de las diferentes versiones de TensorFlow (ya sea para código o datos) y también para desarrolladores que quieran modificar TensorFlow sin perder la compatibilidad.

## Semántica de las versiones 2.0

TensorFlow se rige por la semántica de versiones 2.0 ([semver](http://semver.org)) para esta API pública. Cada versión de lanzamiento tiene la forma `MAJOR.MINOR.PATCH`. Por ejemplo, la versión 1.2.3 de TensorFlow tiene versión 1 en `MAJOR`, versión 2 en `MINOR` y versión 3 en `PATCH`. Los cambios en cada uno de los números tienen los siguientes significados:

- **MAJOR** (<em>mayor</em>): potencialmente, cambios incompatibles hacia atrás. Código y datos que funcionaron con lanzamientos importantes anteriores que no necesariamente funcionarán con el lanzamiento nuevo. Sin embargo, en algunos casos, los grafos y puntos de verificación existentes de TensorFlow pueden migrarse al lanzamiento nuevo. Para más detalles sobre la compatibilidad de los datos, consulte [Compatibilidad de grafos y puntos de verificación](#compatibility_of_graphs_and_checkpoints).

- **MINOR** (<em>menor</em>): las funciones hacia atrás, como las mejoras en la velocidad y demás. Aquellos códigos y datos que funcionaron con un lanzamiento menor previo *y* que dependen solamente de la API pública no experimental seguirán funcionando sin cambios. Para más detalles sobre lo que pertenece a la API pública y lo que no, consulte [Qué se incluye](#what_is_covered).

- **PATCH** (<em>parche</em>): reparaciones de errores compatibles hacia atrás.

Por ejemplo, el lanzamiento 1.0.0 introdujo cambios *incompatibles* hacia atrás a partir del lanzamiento 0.12.1. Pero el lanzamiento 1.1.1 fue *compatible* hacia atrás con el lanzamiento 1.0.0. <a name="what_is_covered"></a>

## Qué se incluye

Solamente las API públicas, en las versiones <em>menor</em> y <em>parche</em>, de TensorFlow son compatibles hacia atrás. Las API públicas están compuestas de la siguiente manera:

- Contienen todas las funciones y clases de [Python](https://www.tensorflow.org/api_docs/python) documentadas en el módulo `tensorflow` y en sus submódulos, excepto por lo siguiente:

    - Símbolos privados: cualquier función, clase, etc. cuyo nombre comience con `_`
    - Símbolos experimentales y `tf.contrib`. Para otros detalles, consulte [más adelante](#not_covered).

    Tenga en cuenta que el código en los directorios `examples/` y `tools/` no se puede alcanzar con el módulo `tensorflow` en Python y, por lo tanto, no se incluye en la garantía de compatibilidad.

    Si hay un símbolo disponible a través del módulo `tensorflow` en Python o de sus submódulos, pero no está documentado, entonces **no** se considera parte de la API pública.

- La API de compatibilidad (en Python, el módulo `tf.compat`). A versiones <em>mayores</em>, se pueden lanzar utilidades y otros <em>endpoints</em> para ayudar a los usuarios con la transición a una nueva versión <em>mayores</em>. Estos símbolos de API ya son obsoletos y no tienen más soporte (es decir, no les agregaremos ninguna función, ni repararemos los errores excepto por atender las vulnerabilidades), pero sí rigen para ellos nuestras garantías de compatibilidad.

- La API C de TensorFlow:

    - [tensorflow/c/c_api.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h))

- La API C de TensorFlow Lite:

    - [tensorflow/lite/c/c_api.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api.h)
    - [tensorflow/lite/c/c_api_types.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_types.h).

- Los siguientes archivos de búfer de protocolo:

    - [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    - [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    - [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    - [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    - [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    - [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    - [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    - [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    - [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    - [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>

## Lo que *no* se incluye

Algunas partes de TensorFlow pueden cambiar, en cualquier punto, con formas incompatibles hacia atrás. Entre ellas:

- Las **API** experimentales: para facilitar el desarrollo, exceptuamos de las garantías de compatibilidad algunos símbolos de API claramente marcados como experimentales. En particular lo siguiente no está cubierto por ninguna garantía de compatibilidad:

    - cualquier símbolo en el módulo `tf.contrib` o en sus submódulos;
    - cualquier símbolo (módulo, función, argumento, propiedad, clase o constante) cuyo nombre contenga `experimental` o `Experimental`; o
    - cualquier símbolo cuyo nombre totalmente calificado incluya un módulo o clase que sea experimental. Incluidos los campos y submensajes de cualquier búfer de protocolo llamado `experimental`.

- **Otros lenguajes**: Las API de TensorFlow API en otros lenguajes diferentes de Python o C, como:

    - [C++](../install/lang_c.ipynb) (expuestos a través de archivos de encabezado en [`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc)).
    - [Java](../install/lang_java_legacy.md),
    - [Go](https://github.com/tensorflow/build/blob/master/golang_install_guide/README.md)
    - [JavaScript](https://www.tensorflow.org/js)

- **Detalles de op (operaciones) compuestas:** Muchas funciones públicas de Python se expanden a varias operaciones primitivas en el grafo, y estos detalles serán parte de cualquiera de los grafos guardados en el disco, como los `GraphDef`. Estos detalles pueden cambiar por lanzamientos <em>menores</em>. En particular, las pruebas de regresión con las que se controla la coincidencia exacta entre grafos tienen la tendencia a dividirse en lanzamientos <em>menores</em>, a pesar de que el comportamiento del grafo deba permanecer sin cambios y los puntos de verificación existentes sigan funcionando.

- **Detalles numéricos de punto flotante:** Los valores de puntos flotantes específicos calculados por operaciones pueden cambiar en cualquier momento. Los usuarios deberían confiar solamente en la exactitud y la estabilidad numérica aproximadas, no en los bits específicos calculados. Los cambios de fórmulas numéricas en lanzamientos <em>menores</em> y <em>parche</em> deberían dar por resultado una exactitud similar o mejorada. Con la salvedad de que en el aprendizaje automático la exactitud mejorada de fórmulas específicas puede generar una exactitud inferior para el sistema en general.

- **Números aleatorios:** Los números aleatorios calculados pueden cambiar en cualquier momento. Los usuarios deberán depender solamente de distribuciones aproximadamente correctas y de la solidez estadística, no de los bits específicos calculados. Para más detalles, consulte la guía sobre [generación de números aleatorios](random_numbers.ipynb).

- **Asimetría de versiones en Tensorflow distribuido:** si se ejecutan dos versiones diferentes de TensorFlow en un solo clúster, no tiene soporte. No hay garantías sobre la compatibilidad hacia atrás del protocolo de cableado.

- **Errores (<em>bugs</em>):** nos reservamos el derecho de hacer cambios por comportamientos de incompatibilidad hacia atrás (aunque no de API) si la implementación actual está claramente interrumpida, es decir, si contradice la documentación o si un comportamiento intencional reconocido y bien definido no se implementa como es debido a causa de un error (<em>bug</em>). Por ejemplo, si un optimizador pretende implementar un algoritmo de optimización reconocido, pero no coincide con ese algoritmo debido a un error, corregiremos el optimizador. Nuestra corrección podría romper código por confiar en el comportamiento erróneo para la convergencia. Notificaremos tales cambios en las notas de lanzamiento.

- **API sin uso:** nos reservamos el derecho de hacer cambios por comportamientos de incompatibilidad hacia atrás a las API para las que no encontramos usos documentados (mediante una auditoría del uso de TensorFlow con búsqueda en GitHub ). Antes de hacer cualquier cambio de ese tipo, anunciaremos nuestra intención de hacerlos en la [lista de correos announce@](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce). Brindaremos las instrucciones sobre cómo abordar cualquier rotura (en caso de ser necesario) y esperaremos dos semanas para darle a nuestra comunidad la oportunidad de compartir sus comentarios al respecto.

- **Comportamiento de errores:** podríamos reemplazar errores por comportamientos sin errores. Por ejemplo, podríamos cambiar una función para que calcule un resultado en vez de generar un error, incluso aunque ese error quede documentado. También nos reservamos el derecho de cambiar el texto de los mensajes de error. Además, el tipo de error puede cambiar a menos que se especifique en la documentación el tipo de excepción para una condición de error específica.

<a name="compatibility_of_graphs_and_checkpoints"></a>

## Compatibilidad de SavedModels, grafos y puntos de verificación

SavedModel es el formato de serialización preferido para usar en los programas de TensorFlow. Los SavedModels contienen dos partes: uno o más grafos codificados como `GraphDefs` y un <em>Checkpoint</em> (punto de verificación). Los grafos describen el flujo de datos de las operaciones que se ejecutarán y los puntos de verificación contienen los valores de tensores guardados de las variables de un grafo.

Muchos usuarios de TensorFlow crean SavedModels, y los cargan y los ejecutan con un lanzamiento posterior de TensorFlow. De conformidad con [semver](https://semver.org), los SavedModels escritos con una versión de TensorFlow se pueden cargar y evaluar con una versión posterior de TensorFlow que tenga el mismo lanzamiento <em>mayor</em>.

Ofrecemos garantías extra para SavedModels *compatibles*. Cuando hablamos de un SavedModel que se creó con **API de no compatibilidad, no experimentales y no obsoletas** en la versión <em>mayor</em> de TensorFlow `N` nos referimos a un *SavedModel* compatible con la versión `N`. Cualquier SavedModel compatible con la versión <em>mayor</em> `N` en TensorFlow se puede cargar y ejecutar con la versión <em>mayor</em> `N+1` de TensorFlow. Sin embargo, la funcionalidad requerida para construir o modificar un modelo como este puede ya no estar disponible, en cuyo caso la garantía solamente regirá para el SavedModel sin modificar.

Nos esforzaremos por preservar la compatibilidad hacia atrás el mayor tiempo posible, de modo que los archivos serializados puedan usarse durante períodos prolongados.

### Compatibilidad con GraphDef

Los grafos se serializan a través del búfer de protocolo `GraphDef`. Para facilitar los cambios de incompatibilidad hacia atrás de los grafos, cada `GraphDef` tiene un número de versión separado del de la versión de TensorFlow. Por ejemplo, la versión 17 de `GraphDef` dejó obsoleta la op `inv` a cambio de `reciprocal`. La semántica es la siguiente:

- Cada versión de TensorFlow es compatible con un intervalo de las versiones de `GraphDef`. Este intervalo será constante en los lanzamientos <em>parche</em> y solamente podrá llegar hasta lanzamientos <em>menores</em>. Pero la compatibilidad para una versión `GraphDef` ya no estará vigente para los casos de lanzamientos <em>mayores</em> de TensorFlow (alineado solamente con la compatibilidad de versiones garantizada para SavedModels).

- Los grafos nuevos que se crean se asignan al último número de versión de `GraphDef`.

- Si una versión dada de TensorFlow es compatible con la versión de `GraphDef` de un grafo, se cargará y evaluará con el mismo comportamiento que la versión de TensorFlow que se usó para generarlo (excepto por los detalles numéricos de punto flotante y los números aleatorios, tal como se explicó arriba), independientemente de cuál sea la versión <em>mayor</em> de TensorFlow. En particular, un GraphDef que es compatible con un archivo de punto de verificación en una versión de TensorFlow (como es en el caso de un SavedModel) seguirá siendo compatible con ese punto de verificación y con las versiones posteriores, siempre que el GraphDef siga recibiendo soporte.

    Tenga en cuenta que solamente se aplica a grafos serializados en GraphDefs (y SavedModels): el *código* que lee un punto de verificación puede no estar disponible para leer los puntos de verificación generados por el mismo código que ejecutan una versión diferente de TensorFlow.

- Si la cota <em>superior</em> de `GraphDef` se aumenta a X en un lanzamiento (<em>menor</em>), al menos, pasarán seis meses antes de que la cota *inferior* se aumente a X.  Por ejemplo (aquí usamos números de versiones hipotéticas):

    - TensorFlow 1.2 puede ser compatible con las versiones 4 a 7 de `GraphDef`.
    - TensorFlow 1.3 podría agregar la versión 8 de `GraphDef` y ser compatible con las versiones 4 a 8.
    - Al menos seis meses después, TensorFlow 2.0.0 podría dejar de ser compatible con las versiones 4 a 7, dejando solamente la versión 8.

    Tenga en cuenta que dado que las versiones <em>mayor</em> de TensorFlow, por lo general, se publican con más de 6 meses de separación, las garantías para SavedModels compatibles detalladas anteriormente son mucho más persistentes y duran más que los 6 meses de garantía de GraphDefs.

Finalmente, cuando se deje de brindar soporte a una versión de un `GraphDef`, intentaremos brindar las herramientas necesarias para convertir automáticamente los grafos a las nuevas versiones de `GraphDef` compatibles.

## Compatibilidad de grafos y puntos de verificación cuando se extiende TensorFlow

Esta sección es relevante solamente cuando se hacen cambios incompatibles en el formato del `GraphDef`, tales como agregar o quitar ops, o cambiar la funcionalidad de ops existentes. La sección anterior probablemente sea suficiente para la mayoría de los usuarios.

<a id="backward_forward"></a>

### Compatibilidad hacia atrás y parcial hacia adelante

Nuestro esquema de versionado tiene tres requerimientos:

- **Compatibilidad hacia atrás** para apoyar la carga de grafos y puntos de verificación creados con versiones anteriores de TensorFlow.
- **Compatibilidad hacia adelante** para escenarios en que el productor de un grafo o de un punto de verificación cambia a una versión más nueva de TensorFlow antes que el consumidor.
- Se adapta TensorFlow a la evolución en casos de incompatibilidad. Por ejemplo, se quitan operaciones, y se agregan o eliminan atributos.

Advierta que si bien el mecanismo de versionado de  `GraphDef` está separado de las versiones de TensorFlow, los cambios en el formato `GraphDef` que son incompatibles hacia atrás aún permanecen restringidos por el versionado semántico. Significa que la funcionalidad solamente se puede eliminar o cambiar entre versiones `MAJOR` de TensorFlow (como de `1.7` a `2.0`). Además, la compatibilidad hacia adelante se aplica dentro de los lanzamientos parche (por ejemplo, de `1.x.1` a `1.x.2`).

Para lograr la compatibilidad hacia atrás y hacia adelante, y saber cuándo aplicar los cambios en los formatos, grafos y puntos de verificación, conserve los metadatos que describan cuándo se produjeron. En las siguientes secciones se detallan la implementación e instrucciones de TensorFlow para las versiones de `GraphDef` que evolucionan.

### Esquemas de versionado de datos independientes

Hay diferentes versiones de datos para grafos y puntos de verificación. Los dos formatos de datos evolucionan a velocidades diferentes entre sí y con respecto a TensorFlow. Ambos sistemas de versionado se definen en [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h). Cada vez que se agrega una versión nueva, también se agrega una nota en el encabezado indicando qué cambió y en qué fecha.

### Datos, productores y consumidores

Distinguimos los siguientes tipos de información sobre versiones de datos:

- **Productores**: binarios que producen datos. Los productores tienen una versión propia (`producer`) y una versión de consumidor mínima con las que son compatibles (`min_consumer`).
- **Consumidores**: binarios que consumen datos. Los consumidores tienen una versión propia (`consumer`) y una versión de productor mínima con las que son compatibles (`min_producer`).

Cada porción de datos versionados tiene un campo [`VersionDef versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto) en el que se registra el `producer` que generó los datos, el `min_consumer` con el que es compatible y una lista de versiones de `bad_consumers` que no se admiten.

Por defecto, cuando un productor genera algunos datos, esos datos heredan las versiones de `producer` y `min_consumer` del productor. Los `bad_consumers` se pueden preparar si se sabe que hay versiones de consumidor específicas que contienen errores y que se deben evitar. Un consumidor puede aceptar una porción de datos si lo siguiente es verdadero:

- `consumer` &gt;= `min_consumer` de los datos
- `producer` de los datos&gt;= `min_producer` del consumidor
- `consumer` que no se encuentra en los `bad_consumers` de los datos

Dado que tanto los productores como los consumidores parten de la misma base de código de TensorFlow, [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) contiene una versión principal de los datos que se trata como `producer` o como `consumer`, dependiendo del contexto y del `min_consumer` y el `min_producer` (necesarios para los productores y consumidores respectivamente). Para ser más específicos:

- Para las versiones `GraphDef`, tenemos `TF_GRAPH_DEF_VERSION`, `TF_GRAPH_DEF_VERSION_MIN_CONSUMER` y `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`.
- Para las versiones de puntos de verificación, tenemos `TF_CHECKPOINT_VERSION`, `TF_CHECKPOINT_VERSION_MIN_CONSUMER` y `TF_CHECKPOINT_VERSION_MIN_PRODUCER`.

### Agregado de un atributo nuevo que propone por defecto a una operación existente

Las siguientes instrucciones brindan compatibilidad hacia adelante solamente si el conjunto de operaciones no ha cambiado:

1. Si desea contar con compatibilidad hacia adelante, establezca `strip_default_attrs` como `True` cuando exporta el modelo con los métodos `tf.saved_model.SavedModelBuilder.add_meta_graph_and_variables` y `tf.saved_model.SavedModelBuilder.add_meta_graph` de la clase `SavedModelBuilder`, o con `tf.estimator.Estimator.export_saved_model`
2. Esto quita los atributos con valoraciones predeterminadas al momento de producir o exportar los modelos. De este modo, se garantiza que el `tf.MetaGraphDef` exportado no contendrá el nuevo atributo de operación cuando se use el valor predeterminado.
3. Este control hará posible que los consumidores desactualizados (por ejemplo, los binarios de servicio que quedan retrasados con respecto a los binarios de entrenamiento) continúen cargando los modelos y se eviten interrupciones en el sistema de servicios del modelo.

### Versiones de GraphDef que evolucionan

En esta sección se explica cómo usar este mecanismo de versionado para generar diferentes tipos de cambios en el formato `GraphDef`.

#### Cómo agregar una operación

Agregue la op (operación) nueva a ambos, consumidores y productores, al mismo tiempo y no cambie ninguna versión de `GraphDef`. Este tipo de cambios es compatible hacia atrás automáticamente y no afecta al plan de compatibilidad hacia adelante, ya que los scripts productores existentes no usarán la nueva funcionalidad repentinamente.

#### Cómo agregar una op y cambiar <em>wrappers</em> de Python para usarla

1. Implemente una funcionalidad nueva e incremente la versión de `GraphDef`.
2. Si es posible hacer que los <em>wrappers</em> (envoltorios) usen la funcionalidad nueva solamente en casos que no hayan funcionado antes, es ahora cuando se pueden actualizar.
3. Cambie los <em>wrappers</em> de Python para usar la funcionalidad nueva. No incremente `min_consumer`, ya que los modelos que no usan esta op no se deberían romper.

#### Cómo quitar o restringir la funcionalidad de una op

1. Repare todos los <em>scripts</em> (no TensorFlow en sí) para no usar la operación o funcionalidad no autorizada.
2. Incremente la versión de `GraphDef` e implemente una nueva funcionalidad consumidora que anule la operación o funcionalidad eliminada para GraphDefs en la versión nueva y posteriores. De ser posible, haga que TensorFlow deje de producir `GraphDefs` con la funcionalidad prohibida. Para lograrlo, agregue [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009).
3. A los fines de ocuparse de la compatibilidad hacia atrás espere a un lanzamiento <em>mayor</em>.
4. Aumente `min_producer` a la versión GraphDef de (2) y quite la funcionalidad por completo.

#### Cómo cambiar la funcionalidad de una operación

1. Agregue una op similar nueva con el nombre `SomethingV2` o uno parecido y realice el proceso de agregarla y cambiar a <em>wrappers</em> de Python existentes para usarla. A fin de garantizar la compatibilidad hacia adelante use las comprobaciones sugeridas en [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py) al cambiar los <em>wrappers</em> de Python.
2. Quite la operación anterior (solamente se puede realizar con un cambio de versión <em>mayor</em> debido a la compatibilidad hacia atrás).
3. Aumente `min_consumer` para excluir consumidores con la operación anterior, vuelva a agregar la operación anterior como un alias para `SomethingV2` y realice el proceso para cambiar los <em>wrappers</em> de Python para usarla.
4. Realice el proceso para quitar `SomethingV2`.

#### Prohibición de una sola versión de consumidor no seguro

1. Actualice (<em>bump</em>) la versión de `GraphDef` y agregue la versión mala a `bad_consumers` para todos los GraphDefs nuevos. De ser posible, agregue los `bad_consumers` solamente a los GraphDefs que contengan una operación segura (<em>certain</em>) o una similar.
2. Si los consumidores existentes tienen una versión mala, sáquelos lo antes posible.

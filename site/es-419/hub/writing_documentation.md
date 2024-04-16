# Escribir documentación

Para contribuir a tfhub.dev, se debe proporcionar documentación en formato Markdown. Para obtener una descripción general completa del proceso de contribución de modelos a tfhub.dev, consulte la guía de [contribuir un modelo](contribute_a_model.md).

**Nota:** el término "editor" se usa en toda la documentación; hace referencia al propietario registrado de un modelo alojado en tfhub.dev.

## Tipos de documentación de Markdown

Hay 3 tipos de documentación de Markdown que se usan en tfhub.dev:

- Markdown del editor: información sobre un editor ([consulte la sintaxis de Markdown](#publisher))
- Markdown del modelo: información sobre un modelo específico y cómo usarlo ([consulte la sintaxis de Markdown](#model))
- Markdown de la colección: contiene información sobre una colección de modelos que define el editor ([consulte la sintaxis de Markdown](#collection)).

## Organización de contenidos

Se requiere la siguiente organización de contenido al contribuir al repositorio de [TensorFlow Hub GitHub](https://github.com/tensorflow/tfhub.dev):

- cada directorio de editores está en el directorio `assets/docs`
- cada directorio de editores contiene directorios `collections` y `models` opcionales
- cada modelo debe tener su propio directorio en `assets/docs/<publisher_name>/models`
- cada colección debe tener su propio directorio en `assets/docs/<publisher_name>/collections`

Los Markdown del editor no tienen versiones, mientras que los modelos sí pueden tener diferentes versiones. Cada versión de modelo requiere Markdown independiente con la versión que describe (es decir, 1.md, 2.md) en el nombre de archivo. Las colecciones tienen versiones, pero solo se admite una única versión (1.md).

Todas las versiones de un modelo determinado deben ubicarse en el directorio de modelos.

A continuación se muestra una ilustración de cómo se organiza el contenido de Markdown:

```
assets/docs
├── <publisher_name_a>
│   ├── <publisher_name_a>.md  -> Documentation of the publisher.
│   └── models
│       └── <model_name>       -> Model name with slashes encoded as sub-path.
│           ├── 1.md           -> Documentation of the model version 1.
│           └── 2.md           -> Documentation of the model version 2.
├── <publisher_name_b>
│   ├── <publisher_name_b>.md  -> Documentation of the publisher.
│   ├── models
│   │   └── ...
│   └── collections
│       └── <collection_name>
│           └── 1.md           -> Documentation for the collection.
├── <publisher_name_c>
│   └── ...
└── ...
```

## Formato de Markdown del editor {:#publisher}

La documentación del editor se declara en el mismo tipo de archivos de Markdown que los modelos, con diferencias sintácticas pequeñas.

La ubicación correcta para el archivo del editor en el repositorio de TensorFlow Hub es: [tfhub.dev/assets/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/&lt;publisher_id&gt;/&lt;publisher_id.md&gt;

Consulte el ejemplo de documentación mínima del editor para el editor "vtab":

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

En el ejemplo anterior se especifica la identificación del editor, el nombre del editor, la ruta al ícono que se debe usar y una documentación de Markdown de formato libre más extensa. Tenga en cuenta que la identificación del editor solo debe contener letras minúsculas, dígitos y guiones.

### Pautas sobre el nombre del editor

Su nombre de editor puede ser su nombre de usuario de GitHub o el nombre de la organización de GitHub que administra.

## Formato de Markdown de la página del modelo {:#model}

La documentación del modelo es un archivo Markdown con alguna sintaxis complementaria. Consulte el siguiente ejemplo para ver un ejemplo mínimo o [un archivo Markdown de ejemplo más realista](https://github.com/tensorflow/tfhub.dev/blob/master/examples/docs/tf2_model_example.md).

### Documentación de ejemplo

Una documentación del modelo de alta calidad contiene fragmentos de código, información sobre cómo se entrenó el modelo y su uso previsto. También se deben aprovechar las propiedades de metadatos específicas del modelo [que se explican a continuación](#metadata) para que los usuarios puedan encontrar sus modelos en tfhub.dev más rápido.

```markdown
# Module google/text-embedding-model/1

Simple one sentence description.

<!-- asset-path: https://path/to/text-embedding-model/model.tar.gz -->
<!-- task: text-embedding -->
<!-- fine-tunable: true -->
<!-- format: saved_model_2 -->

## Overview

Here we give more information about the model including how it was trained,
expected use cases, and code snippets demonstrating how to use the model:

```

### Implementaciones de modelos y agrupaciones de implementaciones

tfhub.dev permite publicar implementaciones de TF.js, TFLite y Coral de un SavedModel de TensorFlow.

La primera línea del archivo Markdown debe especificar el tipo de formato:

- `# Module publisher/model/version` para SavedModels
- `# Tfjs publisher/model/version` para implementaciones de TF.js
- `# Lite publisher/model/version` para implementaciones de Lite
- `# Coral publisher/model/version` para implementaciones de Coral

Es una buena idea que estos diferentes formatos del mismo modelo conceptual aparezcan en la misma página del modelo en tfhub.dev. Para asociar una implementación determinada de TF.js, TFLite o Coral a un modelo SavedModel de TensorFlow, especifique la etiqueta parent-model:

```markdown
<!-- parent-model: publisher/model/version -->
```

Es posible que a veces quiera publicar una o más implementaciones sin un SavedModel de TensorFlow. En ese caso, deberá crear un modelo de marcador de posición y especificar su identificador en la etiqueta `parent-model`. El Markdown de marcador de posición es idéntico al Markdown de modelo de TensorFlow, excepto que la primera línea es: `# Placeholder publisher/model/version` y no requiere la propiedad `asset-path`.

### Propiedades de metadatos específicas del Markdown de modelo {:#metadata}

Los archivos Markdown pueden contener propiedades de metadatos. Se usan para proporcionar filtros y etiquetas para ayudar a los usuarios a encontrar su modelo. Los atributos de metadatos se incluyen como comentarios de Markdown después de la breve descripción del archivo Markdown, por ejemplo

```markdown
# Module google/universal-sentence-encoder/1
Encoder of greater-than-word length text trained on a variety of data.

<!-- task: text-embedding -->
...
```

Se admiten las siguientes propiedades de metadatos:

- `format`: Para modelos de TensorFlow: el formato TensorFlow Hub del modelo. Los valores válidos son `hub` cuando se exporta el modelo mediante el [formato hub TF1](exporting_hub_format.md) heredado o `saved_model_2` cuando se exporta el modelo mediante un [TF2 SavedModel](exporting_tf2_saved_model.md).
- `asset-path`: la ruta remota a los activos del modelo real que se van a cargar y que puede leer cualquier usuario, como a un depósito de Google Cloud Storage. Se debe permitir que la URL se obtenga del archivo robots.txt (por este motivo, "https://github.com/.*/releases/download/.*" no es compatible ya que https://github.com/robots.txt lo prohíbe). Consulte [a continuación](#model-specific-asset-content) para obtener más información sobre el tipo de archivo y el contenido que se requiere.
- `parent-model`: Para modelos TF.js/TFLite/Coral: identificador del SavedModel/marcador de posición que lo acompañan
- `fine-tunable`: Booleano, si el usuario puede ajustar el modelo.
- `task`: el dominio del problema, por ejemplo, "incrustación de texto". Todos los valores que se admiten se definen en [task.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/task.yaml).
- `dataset`: el conjunto de datos en el que se entrenó el modelo, por ejemplo, "wikipedia". Todos los valores que se admiten se definen en[dataset.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/dataset.yaml).
- `network-architecture`: la arquitectura de red en la que se basa el modelo, por ejemplo, "mobilenet-v3". Todos los valores que se admiten se definen en [network_architecture.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/network_architecture.yaml).
- `language`: el código de idioma del idioma en el que se entrenó un modelo de texto, por ejemplo, "en". Todos los valores que se admiten se definen en [language.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/language.yaml).
- `license`: la licencia que se aplica al modelo, por ejemplo, "mit". La licencia asumida por defecto para un modelo publicado es la [llicencia Apache 2.0](https://opensource.org/licenses/Apache-2.0). Todos los valores que se admiten se definen en [licencia.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/license.yaml). Tenga en cuenta que la licencia `custom` requerirá una consideración especial caso por caso.
- `colab`: la URL HTTPS a un cuaderno que demuestra cómo se puede usar o entrenar el modelo ([ejemplo](https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/g3doc/en/hub/tutorials/bigbigan_with_tf_hub.ipynb) para [bigbigan-resnet50](https://tfhub.dev/deepmind/bigbigan-resnet50/1)). Debe dirigirse a `colab.research.google.com`. Tenga en cuenta que se puede acceder a los cuadernos de Jupyter alojados en GitHub a través de `https://colab.research.google.com/github/ORGANIZATION/PROJECT/ blob/master/.../my_notebook.ipynb`.
- `demo`: la URL HTTPS a un sitio web que demuestra cómo se puede usar el modelo TF.js ([ejemplo](https://teachablemachine.withgoogle.com/train/pose) para [posenet](https://tfhub.dev/tensorflow/tfjs-model/posenet/mobilenet/float/075/1/default/1)).
- `interactive-visualizer`: nombre del visualizador que debe incrustarse en la página del modelo, por ejemplo, "visión". Mostrar un visualizador permite a los usuarios explorar las predicciones del modelo de forma interactiva. Todos los valores que se admiten se definen en [Interactive_visualizer.yaml](https://github.com/tensorflow/tfhub.dev/blob/master/tags/interactive_visualizer.yaml).

Los tipos de documentación de Markdown admiten diferentes propiedades de metadatos obligatorias y opcionales:

Tipo | Obligatoria | Opcional
--- | --- | ---
Editor |  |
Colección | tarea | conjunto de datos, idioma,
: : : red de arquitectura: |  |
Marcador de posición | tarea | conjunto de datos, ajustable,
: : : visualizador interactivo, idioma, : |  |
: : : licencia, arquitectura de red : |  |
SavedModel | ruta de activos, tarea, | colab, conjunto de datos,
: : ajustable, formato : visualizador interactivo, idioma, : |  |
: : : licencia, arquitectura de red : |  |
Tfjs | ruta de activos, modelo primario | colab, demo, visualizador interactivo
Lite | ruta de activos, modelo primario | colab, visualizador interactivo
Coral | ruta de activos, modelo primario | colab, visualizador interactivo

### Contenido de activos específico del modelo

Según el tipo de modelo, se requieren los siguientes tipos de archivos y contenidos:

- SavedModel: un archivo tar.gz que tiene contenido como este:

```
saved_model.tar.gz
├── assets/            # Optional.
├── assets.extra/      # Optional.
├── variables/
│     ├── variables.data-?????-of-?????
│     └──  variables.index
├── saved_model.pb
├── keras_metadata.pb  # Optional, only required for Keras models.
└── tfhub_module.pb    # Optional, only required for TF1 models.
```

- TF.js: un archivo tar.gz que tiene contenido como este:

```
tf_js_model.tar.gz
├── group*
├── *.json
├── *.txt
└── *.pb
```

- TFLite: un archivo .tflite
- Coral: un archivo .tflite

Para archivos tar.gz: asumiendo que los archivos de su modelo están en el directorio `my_model` (por ejemplo `my_model/saved_model.pb` para SavedModels o `my_model/model.json` para modelos TF.js), puede crear un archivo tar.gz válido con la herramienta [tar](https://www.gnu.org/software/tar/manual/tar.html) a través de `cd my_model && tar -czvf ../model.tar.gz *`.

Por lo general, todos los archivos y directorios (ya sean comprimidos o sin comprimir) deben comenzar con un carácter de palabra, por lo que, por ejemplo, los puntos no son un prefijo válido de nombres de archivos/directorios.

## Formato de Markdown de la página de colección {:#collection}

Las colecciones son una característica de tfhub.dev que permite a los editores agrupar modelos relacionados para mejorar la experiencia de búsqueda del usuario.

Consulte la [lista de todas las colecciones](https://tfhub.dev/s?subtype=model-family) en tfhub.dev.

La ubicación correcta para el archivo de colección en el repositorio [github.com/tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) es [activos/docs](https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs)/<b>nombre_del_editor&gt;</b>/collections/<b>&lt;nombre_de_la_colección&gt;</b>/<b>1</b>.md

Aquí tiene un ejemplo mínimo que se incluiría en activos/docs/<b>vtab</b>/collections/<b>benchmark</b>/<b>1</b>.md. Tenga en cuenta que el nombre de la colección en la primera línea no incluye la parte de `collections/` que se incluyen en la ruta del archivo.

```markdown
# Collection vtab/benchmark/1
Collection of visual representations that have been evaluated on the VTAB
benchmark.

<!-- task: image-feature-vector -->

## Overview
This is the list of visual representations in TensorFlow Hub that have been
evaluated on VTAB. Results can be seen in
[google-research.github.io/task_adaptation/](https://google-research.github.io/task_adaptation/)

#### Models
|                   |
|-------------------|
| [vtab/sup-100/1](https://tfhub.dev/vtab/sup-100/1)   |
| [vtab/rotation/1](https://tfhub.dev/vtab/rotation/1) |
|------------------------------------------------------|
```

En el ejemplo se especifica el nombre de la colección, una breve descripción de una oración, los metadatos del dominio del problema y la documentación de Markdown de formato libre.

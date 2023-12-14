# Importar modelos basados en GraphDef de TensorFlow a TensorFlow.js

Los modelos basados en GraphDef de TensorFlow (por lo común creados a través de la API de Python) se pueden guardar en uno de los siguientes formatos:

1. [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load) de TensorFlow
2. Modelo <em>congelado</em>
3. [Módulo de Tensorflow Hub](https://www.tensorflow.org/hub/)

Todos los formatos anteriores se pueden convertir con el [conversor de TensorFlow.js](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) a un formato que se puede cargar directamente en TensorFlow.js para realizar inferencias.

(Nota: TensorFlow ha dejado de usar el formato de agrupamiento de sesiones. Por lo tanto, migre sus modelos al formato SavedModel).

## Requisitos

Para el proceso de conversión se requiere de un entorno Python. Probablemente, le resulte conveniente mantener uno aislado usando [pipenv](https://github.com/pypa/pipenv) o [virtualenv](https://virtualenv.pypa.io). Para instalar el conversor, ejecute el siguiente comando:

```bash
 pip install tensorflowjs
```

Importar un modelo de TensorFlow a TensorFlow.js es un proceso de dos pasos. Primero, hay que convertir un modelo TensorFlow.js al formato web y luego hay que cargarlo en TensorFlow.js.

## Paso 1: convertir un modelo de TensorFlow al formato web de TensorFlow.js

Ejecute el "script" del conversor provisto con el paquete pip:

Uso: ejemplo de un SavedModel:

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Ejemplo de un modelo <em>congelado</em>:

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Ejemplo de un módulo de Tensorflow Hub:

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

Argumentos posicionales | Descripción
--- | ---
`input_path` | La ruta completa del directorio del modelo guardado, del directorio del agrupamiento (<em>bundle</em>) de la sesión, del archivo del modelo congelado, o la ruta o handle del módulo de TensorFlow Hub.
`output_path` | La ruta para todos los artefactos de salida.

Opciones | Descripción
--- | ---
`--input_format` | El formato del modelo de entrada usa tf_saved_model para SavedModel, tf_frozen_model para el modelo congelado, tf_session_bundle para el agrupamiento de sesiones, tf_hub para módulos de TensorFlow Hub y keras para HDF5 de Keras.
`--output_node_names` | Los nombres de los nodos de salida, separados por comas.
`--saved_model_tags` | Solamente se usa para la conversión de SavedModel. Son etiquetas para el MetaGraphDef a cargar, en un formato separado por comas. Toma por defecto `serve`.
`--signature_name` | Solamente aplicable a la conversión de módulos de TensorFlow Hub (firma para cargar). Propone por defecto `default`. Consulte https://www.tensorflow.org/hub/common_signatures/.

Use el siguiente comando para obtener un mensaje de ayuda detallado:

```bash
tensorflowjs_converter --help
```

### Archivos generados por conversor

El script de conversión que vimos arriba produce dos tipos de archivos:

- `model.json` (el manifiesto de pesos y grafos del flujo de datos)
- `group1-shard\*of\*` (la colección de archivos de pesos binarios)

Por ejemplo, a continuación se encuentra la salida de la conversión de MobileNet v2:

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## Paso 2: cargar y ejecutar en el navegador

1. Instale el paquete npm del tfjs-converter

`yarn add @tensorflow/tfjs` o `npm install @tensorflow/tfjs`

1. Instancie [la clase FrozenModel](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts) y ejecute la inferencia.

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

Consulte nuestro [demo sobre MobileNet](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/demo/mobilenet).

La API `loadGraphModel` acepta un parámetro `LoadOptions` adicional, que se puede usar para enviar credenciales o encabezados personalizados junto con la solicitud. Para más detalles, consulte la [documentación sobre loadGraphModel()](https://js.tensorflow.org/api/1.0.0/#loadGraphModel).

## Operaciones admitidas

Actualmente, TensorFlow.js admite una cantidad limitada de operaciones de TensorFlow. Si su modelo usa operaciones que no son compatibles, el script `tensorflowjs_converter` fallará e imprimirá un lista de las operaciones incompatibles en el modelo. Informe el [problema](https://github.com/tensorflow/tfjs/issues) de cada una de estas operaciones para que sepamos con cuáles necesita ayuda.

## Carga de los pesos solos

Si prefiere cargar solamente los pesos, puede usar el siguiente fragmento de código.

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

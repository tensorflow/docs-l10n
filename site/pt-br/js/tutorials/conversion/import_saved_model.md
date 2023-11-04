# Como importar modelos baseados no TensorFlow GraphDef para TensorFlow.js

Os modelos baseados no TensorFlow GraphDef (geralmente criados via API do Python) podem ser salvos nos seguintes formatos:

1. [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load) do TensorFlow
2. Modelo congelado
3. [Módulo do Tensorflow Hub](https://www.tensorflow.org/hub/)

Todos os formatos acima podem ser convertidos pelo [conversor de TensorFlow.js](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) para um formato que pode ser carregado diretamente em TensorFlow.js para inferência.

Observação: o TensorFlow descontinuou o formato de pacote de sessão. Migre seus modelos para o formato SavedModel.

## Requisitos

O procedimento de conversão requer um ambiente Python. É interessante manter um ambiente isolado usando [pipenv](https://github.com/pypa/pipenv) ou [virtualenv](https://virtualenv.pypa.io). Para instalar o conversor, execute o seguinte comando:

```bash
 pip install tensorflowjs
```

Importar um modelo do TensorFlow para TensorFlow.js é um processo composto por duas etapas. Primeiro, converta um modelo existente para o formato web TensorFlow.js e depois carregue-o em TensorFlow.js.

## Etapa 1 – Converta um modelo do TensorFlow existente para o formato web TensorFlow.js

Execute o script conversor fornecido pelo pacote pip:

Uso – Exemplo com SavedModel:

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Exemplo com modelo congelado:

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Exemplo com Módulo do TensorFlow Hub

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

Argumentos posicionais | Descrição
--- | ---
`input_path` | Caminho completo do diretório do modelo salvo, diretório do pacote de sessão, arquivo do modelo congelado ou identificador ou caminho do módulo do TensorFlow Hub.
`output_path` | Caminho de todos os artefatos de saída.

Opções | Descrição
--- | ---
`--input_format` | Formato do modelo de entrada. Use tf_saved_model para SavedModel, tf_frozen_model para modelo congelado, tf_session_bundle para pacote de sessão, tf_hub para módulo do TensorFlow Hub e keras para HDF5 do Keras.
`--output_node_names` | Nomes dos nós de saída, separados por vírgula.
`--saved_model_tags` | Aplicável somente à conversão de SavedModel. Marcas do MetaGraphDef a carregar em formato separado por vírgula. O padrão é `serve`.
`--signature_name` | Aplicável somente à conversão de módulo do TensorFlow Hub. Assinatura a carregar. O padrão é `default`. Confira https://www.tensorflow.org/hub/common_signatures/.

Use o comando abaixo para ver uma mensagem de ajuda detalhada:

```bash
tensorflowjs_converter --help
```

### Arquivos gerados pelo conversor

O script de conversão acima gera dois tipos de arquivo:

- `model.json` (grafo do fluxo de dados e manifesto de pesos)
- `group1-shard\*of\*` (coleção de arquivos binários de pesos)

Por exemplo, veja abaixo a saída ao converter de MobileNet v2:

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## Etapa 2 – Carregue e execute no navegador

1. Instale o pacote npm tfjs-converter

`yarn add @tensorflow/tfjs` ou `npm install @tensorflow/tfjs`

1. Instancie a [classe FrozenModel](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts) e execute a inferência.

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

Confira nossa [demonstração de MobileNet](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/demo/mobilenet).

A API `loadGraphModel` aceita o parâmetro adicional `LoadOptions`, que pode ser usado para enviar credenciais ou cabeçalhos personalizados junto com a solicitação. Confira mais detalhes na [documentação de loadGraphModel()](https://js.tensorflow.org/api/1.0.0/#loadGraphModel).

## Operações permitidas

Atualmente, TensorFlow.js oferece suporte a um conjunto limitado de operações do TensorFlow. Se o seu modelo usar uma operação não permitida, haverá falha no script `tensorflowjs_converter`, será exibida via print uma lista das operações não permitidas em seu modelo. Registre um [issue](https://github.com/tensorflow/tfjs/issues) para cada operação para nos avisar de quais operações você precisa.

## Carregamento somente dos pesos

Se você preferir carregar somente os pesos, pode usar o trecho de código abaixo.

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

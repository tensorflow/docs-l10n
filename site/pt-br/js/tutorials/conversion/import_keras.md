# Como importar um modelo do Keras para TensorFlow.js

Os modelos do Keras (geralmente criados via API do Python) podem ser salvos em [diversos formatos](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).  O formato "modelo inteiro" pode ser convertido para o formato TensorFlow.js Layers, que pode ser carregado diretamente em TensorFlow.js para inferência ou mais treinamento.

O formato de destino TensorFlow.js Layers é um diretório contendo um arquivo `model.json` e um conjunto de arquivos de pesos fragmentados em formato binário. O arquivo `model.json` contém tanto a topologia do modelo (também chamada de "arquitetura" ou "grafo": uma descrição das camadas e como elas estão conectadas) e um manifesto dos arquivos de pesos.

## Requisitos

O procedimento de conversão requer um ambiente Python. É interessante manter um ambiente isolado usando [pipenv](https://github.com/pypa/pipenv) ou [virtualenv](https://virtualenv.pypa.io). Para instalar o conversor, use `pip install tensorflowjs`.

Importar um modelo do Keras para TensorFlow.js é um processo composto por duas etapas. Primeiro, converta um modelo do Keras existente para o formato TF.js Layers e depois carregue-o em TensorFlow.js.

## Etapa 1 – Converta um modelo do Keras existente para o formato TF.js Layers

Geralmente, os modelos do Keras são salvos via `model.save(filepath)`, o que gera um único arquivo HDF5 (.h5) contendo tanto a topologia do modelo quanto seus pesos. Para converter esse tipo de arquivo para o formato TF.js Layers, execute o comando abaixo, em que *`path/to/my_model.h5`* é o arquivo .h5 fonte do Keras, e *`path/to/tfjs_target_dir`* é o diretório de saída de destino para os arquivos TF.js:

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## Alternativa – Use a API do Python para exportar diretamente para o formato TF.js Layers

Se você tiver um modelo do Keras no Python, pode exportá-lo diretamente para o formato TensorFlow.js Layers da seguinte forma:

```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## Etapa 2 – Carregue o modelo em TensorFlow.js

Use um servidor web fornecer os arquivos convertidos do modelo que você gerou na etapa 1. Talvez você precise configurar o servidor para [permitir o compartilhamento de recursos com origens diferentes (CORS, na sigla em inglês)](https://enable-cors.org/) para que seja possível buscar os arquivos em JavaScript.

Em seguida, forneça a URL ao arquivo model.json para carregar o modelo em TensorFlow.js:

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

Agora o modelo está pronto para inferência, avaliação e retreinamento. Por exemplo, o modelo carregado pode ser usado imediatamente para fazer uma previsão:

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

Muitos dos [exemplos de TensorFlow.js](https://github.com/tensorflow/tfjs-examples) seguem esta estratégia de uso de modelos pré-treinados que foram convertidos e hospedados no Google Cloud Storage.

Você deve referenciar o modelo inteiro usando o nome de arquivo `model.json`. `loadModel(...)` busca `model.json` e depois faz solicitações HTTP(S) adicionais para obter os arquivos de pesos fragmentados referenciados no manifesto de pesos de `model.json`. Com esta estratégia, o navegador pode fazer cache de todos esses arquivos (e pode usar servidores de cache adicionais na Internet), pois `model.json` e os fragmentos de pesos são menores que o limite típico de tamanho de arquivos de cache. Portanto, é provável que um modelo seja carregado mais rapidamente em ocasiões subsequentes.

## Recursos disponíveis

Atualmente, o formato TensorFlow.js Layers tem suporte somente a modelos do Keras que usem os constructos padrão do Keras. Modelos que usem operações ou camadas sem suporte – como camadas personalizadas, camadas lambda, perdas personalizadas ou métricas personalizadas – não podem ser importadas automaticamente, pois dependem de código Python que não pode ser convertido em JavaScript de uma maneira confiável.

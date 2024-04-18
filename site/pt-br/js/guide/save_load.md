# Salvar e carregar modelos

O TensorFlow.js conta com funcionalidades para salvar e carregar modelos que foram criados com a API [`Layers`](https://js.tensorflow.org/api/0.14.2/#Models) ou convertidos a partir de modelos existentes do TensorFlow. Podem ser modelos que você mesmo treinou ou que tenham sido treinados por outras pessoas. Um importante benefício de se usar a API Layers é que os modelos criados com ela são serializáveis, e é esse aspecto que vamos explorar neste tutorial.

O foco deste tutorial é como salvar e carregar modelos do TensorFlow.js (identificáveis como arquivos JSON). Também podemos importar modelos do TensorFlow em Python. O carregamento desses modelos é discutido nos seguintes tutoriais:

- [Importar modelos do Keras](../tutorials/conversion/import_keras.md)
- [Importar modelos do Graphdef](../tutorials/conversion/import_saved_model.md)

## Salve um tf.Model

Tanto [`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) quanto [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model) contam com uma função [`model.save`](https://js.tensorflow.org/api/0.14.2/#tf.Model.save) que permite salvar a *topologia* e os *pesos* de um modelo.

- Topologia – É um arquivo que descreve a arquitetura de um modelo (isto é, quais operações ele usa). Contém referências aos pesos do modelo que são armazenados externamente.

- Pesos – São arquivos binários que armazenam os pesos de um determinado modelo em um formato eficiente. Geralmente, são armazenados na mesma pasta que a topologia.

Vamos conferir o código para salvar um modelo:

```js
const saveResult = await model.save('localstorage://my-model-1');
```

Algumas considerações:

- O método `save` recebe um argumento de string tipo URL que começa com um **esquema**. Ele descreve o tipo de destino no qual estamos tentando salvar o modelo. No exemplo acima, o esquema é `localstorage://`
- Após o esquema, temos o **caminho**. No exemplo acima, o caminho é `my-model-1`.
- O método `save` é assíncrono.
- O valor de retorno de `model.save` é um objeto JSON que armazena informações como os tamanhos em bytes da topologia e dos pesos do modelo.
- O ambiente usado para salvar o modelo não afeta em quais ambientes podemos carregá-lo. Salvar um modelo no Node.js não impede que ele seja carregado no navegador.

Vamos avaliar os diferentes esquemas disponíveis abaixo.

### Armazenamento local (somente navegador)

**Esquema:** `localstorage://`

```js
await model.save('localstorage://my-model');
```

O modelo é salvo com o nome `my-model` no [armazenamento local](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) do navegador. Ele será persistente entre as atualizações, embora o armazenamento local possa ser excluído pelos usuários ou pelo próprio navegador, caso o espaço de armazenamento se torne uma preocupação. Cada navegador também define seu próprio limite de quantos dados podem ser armazenados no armazenamento local para um determinado domínio.

### IndexedDB (somente navegador)

**Esquema:** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

O modelo é salvo no armazenamento [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) do navegador. Da mesma forma que o armazenamento local, ele também será persistente entre as atualizações. Além disso, ele costuma ter limites de tamanho maiores para os objetos armazenados.

### Downloads de arquivos (somente navegador)

**Esquema:** `downloads://`

```js
await model.save('downloads://my-model');
```

Esse código fará o navegador baixar os arquivos do modelo para a máquina do usuário. Serão gerados dois arquivos:

1. Um arquivo JSON de nome `[my-model].json` com a topologia e a referência para o arquivo de pesos descrito abaixo.
2. Um binário com os valores de pesos, com o nome `[my-model].weights.bin`.

Você pode alterar o nome `[my-model]` para baixar arquivos com um nome diferente.

Como o arquivo `.json` aponta para `.bin` usando um caminho relativo, os dois arquivos devem estar na mesma pasta.

> Observação: alguns navegadores exigem que os usuários concedam permissão antes que mais de um arquivo possa ser baixado ao mesmo tempo.

### Solicitação HTTP(S)

**Esquema:** `http://` ou `https://`

```js
await model.save('http://model-server.domain/upload')
```

Será criada uma solicitação web para salvar um modelo em um servidor remoto. Você deve controlar esse servidor remoto para garantir que ele consiga tratar a solicitação.

O modelo será enviado ao servidor HTTP especificado por uma solicitação [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST). O corpo da solicitação POST fica no formato `multipart/form-data` e consiste de dois arquivos:

1. Um arquivo JSON de nome `model.json` com a topologia e a referência para o arquivo de pesos descrito abaixo.
2. Um binário com os valores de pesos, com o nome `model.weights.bin`.

Observe que os nomes dos arquivos serão sempre como especificado acima (o nome está incorporado à função). Esta [documentação da API](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest) contém um trecho de código Python que demonstra como usar o framework web [flask](http://flask.pocoo.org/) para tratar a solicitação originada pelo `save`.

Geralmente, você precisará passar mais argumentos ou headers na solicitação para o servidor HTTP (por exemplo, para autenticação ou se desejar especificar uma pasta na qual o modelo deve ser salvo). Você pode obter controle refinado sobre esses aspectos das solicitações de `save` substituindo o argumento de string de URL em `tf.io.browserHTTPRequest`. Esta API oferece maior flexibilidade no controle de solicitações HTTP.

Por exemplo:

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

### Sistema de arquivos nativo (somente Node.js)

**Esquema:** `file://`

```js
await model.save('file:///path/to/my-model');
```

Ao executar no Node.js, também temos acesso direto ao sistema de arquivos e podemos salvar modelos nele. O comando acima salva dois arquivos no `path` (caminho) especificado após o `scheme` (esquema).

1. Um arquivo JSON de nome `[model].json` com a topologia e a referência para o arquivo de pesos descrito abaixo.
2. Um binário com os valores de pesos, com o nome `[model].weights.bin`.

Observe que os nomes dos arquivos serão sempre como especificado acima (o nome está incorporado à função).

## Como carregar um tf.Model

Dado um modelo salvo usando um dos métodos acima, podemos carregá-lo usando a API `tf.loadLayersModel`.

Vamos conferir o código para carregar um modelo:

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

Algumas considerações:

- Assim como `model.save()`, a função `loadLayersModel` recebe um argumento de string tipo URL que começa com um **esquema**. Ele descreve o tipo de destino a partir do qual estamos tentando carregar o modelo.
- Após o esquema, temos o **caminho**. No exemplo acima, o caminho é `my-model-1`.
- A string tipo URL pode ser substituída por um objeto que obedeça à interface IOHandler.
- A função `tf.loadLayersModel()` é assíncrona.
- O valor de retorno de `tf.loadLayersModel` é `tf.Model`

Vamos avaliar os diferentes esquemas disponíveis abaixo.

### Armazenamento local (somente navegador)

**Esquema:** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

Esse código carrega um modelo com nome `my-model` a partir do [armazenamento local](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage) do navegador.

### IndexedDB (somente navegador)

**Esquema:** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```

Esse código carrega um modelo a partir do armazenamento [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) do navegador.

### HTTP(S)

**Esquema:** `http://` ou `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```

Esse código carrega um modelo a partir de um endpoint HTTP. Após carregar o arquivo `json`, a função solicitará os arquivos `.bin` correspondentes que o arquivo `json` referencia.

> OBSERVAÇÃO: essa implementação depende da presença do método [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch). Se você estiver usando um ambiente que não forneça o método fetch nativamente, pode fornecer um método global chamado [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) que obedeça a essa interface ou pode usar uma biblioteca, como [`node-fetch`](https://www.npmjs.com/package/node-fetch).

### Sistema de arquivos nativo (somente Node.js)

**Esquema:** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

Ao executar no Node.js, também temos acesso direto ao sistema de arquivos e podemos carregar modelos de lá. Observe que, na chamada à função acima, referenciamos o arquivo model.json (enquanto, ao salvar, especificamos uma pasta). Os arquivos `.bin` correspondentes devem estar na mesma pasta que o arquivo `json`.

## Como carregar modelos com IOHandlers

Se os esquemas acima não forem suficientes para sua necessidade, você pode implementar um comportamento de carregamento personalizado com um `IOHandler`. Um `IOHandler` fornecido pelo TensorFlow.js é [`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles), que permite aos usuários do navegador carregarem arquivos do modelo no navegador. Confira mais informações na [documentação](https://js.tensorflow.org/api/latest/#io.browserFiles).

# Como salvar e carregar modelos com IOHandlers personalizados

Se os esquemas acima não forem suficientes para suas necessidades de carregamento ou salvamento, você pode implementar um comportamento de serialização personalizado com um `IOHandler`.

`IOHandler` é um objeto com um método `save` e `load`.

A função `save` recebe um parâmetro que obedece à interface [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) e deve retornar uma promise que resolve para um objeto [SaveResult](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L107).

A função `load` não recebe parâmetros e deve retornar uma promise que resolve para um objeto [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165). Esse é o mesmo objeto passado para `save`.

Confira um exemplo de como implementar um IOHandler em [BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts).

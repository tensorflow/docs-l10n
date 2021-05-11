# モデルの保存と読み込み

TensorFlow.js は、[`Layers`](https://js.tensorflow.org/api/0.14.2/#Models) API で作成されたモデルまたは既存の TensorFlow モデルから変換されたモデルを保存して読み込む機能を提供しています。これらは、自分でトレーニングしたモデルか、他の人がトレーニングしたモデルです。Layers API を使用する主な利点は、作成したモデルをシリアル化できることです。このチュートリアルでは、この方法について説明します。

このチュートリアルでは、TensorFlow.js モデル (JSON ファイルで識別可能) の保存と読み込みに焦点が当てられています。TensorFlow Python モデルをインポートすることも可能です。これらのモデルの読み込みは、次の 2 つのチュートリアルで解説されています。

- [Keras モデルをインポートする](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/js/tutorials/conversion/import_keras.md)
- [Graphdef モデルをインポートする](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/js/tutorials/conversion/import_saved_model.md)

## tf.Model の保存

[`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) と [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model) は両方とも関数 [`model.save`](https://js.tensorflow.org/api/0.14.2/#tf.Model.save) を提供しており、モデルの*トポロジ*と*重み*を保存できます。

- トポロジ: これは、モデルのアーキテクチャ (モデルが使用する演算) を説明するファイルです。外部に保存されているモデルの重みへの参照が含まれています。

- 重み: これらは、特定のモデルの重みを効率的な形式で保存するバイナリファイルです。通常、トポロジと同じフォルダに保存されます。

モデルを保存するためのコードを見てみましょう。

```js
const saveResult = await model.save('localstorage://my-model-1');
```

注意事項がいくつかあります。

- `save` メソッドは、**スキーム**で始まる URL のような文字列引数を取ります。スキームはモデルの保存先の種類を示します。上記の例では、`localstorage://` がスキームです。
- スキームの後には、**パス**が続きます。上記の例では、`my-model-1` がパスです。
- `save` メソッドは非同期です。
- `model.save` の戻り値は、モデルのトポロジのバイトサイズや重みなどの情報をもつ JSON オブジェクトです。
- モデルの保存に使用される環境は、モデルを読み込む環境には影響しません。node.js にモデルを保存しても、ブラウザへの読み込みが妨げられることはありません。

以下では、利用可能なさまざまなスキームを説明します。

### ローカルストレージ (ブラウザのみ)

**スキーム:** `localstorage://`

```js
await model.save('localstorage://my-model');
```

このスキームは、モデルを `my-model` という名前でブラウザの[ローカルストレージ](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)に保存します。これは更新の間も持続しますが、スペースが問題になる場合は、ユーザーまたはブラウザによってローカルストレージが消去されるようになっています。また、特定のドメインのローカルストレージに保存できるデータ量は、ブラウザごとに制限されています。

### IndexedDB (ブラウザのみ)

**スキーム:** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

このスキームは、モデルをブラウザの [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) ストレージに保存します。ローカルストレージと同様に、更新の間も保持されますが、保存されるオブジェクトのサイズに対する制限も大きくなる傾向があります。

### ファイルダウンロード (ブラウザのみ)

**スキーム:** `downloads://`

```js
await model.save('downloads://my-model');
```

このスキームでは、ブラウザによってモデルファイルがユーザーのマシンにダウンロードされます。以下の 2 つのファイルが作成されます。

1. `[my-model].json` という名前のテキスト JSON ファイル。これには、トポロジと、以下で説明する重みファイルへの参照が含まれます。
2. `[my-model].weights.bin` という名前の重み値を含むバイナリファイル。

名前 `[my-model]` を変更すると、別の名前のファイルを取得できます。

`.json` ファイルは相対パスを使用して `.bin` をポイントするため、2 つのファイルは同じフォルダにある必要があります。

> 注意: 一部のブラウザでは、複数のファイルを同時にダウンロードする前に、ユーザーに権限を付与しておく必要があります。

### HTTP(S) リクエスト

**スキーム:** `http://` または `https://`

```js
await model.save('http://model-server.domain/upload')
```

このスキームは、モデルをリモートサーバーに保存する Web リクエストを作成します。リクエストを確実に処理できるようにリモートサーバーを制御する必要があります。

モデルは、[POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) リクエストを介して指定された HTTP サーバーに送信されます。POST の本文は `multipart/form-data` 形式で、2 つのファイルで構成されています。

1. `[my-model].json` という名前のテキスト JSON ファイル。これには、トポロジと、以下で説明する重みファイルへの参照が含まれます。
2. `model.weights.bin` という名前の重み値を含むバイナリファイル。

2 つのファイルの名前は常に上記で指定されたとおりになることに注意してください（名前は関数に組み込まれています）。この [API ドキュメント](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest)には、<a>Flask</a> ウェブフレームワークを使用して <code>save</code> からのリクエストを処理する方法を示す Python コードスニペットが含まれています。

多くの場合、HTTP サーバーに追加の引数またはリクエストヘッダーを渡す必要があります（認証、またはモデルを保存するフォルダを指定するため）。`tf.io.browserHTTPRequest` の URL 文字列引数を置き換えることにより、`save` からのリクエストのこれらの側面を細かく制御できます。この API は、HTTP リクエストを制御する際の柔軟性を高めます。

例:

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

### ネイティブファイルシステム (Node.js のみ)

**スキーム:** `file://`

```js
await model.save('file:///path/to/my-model');
```

Node.js で実行すると、ファイルシステムに直接アクセスしてモデルを保存することもできます。上記のコマンドは、`scheme` の後に指定された `path` に 2 つのファイルを保存します。

1. `[model].json` という名前のテキスト JSON ファイル。これには、以下で説明するトポロジと重みファイルへの参照が含まれます。
2. `[model].weights.bin` という名前の重み値を含むバイナリファイル。

2 つのファイルの名前は常に上記で指定されたとおりになることに注意してください（名前は関数に組み込まれています）。

## tf.Model モデルの読み込み

上記のメソッドのいずれかを使用して保存されたモデルの場合、`tf.loadLayersModel`API を使用して読み込むことができます。

モデルを読み込むためのコードを見てみましょう。

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

注意事項がいくつかあります。

- `model.save()` と同様に、`loadLayersModel` 関数は、**スキーム**で始まる URL のような文字列引数を取ります。これは、モデルの読み込み先の種類を示しています。
- スキームの後には、**パス**が続きます。上記の例では、`my-model-1` がパスです。
- URL のような文字列は、IOHandler インターフェースに一致するオブジェクトに置き換えることができます。
- `tf.loadLayersModel()` 関数は非同期です。
- `tf.loadLayersModel` の戻り値は `tf.Model`です。

以下では、利用可能なさまざまなスキームを説明します。

### ローカルストレージ (ブラウザのみ)

**スキーム:** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

このスキームは、`my-model` という名前のモデルをブラウザの[ローカルストレージ](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)から読み込みます。

### IndexedDB (ブラウザのみ)

**スキーム:** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```

このスキームは、ブラウザの [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) ストレージからモデルを読み込みます。

### HTTP(S)

**スキーム:** `http://` または `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```

このスキームは、http エンドポイントからモデルを読み込みます。`json` ファイルを読み込んだ後、関数は、`json` ファイルが参照する、対応する `.bin` ファイルをリクエストします。

> 注意: この実装は [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) メソッドがあることに依存しています。ネイティブで fetch メソッドを提供していない環境を使用している場合は、そのインターフェースを満たす [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) を提供するか、`node-fetch` のようなライブラリを使用してください[https://www.npmjs.com/package/node-fetch]。

### ネイティブファイルシステム (Node.js のみ)

**スキーム:** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

Node.js で実行すると、ファイルシステムに直接アクセスすることもでき、そこからモデルを読み込めます。上記の関数呼び出しでは、model.json ファイル自体を参照していることに注意してください（保存時にフォルダを指定します）。対応する`.bin`ファイルは、`json`ファイルと同じフォルダにある必要があります。

## IOHandler を使用したモデルの読み込み

上記のスキームを使ってニーズに対応できない場合は、`IOHandler` を使用してカスタムの読み込み機能を実装できます。TensorFlow.js が提供する  `IOHandler` には [`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles) があり、ブラウザのユーザーがブラウザにモデルファイルをアップロードできるようになっています。詳細については[ドキュメント](https://js.tensorflow.org/api/latest/#io.browserFiles)を参照してください。

# カスタム IOHandler を使用したモデルの保存と読み込み

上記のスキームが読み込みや保存のニーズを十分に満たさない場合は、`IOHandler` を実装することで、カスタムのシリアル化動作を実装できます。

`IOHandler` は、`save` および `load` メソッドを持つオブジェクトです。

`save` 関数は、<a>ModelArtifacts</a> インターフェースと一致するパラメータを 1 つ取り、<a>SaveResult</a> オブジェクトに解決される promise を返す必要があります。

`load` 関数はパラメータを取らず、[ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) オブジェクトに解決される promise を返す必要があります。これは、`save` に渡されるオブジェクトと同じです。

IOHandler の実装方法の例については、[BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts) を参照してください。

# 모델 저장하기 및 로드하기

TensorFlow.js는 [`Layers`](https://js.tensorflow.org/api/0.14.2/#Models) API로 생성되었거나 기존 TensorFlow 모델에서 변환된 모델을 저장하고 로드하는 기능을 제공합니다. 이러한 모델은 본인이 직접 훈련한 모델이거나 타인이 훈련한 모델일 수 있습니다. Layers API를 사용 시 얻을 수 있는 주요 이점은 만든 모델을 직렬화할 수 있다는 점이며 해당 내용을 이 튜토리얼에서 살펴보고자 합니다.

이 튜토리얼은 TensorFlow.js 모델(JSON 파일로 식별 가능) 저장 및 로드에 중점을 둡니다. TensorFlow Python 모델을 가져올 수도 있습니다. 이러한 모델 로드는 다음 두 가지 튜토리얼에서 다루고 있습니다.

- [Keras 모델 가져오기](../tutorials/conversion/import_keras.md)
- [Graphdef 모델 가져오기](../tutorials/conversion/import_saved_model.md)

## tf.Model 저장하기

[`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) 및 [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model) 모두 모델의 <em>토폴로지</em>와 <em>가중치</em>를 저장할 수 있는 <a><code>model.save</code></a> 함수를 제공합니다.

- 토폴로지: 모델의 아키텍처(즉, 사용하는 연산)를 설명하는 파일입니다. 해당 파일은 외부에 저장된 모델의 가중치에 대한 참조를 포함합니다.

- 가중치: 주어진 모델의 가중치를 효율적인 형식으로 저장하는 바이너리 파일입니다. 일반적으로 토폴로지와 같은 폴더에 저장됩니다.

모델을 저장하는 코드가 어떻게 보이는지 살펴보겠습니다.

```js
const saveResult = await model.save('localstorage://my-model-1');
```

다음은 몇 가지 유의 사항입니다.

- `save` 메서드는 **scheme로** 시작하는 URL과 유사한 문자열 인수를 사용하며 이는 모델을 저장하려는 대상 유형을 설명합니다. 위의 예에서 체계는 `localstorage://`입니다.
- 체계 뒤에는 **경로**가 있습니다. 위의 예에서 경로는 `my-model-1`입니다.
- `save` 메서드는 비 동기식입니다.
- `model.save`의 반환 값은 모델 토폴로지 및 가중치의 바이트 크기와 같은 정보를 전달하는 JSON 객체입니다.
- 모델을 저장하는 데 사용되는 환경은 모델을 로드할 수 있는 환경에 영향을 주지 않습니다. node.js에 모델을 저장해도 저장된 모델이 브라우저에 로드되는 것을 막지는 않습니다.

아래에서 사용 가능한 다양한 체계를 살펴보겠습니다.

### 로컬 저장소(브라우저 전용)

**체계:** `localstorage://`

```js
await model.save('localstorage://my-model');
```

브라우저의 [로컬 저장소](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)에 `my-model`이라는 이름으로 모델이 저장됩니다. 공간이 문제가 되는 경우 사용자 또는 브라우저 자체가 로컬 저장소를 비울 수 있지만, 이는 새로 고침간에 계속 유지됩니다. 또한 각 브라우저는 특정 도메인의 로컬 저장소에 저장할 수 있는 데이터양에 대한 자체 제한을 설정합니다.

### IndexedDB(브라우저 전용)

**체계:** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

모델이 브라우저의 [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) 저장소에 저장됩니다. 로컬 저장소와 마찬가지로 새로 고침 사이에 유지되지만, 이는 저장된 객체의 크기에 더 큰 제한을 주게 됩니다.

### 파일 다운로드(브라우저 전용)

**체계:** `downloads://`

```js
await model.save('downloads://my-model');
```

브라우저가 모델 파일을 사용자의 컴퓨터로 다운로드합니다. 두 개의 파일이 생성됩니다.

1. `[my-model].json` 이라는 텍스트 JSON 파일(아래 설명된 가중치 파일에 대한 토폴로지 및 참조를 전달함)
2. `[my-model].weights.bin` 이라는 가중치 값을 전달하는 바이너리 파일

`[my-model]` 이름을 변경하여 다른 이름의 파일을 가져올 수 있습니다.

`.json` 파일은 상대 경로를 사용하여 `.bin` 파일을 가리키므로 두 파일은 같은 폴더에 있어야 합니다.

> 참고: 일부 브라우저에서는 두 개 이상의 파일을 동시에 다운로드하기 전에 사용자에게 권한을 부여해야 합니다.

### HTTP(S) 요청

**체계:** `http://` 또는 `https://`

```js
await model.save('http://model-server.domain/upload')
```

모델을 리모트 서버에 저장하도록 웹 요청이 생성됩니다. 요청이 처리될 수 있도록 사용자가 리모트 서버를 제어해야 합니다.

모델은 [POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) 요청을 통해 지정된 HTTP 서버로 전송됩니다. POST의 본문은 `multipart/form-data` 형식이며 두 개의 파일로 구성됩니다.

1. `model.json` 이라는 텍스트 JSON 파일(아래 설명된 가중치 파일에 대한 토폴로지 및 참조를 전달함)
2. `model.weights.bin` 이라는 가중치 값을 전달하는 바이너리 파일

두 파일의 이름은 항상 위에 지정된 것과 정확히 일치합니다(이름은 함수에 내장됨). 이 [api 문서](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest)에는 [플라스크](http://flask.pocoo.org/) 웹 프레임워크를 사용하여 `save`에서 시작된 요청을 처리하는 방법을 보여주는 Python 코드 조각이 포함되어 있습니다.

더 많은 인수 또는 요청 헤더를 HTTP 서버에 전달해야 하는 경우가 종종 있습니다(예: 인증하거나 모델을 저장해야 하는 폴더를 지정하려는 경우). 요청의 이런 측면도 `tf.io.browserHTTPRequest`의 URL 문자열 인수를 대체하여 `save`에서 세밀하게 제어할 수 있습니다. 이 API는 HTTP 요청을 제어하는 데 더 큰 유연성을 제공합니다.

아래 예제를 봅시다.

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```

### 네이티브 파일 시스템(Node.js만 해당)

**체계:** `file://`

```js
await model.save('file:///path/to/my-model');
```

Node.js에서 실행할 때 파일 시스템에 직접 액세스할 수 있으며 모델을 저장할 수 있습니다. 위의 명령은 `scheme`에 따라 지정된 `path`에 두 개의 파일을 저장합니다.

1. `[model].json` 이라는 텍스트 JSON 파일(아래 설명된 가중치 파일에 대한 토폴로지 및 참조를 전달함)
2. `[model].weights.bin` 이라는 가중치 값을 포함하는 바이너리 파일

두 파일의 이름은 항상 위에 지정된 것과 정확히 일치합니다(이름은 함수에 내장됨).

## tf.Model 로드하기

위의 방법 중 하나를 사용하여 저장된 모델이 주어지면 `tf.loadLayersModel` API를 사용하여 로드할 수 있습니다.

모델을 로드하는 코드가 어떻게 보이는지 살펴보겠습니다.

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

다음은 몇 가지 유의 사항입니다.

- 마찬가지로 `model.save()`의 `loadLayersModel` 함수는 **체계**와 시작하는 URL과 같은 문자열 인수를 취하며, 모델을 로드하려는 대상 유형을 설명합니다.
- 체계 뒤에는 **경로**가 있습니다. 위의 예에서 경로는 `my-model-1`입니다.
- URL과 유사한 문자열은 IOHandler 인터페이스와 일치하는 객체로 대체될 수 있습니다.
- `tf.loadLayersModel()` 함수는 비 동기식입니다.
- `tf.loadLayersModel`의 반환 값은 `tf.Model`입니다.

아래에서 사용 가능한 다양한 체계를 살펴보겠습니다.

### 로컬 저장소(브라우저 전용)

**체계:** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

브라우저의 [로컬 저장소](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)에서 `my-model` 이라는 모델이 로드됩니다.

### IndexedDB(브라우저 전용)

**체계:** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```

브라우저의 [IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) 저장소에서 모델이 로드됩니다.

### HTTP(S)

**체계:** `http://` 또는 `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```

http 엔드 포인트에서 모델을 로드합니다. `json` 파일을 로드한 후 함수는 `json` 파일이 참조하는 해당 `.bin` 파일을 요청합니다.

> 참고: 이 구현 방식은 [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) 메서드의 존재에 의존합니다. 기본적으로 fetch 메서드를 제공하지 않는 환경에 있는 경우 해당 인터페이스를 충족하는 전역 메서드 이름 [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)를 제공하거나 (`node-fetch`)와 같은 라이브러리를 사용할 수 있습니다. [https://www.npmjs.com/package/node-fetch]

### 네이티브 파일 시스템(Node.js만 해당)

**체계:** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

Node.js에서 실행할 때 파일 시스템에 직접 액세스할 수 있으며 여기에서 모델을 로드 할 수 있습니다. 위의 함수 호출에서 model.json 파일 자체를 참조합니다(저장할 때 폴더를 지정함). 해당 `.bin` 파일은 `json` 파일과 같은 폴더에 있어야 합니다.

## IOHandlers로 모델 로드하기

위의 체계에서 요구가 충족되지 않은 경우 `IOHandler`를 사용하여 사용자 정의 로드 동작을 구현할 수 있습니다. TensorFlow.js가 제공하는 `IOHandler` 중 하나는 브라우저 사용자가 브라우저에서 모델 파일을 업로드할 수 있게 해주는 [`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles)입니다. 자세한 내용은 [설명서](https://js.tensorflow.org/api/latest/#io.browserFiles)를 참조하세요.

# 사용자 정의 IOHandlers로 모델 저장하기 및 로드하기

위의 체계에서 로드 또는 저장 요구가 충족되지 않은 경우 `IOHandler`를 구현하여 사용자 정의 직렬화 동작을 구현할 수 있습니다.

`IOHandler`는 `save` 및 `load` 메서드가 있는 객체입니다.

`save` 함수는 [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) 인터페이스와 일치하는 하나의 매개변수를 취하며 [SaveResult](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L107) 객체로 확인되는 프라미스를 반환해야 합니다.

`load` 함수는 매개변수를 사용하지 않으며 [ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165) 객체로 확인되는 프라미스를 반환해야 합니다. 이 객체는 `save`에 전달되는 것과 같은 객체입니다.

IOHandler를 구현하는 방법은 [BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts)의 예제를 참조하세요.

# TensorFlow GraphDef ベースのモデルを TensorFlow.js にインポートする

TensorFlow GraphDef ベースのモデル（通常は Python API で作成されたモデル）は、以下のいずれかの形式で保存することができます。

1. TensorFlow [SavedModel](https://www.tensorflow.org/tutorials/keras/save_and_load)
2. [凍結モデル](https://www.tensorflow.org/mobile/prepare_models#how_do_you_get_a_model_you_can_use_on_mobile)
3. [Tensorflowハブモジュール](https://www.tensorflow.org/hub/)

上記の形式はすべて、[TensorFlow.js コンバータ](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)を使用して TensorFlow.js に直接読み込める形式に変換し、推論に利用することができます。

（注意: セッションバンドル形式は TensorFlow では非推奨なので、モデルを SavedModel 形式に移行してください。）

## 要件

変換プロシージャには Python 環境が必要です。これは [pipenv](https://github.com/pypa/pipenv) や [virtualenv](https://virtualenv.pypa.io) を使用して隔離しておくことをお勧めします。コンバータのインストールには以下のコマンドを実行します。

```bash
 pip install tensorflowjs
```

TensorFlow モデルを TensorFlow.js にインポートするには、2 つのステップがあります。1 番目に既存のモデルを TensorFlow.js Web 形式に変換し、2 番目にそれを TensorFlow.js に読み込みます。

## ステップ 1. 既存の TensorFlow モデルを TensorFlow.js Web 形式に変換する

pip パッケージで提供されているコンバータのスクリプトを実行します。

SavedModel の使用例。

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

凍結モデルの使用例:

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Tensorflow Hub モジュールの使用例:

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

位置指定引数 | 説明
--- | ---
`input_path` | SavedModel ディレクトリ、セッションバンドルディレクトリ、凍結モデルファイル、あるいは TensorFlow Hub モジュールのハンドルまたはパスのフルパス。
`output_path` | すべての中間生成物のパス。

オプション | 説明
--- | ---
`--input_format` | 入力モデルのフォーマットは、SavedModel には tf_saved_model、凍結モデルには tf_frozen_model、セッションバンドルには tf_session_bundle、TensorFlow Hub モジュールには tf_hub、Keras HDF5 には Keras を使用する。
`--output_node_names` | カンマで区切られた出力ノードの名前。
`--saved_model_tags` | SavedModel 変換のみに適用され、MetaGraphDef のタグを読み込む。カンマで区切られた形式で表示される。デフォルトは`serve`。
`--signature_name` | TensorFlow Hub モジュール変換のみに適用され、シグネチャを読み込む。デフォルトは`default`。詳細は https://www.tensorflow.org/hub/common_signatures/ を参照。

ヘルプメッセージの詳細を表示するには、以下のコマンドを使用します。

```bash
tensorflowjs_converter --help
```

### コンバータで生成したファイル

上記の変換スクリプトは、2 種類のファイルを生成します。

- `model.json`（データフローグラフと重みマニフェスト）
- `group1-shard\*of\*`（バイナリ重みファイルのコレクション）

例えば、以下は MobileNet v2 を変換した場合の出力です。

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## Step 2: ブラウザで読み込んで実行する

1. 下記の tfjs-converter npm パッケージをインストールします。

`yarn add @tensorflow/tfjs` または `npm install @tensorflow/tfjs`

1. [凍結モデルクラス](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts)をインスタンス化して推論を実行します。

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat));
```

[MobileNet デモ](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/demo/mobilenet)をご覧ください。

`loadGraphModel` API は追加の`LoadOptions`パラメータを受け入れるため、これを使用して認証情報やカスタムヘッダをリクエストと共に送信することができます。詳細は [loadGraphModel() ドキュメント](https://js.tensorflow.org/api/1.0.0/#loadGraphModel)をご覧ください。

## サポートする演算

現在、TensorFlow.js がサポートする TensorFlow 演算は限られています。モデルがサポートされていない演算を使用している場合、`tensorflowjs_converter` スクリプトは失敗し、モデル内にあるサポートされていない演算のリストを出力します。各演算ごとに 1 つずつ [issue](https://github.com/tensorflow/tfjs/issues) を発行して、サポートが必要な演算を報告してください。

## 重みだけを読み込む

重みだけを読み込む場合には、以下のコードスニペットを使用します。

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```

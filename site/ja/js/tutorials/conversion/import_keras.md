# Keras モデルを TensorFlow.js にインポートする

Keras モデル（通常は Python API を介して作成します）は、[複数の形式のうちのいずれか](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)で保存することができます。「モデル全体」形式は TensorFlow.js レイヤー形式に変換することが可能で、これを直接 TensorFlow.js に読み込んで推論や後続の段階でのトレーニングに利用することができます。

ターゲットの TensorFlow.js レイヤー形式は、`model.json`ファイルとバイナリ形式のシャードされた重みファイルセットを含んだディレクトリです。`model.json`ファイルには、モデルのトポロジー（別称「アーキテクチャ」または「グラフ」。レイヤーとレイヤーの接続状態に関する説明）および重みファイルのマニフェストの両方が含まれています。

## 要件

変換プロシージャには Python 環境が必要です。これは [pipenv](https://github.com/pypa/pipenv) や [virtualenv](https://virtualenv.pypa.io) を使用して隔離しておくことをお勧めします。コンバータのインストールには`pip install tensorflowjs`を使用します。

Keras モデルを TensorFlow.js にインポートするには、2 つのステップがあります。1 番目に既存の Keras モデルを TensorFlow.js レイヤー形式に変換し、2 番目にそれを TensorFlow.js に読み込みます。

## ステップ 1. 既存の Keras モデルを TensorFlow.js レイヤー形式に変換する

通常、Keras のモデルは`model.save(filepath)`で保存し、モデルのトポロジーと重みの両方を含む単一の HDF5 (.h5) ファイルを生成します。このようなファイルを TensorFlow.js レイヤー形式に変換する場合は、以下のコマンドを実行します。*`path/to/my_model.h5`* はソースの Keras .h5 ファイルで、*`path/to/tfjs_target_dir`* はターゲットの TensorFlow.js ファイルの出力ディレクトリです。

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## オプション: Python API を使用して TensorFlow.js レイヤー形式に直接エクスポートする

Keras モデルが Python で作成されている場合には、以下のように TensorFlow.js のレイヤー形式で直接エクスポートすることが可能です。

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

## ステップ 2: TensorFlow.js にモデルを読み込む

ステップ 1 で生成した、変換されたモデルファイルを提供するには、Web サーバーを使用します。JavaScript のファイルの取得には、サーバー設定で[クロスオリジンリソースシェアリング (CORS) の許可](https://enable-cors.org/)が必要な場合があるので、注意してください。

次に、model.json ファイルの URL を指定してモデルを TensorFlow.js に読み込みます。

```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

これでモデルを推論、評価、または再トレーニングする準備ができました。例えば、読み込んだモデルは即時に予測に使用することができます。

```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

多くの [TensorFlow.js の例](https://github.com/tensorflow/tfjs-examples) はこのアプローチを採用しており、Google Cloud Storage 上で変換されてホストされている事前トレーニング済みモデルを使用しています。

`model.json` ファイル名を使用すると、モデル全体を参照することに注意してください。`loadModel(...)`は`model.json`をフェッチし、それから HTTP(S) リクエストを追加して `model.json`の重みマニフェストで参照されるシャードされた重みファイルを取得します。このアプローチでは、`model.json`と重みのシャードがどちらも一般的なキャッシュファイルのサイズ制限より小さいため、これらのファイルをすべてブラウザによって（そして恐らくインターネット上の追加のキャッシュサーバーによって）キャッシュすることが可能になります。これにより、後の場面でもモデルの読み込み速度の高速化が期待されます。

## サポートする機能

現在、TensorFlow.js レイヤーは標準の Keras 構造を使用した Keras モデルのみをサポートしています。サポートされていない演算やレイヤー（例えばカスタムレイヤー、Lambda レイヤー、カスタム損失、カスタムメトリクスなど）を使用したモデルは確実に JavaScript に変換できない Python コードに依存しているため、自動的にインポートすることはできません。

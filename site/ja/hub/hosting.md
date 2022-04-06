<!--* freshness: { owner: 'maringeo' reviewed: '2021-12-13' review_interval: '6 months'} *-->

# モデルのホスティングプロトコル

このドキュメントでは、すべてのモデルタイプ（TFJS、TF Lite、および TensorFlow モデル）を [tfhub.dev](https://tfhub.dev) にホストする際に使用される URL の表記法を説明します。また、`tensorflow_hub` ライブラリが [tfhub.dev](https://tfhub.dev) と対応サービスから TensorFlow プログラムに TensorFlow モデルを読み込むために実装する HTTP(S) ベースのプロトコルについても説明します。

モデルの読み込みと、ブラウザでモデルドキュメントの閲覧できるようにする上で、コードに同じ URL を使用できるという主な特徴があります。

## 全般的な URL 表記法

[tfhub.dev](https://tfhub.dev) では、次のフォーマットがサポートされています。

- TF Hub パブリッシャは `https://tfhub.dev/<publisher>` に従います。
- TF Hub コレクションは `https://tfhub.dev/<publisher>/collection/<collection_name>` に従います。
- TF Hub モデルにはバージョン管理された `https://tfhub.dev/<publisher>/<model_name>/<version>` url と、モデルの最新バージョンに解決するバージョン管理されていない url `https://tfhub.dev/<publisher>/<model_name>` があります。

TF Hub モデルは、URL パラメータを [tfhub.dev](https://tfhub.dev) モデル URL にアペンドすれば、圧縮アセットとしてダウンロードすることができますが、これを実現するために必要な URL パラメータはモデルの種類によって異なります。

- TensorFlow モデル（SavedModel 形式と TF1 Hub 形式）: TensorFlow モデル URL に `?tf-hub-format=compressed` をアペンドします。
- TFJS モデル: TFJS モデル URL に `?tfjs-format=compressed` をアペンドして圧縮されたアセットをダウンロードするか、リモートストレージの場合は `/model.json?tfjs-format=file` をアペンドして読み取ります。
- TF Lite モデル: TF Lite モデル URL に `?lite-format=tflite` をアペンドします。

例を示します。

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">種類</td>
    <td style="text-align: center; background-color: #D0D0D0">モデル URL</td>
    <td style="text-align: center; background-color: #D0D0D0">ダウンロードの形式</td>
    <td style="text-align: center; background-color: #D0D0D0">URL パラメータ</td>
    <td style="text-align: center; background-color: #D0D0D0">ダウンロード URL</td>
  </tr>
  <tr>
    <td>TensorFlow（SavedModel、TF1 Hub 形式）</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>.tar.gz</td>
    <td>?tf-hub-format=compressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=compressed</td>
  </tr>
  <tr>
    <td>TF Lite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1</td>
    <td>.tflite</td>
    <td>?lite-format=tflite</td>
    <td>https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.tar.gz</td>
    <td>?tfjs-format=compressed</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1?tfjs-format=compressed</td>
  </tr>
</table>

また、モデルには、ダウンロードせずにリモートストレージから直接読み取ることのできる形式でホストされているものもあります。特に、TF.js モデルをブラウザで実行している場合や [Colab](https://colab.research.google.com/) で SavedModel を読み込む場合など、利用できるローカルストレージがない場合に役立つ方法です。ローカルにダウンロードせずに、リモートにホストされているモデルを読み取ると、レイテンシが高まる可能性があることに注意してください。

<table style="width: 100%;">
  <tr style="text-align: center">
    <col style="width: 10%">
    <col style="width: 20%">
    <col style="width: 15%">
    <col style="width: 30%">
    <col style="width: 25%">
    <td style="text-align: center; background-color: #D0D0D0">種類</td>
    <td style="text-align: center; background-color: #D0D0D0">モデル URL</td>
    <td style="text-align: center; background-color: #D0D0D0">レスポンスの型</td>
    <td style="text-align: center; background-color: #D0D0D0">URL パラメータ</td>
    <td style="text-align: center; background-color: #D0D0D0">リクエスト URL</td>
  </tr>
  <tr>
    <td>TensorFlow（SavedModel、TF1 Hub 形式）</td>
    <td>https://tfhub.dev/google/spice/2</td>
    <td>文字列（圧縮されていないモデルが保存されている GCS フォルダへのパス）</td>
    <td>?tf-hub-format=uncompressed</td>
    <td>https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed</td>
  </tr>
  <tr>
    <td>TF.js</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1</td>
    <td>.json</td>
    <td>?tfjs-format=file</td>
    <td>https://tfhub.dev/google/tfjs-model/spice/2/default/1/model.json?tfjs-format=file</td>
  </tr>
</table>

## tensorflow_hub ライブラリのプロトコル

このセクションでは、tensorflow_hub ライブラリと使用するために [tfhub.dev](https://tfhub.dev) にモデルをホストする方法を説明します。tensorflow_hub ライブラリと連携する独自のモデルリポジトリをホストする場合は、HTTP(s) 配信サービスでこのプロトコルの実装が行われている必要があります。

このセクションは、TF Lite と TFJS モデルには触れていません。これらのモデルは、`tensorflow_hub` ライブラリでダウンロードできないためです。これらのモデルをホストする方法については、[上記](#general-url-conventions)をご確認ください。

### 圧縮によるホスティング

モデルは 圧縮された tar.gz ファイルとして [tfhub.dev](https://tfhub.dev) に保存されています。デフォルトでは、tensorflow_hub ライブラリは自動的に圧縮モデルをダウンロードするようになっています。また、モデル URLに `?tf-hub-format=compressed` をアペンドすると、手動でダウンロードすることも可能です。以下に例を示します。

```shell
wget https://tfhub.dev/tensorflow/albert_en_xxlarge/1?tf-hub-format=compressed
```

アーカイブのルートは、モデルディレクトリのルートで、この例のように SavedModel が含まれています。

```shell
# Create a compressed model from a SavedModel directory.
$ tar -cz -f model.tar.gz --owner=0 --group=0 -C /tmp/export-model/ .

# Inspect files inside a compressed model
$ tar -tf model.tar.gz
./
./variables/
./variables/variables.data-00000-of-00001
./variables/variables.index
./assets/
./saved_model.pb
```

レガシーの [TF1 Hub 形式](https://www.tensorflow.org/hub/tf1_hub_module)で使用する Tarball には、`./tfhub_module.pb` ファイルも含まれています。

`tensorflow_hub` ライブラリモデルの読み込み API の 1 つが呼び出されると（[hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer)、[hub.load](https://www.tensorflow.org/hub/api_docs/python/hub/load) など）、ライブラリはモデルをダウンロードして解凍し、ローカルにキャッシュします。`tensorflow_hub` ライブラリは、無期限にキャッシュできるように、モデル URL がバージョン管理されており、あるバージョンのモデルコンテンツがミュータブルではないことを期待しています。詳細は、[モデルのキャッシング](caching.md)をご覧ください。

![](https://raw.githubusercontent.com/tensorflow/hub/master/docs/images/library_download_cache.png)

### 非圧縮によるホスティング

環境変数 `TFHUB_MODEL_LOAD_FORMAT` またはコマンドラインのフラグ `--tfhub_model_load_format` が `UNCOMPRESSED` に設定されている場合、モデルはローカルにダウンロードして解凍される代わりに、リモートストレージから（GCS）から直接読み取られます。この動作が有効である場合、ライブラリはモデル URL に `?tf-hub-format=uncompressed` をアペンドします。そのリクエストは圧縮されていないモデルファイルを含む GCS のフォルダへのパスを返します。たとえば、以下の URL <br> `https://tfhub.dev/google/spice/2?tf-hub-format=uncompressed` <br> は 303 レスポンスの本文で<br> `gs://tfhub-modules/google/spice/2/uncompressed` を返します。ライブラリはこの GCS の場所からモデルを読み取ります。

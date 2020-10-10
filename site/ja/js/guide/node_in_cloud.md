# クラウドプラットフォームに tfjs-node プロジェクトをデプロイする

このドキュメントでは、クラウドプラットフォームで @tensorflow/tfjs-node パッケージを使用して Node.js プロセスを実行する方法について解説します。

tfjs-node@1.2.4 以降、クラウドプラットフォームで Node.js プロジェクトを実行するには、追加の構成は必要ありません。このガイドでは、Heroku と GCloud の @tensorflow/tfjs-examples リポジトリで[mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node)サンプルを実行する方法を紹介します。Heroku の Node.js サポートは、この[記事](https://devcenter.heroku.com/articles/nodejs-support)に記載されています。Google Cloud Platform で Node.js を実行する方法は、[こちら](https://cloud.google.com/nodejs/docs/)に記載されています。

## Heroku に Node.js プロジェクトをデプロイする

### 前提条件

1. Node.js と npm がインストールされていること
2. Heroku アカウント
3. Heroku CLI

### Node.js アプリの作成

1. フォルダを作成し、`data.js`、`main.js`、`model.js`および`package.json`ファイルを [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) の例からコピーします。
2. @tensorflow/tfjs-node の依存関係が@1.2.4 以降であることを確認してください。

### アプリを構築してローカルで実行する

1. ローカルディレクトリで`npm install`コマンドを実行して、`package.json`ファイルで宣言されている依存関係をインストールします。tfjs-node パッケージがインストールされ、libtensorflow がダウンロードされていることを確認します。

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. `npm start`を実行してモデルをローカルでトレーニングします。

```
$ npm start
> tfjs-examples-mnist-node@0.1.0 start /mnist-node
> node main.js

2019-07-30 17:33:34.109195: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-07-30 17:33:34.147880: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3492175000 Hz
2019-07-30 17:33:34.149030: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x52f7090 executing computations on platform Host. Devices:
2019-07-30 17:33:34.149057: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>

Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
========================>----------------------------------------------------------------------------------: 35.5
```

1. .gitignore ファイル内の node_modules などのビルドアーティファクトを無視してください。

### Heroku アプリを作成してデプロイする

1. Heroku Web サイトで新しいアプリを作成する
2. 変更をコミットして Heroku マスターにプッシュする

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

1. ビルドログで、tfjs-node パッケージが TensorFlow C Library をダウンロードし、TensorFlow Node.js ネイティブアドオンを読み込んでいることを確認します。

```
remote: -----> Installing dependencies
remote:        Installing node modules (package.json)
remote:
remote:        > @tensorflow/tfjs-node@1.2.5 install /tmp/build_de800e169948787d84bcc2b9ccab23f0/node_modules/@tensorflow/tfjs-node
remote:        > node scripts/install.js
remote:
remote:        CPU-linux-1.2.5.tar.gz
remote:        * Downloading libtensorflow
remote:
remote:        * Building TensorFlow Node.js bindings
remote:        added 92 packages from 91 contributors and audited 171 packages in 9.983s
remote:        found 0 vulnerabilities
remote:
```

Heroku のプロセスログで、モデルのトレーニングログを確認します。

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

また、Heroku [コンソール](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview)でプロセスを開始またはデバッグすることもできます。

### バージョン 1.2.4 より前の tfjs-node の使用

バージョン 1.2.4 より前の tfjs-node パッケージを使用している場合、パッケージには、ソースファイルからノードのネイティブアドオンをコンパイルするための g ++ が必要です。スタックに Linux ビルド必須パッケージが含まれていることを確認する必要があります (新しいバージョンのスタックにはデフォルトで含まれていない場合があります)。

## Node.js プロジェクトを Google Cloud プラットフォームにデプロイする

###前提条件

1. 請求先アカウントのある有効な Google Cloud プロジェクトを持っていること
2. Google Cloud [クライアントツール](https://cloud.google.com/storage/docs/gsutil_install)がインストールされていること
3. [Node.js ランタイム](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime) を構成するために app.yaml ファイルを追加すること

### アプリを GCloud にデプロイする

`gcloud app deploy`を実行して、ローカルコードと構成を App Engine にデプロイします。デプロイログで、tfjs-node がインストールされていることを確認します。

```
$ gcloud app deploy

Step #1:
Step #1: > @tensorflow/tfjs-node@1.2.5 install /app/node_modules/@tensorflow/tfjs-node
Step #1: > node scripts/install.js
Step #1:
Step #1: CPU-linux-1.2.5.tar.gz
Step #1: * Downloading libtensorflow
Step #1:
Step #1: * Building TensorFlow Node.js bindings
Step #1: added 88 packages from 85 contributors and audited 171 packages in 13.392s
Step #1: found 0 vulnerabilities
```

アプリのログで、モデルのトレーニングプロセスを確認できます。

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0

Epoch 1 / 20
===============================================================================>
745950ms 14626us/step - acc=0.920 loss=0.247 val_acc=0.987 val_loss=0.0445
Loss: 0.247 (train), 0.044 (val); Accuracy: 0.920 (train), 0.987 (val) (14.62 ms/step)
Epoch 2 / 20
===============================================================================>
818140ms 16042us/step - acc=0.980 loss=0.0655 val_acc=0.989 val_loss=0.0371
Loss: 0.066 (train), 0.037 (val); Accuracy: 0.980 (train), 0.989 (val) (16.04 ms/step)
Epoch 3 / 20
Epoch 3 / 20
```

# 在云平台上部署 tfjs-node 项目

本文档介绍了如何在云平台上使用 @tensorflow/tfjs-node 软件包运行 Node.js 进程。

从 tfjs-node@1.2.4 开始，在云平台上运行 Node.js 项目便已不需要其他配置。本指南将展示如何在 Heroku 和 GCloud 上运行 @tensorflow/tfjs-examples 仓库中的 [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) 示例。[本文](https://devcenter.heroku.com/articles/nodejs-support)介绍了 Heroku 的 Node.js 支持。[此处](https://cloud.google.com/nodejs/docs/)介绍了在 Google Cloud Platform 上运行 Node.js。

## 在 Heroku 上部署 Node.js 项目

### 前提条件

1. 安装 Node.js 和 npm
2. Heroku 帐户
3. Heroku CLI

### 创建 Node.js 应用

1. 创建一个文件夹，然后从 [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) 示例中复制 `data.js`、`main.js`、`model.js` 和 `package.json` 文件。
2. 确保 @tensorflow/tfjs-node 依赖项为 @1.2.4 或更高版本。

### 在本地构建并运行您的应用

1. 在本地目录中运行 `npm install` 命令以安装 `package.json` 文件中声明的依赖项。您应当能够看到 tfjs-node 软件包已安装并且 libtensorflow 已下载。

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. 通过运行 `npm start` 以在本地训练模型。

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

1. 确保忽略 .gitignore 文件中的构建工件，例如 node_modules。

### 创建并部署 Heroku 应用

1. 在 Heroku 网站上创建一个新应用
2. 提交变更并推送到 heroku master

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

1. 在构建日志中，您应当能够看到 tfjs-node 软件包下载了 TensorFlow C 库并加载了 TensorFlow Node.js 原生插件：

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

在 Heroku 上的进程日志中，您应当能够看到模型训练日志：

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

您也可以在 Heroku [控制台](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview)中启动或调试进程。

### 使用 1.2.4 版本之前的 tfjs-node

如果您使用 1.2.4 之前版本的 tfjs-node 软件包，则软件包需要 g++ 才能从源文件编译节点原生插件。您必须确保您的堆栈具有 Linux build-essential 软件包（较新版本的堆栈在默认情况下可能没有该软件包）。

## 在 Google Cloud Platform 上部署 Node.js 项目

###前提条件

1. 具备有效的 Google Cloud 项目和计费帐号
2. 安装 Google Cloud [客户端工具](https://cloud.google.com/storage/docs/gsutil_install)
3. 添加 app.yaml 文件以配置 [Node.js 运行时](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)

### 将应用部署到 GCloud

运行 `gcloud app deploy` 以将本地代码和配置部署到 App Engine。在部署日志中，您应当能够看到 tfjs-node 已安装：

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

在应用日志中，您应当能够看到模型训练进程：

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

# 在您的云平台上部署 tfjs-node 项目

此文档叙述了如何在云平台上使用 @tensorflow/tfjs-node 包运行一个 Node.js 进程。

自 tfjs-node@1.2.4 起，在云平台上运行 Node.js 不需要额外的配置。此教程演示了如何运行 Heroku 和 GCloud 上的 [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) 仓库中的示例。Heroku 的 Node.js 支持请参阅此[文档](https://devcenter.heroku.com/articles/nodejs-support)。在 Google Cloud Platform 上运行 Node.js 另见[这里](https://cloud.google.com/nodejs/docs/).

## 在 Heroku 上部署 Node.js 项目

### 准备工作

1. 已安装 Node.js 和 npm
2. Heroku 账户
3. Heroku 命令行工具

### 创建 Node.js 应用程序

1. 创建一个文件夹并从 [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) 示例中复制 `data.js`、`main.js`、`model.js` 和 `package.json` 文件。
2. 确保 @tensorflow/tfjs-node 依赖处于 @1.2.4 或更新的版本。

### 在本地构建并运行您的应用程序

1. 在您的本地文件夹运行 `npm install` 命令以安装在 `package.json` 文件中声明的依赖。您将能看到 tfjs-node 已安装，且 libtensorflow 已下载。

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. 在本地通过运行 `npm start` 以训练模型。

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

3. 确保您已在 .gitignore 文件中设置忽略了构建用的组件，如 node_modules。

### 创建并部署 Heroku 应用程序

1. 在 Heroku 网站上创建一个新的应用程序
2. Commit 您的更改并 push 到 heroku master 分支

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

3. 在构建日志中，您将能看到 tfjs-library 软件包正在下载 TensorFlow C 支持库和加载 TensorFlow Node.js 原生插件：

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

在 Heroku 的进程日志中，您将能看到模型训练日志：

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

您也可以在 Heroku [控制台](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview)中运行或调试进程。

### 使用 1.2.4 版本之前的 tfjs-node

如果您正在使用 1.2.4 版本之前的 tfjs-node，需要使用 g++ 将源文件编译成 node 的原生插件。您必须确保您拥有 Linux build-essential 软件包（新版本的工具栈可能默认不配备）。

## 在 Google Cloud Platform 上部署 Node.js 项目

### 准备工作

1. 拥有一个有效的 Google Cloud Project 的付费账户
2. 安装 Google Cloud [客户端工具]](https://cloud.google.com/storage/docs/gsutil_install)
3. 添加 app.yaml 文件以配置 [Node.js 运行时](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)

### 将应用程序部署至 GCloud

通过执行 `gcloud app deploy` 部署应用引擎的本地代码并加载配置文件。在部署日志中您将能看到 tfjs-node 已安装：

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

在应用日志中，您将能看到模型的训练过程：
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

# Deploy tfjs-node project on cloud platform

This doc describes how to run a Node.js process with @tensorflow/tfjs-node package on cloud platforms.

Starting from tfjs-node@1.2.4, running Node.js project on cloud platforms does not require additional configuration. This guide will show how to run the [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) example in the @tensorflow/tfjs-examples repository on Heroku and GCloud. Heroku’s Node.js support is documented in this [article](https://devcenter.heroku.com/articles/nodejs-support). Running Node.js on Google Cloud Platform is documented [here](https://cloud.google.com/nodejs/docs/).

## Deploy Node.js project on Heroku

### Prerequisites

1. Node.js and npm installed
2. Heroku account
3. Heroku CLI

### Create the Node.js app

1. Create a folder and copy the `data.js`, `main.js`, `model.js` and `package.json` files from the [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) example.
2. Make sure the @tensorflow/tfjs-node dependency is @1.2.4 or newer version.

### Build your app and run it locally

1. Run the `npm install` command in your local directory to install the dependencies that are declared in the `package.json` file. You should be able to see that the tfjs-node package is installed and libtensorflow is downloaded.

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

2. Train the model locally by running `npm start`.

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

3. Make sure you ignore build artifacts, such as node_modules, in your .gitignore file.

### Create and deploy the Heroku app

1. Create a new app on the Heroku website
2. Commit your change and push to heroku master

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

3. In the build logs, you should be able to see the tfjs-node package downloading the TensorFlow C Library and loading TensorFlow Node.js native addon:

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

In the process logs on Heroku, you should be able to see the model training logs:

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

You can also start or debug the process in Heroku [console](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview).

### Using tfjs-node prior to version 1.2.4

If you are using tfjs-node package before version 1.2.4, the package requires g++ to compile the node native addon from source files. You will have to make sure your stack has the Linux build-essential package (newer version stack may not have it on default).

## Deploy Node.js project on Google Cloud Platform

###Prerequisites

1. Have a valid Google Cloud Project with billing account
2. Install Google Cloud [client tool](https://cloud.google.com/storage/docs/gsutil_install)
3. Add app.yaml file to configure the [Node.js Runtime](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)

### Deploy app to GCloud

Run `gcloud app deploy` to deploy the local code and configurations to App Engine. In the deploy logs you should be able to see that tfjs-node is installed:

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

In the apps logs, you should be able to see the model training process:
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



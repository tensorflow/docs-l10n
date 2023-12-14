# Implementación de un proyecto tfjs-node en una plataforma en la nube

En este documento se describe cómo ejecutar un proceso de Node.js con el paquete @tensorflow/tfjs-node en plataformas en la nube.

Partiendo de tfjs-node@1.2.4, para ejecutar un proyecto de Node.js en plataformas en la nube no se necesita contar con ninguna configuración adicional. En esta guía mostraremos cómo ejecutar el ejemplo [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) del repositorio @tensorflow/tfjs-en Heroku y GCloud. La compatibilidad de Node.js con Heroku está documentada en [este artículo](https://devcenter.heroku.com/articles/nodejs-support). La ejecución de Node.js en Google Cloud está documentada [aquí](https://cloud.google.com/nodejs/docs/).

## Implementación de un proyecto Node.js en Heroku

### Requisitos previos

1. Node.js y npm instalados
2. Cuenta de Heroku
3. CLI de Heroku

### Creación de la aplicación Node.js

1. Cree una carpeta y copie los archivos `data.js`, `main.js`, `model.js` y `package.json` del ejemplo [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node).
2. Compruebe que la dependencia de @tensorflow/tfjs-node sea @1.2.4 o una versión posterior.

### Creación de la aplicación y ejecución local

1. Ejecute el comando `npm install` en el directorio local para instalar las dependencias que se declaran en el archivo `package.json`. Debería poder ver que el paquete está instalado y que <em>libtensorflow se ha descargado</em>.

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. Entrene el modelo localmente ejecutando `npm start`.

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

1. No olvide ignorar los artefactos de construcción, como los node_modules, de su archivo .gitignore.

### Creación e implementación de la aplicación en Heroku

1. Cree una aplicación nueva en el sitio web de Heroku
2. Haga el cambio y envíelo al máster de Heroku

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

1. En los registros de construcción, si descarga la biblioteca de TensorFlow para C y carga el complemento nativo para TensorFlow Node.js, debería poder ver el paquete tfjs-node:

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

En los registros de proceso de Heroku, debería poder ver los registros de entrenamiento del modelo:

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

También puede iniciar o depurar el proceso en la [consola](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview) de Heroku.

### Uso de tfjs-node antes de la versión 1.2.4

Si usa el paquete tfjs-node anterior a la versión 1.2.4, será necesario contar con g++ para compilar el complemento nativo del nodo a partir de los archivos de origen. Deberá asegurarse de tener el paquete esencial para construcciones de Linux (probablemente no haya una versión más nueva por defecto).

## Implementación de un proyecto Node.js en Google Cloud

###Requisitos previos

1. Tener un proyecto en Google Cloud con cuenta de facturación.
2. Instalar la [herramienta de clientes](https://cloud.google.com/storage/docs/gsutil_install) de Google Cloud.
3. Agregar el archivo app.yaml para configurar el [tiempo de ejecución de Node.js](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)

### Implementación de la aplicación en GCloud

Ejecute `gcloud app deploy` para implementar el código local y las configuraciones en la App Engine. En los registros de implementación debería poder ver que tfjs-node está instalado:

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

En los registros de las aplicaciones, debería poder ver el proceso de entrenamiento del modelo:

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

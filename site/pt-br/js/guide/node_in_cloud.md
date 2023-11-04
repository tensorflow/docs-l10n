# Implante um projeto tfjs-node em uma plataforma em nuvem

Este documento descreve como executar um processo Node.js com o pacote @tensorflow/tfjs-node em plataformas em nuvem.

A partir do tfjs-node@1.2.4, executar um projeto Node.js em plataformas em nuvem não requer configurações adicionais. Este guia mostra como executar o exemplo [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) do repositório @tensorflow/tfjs-examples no Heroku e GCloud. O suporte ao Node.js do Heroku está documentado neste [artigo](https://devcenter.heroku.com/articles/nodejs-support). Confira a documentação de como executar o Node.js no Google Cloud Platform [aqui](https://cloud.google.com/nodejs/docs/).

## Implante um projeto Node.js no Heroku

### Pré-requisitos

1. Instalação de Node.js e npm
2. Conta do Heroku
3. CLI do Heroku

### Crie a aplicação Node.js

1. Crie uma pasta e copie os arquivos `data.js`, `main.js`, `model.js` e `package.json` do exemplo [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node).
2. Confirme se a dependência @tensorflow/tfjs-node está na versão 1.2.4 ou superior.

### Compile a aplicação e execute localmente

1. Execute o comando `npm install` em seu diretório local para instalar as dependências declaradas no arquivo `package.json`. Você deverá ver que o pacote tfjs-node foi instalado e que libtensorflow foi baixado.

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. Execute `npm start` para treinar o modelo localmente.

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

1. Você deve ignorar os artefatos de compilação, como node_modules, em seu arquivo .gitignore.

### Crie e implante a aplicação Heroku

1. Crie uma nova aplicação no site do Heroku
2. Faça o commit da alteração e faça o push no master do Heroku

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

1. Nos logs de compilação, você deverá ver o pacote tfjs-node baixando a Biblioteca C do TensorFlow e carregando o complemento nativo Node.js do TensorFlow:

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

Nos logs de processo do Heroku, você deverá ver os logs de treinamento do modelo:

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

Você também pode iniciar ou depurar o processo no [console](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview) do Heroku.

### Usando tfjs-node antes da versão 1.2.4

Se você estiver usando o pacote tfjs-node antes da versão 1.2.4, ele requer g++ pra compilar o complemento nativo do nó a partir dos arquivos fonte. Sua pilha precisará ter o pacote build-essential do Linux (ele poderá não estar por padrão na pilha de versões mais recentes).

## Implante um projeto Node.js no Google Cloud Platform

###Pré-requisitos

1. Ter uma conta do Google Cloud Platform válida, com cobrança
2. Instalar a [ferramenta cliente](https://cloud.google.com/storage/docs/gsutil_install) do Google Cloud
3. Adicionar o arquivo app.yaml para configurar o [Runtime do Node.js](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)

### Implante a aplicação no GCloud

Execute `gcloud app deploy` para implantar o código e configurações locais no App Engine. Nos logs de implantação, você deverá ver que o tfjs-node foi instalado:

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

Nos logs de aplicações, você deverá ver o processo de treinamento do modelo:

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

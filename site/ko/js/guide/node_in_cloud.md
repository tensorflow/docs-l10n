# 클라우드 플랫폼에 tfjs-node 프로젝트 배포하기

이 문서에서는 클라우드 플랫폼에서 @tensorflow/tfjs-node 패키지를 사용하여 Node.js 프로세스를 실행하는 방법을 설명합니다.

tfjs-node@1.2.4부터 클라우드 플랫폼에서 Node.js 프로젝트를 실행하는 데 추가 구성이 필요하지 않습니다. 이 가이드에서는 Heroku 및 GCloud의 @tensorflow/tfjs-examples 리포지토리에서 [mnist-node](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-node) 예제를 실행하는 방법을 보여줍니다. Heroku의 Node.js 지원은 해당 <a>문서</a>에 기록되어 있으며 Google Cloud Platform에서 Node.js를 실행하는 방법은 <a>여기</a>에 설명되어 있습니다.

## Heroku에 Node.js 프로젝트 배포하기

### 전제 조건

1. Node.js 및 npm 설치
2. Heroku 계정
3. Heroku CLI

### Node.js 앱 만들기

1. 폴더를 만들고 <a>mnist-node</a> 예제에서 <code>data.js</code>, <code>main.js</code>, `model.js` 및 `package.json` 파일을 복사합니다.
2. @tensorflow/tfjs-node 종속성이 @1.2.4 이상 버전인지 확인합니다.

### 앱을 빌드하고 로컬에서 실행하기

1. 로컬 디렉터리에서 `npm install` 명령을 실행하여 `package.json` 파일에 선언된 종속성을 설치합니다. tfjs-node 패키지가 설치되고 libtensorflow가 다운로드된 것을 확인할 수 있어야 합니다.

```
$ npm install
> @tensorflow/tfjs-node@1.2.5 install mnist-node/node_modules/@tensorflow/tfjs-node
> node scripts/install.js

CPU-linux-1.2.5.tar.gz
* Downloading libtensorflow
[==============================] 22675984/bps 100% 0.0s
* Building TensorFlow Node.js bindings
```

1. `npm start`를 실행하여 모델을 로컬에서 훈련합니다.

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

1. .gitignore 파일에서 node_modules와 같은 빌드 아티팩트는 무시합니다.

### Heroku 앱 만들기 및 배포하기

1. Heroku 웹사이트에서 새 앱을 만듭니다.
2. 변경 사항을 커밋하고 heroku 마스터에게 푸시합니다.

```
$ git init
$ heroku git:remote -a your-app-name
$ git add .
$ git commit -m "First Commit"
$ git push heroku master
```

1. 빌드 로그에서 TensorFlow C 라이브러리를 다운로드하고 TensorFlow Node.js 네이티브 애드온을 로드하는 tfjs-node 패키지를 볼 수 있어야 합니다.

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

Heroku의 프로세스 로그에서 모델 훈련 로그를 볼 수 있어야 합니다.

```
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
Epoch 1 / 20
====>--------------------------------------------------------------------: 221.9
```

Heroku [콘솔](https://devcenter.heroku.com/articles/heroku-dashboard#application-overview)에서 프로세스를 시작하거나 디버그할 수도 있습니다.

### 1.2.4 이전 버전의 tfjs-node 사용하기

1.2.4 이전 버전의 tfjs-node 패키지를 사용하는 경우 패키지는 소스 파일에서 노드 네이티브 애드온을 컴파일하기 위해 g++를 필요로 합니다. 최신 버전 스택에는 기본값이 없을 수도 있기 때문에 스택에 Linux 빌드 필수 패키지가 있는지 반드시 확인해야 합니다.

## Google Cloud Platform에 Node.js 프로젝트 배포하기

### 전제 조건

1. 결제 계정이있는 유효한 Google Cloud 프로젝트가 있어야 합니다.
2. Google Cloud [클라이언트 도구](https://cloud.google.com/storage/docs/gsutil_install)를 설치합니다.
3. app.yaml 파일을 추가하여 [Node.js 런타임](https://cloud.google.com/appengine/docs/flexible/nodejs/runtime)을 구성합니다.

### GCloud에 앱 배포하기

`gcloud app deploy`를 실행하여 로컬 코드와 구성을 App Engine에 배포합니다. 배포 로그에서 tfjs-node가 설치된 것을 확인할 수 있습니다.

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

앱 로그에서 모델 훈련 프로세스를 볼 수 있어야 합니다.

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

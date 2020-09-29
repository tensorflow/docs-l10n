<!--* freshness: { owner: 'akhorlin' reviewed: '2020-09-08' } *-->

<!-- Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================-->

# Linux를 사용하여 TensorFlow Hub pip 패키지 만들기

Note: This document is for developers interested in modifying TensorFlow Hub itself. To *use* TensorFlow Hub, see the [Install instructions](installation.md)

If you make changes to TensorFlow Hub pip package, you will likely want to rebuild the pip package from source to try out your changes.

This requires:

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

Alternatively, if you install the protobuf compiler you can [try out your changes without using bazel](#develop).

## Setup a virtualenv

<a id="setup"></a>

### virtualenv 활성화

Install virtualenv if it's not installed already:

```shell
~$ sudo apt-get install python-virtualenv
```

Create a virtual environment for the package creation:

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

And activate it:

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### TensorFlow Hub 저장소를 복제합니다.

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## Test your changes

### Run TensorFlow Hub's tests

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## 패키지 빌드 및 설치

### Build TensorFlow Hub pip packaging script

To build a pip package for TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### TensorFlow Hub pip 패키지 만들기

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### Install and test the pip package (optional)

다음 명령을 실행하여 pip 패키지를 설치합니다.

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

Test import TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## "Developer" install (experimental)

<a id="develop"></a>

Warning: This approach to running TensorFlow is experimental, and not officially supported by the TensorFlow Hub team.

Building the package with bazel is the only officially supported method. However if you are unfamiliar with bazel simpler to work with open source tools. For that you can do a "developer install" of the package.

이 설치 방법을 사용하면 작업 디렉토리를 Python 환경에 설치할 수 있으므로 패키지를 가져올 때 지속적인 변경 사항이 반영됩니다.

### Setup the repository

First setup the virtualenv and repository, as described [above](#setup).

### `protoc` 설치

TensorFlow Hub는 protobufs를 사용하기 때문에 `.proto` 파일에서 필요한 python `_pb2.py` 파일을 생성하려면 protobuf 컴파일러가 필요합니다.

#### On a Mac:

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### On Linux

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### `.proto` 파일 컴파일

처음에는 디렉토리에 `_pb2.py` 파일이 없습니다.

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

Run `protoc` to create them:

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre>tensorflow_hub/image_module_info_pb2.py
tensorflow_hub/module_attachment_pb2.py
tensorflow_hub/module_def_pb2.py
</pre>

Note: Don't forget to recompile the `_pb2.py` files if you make changes to the `.proto` definitions.

### Import directly from the repository

With the `_pb2.py` files in place, you can use try out your modifications directly from the TensorFlow Hub directory:

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### Install in "developer" mode

Or to use this from outside the repository root, you can use the `setup.py develop` installation:

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

Now you can use your local changes in a regular python virtualenv, without the need to rebuild and install the pip package for each new change:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## virtualenv 비활성화

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```

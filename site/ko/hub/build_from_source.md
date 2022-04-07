<!--* freshness: { owner: 'akhorlin' reviewed: '2022-03-19' } *-->

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

참고: 이 문서는 TensorFlow Hub 자체를 수정하는 데 관심이 있는 개발자를 위한 것입니다. TensorFlow Hub를 *사용*하려면 [설치 지침](installation.md)을 참조하세요.

TensorFlow Hub pip 패키지를 변경하는 경우, 소스에서 pip 패키지를 다시 빌드하여 변경 사항을 시도해 볼 수 있습니다.

다음이 필요합니다.

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

또는 protobuf 컴파일러를 설치하는 경우, [bazel을 사용하지 않고 변경 사항을 시도](#develop)할 수 있습니다.

## Setup a virtualenv {:#setup}

### virtualenv 활성화

virtualenv가 아직 설치되지 않은 경우 설치합니다.

```shell
~$ sudo apt-get install python-virtualenv
```

패키지 생성을 위한 가상 환경을 생성합니다.

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

활성화합니다.

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### TensorFlow Hub 저장소를 복제합니다.

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## 변경 사항 테스트하기

### TensorFlow Hub의 테스트 실행하기

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## 패키지 빌드 및 설치

### TensorFlow Hub pip 패키징 스크립트 빌드하기

TensorFlow Hub용 pip 패키지를 빌드하려면:

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### TensorFlow Hub pip 패키지 만들기

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### pip 패키지 설치 및 테스트(선택 사항)

다음 명령을 실행하여 pip 패키지를 설치합니다.

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

TensorFlow Hub 가져오기 테스트:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## "개발자" 설치(실험용)

<a id="develop"></a>

경고: TensorFlow를 실행하는 이 접근 방식은 실험적이며 TensorFlow Hub 팀에서 공식적으로 지원하지 않습니다.

bazel을 사용하여 패키지를 빌드하는 것이 공식적으로 지원되는 유일한 방법입니다. 그러나 bazel에 익숙하지 않은 경우, 오픈 소스 도구로 작업하는 것이 더 간단합니다. 이를 위해 패키지의 "개발자 설치"를 수행할 수 있습니다.

이 설치 방법을 사용하면 작업 디렉토리를 Python 환경에 설치할 수 있으므로 패키지를 가져올 때 지속적인 변경 사항이 반영됩니다.

### 리포지토리 설정

먼저 [위에서](#setup) 설명한 대로 virtualenv 및 리포지토리를 설정합니다.

### `protoc` 설치

TensorFlow Hub는 protobufs를 사용하기 때문에 `.proto` 파일에서 필요한 python `_pb2.py` 파일을 생성하려면 protobuf 컴파일러가 필요합니다.

#### Mac

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### Linux

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### `.proto` 파일 컴파일

처음에는 디렉토리에 `_pb2.py` 파일이 없습니다.

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

`protoc`를 실행하여 생성합니다.

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre>tensorflow_hub/image_module_info_pb2.py
tensorflow_hub/module_attachment_pb2.py
tensorflow_hub/module_def_pb2.py
</pre>

참고: `.proto` 정의를 변경한 경우, `_pb2.py` 파일을 다시 컴파일하는 것을 잊지 마세요.

### 리포지토리에서 직접 가져오기

`_pb2.py` 파일이 제자리에 있으면 TensorFlow Hub 디렉터리에서 직접 수정을 시도할 수 있습니다.

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### "개발자" 모드로 설치하기

또는 리포지토리 루트의 외부에서 사용하려면 `setup.py develop` 설치를 사용할 수 있습니다.

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

이제 각 새 변경 사항에 대해 pip 패키지를 다시 빌드하고 설치할 필요 없이 일반 Python virtualenv에서 로컬 변경 사항을 사용할 수 있습니다.

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## virtualenv 비활성화

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```

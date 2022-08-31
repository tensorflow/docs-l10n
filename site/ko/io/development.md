## 개발

본 문서는 개발 환경을 설정하고 여러 플랫폼의 소스에서 `tensorflow-io` 패키지를 구축하는 데 필요한 정보를 포함합니다. 설정이 완료되면 새로운 ops 추가에 대한 지침으로 [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md)를 참조합니다.

### IDE 설정

TensorFlow I/O 개발을 위한 Visual Studio Code 구성 방법에 대한 지침은 이 [문서](https://github.com/tensorflow/io/blob/master/docs/vscode.md)를 참조합니다.

### Lint

TensorFlow I/O의 코드는 Bazel Buildifier, Clang Format, Black 및 Pyupgrade에 따릅니다. 다음 명령을 사용하여 소스 코드를 검사하고 lint 문제를 식별합니다.

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

Bazel Buildifier 및 Clang Format의 경우, 다음 명령이 lint 오류를 자동으로 식별하고 수정합니다.

```
$ bazel run //tools/lint:lint
```

개별 linter를 사용하여 lint 검사만 하려는 경우, `black`, `pyupgrade`, `bazel` 또는 `clang`을 선별적으로 위의 명령에 전달할 수 있습니다.

예를 들어, 다음을 사용하여 `black` 특정 lint를 검사할 수 있습니다.

```
$ bazel run //tools/lint:check -- black
```

Bazel Buildifier 및 Clang Format을 사용한 Lint 수정은 다음을 사용하여 수행할 수 있습니다.

```
$ bazel run //tools/lint:lint -- bazel clang
```

`black` 및 `pyupgrade`를 사용하여 개별 python 파일에 대해 lint 검사를 수행하려는 경우 다음을 사용할 수 있습니다.

```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

Lint는 다음을 사용하여 개별 python 파일을 black 및 pyupgrade로 수정합니다.

```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

macOS Catalina 10.15.7에서, 시스템 제공 python 3.8.2로 tensorflow-io를 구축하는 것이 가능합니다. 이 작업을 수행하려면 `tensorflow` 및 `bazel`이 모두 필요합니다.

참고: macOS 10.15.7의 시스템 기본 python 3.8.2는 `-arch arm64 -arch x86_64`의 컴플라이어 옵션으로 발생한`regex` 설치 오류를 유발합니다(https://github.com/giampaolo/psutil/issues/1832에 언급된 문제와 유사함). 이 문제를 해결하려면 `ARCHFLAGS="-arch x86_64"를 내보내기`하여 arm64 빌드 옵션을 제거해야 합니다.

```sh
#!/usr/bin/env bash

# Disable arm64 build by specifying only x86_64 arch.
# Only needed for macOS's system default python 3.8.2 on macOS 10.15.7
export ARCHFLAGS="-arch x86_64"

# Use following command to check if Xcode is correctly installed:
xcodebuild -version

# Show macOS's default python3
python3 --version

# Install Bazelisk (manage bazel version implicitly)
brew install bazelisk

# Install tensorflow and configure bazel
sudo ./configure.sh

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

참고: pytest를 실행할 때 python이 빌드 프로세스 후에 생성된 공유 라이브러리를 활용할 수 있도록 `TFIO_DATAPATH=bazel-bin`을 전달해야 합니다.

##### 트러블슈팅

Xcode가 설치되었지만 `$ xcodebuild -version`이 예상 출력에 나타나지 않았다면, 다음과 같은 명령으로 Xcode 명령줄을 사용 가능하게 해야 합니다.

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

변경 내용이 반영되려면 단말기 재시작이 필요할 수 있습니다.

샘플 출력:

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

Linux에서의 tensorflow-io 개발은 macOS와 유사합니다. 필요한 패키지는 gcc, g++, git, bazel, 및 python 3입니다. 그러나 기본 시스템 설치 버전이 아닌 새로운 버전의 gcc 또는 python이 필요할 수 있습니다.

##### Ubuntu 20.04

Ubuntu 20.04는 gcc/g++, git 및 python 3이 필요합니다. 다음은 Ubuntu 20.04에 종속성을 설치하고 공유 라이브러리를 구축합니다.

```sh
#!/usr/bin/env bash

# Install gcc/g++, git, unzip/curl (for bazel), and python3
sudo apt-get -y -qq update
sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
sudo python3 -m pip install -U pip

# Install tensorflow and configure bazel
sudo ./configure.sh

# Alias python3 to python, needed by bazel
sudo ln -s /usr/bin/python3 /usr/bin/python

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

##### CentOS 8

CentOS 8용 공유 라이브러리 구축 단계는 다음을 제외하고 위의 Ubuntu 20.04와 유사합니다.

```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```

대신 이는 gcc/g++, git, unzip/which(bazel의 경우) 및 python3을 설치하는 데 사용되어야 합니다.

##### CentOS 7

CentOS 7에서, 기본 python과 gcc 버전은 너무 오래되어 tensorflow-io 공유 라이브러리(.so)를 구축할 수 없습니다. Developer Toolset으로 제공된 gcc 및 rh-python36을 대신 사용해야 합니다. 또한 libstdc++는 CentOS vs. devtoolset별 새로운 gcc 버전에 설치된 libstdc++의 불일치를 피하기 위해 정적으로 링크되어야 합니다.

또한 파일 시스템 플러그인에 대한 정적 링크 라이브러리의 기호 중복을 피하기 위해 특수 플래그 <br>`--//tensorflow_io/core:static_build`를 Bazel에 전달해야 합니다.

다음은 bazel, devtoolset-9, rh-python36을 설치하고 공유 라이브러리를 구축합니다.

```sh
#!/usr/bin/env bash

# Install centos-release-scl, then install gcc/g++ (devtoolset), git, and python 3
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-9 git rh-python36 make

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install -U pip'

# Install tensorflow and configure bazel with rh-python36
scl enable rh-python36 devtoolset-9 \
    './configure.sh'

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries, notice the passing of --//tensorflow_io/core:static_build
BAZEL_LINKOPTS="-static-libstdc++ -static-libgcc" BAZEL_LINKLIBS="-lm -l%:libstdc++.a" \
  scl enable rh-python36 devtoolset-9 \
    'bazel build -s --verbose_failures $BAZEL_OPTIMIZATION --//tensorflow_io/core:static_build //tensorflow_io/...'

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install pytest'

TFIO_DATAPATH=bazel-bin \
  scl enable rh-python36 devtoolset-9 \
    'python3 -m pytest -s -v tests/test_serialization.py'
```

#### Docker

Python 개발의 경우, [여기](tools/docker/devel.Dockerfile)에서 참조 Dockerfile을 사용하여 소스에서 TensorFlow I/O 패키지 (`tensorflow-io`)를 구축할 수 있습니다. 또한, 사전 구축된 개발 이미지도 다음과 같이 사용할 수 있습니다.

```sh
# Pull (if necessary) and start the devel container
$ docker run -it --rm --name tfio-dev --net=host -v ${PWD}:/v -w /v tfsigio/tfio:latest-devel bash

# Inside the docker container, ./configure.sh will install TensorFlow or use existing install
(tfio-dev) root@docker-desktop:/v$ ./configure.sh

# Clean up exisiting bazel build's (if any)
(tfio-dev) root@docker-desktop:/v$ rm -rf bazel-*

# Build TensorFlow I/O C++. For compilation optimization flags, the default (-march=native)
# optimizes the generated code for your machine's CPU type.
# Reference: https://www.tensorflow.orginstall/source#configuration_options).

# NOTE: Based on the available resources, please change the number of job workers to:
# -j 4/8/16 to prevent bazel server terminations and resource oriented build errors.

(tfio-dev) root@docker-desktop:/v$ bazel build -j 8 --copt=-msse4.2 --copt=-mavx --compilation_mode=opt --verbose_failures --test_output=errors --crosstool_top=//third_party/toolchains/gcc7_manylinux2010:toolchain //tensorflow_io/... //tensorflow_io_gcs_filesystem/...


# Run tests with PyTest, note: some tests require launching additional containers to run (see below)
(tfio-dev) root@docker-desktop:/v$ pytest -s -v tests/
# Build the TensorFlow I/O package
(tfio-dev) root@docker-desktop:/v$ python setup.py bdist_wheel
```

구축에 성공하면 패키지 파일 `dist/tensorflow_io-*.whl`이 생성됩니다.

참고: Python 개발 컨테이너에서 작업하는 경우, 환경 변수 `TFIO_DATAPATH`는`pytest`를 실행하고 `bdist_wheel`를 구축하기 위해 Bazel이 설치한 공유 C++ 라이브러리로<br>tensorflow-io를 지정하도록 자동으로 설정합니다. Python `setup.py`은 `--data [path]`를 인수로 받아들일 수 있습니다. (예: `python setup.py --data bazel-bin bdist_wheel`)

참고: tfio-dev 컨테이너는 개발자들에게 작업하기 쉬운 환경을 제공해 주지만, 출시된 whl 패키지는 manylinux2010 요건으로 인해 다르게 구축되었습니다. 출시된 whl 패키지 생성 방법에 대한 자세한 내용은 [상태 및 CI 구축] 섹션을 참조합니다.

#### Python Wheels

다음과 같은 명령으로 bazel 구축이 완료되면 python wheel을 구축할 수 있습니다.

```
$ python setup.py bdist_wheel --data bazel-bin
```

.whl 파일은 dist 디렉터리에서 사용할 수 있습니다. `bazel-bin`이 `tensorflow_io` 패키지 디렉터리의 외부에 있으므로 setup.py가 필요한 공유 객체를 찾으려면 `--data` 변수로 bazel 바이너리 디렉터리 `bazel-bin`을 전달해야 합니다.

또는 다음을 사용하여 소스 설치를 수행할 수 있습니다.

```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```

같은 이유로 `TFIO_DATAPATH=bazel-bin`를 전달합니다.

`-e`를 사용하여 설치하는 것은 위와 다릅니다.

```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```

이는 `TFIO_DATAPATH=bazel-bin`을 사용하여도 공유 객체를 자동으로 설치되지 않습니다. 대신, 설치 후 프로그램을 실행할 때마다 다음과 같은 `TFIO_DATAPATH=bazel-bin`을 전달해야 합니다.

```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### 테스트

몇몇 테스트는 테스트 컨테이너를 시작하거나 실행하기 전 관련 도구의 로컬 인스턴트를 실행해야 합니다. 예를 들어, kafka, zookeeper 및 schema-registry의 로컬 인스턴스를 시작하는 kafra 연관 테스트를 실행하려면 다음을 사용합니다.

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

`Elasticsearch` 또는 `MongoDB`과 같은 도구와 연관된 `Datasets`를 테스트하려면 시스템에서 도커를 사용할 수 있어야 합니다. 이런 시나리오의 경우, 다음을 사용합니다.

```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

또한 `tensorflow-io`의 몇몇 기능을 테스트하려면 데이터가 `tests` 디렉터리 자체에 제공되었기 때문에 추가 도구를 스핀 업할 필요가 없습니다. 예를 들어, `parquet` 데이터세트와 연관된 테스트를 실행하려면, 다음을 사용합니다.

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```

### R

[여기](R-package/scripts/Dockerfile)에 제공된 참조 Dockerfile을 사용하여 R 패키지를 바로 테스트에 사용할 수 있습니다.

```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

컨테이너 내에서 R 세션을 시작하고, 예시 [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile) [string.seq](R-package/tests/testthat/testdata/string.seq)에서 `SequenceFileDataset`을 인스턴스화한 다음 데이터세트의 [tfdatasets 패키지](https://tensorflow.rstudio.com/tools/tfdatasets/)가 제공하는 [변환 함수](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations)를 다음과 같이 사용할 수 있습니다.

```r
library(tfio)
dataset <- sequence_file_dataset("R-package/tests/testthat/testdata/string.seq") %>%
    dataset_repeat(2)

sess <- tf$Session()
iterator <- make_iterator_one_shot(dataset)
next_batch <- iterator_get_next(iterator)

until_out_of_range({
  batch <- sess$run(next_batch)
  print(batch)
})
```

## Desenvolvimento

Este documento contém as informações necessárias para configurar o ambiente de desenvolvimento e compilar o pacote `tensorflow-io` a partir do código-fonte em diversas plataformas. Após concluir a configuração, consulte as diretrizes de como adicionar novas operações em [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md).

### Configuração do IDE

Confira as instruções de como configurar o Visual Studio Code para desenvolvimento do TensorFlow I/O neste [documento](https://github.com/tensorflow/io/blob/master/docs/vscode.md).

### Lint

O código do TensorFlow I/O está em conformidade com o Bazel Buildifier, Clang Format, Black e Pyupgrade. Use o comando abaixo para verificar o código fonte e identificar problemas de lint:

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

Para o Bazel Buildifier e Clang Format, o comando abaixo vai identificar e corrigir automaticamente qualquer erro de lint:

```
$ bazel run //tools/lint:lint
```

Se você só deseja fazer a verificação de lint usando linters individuais, outra opção é passar seletivamente `black`, `pyupgrade`, `bazel` ou `clang` aos comandos acima.

Por exemplo: uma verificação de lint específica ao `black` pode ser feita usando-se:

```
$ bazel run //tools/lint:check -- black
```

A correção de lint ao utilizar Bazel Buildifier e Clang Format pode ser feita usando-se:

```
$ bazel run //tools/lint:lint -- bazel clang
```

A verificação de lint ao utilizar `black` e `pyupgrade` para um arquivo Python individual pode ser feita usando-se:

```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

A correção de lint de um arquivo Python indivudual com Black e Pyupgrade pode ser feita usando-se:

```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

No macOS Catalina 10.15.7, é possível compilar o tensorflow-io com o Python 3.8.2 fornecido pelo sistema. Tanto `tensorflow` quanto `bazel` são necessários para fazer isso.

OBSERVAÇÃO: o Python 3.8.2 padrão do sistema no macOS 10.15.7 vai gerar o erro de instalação de `regex`, causado pela opção do compilador `-arch arm64 -arch x86_64` (similar ao problema mencionado em https://github.com/giampaolo/psutil/issues/1832). Para resolver esse problema, `export ARCHFLAGS="-arch x86_64"` será necessário para remover a opção de compilação arm64.

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

OBSERVAÇÃO: ao executar pytest, `TFIO_DATAPATH=bazel-bin` precisa ser passado para que o Python possa utilizar as bibliotecas compartilhadas geradas após o processo de compilação.

##### Solução de problemas

Se o Xcode estiver instalado, mas `$ xcodebuild -version` não estiver exibindo a saída esperada, talvez você precise ativar a linha de comando do Xcode com o comando:

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

Pode ser necessário reinicializar o terminal para que as mudanças entrem em vigor.

Exemplo de saída:

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

O desenvolvimento do tensorflow-io no Linux é similar à no macOS. Os pacotes necessários são gcc, g++, git, Bazel e Python 3. Porém, pode ser necessárias versões mais recentes do gcc ou Python do que as versões padrão instaladas no sistema.

##### Ubuntu 20.04

O Ubuntu 20.04 requer gcc/g++, git e Python 3. O código abaixo vai instalar as dependências e compilar as bibliotecas compartilhadas no Ubuntu 20.04:

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

As etapas para compilar bibliotecas compartilhadas para o CentOS 8 são similares às do Ubuntu 20.04 acima, exceto que

```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```

deve ser usado para instalar gcc/g++, git, unzip/which (para o Bazel) e Python3.

##### CentOS 7

No CentOS 7, a versão padrão de Python e gcc é antiga demais para compilar as bibliotecas compartilhadas (.so) de tensorflow-io. O gcc fornecido pelo Developer Toolset e o rh-python36 devem ser usados. Além disso, libstdc++ precisa ser vinculado estaticamente para evitar a discrepância de libstdc++ instalado no CentOS versus a versão mais nova do gcc por devtoolset.

Além disso, o sinalizador adicional `--//tensorflow_io/core:static_build` precisa ser passado ao Bazel para evitar a duplicação de símbolos em bibliotecas vinculadas estatisticamente para plug-ins do sistema de arquivos.

O código abaixo vai instalar bazel, devtoolset-9, rh-python36 e compilar as bibliotecas compartilhadas:

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

Para desenvolvimento no Python, [este Dockerfile de referência](tools/docker/devel.Dockerfile) pode ser usado para compilar o pacote do TensorFlow I/O (`tensorflow-io`) pelo fonte. Além disso, as imagens devel pré-criadas também podem ser usadas:

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

Um arquivo de pacote `dist/tensorflow_io-*.whl` será gerado após a compilação bem-sucedida.

OBSERVAÇÃO: ao trabalhar no container de desenvolvimento do Python, uma variável de ambiente `TFIO_DATAPATH` é definida automaticamente para apontar tensorflow-io para as bibliotecas C++ compartilhadas compiladas pelo Bazel para executar `pytest` e compilar `bdist_wheel`. `setup.py` do Python também pode aceitar `--data [path]` como argumento. Por exemplo: `python setup.py --data bazel-bin bdist_wheel`.

OBSERVAÇÃO: embora o container tfio-dev proporcione aos desenvolvedores uma facilidade de trabalhar com o ambiente, os pacotes whl lançados são compilados de maneira diferente devido os requisitos do manylinux2010. Confira mais detalhes de como os pacotes whl lançados são gerados na seção [Status da compilação e CI].

#### Wheels do Python

É possível criar wheels do Python após a conclusão da compilação do Bazel com o seguinte comando:

```
$ python setup.py bdist_wheel --data bazel-bin
```

O arquivo .whl ficará disponível no diretório dist. Observe que o diretório de binários do Bazel `bazel-bin` precisa ser passado com o argumento `--data` para que setup.py localize os objetos compartilhados necessários, pois`bazel-bin` está fora do diretório de pacotes `tensorflow_io`.

Outra opção para instalar o fonte é usando:

```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```

com `TFIO_DATAPATH=bazel-bin` passado pelo mesmo motivo.

Observe que a instalação com `-e` é diferente da acima. O comando

```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```

não instalará o objeto compartilhado automaticamente mesmo com `TFIO_DATAPATH=bazel-bin`. Em vez disso, `TFIO_DATAPATH=bazel-bin` precisa ser passado toda vez que o programa é executado após a instalação:

```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### Testes

Para alguns testes, é necessário abrir um container de teste ou iniciar uma instância local da ferramenta associada antes da execução. Por exemplo, para executar testes relacionados ao kafka que iniciarão uma instância local do kafka, zookeeper e schema-registry, use:

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

Para testar `Datasets` associados a ferramentas como `Elasticsearch` ou `MongoDB`, o docker precisa estar disponível no sistema. Nesses casos, use:

```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

Além disso, para testar alguns recursos do `tensorflow-io`, você não precisa iniciar nenhuma ferramenta adicional, pois os dados foram fornecidos no próprio diretório `tests`. Por exemplo: para executar testes relacionados a datasets do `parquet`, use:

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```

### R

Fornecemos um Dockerfile de referência [aqui](R-package/scripts/Dockerfile) para que você possa usar o pacote do R diretamente para testes. Você pode compilá-lo com este comando:

```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

Dentro do container, você pode iniciar sua sessão do R, instanciar um `SequenceFileDataset` a partir de um [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile) [string.seq](R-package/tests/testthat/testdata/string.seq) de exemplo e depois usar qualquer [função de transformação](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations) fornecida pelo [pacote tfdatasets](https://tensorflow.rstudio.com/tools/tfdatasets/) no dataset da seguinte forma:

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

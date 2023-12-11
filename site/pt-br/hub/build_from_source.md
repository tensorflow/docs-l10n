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

# Como criar o pacote pip para o TensorFlow Hub usando o Linux

Observação: este documento é destinado a desenvolvedores interessados em modificar o TensorFlow Hub. Para *usar* o TensorFlow Hub, confira as [instruções de instalação](installation.md).

Se você fizer alterações no pacote pip do TensorFlow Hub, provavelmente vai querer recompilar o pacote pip pelo código-fonte para testar as mudanças.

Para fazer isso, são necessários:

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

Outra opção é instalar o compilador protobuf – você pode [testar as alterações sem usar o bazel](#develop).

## Configure um virtualenv {:#setup}

### Ative o virtualenv

Instale o virtualenv, caso ainda não esteja instalado:

```shell
~$ sudo apt-get install python-virtualenv
```

Crie um ambiente virtual para a criação do pacote:

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

E ative-o:

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### Clone o repositório do TensorFlow Hub

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## Teste as alterações

### Execute os testes do TensorFlow Hub

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## Compile e instale o pacote

### Compile o script de criação do pacote pip para o TensorFlow Hub

Para criar um pacote pip para o TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### Crie o pacote pip para o TensorFlow Hub

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### Instale e teste o pacote pip (opcional)

Execute os comandos abaixo para instalar o pacote pip:

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

Teste a importação do TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## Instalação de "desenvolvedor" (experimental)

<a id="develop"></a>

Atenção: esta estratégia de executar o TensorFlow é experimental. Não há suporte oficial da equipe do TensorFlow Hub.

Criar o pacote usando o bazel é o único método com suporte oficial. Entretanto, se você não conhecer muito bem o bazel, é mais simples trabalhar com ferramentas de código aberto. Para isso, você pode fazer uma "instalação de desenvolvedor" do pacote.

Com este método de instalação, você pode instalar seu diretório funcional em seu ambiente Python para que as alterações em andamento sejam refletidas ao importar o pacote.

### Configure o repositório

Primeiro, configure o virtualenv e o repositório conforme descrito [acima](#setup).

### Instale o `protoc`

Como o TensorFlow Hub usa protobufs, você precisará do compilador protobuf para criar os arquivos `_pb2.py` do Python necessários a partir dos arquivos `.proto` .

#### No Mac:

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### No Linux

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### Compile os arquivos `.proto`

Inicialmente, não há nenhum arquivo `_pb2.py` no diretório:

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

Execute `protoc` para criá-los:

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre>tensorflow_hub/image_module_info_pb2.py
tensorflow_hub/module_attachment_pb2.py
tensorflow_hub/module_def_pb2.py
</pre>

Observação: não se esqueça de recompilar os arquivos `_pb2.py` se você fizer alterações nas definições de `.proto`.

### Importe diretamente do repositório

Com os arquivos `_pb2.py` em seu devido lugar, você pode testar suas modificações diretamente pelo diretório do TensorFlow Hub:

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### Instale no modo "desenvolvedor"

Ou então, para usar isso fora da raiz do repositório, você pode utilizar a instalação `setup.py develop`:

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

Agora, você pode usar suas alterações locais em um virtualenv comum do Python sem a necessidade de recompilar e instalar o pacote pip a cada nova mudança:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## Desative o virtualenv

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```

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

# Crear el paquete pip de TensorFlow Hub con Linux

Nota: Este documento es para los desarrolladores que les gustaría modificar TensorFlow Hub. Para *usar* TensorFlow Hub, consulte las [instrucciones de instalación](installation.md)

Si realiza cambios en el paquete pip de TensorFlow Hub, es probable que quiera reconstruir el paquete pip desde el origen para probar los cambios.

Se requiere:

- Python
- TensorFlow
- Git
- [Bazel](https://docs.bazel.build/versions/master/install.html)

De forma alternativa, si instala el compilador protobuf, puede [probar los cambios sin usar bazel](#develop).

## Configurar un virtualenv {:#setup}

### Activar virtualenv

Instale virtualenv si aún no está instalado:

```shell
~$ sudo apt-get install python-virtualenv
```

Cree un entorno virtual para la creación del paquete:

```shell
~$ virtualenv --system-site-packages tensorflow_hub_env
```

Y actívalo:

```shell
~$ source ~/tensorflow_hub_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow_hub_env/bin/activate.csh  # csh or tcsh
```

### Clonar el repositorio de TensorFlow Hub.

```shell
(tensorflow_hub_env)~/$ git clone https://github.com/tensorflow/hub
(tensorflow_hub_env)~/$ cd hub
```

## Pruebe sus cambios

### Ejecute las pruebas de TensorFlow Hub

```shell
(tensorflow_hub_env)~/hub/$ bazel test tensorflow_hub:all
```

## Construir e instalar el paquete

### Cree una secuencia de comandos de empaquetado de pips de TensorFlow Hub

Para crear un paquete pip para TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ bazel build tensorflow_hub/pip_package:build_pip_package
```

### Cree el paquete de pips de TensorFlow Hub

```shell
(tensorflow_hub_env)~/hub/$ bazel-bin/tensorflow_hub/pip_package/build_pip_package \
/tmp/tensorflow_hub_pkg
```

### Instale y pruebe el paquete pip (opcional)

Ejecute los siguientes comandos para instalar el paquete pip.

```shell
(tensorflow_hub_env)~/hub/$ pip install /tmp/tensorflow_hub_pkg/*.whl
```

Pruebe la importación de TensorFlow Hub:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## Instalación de "desarrollador" (experimental)

<a id="develop"></a>

Advertencia: este enfoque para ejecutar TensorFlow es experimental y no cuenta con el respaldo oficial del equipo de TensorFlow Hub.

Crear el paquete con bazel es el único método oficialmente admitido. Sin embargo, si no tiene conocimientos de bazel, será más fácil trabajar con herramientas de código abierto. Para eso, puede realizar una "instalación de desarrollador" del paquete.

Este método de instalación le permite instalar el directorio de trabajo en su entorno Python, de modo que los cambios continuos se reflejen cuando importe el paquete.

### Configurar el repositorio

Primero configure virtualenv y el repositorio, como se describe [anteriormente](#setup).

### Instalar `protoc`

Ya que TensorFlow Hub usa protobufs, necesitará el compilador de protobuf para crear los archivos `_pb2.py` de Python necesarios con los archivos `.proto`.

#### En una Mac:

```
(tensorflow_hub_env)~/hub/$ brew install protobuf
```

#### En Linux

```
(tensorflow_hub_env)~/hub/$ sudo apt install protobuf-compiler
```

### Compilar los archivos `.proto`

Al principio, no hay archivos `_pb2.py` en el directorio:

```
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

Ejecute `protoc` para crearlos:

```
(tensorflow_hub_env)~/hub/$ protoc -I=tensorflow_hub --python_out=tensorflow_hub tensorflow_hub/*.proto
(tensorflow_hub_env)~/hub/$ ls -1 tensorflow_hub/*_pb2.py
```

<pre>tensorflow_hub/image_module_info_pb2.py
tensorflow_hub/module_attachment_pb2.py
tensorflow_hub/module_def_pb2.py
</pre>

Nota: Recuerde volver a compilar los archivos `_pb2.py` si realiza cambios en las definiciones `.proto`.

### Importar directamente desde el repositorio

Una vez que se ubicados los archivos `_pb2.py`, puede probar sus modificaciones directamente desde el directorio de TensorFlow Hub:

```
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

### Instalar en modo "desarrollador"

O para usar esto desde fuera de la raíz del repositorio, puede usar la instalación `setup.py develop`:

```
(tensorflow_hub_env)~/hub/$ python tensorflow_hub/pip_package/setup.py develop
```

Ahora puede usar sus cambios locales en un virtualenv de Python normal, sin la necesidad de reconstruir e instalar el paquete pip para cada nuevo cambio:

```shell
(tensorflow_hub_env)~/hub/$ cd ..  # exit the directory to avoid confusion
(tensorflow_hub_env)~/$ python -c "import tensorflow_hub as hub"
```

## Desactivar el virtualenv

```shell
(tensorflow_hub_env)~/hub/$ deactivate
```

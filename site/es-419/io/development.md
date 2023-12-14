## Desarrollo

El documento contiene la información necesaria para configurar el entorno de desarrollo y construir el paquete `tensorflow-io` desde el código fuente en varias plataformas. Una vez que complete la configuración, consulte [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md) para obtener las pautas sobre cómo agregar nuevas operaciones.

### Configurar el entorno de desarrollo integrado

Para obtener instrucciones sobre cómo configurar Visual Studio Code para desarrollar TensorFlow I/O, consulte este [documento](https://github.com/tensorflow/io/blob/master/docs/vscode.md).

### Lint

El código de TensorFlow I/O se ajusta a Bazel Buildifier, Clang Format, Black y Pyupgrade. Use el siguiente comando para verificar el código fuente e identificar problemas de lint:

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

Para Bazel Buildifier y Clang Format, el siguiente comando identificará y corregirá automáticamente cualquier error de lint:

```
$ bazel run //tools/lint:lint
```

De forma alternativa, si solo desea realizar una verificación de lint con linters individuales, puede pasar selectivamente `black`, `pyupgrade`, `bazel` o `clang` en los comandos anteriores.

Por ejemplo, se puede realizar una verificación de lint específica `black` con:

```
$ bazel run //tools/lint:check -- black
```

La corrección de lint con Bazel Buildifier y Clang Format se puede realizar con:

```
$ bazel run //tools/lint:lint -- bazel clang
```

La verificación de lint con `black` y `pyupgrade` para un archivo de Python individual se puede realizar con:

```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

La corrección de lint de un archivo python individual con black y pyupgrade con:

```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

En macOS Catalina 10.15.7, es posible compilar tensorflow-io con Python 3.8.2 que viene con el sistema. Se necesitan tanto `tensorflow` como `bazel` para hacerlo.

NOTA: El Python 3.8.2 predeterminado del sistema en macOS 10.15.7 provocará un error de instalación `regex` que ocurre por la opción del compilador `-arch arm64 -arch x86_64` (similar al problema mencionado en https://github.com/giampaolo/psutil/issues/1832). Para solucionar este problema, necesitamos `export ARCHFLAGS="-arch x86_64"` para eliminar la opción de compilación arm64.

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

NOTA: Al ejecutar pytest, se debe pasar `TFIO_DATAPATH=bazel-bin` para que Python pueda usar las bibliotecas compartidas generadas después del proceso de compilación.

##### Solución de problemas

Si Xcode está instalado, pero `$ xcodebuild -version` no muestra el resultado esperado, es posible que deba habilitar la línea de comando de Xcode con el comando:

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

Es posible que sea necesario reiniciar la terminal para que los cambios surtan efecto.

Salida de muestra:

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

El desarrollo de tensorflow-io en Linux es similar a macOS. Los paquetes requeridos son gcc, g++, git, bazel y python 3. Sin embargo, es posible que se requieran versiones más nuevas de gcc o python, distintas a las instaladas por defecto en el sistema.

##### Ubuntu 20.04

Ubuntu 20.04 requiere gcc/g++, git y python 3. En la siguiente celda se instalarán dependencias y se crearán las bibliotecas compartidas en Ubuntu 20.04:

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

Los pasos para crear bibliotecas compartidas para CentOS 8 son similares a los de Ubuntu 20.04 mencionados anteriormente, excepto que

```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```

debería usarse en su lugar para instalar gcc/g++, git, unzip/what (para bazel) y python3.

##### CentOS 7

En CentOS 7, las versiones predeterminadas de python y gcc son demasiado antiguas para crear las bibliotecas compartidas de tensorflow-io (.so). En su lugar, se debe usar el gcc proporcionado por Developer Toolset y rh-python36. Además, libstdc++ debe vincularse estáticamente para evitar discrepancias entre libstdc++ instalado en CentOS y la versión más reciente de gcc de devtoolset.

Además, se debe pasar un indicador especial `--//tensorflow_io/core:static_build` a Bazel para evitar la duplicación de símbolos en bibliotecas vinculadas estáticamente para complementos del sistema de archivos.

En la celda siguinte se instalará bazel, devtoolset-9, rh-python36 y se crearán las bibliotecas compartidas:

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

Para el desarrollo de Python, se puede usar un Dockerfile de referencia, [aquí](tools/docker/devel.Dockerfile), para crear el paquete de I/O de TensorFlow (`tensorflow-io`) desde el código fuente. Además, también se pueden usar las imágenes de desarrollo prediseñadas:

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

Se generará un archivo de paquete `dist/tensorflow_io-*.whl` después de que la compilación sea exitosa.

NOTA: Cuando se trabaja en el contenedor de desarrollo de Python, una variable de entorno `TFIO_DATAPATH` se configura automáticamente para apuntar tensorflow-io a las bibliotecas C++ compartidas creadas por Bazel para ejecutar `pytest` y compilar `bdist_wheel`. Python `setup.py` también puede aceptar `--data [path]` como argumento, por ejemplo `python setup.py --data bazel-bin bdist_wheel`.

NOTA: Si bien el contenedor tfio-dev ofrece a los desarrolladores un entorno fácil de trabajar, los paquetes whl publicados se crean de manera diferente debido a muchos requisitos de Linux2010. Consulte la sección [Estado de compilación y CI] para obtener más detalles sobre cómo se generan los paquetes whl publicados.

#### Ruedas de Python

Es posible generar ruedas de Python después de completar la generación de Bazel con el siguiente comando:

```
$ python setup.py bdist_wheel --data bazel-bin
```

El archivo .whl estará disponible en el directorio dist. Tenga en cuenta que el directorio binario de bazel, `bazel-bin` debe pasarse con `--data` args para que setup.py ubique los objetos compartidos necesarios, ya que `bazel-bin` está fuera del directorio del paquete `tensorflow_io`.

Alternativamente, la instalación del código fuente se puede realizar con:

```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```

con `TFIO_DATAPATH=bazel-bin` pasado por el mismo motivo.

Tenga en cuenta que la instalación con `-e` es diferente a la anterior. El

```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```

no instalará el objeto compartido automáticamente incluso con `TFIO_DATAPATH=bazel-bin`. En su lugar, se debe pasar `TFIO_DATAPATH=bazel-bin` cada vez que se ejecuta el programa después de la instalación:

```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### Prueba

Algunas pruebas requieren iniciar un contenedor de prueba o iniciar una instancia local de la herramienta asociada antes de ejecutarse. Por ejemplo, para ejecutar pruebas relacionadas con Kafka que iniciarán una instancia local de Kafka, zookeeper y esquema-registro, use:

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

Las pruebas de `Datasets` asociadas con herramientas como `Elasticsearch` o `MongoDB` requieren que Docker esté disponible en el sistema. En tales escenarios, use:

```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

Además, probar algunas características de `tensorflow-io` no requiere que active ninguna herramienta adicional, ya que los datos se proporcionan en el directorio `tests`. Por ejemplo, para ejecutar pruebas relacionadas con conjuntos de datos `parquet`, use:

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```

### R

[Aquí](R-package/scripts/Dockerfile) le proporcionamos un Dockerfile de referencia para que pueda usar el paquete R directamente para realizar pruebas. Puede generarlo a través de:

```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

Dentro del contenedor, puede iniciar su sesión de R, crear una instancia de un `SequenceFileDataset` a partir de un ejemplo [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile) [string.seq](R-package/tests/testthat/testdata/string.seq) y luego usar cualquier [función de transformación](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations) proporcionada por [el paquete tfdatasets](https://tensorflow.rstudio.com/tools/tfdatasets/) en el conjunto de datos como se muestra a continuación:

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

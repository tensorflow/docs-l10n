# Instalar la optimización del modelo TensorFlow

Se recomienda crear un entorno virtual Python antes de proceder a la instalación. Consulte la [guía](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended) de instalación de TensorFlow para obtener más información.

### Compilaciones estables

Para instalar la última versión, ejecute el siguiente comando:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow-model-optimization
```

Para obtener detalles de la versión, consulte nuestras [notas de la versión](https://github.com/tensorflow/model-optimization/releases).

Para conocer la versión requerida de TensorFlow y otra información de compatibilidad, consulte la sección Matriz de compatibilidad de la API de la página de Descripción general para conocer la técnica que desea usar. Por ejemplo, para la eliminación de entradas, la página de Descripción general está [aquí](https://www.tensorflow.org/model_optimization/guide/pruning).

Nota: Dado que TensorFlow *no* está incluido como una dependencia del paquete de optimización del modelo de TensorFlow (en `setup.py`), debemos instalar explícitamente el paquete de TensorFlow (`tf-nightly` o `tf-nightly-gpu`). Esto nos permite mantener un paquete en lugar de paquetes separados para TensorFlow habilitado para CPU y GPU.

### Instalar desde la fuente

También puede instalar desde el código fuente. Esto requiere el sistema de compilación [Bazel.](https://bazel.build/)

```shell
# To install dependencies on Ubuntu:
# sudo apt-get install bazel git python-pip
# For other platforms, see Bazel docs above.
git clone https://github.com/tensorflow/model-optimization.git
cd model-optimization
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```

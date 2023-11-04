# Cómo instalar TensorFlow Graphics

## Compilación estable

TensorFlow Graphics depende de [TensorFlow](https://www.tensorflow.org/install) 1.13.1 o versiones posteriores. También se admiten compilaciones nocturnas de TensorFlow (tf-nightly).

Para instalar la última versión de CPU desde [PyPI](https://pypi.org/project/tensorflow-graphics/), ejecute el siguiente comando:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

y para instalar la última versión de GPU, ejecute el siguiente comando:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

Para obtener ayuda adicional sobre la instalación, orientación sobre la instalación de requisitos previos y (opcionalmente) configurar entornos virtuales, consulte la [guía de instalación de TensorFlow](https://www.tensorflow.org/install).

## Cómo instalar desde la fuente: macOS/Linux

También se puede instalar desde la fuente si se ejecutan los siguientes comandos:

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## Cómo instalar paquetes opcionales: Linux

Para utilizar el cargador de datos EXR de TensorFlow Graphics, debe instalar OpenEXR. Esto se puede hacer al ejecutar los siguientes comandos:

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```

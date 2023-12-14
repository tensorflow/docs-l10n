# Instalar TensorFlow Lattice

Hay varias formas de configurar su entorno para usar TensorFlow Lattice (TFL).

- La forma más sencilla de aprender y usar TFL no requiere instalación: ejecute cualquiera de los tutoriales (por ejemplo, [el tutorial de estimadores prediseñados](tutorials/canned_estimators.ipynb)).
- Para usar TFL en una máquina local, instale el paquete pip `tensorflow-lattice`.
- Si tiene una configuración para una sola máquina, puede generar el paquete desde el código fuente.

## Instalar TensorFlow Lattice con pip

Instale con pip.

```shell
pip install --upgrade tensorflow-lattice
```

## Generar desde el código fuente

Clone el repositorio de github:

```shell
git clone https://github.com/tensorflow/lattice.git
```

Construya el paquete pip desde la fuente:

```shell
python setup.py sdist bdist_wheel --universal --release
```

Instale el paquete:

```shell
pip install --user --upgrade /path/to/pkg.whl
```

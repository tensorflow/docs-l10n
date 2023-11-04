# Instale o TensorFlow Lattice

Há várias maneiras de configurar seus ambientes para usar o TensorFlow Lattice (TFL).

- A maneira mais fácil de aprender e usar o TFL não exige instalação: realize qualquer um dos tutoriais (por exemplo, [tutorial dos estimadores predefinidos](tutorials/canned_estimators.ipynb)).
- Para usar o TFL em uma máquina local, instale o pacote pip `tensorflow-lattice`.
- Se você tiver uma configuração de máquina única, é possível criar o pacote a partir do código-fonte.

## Instale o TensorFlow Lattice usando o pip

Instale usando o pip.

```shell
pip install --upgrade tensorflow-lattice
```

## Crie a partir do código-fonte

Clone o repositório do github:

```shell
git clone https://github.com/tensorflow/lattice.git
```

Crie o pacote pip a partir do código-fonte:

```shell
python setup.py sdist bdist_wheel --universal --release
```

Instale o pacote:

```shell
pip install --user --upgrade /path/to/pkg.whl
```

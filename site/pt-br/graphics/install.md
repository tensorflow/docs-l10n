# Instalação do TensorFlow Graphics

## Builds estáveis

O TensorFlow Graphics depende do [TensorFlow](https://www.tensorflow.org/install) 1.13.1 ou superior. Também há suporte aos builds noturnos do TensorFlow (tf-nightly).

Para instalar a versão para CPU mais recente pelo [PyPI](https://pypi.org/project/tensorflow-graphics/), execute:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics
```

Para instalar a versão para CPU mais recente, execute:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --upgrade tensorflow-graphics-gpu
```

Caso precise de mais ajuda para instalação, orientações sobre os pré-requisitos de instalação e opcionalmente configurar ambientes virtuais, confira o [guia de instalação do TensorFlow](https://www.tensorflow.org/install).

## Instalação usando o fonte – macOS/Linux

Também é possível instalar usando o fonte. Basta executar os seguintes comandos:

```shell
git clone https://github.com/tensorflow/graphics.git
sh build_pip_pkg.sh
pip install --upgrade dist/*.whl
```

## Instalação de pacotes opcionais – Linux

Para usar o carregador de dados EXR do TensorFlow Graphics, o OpenEXR precisa ser instalado, o que pode ser feito executando-se os seguintes comandos:

```
sudo apt-get install libopenexr-dev
pip install --upgrade OpenEXR
```

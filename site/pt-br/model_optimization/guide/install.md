# Instale o TensorFlow Model Optimization

É recomendável criar um ambiente virtual do Python antes de continuar com a instalação. Veja o [guia](https://www.tensorflow.org/install/pip#2.-create-a-virtual-environment-recommended) de instalação do TensorFlow para mais informações.

### Compilações estáveis

Para instalar a ultima versão, execute o código a seguir:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tensorflow-model-optimization
```

Para detalhes da versão, confira nossas [notas da versão](https://github.com/tensorflow/model-optimization/releases).

Para a versão exigida do TensorFlow e outras informações de compatibilidade, confira a seção "Matriz de compatibilidade da API" na página "Visão geral" para a técnica que você pretende usar. Por exemplo, para o pruning, a página "Visão geral" está [aqui](https://www.tensorflow.org/model_optimization/guide/pruning).

Como o TensorFlow *não* está incluído como dependência do pacote do TensorFlow Model Optimization (em `setup.py`), você precisa instalar explicitamente o pacote do TensorFlow (`tf-nightly` ou `tf-nightly-gpu`). Dessa forma, é possível manter um pacote em vez de pacotes separados para o TensorFlow compatível com CPU e com GPU.

### Instalação a partir do código-fonte

Você também pode instalar a partir do código-fonte. Isso exige o sistema de build [Bazel](https://bazel.build/).

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

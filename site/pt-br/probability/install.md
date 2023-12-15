# Instalação

## Builds estáveis

Instale a última versão do TensorFlow Probability:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell"> pip install --upgrade tensorflow-probability </pre>

O TensorFlow Probability depende de uma versão estável recente do [TensorFlow](https://www.tensorflow.org/install) (pacote pip `tensorflow`). Confira os detalhes das dependências entre o TensorFlow e o TensorFlow Probability nas [notas de versão do TFP](https://github.com/tensorflow/probability/releases).

Observação: como o TensorFlow *não* está incluído como dependência do pacote do TensorFlow Probability (em `setup.py`), você precisa instalar explicitamente o pacote do TensorFlow (`tensorflow` ou `tensorflow-gpu`). Dessa forma, é possível manter um pacote em vez de pacotes separados para o TensorFlow compatível com CPU e com GPU.

Para forçar a instalação do Python 3 especificamente, substitua `pip` por `pip3` nos comandos acima. Caso precise de mais ajuda para instalação, orientações sobre os pré-requisitos de instalação e opcionalmente configurar ambientes virtuais, confira o [guia de instalação do TensorFlow](https://www.tensorflow.org/install).

## Builds noturnas

Também há builds noturnas do TensorFlow Probability no pacote pip `tfp-nightly`, que depende de `tf-nightly` ou de `tf-nightly-gpu`. As builds noturnas incluem recursos mais novos, mas podem ser menos estáveis do que as versões normais.

## Instalar a partir do código-fonte

Também é possível instalar a partir do código-fonte, o que exige o sistema de builds [Bazel](https://bazel.build/){:.external}. Recomendamos que você instale a build noturna do TensorFlow (`tf-nightly`) antes de tentar compilar o TensorFlow Probability a partir do código-fonte.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get install bazel git python-pip</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user tf-nightly</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/probability.git</code>
  <code class="devsite-terminal">cd probability</code>
  <code class="devsite-terminal">bazel build --copt=-O3 --copt=-march=native :pip_pkg</code>
  <code class="devsite-terminal">PKGDIR=$(mktemp -d)</code>
  <code class="devsite-terminal">./bazel-bin/pip_pkg $PKGDIR</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user $PKGDIR/*.whl</code>
</pre>

<!-- common_typos_enable -->

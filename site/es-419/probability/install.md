# Instalación

## Compilaciones estables

Instale la última versión de TensorFlow Probability:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell"> pip install --upgrade tensorflow-probability </pre>

TensorFlow Probability depende de una versión estable reciente de [TensorFlow](https://www.tensorflow.org/install) (paquete pip `tensorflow`). Consulte las [notas de la versión de TFP](https://github.com/tensorflow/probability/releases) para obtener más información sobre las dependencias entre TensorFlow y TensorFlow Probability.

Nota: Dado que TensorFlow *no* está incluido como una dependencia del paquete TensorFlow Probability (en `setup.py`), debemos instalar explícitamente el paquete TensorFlow (`tensorflow` o `tensorflow-gpu`). Esto nos permite mantener un paquete en lugar de paquetes separados para TensorFlow habilitado para CPU y GPU.

Para forzar una instalación específica de Python 3, reemplace `pip` con `pip3` en los comandos anteriores. Para obtener ayuda adicional para la instalación, orientación para instalar los requisitos previos y (opcionalmente) configurar entornos virtuales, consulte la [guía de instalación de TensorFlow](https://www.tensorflow.org/install).

## Compilaciones nocturnas

También hay compilaciones nocturnas de TensorFlow Probability en el paquete pip `tfp-nightly`, que dependen de una de `tf-nightly` y `tf-nightly-gpu`. Las compilaciones nocturnas incluyen funciones más nuevas, pero podrían ser menos estables que las versiones oficiales.

## Instalación desde la fuente

También puede instalar desde la fuente. Para esto se requiere el sistema de compilación [Bazel](https://bazel.build/) {:.external}. Se recomienda encarecidamente que instale la compilación nocturna de TensorFlow (`tf-nightly`) antes de tratar de compilar TensorFlow Probability desde la fuente.

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

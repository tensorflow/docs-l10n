# Instalación del paquete de aprendizaje estructurado neuronal.

Hay varias formas de configurar su entorno para usar el aprendizaje estructurado neuronal (NSL) en TensorFlow:

- La forma más sencilla de aprender y usar NSL no requiere instalación; ejecute los tutoriales de directamente desde el navegador con [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).
- Para usar NSL en una máquina local, instale el [paquete de NSL](#install-neural-structured-learning-using-pip) con el administrador de paquetes `pip` de Python.
- Si tiene una configuración para una sola máquina, [construya el paquete NSL](#build-the-neural-structured-learning-pip-package) a partir del código fuente.

Nota: Para usar NSL es necesario contar con TensorFlow en una versión 1.15 o posterior. NSL también es compatible con TensorFlow 2.x a excepción de v2.1, que contiene un error que es incompatible con NSL.

## Instalación del paquete de aprendizaje estructurado neuronal con pip

#### 1. Instale el entorno de desarrollo de Python.

En Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

En macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Cree un entorno virtual.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Nota: Para salir del entorno virtual, ejecute `deactivate`.

#### 3. Instale TensorFlow

Soporte para CPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

Soporte para GPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 4. Instale el paquete `pip` para el aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade neural_structured_learning</code>
</pre>

#### 5. (Opcional) Pruebe el aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

Excelente, el aprendizaje estructurado neuronal ya está instalado.

## Creación del paquete pip para el  aprendizaje estructurado neuronal

#### 1. Instale el entorno de desarrollo de Python.

En Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

En macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Instale Bazel.

[Instale Bazel](https://docs.bazel.build/versions/master/install.html), la herramienta de construcción que se usa para compilar  aprendizaje estructurado neuronal.

#### 3. Clone el repositorio de  aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/neural-structured-learning.git</code>
</pre>

#### 4. Cree un entorno virtual.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Nota: Para salir del entorno virtual, ejecute `deactivate`.

#### 5. Instale TensorFlow

Tenga en cuenta que para NSL se necesita una versión 1.15 o posterior de TensorFlow. NSL también es compatible con TensorFlow 2.0.

Soporte para CPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

Soporte para GPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 6. Instale las dependencias para el  aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">cd neural-structured-learning</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement neural_structured_learning/requirements.txt</code>
</pre>

#### 7. (Opcional) Haga la prueba unitaria del  aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //neural_structured_learning/...</code>
</pre>

#### 8. Cree el paquete pip.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python setup.py bdist_wheel --universal --dist-dir="./wheel"</code>
</pre>

#### 9. Instale el paquete pip.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade ./wheel/neural_structured_learning*.whl</code>
</pre>

#### 10. Pruebe el aprendizaje estructurado neuronal.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

Excelente, se ha creado el paquete de  aprendizaje estructurado neuronal.

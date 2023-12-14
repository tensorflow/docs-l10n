# Cómo instalar TensorFlow Federated

Hay un par de maneras de configurar el entorno para usar TensorFlow Federated (TFF):

- La forma más sencilla de aprender y usar TFF no requiere instalación; ejecute los tutoriales de TensorFlow Federated directamente en su navegador con [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).
- Para usar TensorFlow Federated en una máquina local, [instale el paquete de TFF](#install-tensorflow-federated-using-pip) con el administrador de paquetes `pip` de Python.
- Si tiene una configuración para una sola máquina, [compile el paquete de TFF desde el código fuente](#build-the-tensorflow-federated-python-package-from-source).

## Cómo usar `pip` para instalar TensorFlow Federated

### 1. Instale el entorno de desarrollo de Python

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
</pre>

### 2. Cree un entorno virtual

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">python3 -m venv "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "pip"</code>
</pre>

Nota: Para salir del entorno virtual, ejecute `deactivate`.

### 3. Instale el paquete de TensorFlow Federated publicado para Python

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade tensorflow-federated</code>
</pre>

### 4. Pruebe Tensorflow Federated

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Éxito: Ya se instaló el último paquete de TensorFlow Federated de Python.

## Cómo compilar el paquete de TensorFlow Federated para Python desde el código fuente

Compilar el paquete TensorFlow Federated para Python desde el código fuente es útil si desea hacer lo siguiente:

- Hacer cambios en TensorFlow Federated y probar esos cambios en un componente que use TensorFlow Federated antes de enviar o publicar esos cambios.
- Utilice los cambios que se enviaron a TensorFlow Federated pero que no se publicaron.

### 1. Instale el entorno de desarrollo de Python

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
</pre>

### 2. Instale Bazel

[Instale Bazel](https://docs.bazel.build/versions/master/install.html), la herramienta de compilación utilizada para compilar Tensorflow Federated.

### 3. Clone el repositorio de Tensorflow Federated

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/federated.git</code>
<code class="devsite-terminal">cd "federated"</code>
</pre>

### 4. Cree un entorno virtual

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">python3 -m venv "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "pip"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install numpy</code>
</pre>

### 5. Compile el paquete de TensorFlow Federated para Python

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/tensorflow_federated"</code>
<code class="devsite-terminal">bazel run //tensorflow_federated/tools/python_package:build_python_package -- \
    --output_dir="/tmp/tensorflow_federated"</code>
</pre>

### 6. Salga del entorno virtual

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">deactivate</code>
</pre>

### 7. Cree un nuevo proyecto

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">mkdir "/tmp/project"</code>
<code class="devsite-terminal">cd "/tmp/project"</code>
</pre>

### 8. Cree un nuevo entorno virtual

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">python3 -m venv "venv"</code>
<code class="devsite-terminal">source "venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "pip"</code>
</pre>

Nota: Para salir del entorno virtual, ejecute `deactivate`.

### 9. Instale el paquete de TensorFlow Federated para Python

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade "/tmp/tensorflow_federated/"*".whl"</code>
</pre>

### 10. Pruebe Tensorflow Federated

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"</code>
</pre>

Éxito: Se ha compilado e instalado un paquete de TensorFlow Federated para Python desde el código fuente.

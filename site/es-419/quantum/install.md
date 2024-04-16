# Instalación de TensorFlow Quantum

Hay algunas maneras diferentes de preparar el entorno para usar TensorFlow Quantum (TFQ):

- La manera más fácil para aprender y usar TFQ no requiere instalaciones. Solamente hay que ejecutar los [tutoriales de TensorFlow Quantum](./tutorials/hello_many_worlds.ipynb) directamente en el navegador con [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
- Para usar TensorFlow Quantum en una máquina local, instale el paquete de TFQ con el administrador de paquetes pip de Python.
- O también se puede crear TensorFlow Quantum a partir de código fuente.

TensorFlow Quantum es compatible con Python 3.7, 3.8 y 3.9, y depende directamente de [Cirq](https://github.com/quantumlib/Cirq).

## Paquete de Pip

### Requisitos

- pip 19.0 o posterior (necesita soporte de `manylinux2010`)
- [TensorFlow == 2.11.0](https://www.tensorflow.org/install/pip)

Consulte la [guía de instalación de TensorFlow](https://www.tensorflow.org/install/pip) para preparar el entorno de desarrollo de Python y un entorno virtual (opcional).

Actualice `pip` e instale TensorFlow

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.11.0</code>
</pre>

<!-- common_typos_enable -->

### Instalación del paquete

Instale la última versión estable de TensorFlow Quantum:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

Excelente, ahora TensorFlow Quantum está instalado.

Las versiones nocturnas que puedan depender de versiones más nuevas de TensorFlow se pueden instalar con:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## Construcción a partir del código fuente

Hay pruebas que indican que los siguientes pasos funcionan con sistemas del estilo Ubuntu.

### 1. Preparación de un entorno de desarrollo Python 3

Lo primero que necesitaremos son las herramientas de desarrollo de Python 3.8.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. Creación de un entorno virtual

Vamos al directorio del lugar de trabajo y hacemos un entorno virtual para el desarrollo de TFQ.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Instalación de Bazel

Tal como se indicó en la [guía para construir a partir de código fuente](https://www.tensorflow.org/install/source#install_bazel) de TensorFlow, será necesario contar con un sistema <a href="https://bazel.build/" class="external">Bazel</a>.

En nuestras últimas construcciones usamos TensorFlow 2.11.0. Para garantizar la compatibilidad usamos la versión 5.3.0 de `bazel`. Para eliminar cualquier otra versión de Bazel:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

Descargamos e instalamos la versión 5.3.0 de `bazel`:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel_5.3.0-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_5.3.0-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

Para evitar la actualización automática de `bazel` a una versión incompatible, ejecutamos lo siguiente:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

Finalmente, confirmamos la instalación de la versión correcta de `bazel`:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>

<!-- common_typos_enable -->

### 4. Construcción de TensorFlow a partir de código fuente

En este caso, adaptamos las instrucciones de la guía para [construir a partir de la fuente](https://www.tensorflow.org/install/source) de TensorFlow. Para más detalles, consulte el enlace. TensorFlow Quantum es compatible con la versión 2.11.0 de TensorFlow.

Descargue el <a href="https://github.com/tensorflow/tensorflow" class="external">código fuente de TensorFlow</a>:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.11.0</code>
</pre>

Preste atención a que el entorno virtual creado en el paso 2 esté activo. Después, instale las dependencias de TensorFlow:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.24.2</code>
  <code class="devsite-terminal">pip install packaging requests</code>
</pre>

<!-- common_typos_enable -->

Configure la construcción de TensorFlow. Cuando se pidan las ubicaciones de la biblioteca y el intérprete de Python, no olvide especificar las ubicaciones dentro de la capeta del entorno virtual. Las opciones restantes se pueden dejar a los valores predeterminados.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

Construya el paquete TensorFlow (desde TF v2.8, `_GLIBCXX_USE_CXX11_ABI` se establece en 1, y los códigos c++ se compilan todos con `-std=c++17`):

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

Nota: Construir un paquete puede demorar más de una hora.

Después de terminar de construir, instale el paquete y deje el directorio de TensorFlow:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. Descarga de TensorFlow Quantum

Para las contribuciones usamos el [flujo de trabajo de solicitudes de bifurcación y extracción (<em>fork and pull</em>)](https://guides.github.com/activities/forking/). Después de bifurcar a partir de la página de GitHub de [TensorFlow Quantum](https://github.com/tensorflow/quantum), descargamos el código fuente del bifurcador e instalamos los requisitos:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/&lt;var&gt;username&lt;/var&gt;/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

### 6. Construya el paquete de pip para TensorFlow Quantum

Construya el paquete de pip de TensorFlow Quantum e instale:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
</pre>

<!-- common_typos_enable -->

Para confirmar si TensorFlow Quantum se ha instalado correctamente, puede ejecutar las siguientes pruebas:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

Excelente, ahora TensorFlow Quantum está instalado.

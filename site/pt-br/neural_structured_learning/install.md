# Instale o Neural Structured Learning

Há várias maneiras de configurar seu ambiente para usar o Neural Structured Learning (NSL) no TensorFlow:

- A maneira mais fácil de aprender e usar o NSL não requer instalação: basta executar os tutoriais NSL diretamente no seu navegador usando [o Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb).
- Para usar o NSL numa máquina local, instale o [pacote NSL](#install-neural-structured-learning-using-pip) com o gerenciador de pacotes `pip` do Python.
- Se você tiver uma configuração de máquina exclusiva, [compile o NSL](#build-the-neural-structured-learning-pip-package) a partir do código-fonte.

Observação: o NSL requer uma versão 1.15 do TensorFlow ou superior. O NSL também suporta o TensorFlow 2.x, com exceção da v2.1, que contém um bug incompatível com o NSL.

## Instale o Neural Structured Learning usando o pip

#### 1. Instale o ambiente de desenvolvimento Python.

No Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

No macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Crie um ambiente virtual.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Observação: Para sair do ambiente virtual, execute `deactivate`.

#### 3. Instale o TensorFlow

Suporte de CPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

Suporte de GPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 4. Instale o pacote `pip` do Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade neural_structured_learning</code>
</pre>

#### 5. (Opcional) Teste o Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

Sucesso: o Neural Structured Learning já está instalado.

## Compile o pacote pip do Neural Structured Learning

#### 1. Instale o ambiente de desenvolvimento Python.

No Ubuntu:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install python3-dev python3-pip  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

No macOS:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew update</code>
<code class="devsite-terminal">brew install python  # Python 3</code>
<code class="devsite-terminal">sudo pip3 install --upgrade virtualenv  # system-wide install</code>
</pre>

#### 2. Instale o Bazel.

[Instale o Bazel](https://docs.bazel.build/versions/master/install.html), a ferramenta de build usada para compilar o Neural Structured Learning.

#### 3. Clone o repositório do Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/neural-structured-learning.git</code>
</pre>

#### 4. Crie um ambiente virtual.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv --python python3 "./venv"</code>
<code class="devsite-terminal">source "./venv/bin/activate"</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade pip</code>
</pre>

Observação: Para sair do ambiente virtual, execute `deactivate`.

#### 5. Instale o Tensorflow

Observe que o NSL requer uma versão 1.15 do TensorFlow ou superior. O NSL também suporta o TensorFlow 2.0.

Suporte de CPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow&gt;=1.15.0'</code>
</pre>

Suporte de GPU:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install 'tensorflow-gpu&gt;=1.15.0'</code>
</pre>

#### 6. Instale as dependências do Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">cd neural-structured-learning</code>
<code class="devsite-terminal tfo-terminal-venv">pip install --requirement neural_structured_learning/requirements.txt</code>
</pre>

#### 7. (Opcional) Rode os testes unitários do Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">bazel test //neural_structured_learning/...</code>
</pre>

#### 8. Compile o pacote pip.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python setup.py bdist_wheel --universal --dist-dir="./wheel"</code>
</pre>

#### 9. Instale o pacote pip.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">pip install --upgrade ./wheel/neural_structured_learning*.whl</code>
</pre>

#### 10. Teste o Neural Structured Learning.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-venv">python -c "import neural_structured_learning as nsl"</code>
</pre>

Sucesso: O pacote Neural Structured Learning foi compilado.

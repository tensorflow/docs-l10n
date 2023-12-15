# Instale o TensorFlow Quantum

Há algumas maneiras de configurar seu ambiente para usar o TensorFlow Quantum (TFQ).

- A maneira mais fácil de aprender e usar o TFQ não exige instalação: execute os [tutoriais do TensorFlow Quantum](./tutorials/hello_many_worlds.ipynb) diretamente no seu navegador usando o [Google Colab](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb).
- Para usar o TensorFlow Quantum em uma máquina local, instale o pacote do TFQ usando o gerenciador de pacotes pip do Python.
- Ou compile o TensorFlow Quantum a partir do código-fonte.

O TensorFlow Quantum é compatível com o Python 3.7, 3.8 e 3.9, além de depender diretamente do [Cirq](https://github.com/quantumlib/Cirq).

## Pacote pip

### Requisitos

- pip 19.0 ou mais recente (requer o suporte a `manylinux2010`)
- [TensorFlow == 2.11.0](https://www.tensorflow.org/install/pip)

Veja o [guia de instalação do TensorFlow](https://www.tensorflow.org/install/pip) para configurar o ambiente de desenvolvimento Python e um ambiente virtual (opcional).

Faça upgrade do `pip` e instale o TensorFlow:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install --upgrade pip</code>
  <code class="devsite-terminal">pip3 install tensorflow==2.11.0</code>
</pre>

<!-- common_typos_enable -->

### Instale o pacote

Instale a última versão estável do TensorFlow Quantum:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tensorflow-quantum</code>
</pre>

<!-- common_typos_enable -->

Sucesso: o TensorFlow Quantum foi instalado.

As versões noturnas, que talvez dependam da versão mais recente do TensorFlow, podem ser instaladas com:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip3 install -U tfq-nightly</code>
</pre>

<!-- common_typos_enable -->

## Compile a partir do código-fonte

As seguintes etapas foram testadas para sistemas como o Ubuntu.

### 1. Configure um ambiente de desenvolvimento Python 3

Primeiro, precisamos das ferramentas de desenvolvimento do Python 3.8.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt update</code>
  <code class="devsite-terminal">sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3.8</code>
  <code class="devsite-terminal">sudo apt install python3.8 python3.8-dev python3.8-venv python3-pip</code>
  <code class="devsite-terminal">python3.8 -m pip install --upgrade pip</code>
</pre>

<!-- common_typos_enable -->

### 2. Crie um ambiente virtual

Acesse o diretório do seu espaço de trabalho e crie um ambiente virtual para o desenvolvimento do TFQ.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">python3.8 -m venv quantum_env</code>
  <code class="devsite-terminal">source quantum_env/bin/activate</code>
</pre>

<!-- common_typos_enable -->

### 3. Instale o Bazel

Conforme observado no guia [Compile a partir do código-fonte](https://www.tensorflow.org/install/source#install_bazel) do TensorFlow, o sistema de build <a href="https://bazel.build/" class="external">Bazel</a> será necessário.

Nossas últimas compilações de código-fonte usam o TensorFlow 2.11.0. Para garantir a compatibilidade, usamos a versão 5.3.0 do `bazel`. Para remover qualquer versão existente do Bazel:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get remove bazel</code>
</pre>

<!-- common_typos_enable -->

Baixe e instale a versão 5.3.0 do `bazel`:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel_5.3.0-linux-x86_64.deb
</code>
  <code class="devsite-terminal">sudo dpkg -i bazel_5.3.0-linux-x86_64.deb</code>
</pre>

<!-- common_typos_enable -->

Para impedir a atualização automática do `bazel` para uma versão incompatível, execute o seguinte:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-mark hold bazel</code>
</pre>

<!-- common_typos_enable -->

Por fim, confirme a instalação da versão correta do `bazel`:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel --version</code>
</pre>

<!-- common_typos_enable -->

### 4. Compile o TensorFlow a partir do código-fonte

Aqui, adaptamos as instruções do guia [Compile a partir do código-fonte](https://www.tensorflow.org/install/source). Veja mais detalhes no link. O TensorFlow Quantum é compatível com a versão 2.11.0 do TensorFlow.

Baixe o <a href="https://github.com/tensorflow/tensorflow" class="external">código-fonte do TensorFlow</a>:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
  <code class="devsite-terminal">cd tensorflow</code>
  <code class="devsite-terminal">git checkout v2.11.0</code>
</pre>

Garanta que o ambiente virtual criado na etapa 2 esteja ativado. Em seguida, instale as dependências do TensorFlow:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">pip install -U pip six numpy wheel setuptools mock 'future&gt;=0.17.1'</code>
  <code class="devsite-terminal">pip install -U keras_applications --no-deps</code>
  <code class="devsite-terminal">pip install -U keras_preprocessing --no-deps</code>
  <code class="devsite-terminal">pip install numpy==1.24.2</code>
  <code class="devsite-terminal">pip install packaging requests</code>
</pre>

<!-- common_typos_enable -->

Configure a compilação do TensorFlow. Quando forem solicitados os locais da biblioteca e do interpretador Python, especifique-os dentro da pasta do ambiente virtual. As opções restantes podem permanecer com os valores padrão.

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure</code>
</pre>

<!-- common_typos_enable -->

Compile o pacote do TensorFlow (desde o TF v2.8, `_GLIBCXX_USE_CXX11_ABI` é definido como 1, e os códigos c++ são todos compilados com `-std=c++17`):

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" //tensorflow/tools/pip_package:build_pip_package</code>
</pre>

<!-- common_typos_enable -->

Observação: a compilação do pacote pode levar mais de uma hora.

Quando a compilação for concluída, instale o pacote e saia do diretório do TensorFlow:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg</code>
  <code class="devsite-terminal">pip install /tmp/tensorflow_pkg/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
  <code class="devsite-terminal">cd ..</code>
</pre>

<!-- common_typos_enable -->

### 5. Baixe o TensorFlow Quantum

Usamos o [fluxo de trabalho fork e pull request](https://guides.github.com/activities/forking/) padrão para contribuições. Depois de fazer o fork a partir da página [TensorFlow Quantum](https://github.com/tensorflow/quantum) do GitHub, baixe o código-fonte do fork e instale os requisitos:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">git clone https://github.com/&lt;var&gt;username&lt;/var&gt;/quantum.git</code>
  <code class="devsite-terminal">cd quantum</code>
  <code class="devsite-terminal">pip install -r requirements.txt</code>
</pre>

<!-- common_typos_enable -->

### 6. Crie o pacote pip para o TensorFlow Quantum

Crie o pacote pip para o TensorFlow Quantum e instale:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./configure.sh</code>
  <code class="devsite-terminal">bazel build -c opt --cxxopt="-O3" --cxxopt="-march=native" --cxxopt="-std=c++17" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" release:build_pip_package</code>
  <code class="devsite-terminal">bazel-bin/release/build_pip_package /tmp/tfquantum/</code>
  <code class="devsite-terminal">python3 -m pip install /tmp/tfquantum/&lt;var&gt;name_of_generated_wheel&lt;/var&gt;.whl</code>
</pre>

<!-- common_typos_enable -->

Para confirmar que o TensorFlow Quantum foi instalado com êxito, execute os testes:

<!-- common_typos_disable -->

<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">./scripts/test_all.sh</code>
</pre>

<!-- common_typos_enable -->

Sucesso: o TensorFlow Quantum foi instalado.

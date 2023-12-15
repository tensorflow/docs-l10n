# Contribua com a documentação do TensorFlow

O TensorFlow aceita contribuições para sua documentação. Se você melhora a documentação, você melhora a própria biblioteca do TensorFlow. A documentação em tensorflow.org se enquadra nas seguintes categorias:

- *Referência da API* — a [documentação de referência da API](https://www.tensorflow.org/api_docs/) é gerada a partir de docstrings no [código-fonte do TensorFlow](https://github.com/tensorflow/tensorflow).
- *Documentação narrativa*: são [tutoriais](https://www.tensorflow.org/tutorials), [guias](https://www.tensorflow.org/guide) e outros textos que não fazem parte do código do TensorFlow. Esta documentação está no repositório [tensorflow/docs](https://github.com/tensorflow/docs) no GitHub.
- *Traduções da comunidade* —São guias e tutoriais traduzidos pela comunidade. Todas as traduções da comunidade residem no repositório [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site).

Alguns [projetos do TensorFlow](https://github.com/tensorflow) mantêm os arquivos-fonte da documentação próximos ao código num repositório separado, geralmente num diretório `docs/`. Veja o arquivo `CONTRIBUTING.md` do projeto ou entre em contato com o mantenedor para contribuir.

Para participar da comunidade de documentação do TensorFlow:

- Monitore o repositório GitHub [tensorflow/docs](https://github.com/tensorflow/docs).
- Siga a tag [docs](https://discuss.tensorflow.org/tag/docs) no [Fórum do TensorFlow](https://discuss.tensorflow.org/).

## Referência da API

Para mais detalhes, use o [guia do contribuidor de documentação da API do TensorFlow](docs_ref.md). Ele vai mostrar como encontrar o [arquivo-fonte](https://www.tensorflow.org/code/tensorflow/python/) e como editar a <a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a> do símbolo. Muitas páginas de referência da API em tensorflow.org incluem um link para o arquivo-fonte onde o símbolo é definido. Docstrings suportam <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a> e podem ser visualizados (de forma aproximada) usando qualquer <a href="http://tmpvar.com/markdown.html" class="external">visualizador de Markdown</a>.

### Versões e branches

A versão da [Referência da API](https://www.tensorflow.org/api_docs/python/tf) do site refere-se ao binário estável mais recente, o que corresponde ao pacote instalado com `pip install tensorflow`.

O pacote TensorFlow padrão é criado a partir do branch estável `rX.x` no repositório principal <a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>. A documentação de referência é gerada a partir de comentários de código e docstrings no código-fonte para <a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>, <a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a> e<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>.

Versões anteriores da documentação do TensorFlow estão disponíveis como [branches rX.x](https://github.com/tensorflow/docs/branches) no repositório TensorFlow Docs. Esses branches são adicionados quando uma nova versão é lançada.

### Geração dos documentos da API

Observação: esta etapa não é necessária para editar ou visualizar documentos da API, apenas para gerar o HTML usado em tensorflow.org.

#### Referência Python

O pacote `tensorflow_docs` inclui o gerador para a [documentação de referência da API Python](https://www.tensorflow.org/api_docs/python/tf). Para instalar:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

Para gerar a documentação de referência do TensorFlow 2, use o script `tensorflow/tools/docs/generate2.py`:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

Observação: este script usa o pacote TensorFlow *instalado* para gerar documentação e funciona apenas para o TensorFlow 2.x.

## Documentação narrativa

[Os guias](https://www.tensorflow.org/guide) e [tutoriais](https://www.tensorflow.org/tutorials) do TensorFlow são escritos como arquivos <a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> e notebooks <a href="https://jupyter.org/" class="external">Jupyter</a> interativos. Os notebooks podem ser executados em seu navegador usando o <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>. A documentação narrativa em [tensorflow.org](https://www.tensorflow.org) é construída a partir do branch <code>master</code> de <a>tensorflow/docs</a>. Versões mais antigas estão disponíveis no GitHub nos branches de lançamento `rX.x`

### Alterações simples

A maneira mais fácil de fazer atualizações na documentação em arquivos Markdown é usar o <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">editor de arquivos baseado na web</a> do GitHub. Navegue no repositório [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) para encontrar o Markdown que corresponde aproximadamente à estrutura da URL <a href="https://www.tensorflow.org">tensorflow.org</a>. No canto superior direito da visualização do arquivo, clique no ícone de lápis <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> para abrir o editor de arquivos. Edite o arquivo e submeta um novo pull request.

### Instale um repositório Git local

Para edições em múltiplos arquivos ou atualizações mais complexas, é melhor usar um workflow Git local para criar um pull request.

Observação: O <a href="https://git-scm.com/" class="external">Git</a> é o sistema de controle de versão (VCS) de código aberto usado para rastrear alterações em código-fonte. O <a href="https://github.com" class="external">GitHub</a> é um serviço online que fornece ferramentas de colaboração que funcionam com Git. Consulte a <a href="https://help.github.com" class="external">Ajuda do GitHub</a> para configurar sua conta GitHub e começar.

As etapas Git a seguir são necessárias apenas na primeira vez que você configurar um projeto local.

#### Faça um fork do repositório tensorflow/docs

Na página <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> GitHub, clique no botão *Fork* <svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> para criar sua própria cópia do repositório na sua conta GitHub. Uma vez feito o fork, você será responsável por manter sua cópia do repositório em dia com o repositório upstream do TensorFlow.

#### Clone seu repositório

Baixe uma cópia do *seu* repositório remoto <var>nome_de_usuário</var>/docs para sua máquina local. Este é o diretório de trabalho onde você fará alterações:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### Adicione um repositório upstream para se manter em dia (opcional)

Para manter seu repositório local sincronizado com `tensorflow/docs`, adicione um repositório *upstream* remoto para baixar as alterações mais recentes.

Observação: não deixe de atualizar seu repositório local *antes* de iniciar uma contribuição. As sincronizações regulares com o repositório upstream reduzem a chance de um <a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">conflito de merge</a> quando você envia seu pull request.

Adicione um repositório remoto:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

Para atualizar:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### Workflow do GitHub

#### 1. Crie um novo branch

Depois de atualizar seu repositório a partir de `tensorflow/docs`, crie um novo branch a partir do branch *master* local:

<pre class="prettyprint lang-bsh">
&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. Faça alterações

Edite os arquivos em seu editor favorito e siga o [guia de estilo para documentação do TensorFlow](./docs_style.md).

Faça commit da alteração do seu arquivo:

<pre class="prettyprint lang-bsh">
# View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

Adicione mais commits, conforme necessário.

#### 3. Crie um pull request

Faça upload do seu branch local para seu repositório GitHub remoto ( <var>github.com/nome_de_usuário/docs</var>):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

Após a conclusão do push, uma mensagem poderá exibir uma URL para enviar automaticamente um pull request ao repositório upstream. Caso contrário, acesse o repositório <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> — ou seu próprio repositório — e o GitHub solicitará que você crie um pull request.

#### 4. Revise

Mantenedores e outros colaboradores revisarão seu pull requesst. Por favor, participe da discussão e faça as alterações solicitadas. Quando seu pull request for aprovado, será feito o merge dele com o repositório upstream do TensorFlow docs.

Sucesso: suas alterações foram aceitas na documentação do TensorFlow.

Há uma etapa de publicação separada para atualizar o [tensorflow.org](https://www.tensorflow.org) do repositório GitHub. Normalmente, as alterações são agrupadas em lote e o site é atualizado numa cadência regular.

## Notebooks interativos

Embora seja possível editar o arquivo JSON do notebook com o <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">editor de arquivos baseado na web</a> do GitHub, isto não é recomendado, pois JSON malformado pode corromper o arquivo. Certifique-se de testar o notebook antes de enviar um pull request.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">O Google Colaboratory</a> é um ambiente de notebook hospedado que facilita a edição e a execução de documentação em notebooks. Os notebooks no GitHub são carregados no Google Colab passando o caminho para a URL do Colab, por exemplo, para o notebook localizado no GitHub aqui: <a href="https://github.com/tensorflow/docs-l10n/blob/master/site/pt-br/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

Existe uma extensão Chrome <a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a> que faz essa substituição de URL ao navegar num notebook no GitHub. Isto é útil ao abrir um notebook no seu fork do repositório, porque os botões superiores sempre vinculam ao branch `master` do TensorFlow Docs.

### Formatação dos notebooks

Uma ferramenta de formatação de notebooks deixa os diffs das fontes do notebook Jupyter consistentes e mais fáceis de revisar. Como os ambientes de autoria de notebooks diferem em relação à saída do arquivo, recuo, metadados e outros campos não especificados; `nbfmt` usa padrões opinativos com preferência para o workflow do Colab de documentação do TensorFlow. Para formatar um notebook, instale as <a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools/" external="class">ferramentas de notebook para documentação do TensorFlow</a> e execute a ferramenta `nbfmt`:

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

Para projetos de documentação do TensorFlow, notebooks *sem* células de saída são executados e testados; notebooks *com* células de saída salvas são publicados como estão. O `nbfmt` respeita o estado do notebook e usa a opção `--remove_outputs` para remover explicitamente as células de saída.

Para criar um novo notebook, copie e edite o <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">modelo de notebook  para documentação do TensorFlow</a>.

### Edição no Colab

Dentro do ambiente do Google Colab, dê um duplo clique nas células para editar blocos de texto e código. Células de texto usam Markdown e devem seguir o [guia de estilo para documentação do TensorFlow](./docs_style.md).

Baixe arquivos de notebook do Colab com *File &gt; Download .pynb*. Faça um commit desse arquivo para seu [repositório Git local](##set_up_a_local_git_repo) e envie um pull request.

Para criar um novo notebook, copie e edite o <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">modelo de notebook do TensorFlow</a>.

### Workflow Colab-GitHub

Em vez de baixar um arquivo de notebook e usar um workflow de Git local, você pode editar e atualizar o fork do seu repositório GitHub diretamente no Google Colab:

1. No seu fork do repositório <var>nome_de_usuário</var>/docs, use a interface web do GitHub para <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">criar um novo branch</a>.
2. Navegue até o arquivo do notebook para editar.
3. Abra o notebook no Google Colab: use a troca de URL ou a extensão Chrome *Open in Colab*.
4. Edite o notebook no Colab.
5. Faça commit das alterações no seu repositório do Colab com *File &gt; Save a copy in GitHub...*. A caixa de diálogo salvar deve conter um link para o repositório e branch apropriados. Adicione uma mensagem de commit significativa.
6. Depois de salvar, navegue até seu repositório ou até o repositório <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>, o GitHub deve solicitar que você crie um pull request.
7. O pull request será revisado pelos mantenedores.

Sucesso: suas alterações foram aceitas na documentação do TensorFlow.

## Traduções

A equipe do TensorFlow trabalha com a comunidade e fornecedores para fornecer traduções para o site tensorflow.org. As traduções de notebooks e outros conteúdos técnicos estão localizadas no repositório <a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> no GitHub. Por favor, envie pull requests através do <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">projeto TensorFlow GitLocalize</a>.

A documentação em inglês é a *fonte da verdade* e as traduções devem seguir esstes guias o mais fielmente possível. Dito isto, as traduções são escritas para as comunidades que servem. Se a terminologia, fraseado, estilo ou tom em inglês não traduzir bem para outro idioma, por favor use uma tradução apropriada para o leitor.

O suporte a outros idiomas é determinado por uma série de fatores, incluindo, entre outros, métricas e demanda do site, suporte da comunidade, <a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">proficiência em inglês</a>, preferências do público e outros indicadores. Como cada idioma suportado incorre em um custo, os idiomas que não são mantidos são removidos. O suporte a novos idiomas será anunciado no <a class="external" href="https://blog.tensorflow.org/">blog do TensorFlow</a> ou <a class="external" href="https://twitter.com/TensorFlow">no Twitter</a>.

Se o seu idioma preferido não for suportado, você pode manter um fork da comunidade para contribuidores de código aberto. Esses forks não são publicados no site tensorflow.org.

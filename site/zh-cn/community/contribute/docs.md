# 为 TensorFlow 文档做贡献

TensorFlow欢迎文档贡献 - 如果您改进了文档，等同于改进TensorFlow库本身。 tensorflow.org上的文档分为以下几类：

- *API 文档* —[API 文档](https://tensorflow.google.cn/api_docs/) 经由 [TensorFlow 源代码](https://github.com/tensorflow/tensorflow)中的文档字符串(docstring)生成.
- *叙述文档* —这部分内容为[教程](https://tensorflow.google.cn/tutorials)、 [指南](https://tensorflow.google.cn/guide)以及其他不属于TensorFlow代码的内容. 这部分代码位于GitHub的 [tensorflow/docs](https://github.com/tensorflow/docs) 仓库(repository)中.
- *社区翻译* —这些是经由社区翻译的指南和教程。他们都被存放在 [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) 仓库(repository)中.

一些 [TensorFlow 项目](https://github.com/tensorflow) 将文档源文件保存在单独的存储库中，通常位于`docs/`目录中。 请参阅项目的`CONTRIBUTING.md`文件或联系维护者以参与。

参与到TensorFlow文档社区的方式有:

- 关注GitHub中的 [tensorflow/docs](https://github.com/tensorflow/docs) 仓库(repository).
- 按照[TensorFlow 论坛](https://discuss.tensorflow.org/tag/docs)[上的 docs](https://discuss.tensorflow.org/)标签进行操作。

## API 文档

For details, use the [TensorFlow API docs contributor guide](docs_ref.md). This shows you how to find the [source file](https://www.tensorflow.org/code/tensorflow/python/) and edit the symbol's <a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a>. Many API reference pages on tensorflow.org include a link to the source file where the symbol is defined. Docstrings support <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a> and can be (approximately) previewed using any <a href="http://tmpvar.com/markdown.html" class="external">Markdown previewer</a>.

### 版本(Versions) 和 分支(Branches)

本网站的 [API 文档](https://tensorflow.google.cn/api_docs/python/tf) 版本默认为最新的稳定二进制文件—即与通过`pip install tensorflow`安装的版本所匹配.

默认的TensorFlow 包是根据<a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>仓库(repository)中的稳定分支`rX.x`所构建的。文档则是由 <a href="https://tensorflow.google.cn/code/tensorflow/python/" class="external">Python</a>、 <a href="https://tensorflow.google.cn/code/tensorflow/cc/" class="external">C++</a>与 <a href="https://tensorflow.google.cn/code/tensorflow/java/" class="external">Java</a>代码中的注释与文档字符串所生成。

以前版本的TensorFlow文档在TensorFlow Docs 仓库(repository)中以[rX.x 分支](https://github.com/tensorflow/docs/branches) 的形式提供。在发布新版本时会添加这些分支。

### 构建API文档

注意：编辑或预览API文档字符串不需要此步骤，只需生成tensorflow.org上使用的HTML。

#### Python 文档

`tensorflow_docs`包中包含[Python API 文档](https://tensorflow.google.cn/api_docs/python/tf)的生成器。 安装方式：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

要生成TensorFlow 2.0文档，使用 `tensorflow/tools/docs/generate2.py` 脚本:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

注意：此脚本使用*已安装*的TensorFlow包生成文档并且仅适用于TensorFlow 2.x.

## 叙述文档

TensorFlow [指南](https://tensorflow.google.cn/guide) 和 [教程](https://tensorflow.google.cn/tutorials) 是通过 <a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> 文件和交互式的 <a href="https://jupyter.org/" class="external">Jupyter</a> 笔记本所编写。 可以使用 <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> 在您的浏览器中运行笔记本。 [tensorflow.org](https://tensorflow.google.cn)中的叙述文档是根据 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>的 `master` 分支构建. 旧版本存储在在GitHub 仓库(repository)下的`rX.x`发行版分支中。

### 简单更改

进行简单文档更新和修复的最简单方法是使用GitHub的 <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">Web文件编辑器</a>。 浏览[tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) 仓库(repository) 以寻找与 <a href="https://tensorflow.google.cn">tensorflow.org</a> 中的URL 结构相对应的Markdown或notebook文件。 在文件视图的右上角，单击铅笔图标 <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> 来打开文件编辑器。 编辑文件，然后提交新的拉取请求(pull request)。

### 设置本地Git仓库(repository)

对于多文件编辑或更复杂的更新，最好使用本地Git工作流来创建拉取请求(pull request)。

注意：<a href="https://git-scm.com/" class="external">Git</a> 是用于跟踪源代码更改的开源版本控制系统（VCS）。 <a href="https://github.com" class="external">GitHub</a>是一种在线服务， 提供与Git配合使用的协作工具。请参阅<a href="https://help.github.com" class="external">GitHub Help</a>以设置您的GitHub帐户并开始使用。

只有在第一次设置本地项目时才需要以下Git步骤。

#### 复制(fork) tensorflow/docs 仓库(repository)

在 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 的Github页码中，点击*Fork*按钮 <svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> 在您的GitHub帐户下创建您自己的仓库副本。复制(fork) 完成，您需要保持您的仓库副本副本与上游TensorFlow仓库的同步。

#### 克隆您的仓库(repository)

下载一份您 <var>username</var>/docs 仓库的副本到本地计算机。这是您之后进行操作的工作目录：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 添加上游仓库(upstream repo)以保持最新（可选）

要使本地存储库与`tensorflow/docs`保持同步，需要添加一个*上游(upstream)* 仓库来下载最新的更改。

注意：确保在开始撰稿*之前*更新您的本地仓库。定期向上游同步会降低您在提交拉取请求(pull request)时产生<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">合并冲突(merge conflict)</a>的可能性。

添加远程仓库:

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

更新:

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub 工作流

#### 1. 创建一个新分支

从`tensorflow / docs`更新您的仓库后，从本地*master*分支中创建一个新的分支:

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. 做更改

在您喜欢的编辑器中编辑文件，并请遵守 [TensorFlow文档样式指南](./docs_style.md)。

提交文件更改：

<pre class="prettyprint lang-bsh"># View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

根据需要添加更多提交。

#### 3. 创建一个拉取请求(pull request)

将您的本地分支上传到您的远程GitHub仓库 (github.com/<var>username</var>/docs):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

推送完成后，消息可能会显示一个URL，以自动向上游存储库提交拉取请求。如果没有，请转到 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 仓库—或者您自己的仓库—GitHub将提示您创建拉取请求(pull request)。

#### 4. 审校

维护者和其他贡献者将审核您的拉取请求(pull request)。请参与讨论并根据要求进行修改。当您的请求获得批准后，它将合并到上游TensorFlow文档仓库中。

成功后：您的更改会被TensorFlow文档接受。

从GitHub仓库更新 [tensorflow.org](https://tensorflow.google.cn)是一个单独的步骤。通常情况下，多个更改将被一并处理，并定期上传至网站中。

## 交互式笔记本（notebook）

虽然可以使用GitHub的<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web文本编辑器</a>来编辑笔记本JSON文件，但不推荐使用它，因为格式错误的JSON可能会损坏文件。 确保在提交拉取请求(pull request)之前测试笔记本。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> 是一个托管笔记本环境，可以轻松编辑和运行笔记本文档。 GitHub中的笔记本通过将路径传递给Colab URL（例如，位于GitHub中的笔记本）在Google Colab中加载： <a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a><br> 可以通过以下URL链接在Google Colab中加载: <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

有一个 <a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a> 扩展程序，可以在GitHub上浏览笔记本时执行此URL替换。 这在您复制的仓库中中打开笔记本时非常有用，因为顶部按钮始终链接到TensorFlow Docs的`master`分支。

### 在Colab编辑

在Google Colab环境中，双击单元格以编辑文本和代码块。文本单元格使用Markdown格式，请遵循 [TensorFlow文档样式指南](./docs_style.md).

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

通过点击 *File &gt; Download .pynb* 可以从Colab中下载笔记本文件。 将此文件提交到您的[本地Git仓库](###%E8%AE%BE%E7%BD%AE%E6%9C%AC%E5%9C%B0Git%E4%BB%93%E5%BA%93(repository))后再提交拉取请求。

如需要创建新笔记本，请复制和编辑 <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 笔记本模板</a>.

### Colab-GitHub工作流

您可以直接从Google Colab编辑和更新复制的GitHub仓库，而不是下载笔记本文件并使用本地Git工作流：

成功后：您的更改会被TensorFlow文档接受。

To create a new notebook, copy and edit the <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow notebook template</a>.

### 审校须知

注意：*请勿翻译* tensorflow.org中的API引用.

1. 在您复制的 <var>username</var>/docs 仓库中，使用 GitHub Web 界面<a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">创建新分支</a>。
2. Navigate to the notebook file to edit.
3. 在 Google Colab 中打开笔记本：使用网址替换或 *Open in Colab* Chrome 扩展程序。
4. 在 Colab 中编辑笔记本。
5. 点击 *File &gt; Save a copy in GitHub...* 从 Colab 中向您的仓库提交变更。保存对话框应链接到相应的仓库和分支。添加一条有意义的提交消息。
6. 保存后，浏览到您的仓库或者 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 仓库，GitHub 应提示您创建拉取请求。
7. 维护者会审查拉取请求。

有特定于语言的文档组，使翻译贡献者可以更轻松地进行组织。 如果您是作者，评论者或只是想为社区构建TensorFlow.org内容，请加入：

## 社区翻译

The TensorFlow team works with the community and vendors to provide translations for tensorflow.org. Translations of notebooks and other technical content are located in the <a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> GitHub repo. Please submit pull requests through the <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">TensorFlow GitLocalize project</a>.

The English docs are the *source-of-truth* and translations should follow these guides as close as possible. That said, translations are written for the communities they serve. If the English terminology, phrasing, style, or tone does not translate to another language, please use a translation appropriate for the reader.

Language support is determined by a number of factors including—but not limited to—site metrics and demand, community support, <a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">English proficiency</a>, audience preference, and other indicators. Since each supported language incurs a cost, unmaintained languages are removed. Support for new languages will be announced on the <a class="external" href="https://blog.tensorflow.org/">TensorFlow blog</a> or <a class="external" href="https://twitter.com/TensorFlow">Twitter</a>.

If your preferred language is not supported, you are welcome to maintain a community fork for open source contributors. These are not published to tensorflow.org.

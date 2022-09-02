# 为 TensorFlow 文档做贡献

TensorFlow 欢迎文档贡献 - 如果您改进文档，等同于改进 TensorFlow 库本身。tensorflow.org 上的文档分为以下几类：

- *API 文档* - [API 文档](https://tensorflow.google.cn/api_docs/)由 [TensorFlow 源代码](https://github.com/tensorflow/tensorflow)中的 docstring 生成。
- *叙述文档* - 这些内容为[教程](https://tensorflow.google.cn/tutorials)、[指南](https://tensorflow.google.cn/guide)以及其他不属于 TensorFlow 代码的内容。这种文档位于 [tensorflow/docs](https://github.com/tensorflow/docs) GitHub 仓库中。
- *社区翻译* - 这些是由社区翻译的指南和教程。所有社区翻译都存放在 [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) 仓库中。

某些 [TensorFlow 项目](https://github.com/tensorflow)将文档源文件保存在单独仓库中的代码旁，通常位于 `docs/` 目录中。请参阅项目的 `CONTRIBUTING.md` 文件或联系维护者以做贡献。

参与 TensorFlow 文档社区：

- 关注 [tensorflow/docs](https://github.com/tensorflow/docs) GitHub 仓库。
- 按照[TensorFlow 论坛](https://discuss.tensorflow.org/tag/docs)[上的 docs](https://discuss.tensorflow.org/)标签进行操作。

## API 参考

有关详细信息，请使用 [TensorFlow API 文档贡献者指南](docs_ref.md)。这向您展示了如何找到[源文件](https://www.tensorflow.org/code/tensorflow/python/)并编辑符号的 <a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a>。tensorflow.org 上的许多 API 参考页面都包含指向定义符号的源文件的链接。Docstring 支持 <a href="https://www.python.org/dev/peps/pep-0257/" class="external">Markdown</a> 并且（绝大多数时候）都能使用任意 <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown 预览程序</a>进行浏览。

### 版本和分支

本网站的 [API 参考](https://tensorflow.google.cn/api_docs/python/tf)版本默认为最新的稳定二进制文件，与通过 `pip install tensorflow` 安装的软件包匹配。

默认的 TensorFlow 软件包根据 <a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a> 主仓库中的稳定分支 `rX.x` 构建。参考文档由 <a href="https://tensorflow.google.cn/code/tensorflow/python/" class="external">Python</a>、<a href="https://tensorflow.google.cn/code/tensorflow/cc/" class="external">C++</a> 和 <a href="https://tensorflow.google.cn/code/tensorflow/java/" class="external">Java</a> 源代码中的代码注释与 docstring 生成。

以前版本的 TensorFlow 文档在 TensorFlow Docs 仓库中以 [rX.x 分支](https://github.com/tensorflow/docs/branches)形式提供。在发布新版本时会添加这些分支。

### 构建 API 文档

注：编辑或预览 API docstring 不需要此步骤，只需生成 tensorflow.org 上使用的 HTML。

#### Python 参考

`tensorflow_docs` 软件包中包含 [Python API 参考文档](https://tensorflow.google.cn/api_docs/python/tf)的生成器。要安装，请运行以下代码：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

要生成 TensorFlow 2 参考文档，请使用 `tensorflow/tools/docs/generate2.py` 脚本：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

注：此脚本使用*已安装的* TensorFlow 软件包来生成文档，并且仅适用于 TensorFlow 2.x。

## 叙述文档

TensorFlow [指南](https://tensorflow.google.cn/guide)和[教程](https://tensorflow.google.cn/tutorials)作为 <a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> 文件和交互式 <a href="https://jupyter.org/" class="external">Jupyter</a> 笔记本编写。可以使用 <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> 在您的浏览器中运行笔记本。[tensorflow.org](https://tensorflow.google.cn) 上的叙述文档根据 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 的 `master` 分支构建。旧版本存储在在 GitHub 仓库下的 `rX.x` 版本分支中。

### 简单变更

Markdown 文件进行简单文档更新的最简单方法是使用 GitHub 的 <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">Web 文件编辑器</a>。浏览 [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) 仓库以找到与 <a href="https://www.tensorflow.org">tensorflow.org</a> 网址结构大致对应的 Markdown。在文件视图的右上角，点击铅笔图标 <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> 打开文件编辑器。编辑文件，然后提交新的拉取请求。

### 设置本地 Git 仓库

对于多文件编辑或更复杂的更新，最好使用本地 Git 工作流创建拉取请求。

注：<a href="https://git-scm.com/" class="external">Git</a> 是用于跟踪源代码变更的开源版本控制系统 (VCS)。<a href="https://github.com" class="external">GitHub</a>是一种在线服务，提供可与 Git 配合使用的协作工具。请参阅 <a href="https://help.github.com" class="external">GitHub 帮助</a>来设置您的 GitHub 帐号并开始使用。

只有在第一次设置本地项目时才需要以下 Git 步骤。

#### 复刻 tensorflow/docs 仓库

在 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> Github 页面中，点击 *Fork* 按钮 <svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> 在您的 GitHub 帐号下创建您自己的仓库副本。复刻后，您需要保持您的仓库副本与上游 TensorFlow 仓库同步。

#### 克隆您的仓库

将*您的*远程 <var>username</var>/docs 仓库的副本下载到本地计算机。这是您之后进行更改的工作目录：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 添加上游仓库以保持最新（可选）

要使本地存储库与 `tensorflow/docs` 保持同步，请添加一个*上游*仓库来下载最新变更。

注：确保在开始贡献*之前*更新您的本地仓库。定期向上游同步会降低您在提交拉取请求时产生<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">合并冲突</a>的可能性。

添加远程仓库：

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

更新：

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub 工作流

#### 1. 创建一个新分支

从 `tensorflow/docs` 更新您的仓库后，从本地 *master* 分支创建一个新分支：

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. 进行更改

在您喜欢的编辑器中编辑文件，并请遵守 [TensorFlow 文档风格指南](./docs_style.md)。

提交文件变更：

<pre class="prettyprint lang-bsh"># View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

根据需要添加更多提交。

#### 3. 创建拉取请求

将本地分支上传到您的远程 GitHub 仓库 (github.com/<var>username</var>/docs)：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

推送完成后，消息可能会显示一个网址，以自动向上游仓库提交拉取请求。如果没有，请转到 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 仓库或者您自己的仓库，GitHub 将提示您创建拉取请求。

#### 4. 审查

维护者和其他贡献者将审查您的拉取请求。请参与讨论并根据要求进行修改。当您的请求获得批准后，它将被合并到上游 TensorFlow 文档仓库中。

成功：您的变更已被 TensorFlow 文档接受。

从 GitHub 仓库更新 [tensorflow.org](https://tensorflow.google.cn) 是一个单独的步骤。通常情况下，多个变更将一起处理，并定期上传至网站。

## 交互式笔记本

虽然可以使用 GitHub 的 <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">Web 文件编辑器</a>编辑笔记本 JSON 文件，但不推荐使用，因为格式错误的 JSON 可能会损坏文件。确保先测试笔记本，然后再提交拉取请求。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> 是一个托管笔记本环境，可以轻松编辑和运行笔记本文档。GitHub 中的笔记本通过将路径传递给 Colab 网址加载到 Google Colab 中，例如，位于 GitHub 中以下位置的笔记本： <a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a><br> 可以通过以下网址加载到 Google Colab 中：<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

有一个 <a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a> Chrome 扩展程序，可以在 GitHub 上浏览笔记本时执行此网址替换。这在仓库复刻中打开笔记本时非常有用，因为顶部按钮始终链接到 TensorFlow Docs 的 `master` 分支。

### 笔记本格式设置

借助笔记本格式设置工具，可使 Jupyter 笔记本源差异一直并更易于审查。由于笔记本写作在文件输出、缩进、元数据和其他非指定字段方面不同，`nbfmt` 使用偏好 TensorFlow 文档 Colab 工作流的默认设置。要设置笔记本格式，请安装 <a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools/" external="class">TensorFlow 文档笔记本工具</a>并运行 `nbfmt` 工具：

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

对于 TensorFlow 文档项目，将执行和测试*没有*输出单元的笔记本；而*带有*保存输出单元的笔记本将按原样发布。`nbfmt` 遵从笔记本状态并使用 `--remove_outputs` 选项显式移除输出单元。

要创建新笔记本，请复制并编辑 <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 文档笔记本模板</a>。

### 在 Colab 中编辑

在 Google Colab 环境中，双击单元可以编辑文本块和代码块。文本单元使用 Markdown 并且应遵循 [TensorFlow 文档风格指南](./docs_style.md)。

点击 *File &gt; Download .pynb*，从 Colab 中下载笔记本文件。将此文件提交到您的[本地 Git 仓库](##set_up_a_local_git_repo)并发送拉取请求。

要创建新笔记本，请复制并编辑 <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 笔记本模板</a>。

### Colab-GitHub 工作流

您可以直接 从Google Colab 编辑和更新复刻的 GitHub 仓库，而不用下载笔记本文件并使用本地 Git 工作流：

1. 在您复制的 <var>username</var>/docs 仓库中，使用 GitHub Web 界面<a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">创建新分支</a>。
2. 导航到要编辑的笔记本文件。
3. 在 Google Colab 中打开笔记本：使用网址替换或 *Open in Colab* Chrome 扩展程序。
4. Edit the notebook in Colab.
5. 点击 *File &gt; Save a copy in GitHub...* 从 Colab 中向您的仓库提交变更。保存对话框应链接到相应的仓库和分支。添加一条有意义的提交消息。
6. 保存后，浏览到您的仓库或者 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 仓库，GitHub 应提示您创建拉取请求。
7. 维护者会审查拉取请求。

成功：您的变更已被 TensorFlow 文档接受。

## 翻译

TensorFlow 团队与社区和供应商合作，为 tensorflow.org 提供翻译。笔记本和其他技术内容的翻译位于 <a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> GitHub 仓库中。请通过 <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">TensorFlow GitLocalize 项目</a>提交拉取请求。

英文文档是*事实来源*，翻译应尽可能遵循这些指南。也就是说，翻译是为它们所服务的社区编写的。如果英语术语、措辞、风格或语气不能翻译成另一种语言，请使用适合读者的翻译。

语言支持由多种因素决定，包括但不限于站点指标和需求、社区支持、<a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">英语水平</a>、受众偏好和其他指标。由于每种支持的语言都会产生成本，因此会删除未维护的语言。我们将在 <a class="external" href="https://blog.tensorflow.org/">TensorFlow 博客</a>或 <a class="external" href="https://twitter.com/TensorFlow">Twitter</a> 上公布对新语言的支持。

如果您的首选语言不受支持，欢迎您为开源贡献者维护社区复刻。这些内容不会发布到 tensorflow.org。

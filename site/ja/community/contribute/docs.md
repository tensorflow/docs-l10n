# TensorFlow ドキュメントへの貢献

TensorFlow はドキュメントの貢献を歓迎します-
あなたがドキュメントを改善してくれれば、TensorFlowライブラリそのものが改善されます。
tensorflow.orgにあるドキュメントは以下のカテゴリに分類されます。


* *APIリファレンス* —[API リファレンスドキュメント](https://www.tensorflow.org/api_docs/)は
  [TensorFlowソースコード](https://github.com/tensorflow/tensorflow)の docstring から生成されています。
* *解説ドキュメント* —これは [チュートリアル](https://www.tensorflow.org/tutorials)や
  [ガイド](https://www.tensorflow.org/guide)、その他のTensorFlowのコードの一部ではない文書です。
  これらのドキュメントは GitHubリポジトリの
  [tensorflow/docs](https://github.com/tensorflow/docs)にあります。
* *コミュニティによる翻訳* —これらのガイドやチュートリアルは
  コミュニティが翻訳したものです。
  コミュニティによる翻訳活動はすべて
  [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site)リポジトリで活発に行われています。

いくつかの[TensorFlow プロジェクト](https://github.com/tensorflow)は、
ドキュメント用のソースファイルを別リポジトリのコードの近く、
たいていは `docs/` 配下のディレクトリに置いてあります。
 各プロジェクトの `CONTRIBUTING.md` ファイルを参照するか、
 メンテナに連絡を取り貢献したいという思いを伝えましょう。

TensorFlow docsコミュニティに参加するのであれば、以下の2つを行ってください。

* GitHubの[tensorflow/docs](https://github.com/tensorflow/docs)リポジトリを見てください。
* [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs)フォーラムに参加してください。

## API リファレンス

リファレンスドキュメントを更新するには、
[ソースファイル](https://www.tensorflow.org/code/tensorflow/python/)を見つけて、
シンボルの<a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a>を編集します。
tensorflow.orgの多くのAPIリファレンスページには、
シンボルが定義されているソースファイルへのリンクを含みます。
docstring は<a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown形式</a>をサポートしており、
どんな<a href="http://tmpvar.com/markdown.html" class="external">Markdownプレビューアー</a>でも(ほぼ)プレビューすることができます。

リファレンスドキュメントの品質と、
短期集中型ドキュメント作成(訳注:doc sprints)とコミュニティに
深くかかわる方法を知るために、
[TensorFlow 2 API Docs advice](https://docs.google.com/document/d/1e20k9CuaZ_-hp25-sSd8E8qldxKPKQR-SkwojYr_r-U/preview)を参照してください。

### バージョンとブランチ

サイトの[API リファレンス](https://www.tensorflow.org/api_docs/python/tf)のバージョンは、
通常最新の安定バイナリを元に作成されています。
これは `pip install tensorflow` でインストールされる
パッケージのバージョンと一致しています。

デフォルトの TensorFlow パッケージは、メインリポジトリである
<a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>の安定版ブランチ `rX.x` から構築されます。
リファレンスドキュメントは、
<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>、
<a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>、
そして<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>の、
ソースコードのコメントとdocstringから生成されます。

TensorFlowドキュメントの前のバージョンは、
[ブランチ rX.x](https://github.com/tensorflow/docs/branches) として
TensorFlow Docsリポジトリで手に入れることができます。
新しいバージョンがリリースされると、
新しいブランチが追加されていきます。

### APIドキュメントを構築する

注意：このステップではAPI docstringを編集、
もしくはプレビューする必要はありません。
ただtensorflow.orgで利用されるHTMLを生成するだけです。

#### Python reference

The `tensorflow_docs` package includes the generator for the
[Python API reference docs](https://www.tensorflow.org/api_docs/python/tf). To
install:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

To generate the TensorFlow 2 reference docs, use the
`tensorflow/tools/docs/generate2.py` script:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

Note: This script uses the *installed* TensorFlow package to generate docs and
only works for TensorFlow 2.x.


## Narrative documentation

TensorFlow [guides](https://www.tensorflow.org/guide) and
[tutorials](https://www.tensorflow.org/tutorials) are written as
<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a>
files and interactive
<a href="https://jupyter.org/" class="external">Jupyter</a> notebooks. Notebooks
can be run in your browser using
<a href="https://colab.research.google.com/notebooks/welcome.ipynb"
   class="external">Google Colaboratory</a>.
The narrative docs on [tensorflow.org](https://www.tensorflow.org) are built
from the
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
`master` branch. Older versions are available in GitHub on the `rX.x` release
branches.

### Simple changes

The easiest way to make straightforward documentation updates to Markdown files
is to use GitHub's
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web-based
file editor</a>. Browse the
[tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)
repository to find the Markdown that roughly corresponds to the
<a href="https://www.tensorflow.org">tensorflow.org</a> URL structure. In the
upper right corner of the file view, click the pencil icon
<svg version="1.1" width="14" height="16" viewBox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path></svg>
to open the file editor. Edit the file and then submit a new pull request.

### Set up a local Git repo

For multi-file edits or more complex updates, it's better to use a local Git
workflow to create a pull request.

Note: <a href="https://git-scm.com/" class="external">Git</a> is the open source
version control system (VCS) used to track changes to source code.
<a href="https://github.com" class="external">GitHub</a> is an online service
that provides collaboration tools that work with Git. See the
<a href="https://help.github.com" class="external">GitHub Help</a> to set up
your GitHub account and get started.

The following Git steps are only required the first time you set up a local
project.

#### Fork the tensorflow/docs repo

On the
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
GitHub page, click the *Fork* button
<svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path></svg>
to create your own repo copy under your GitHub account. Once forked, you're
responsible for keeping your repo copy up-to-date with the upstream TensorFlow
repo.

#### Clone your repo

Download a copy of *your* remote <var>username</var>/docs repo to your local
machine. This is the working directory where you will make changes:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:<var>username</var>/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### Add an upstream repo to keep up-to-date (optional)

To keep your local repository in sync with `tensorflow/docs`, add an *upstream*
remote to download the latest changes.

Note: Make sure to update your local repo *before* starting a contribution.
Regular syncs to upstream reduce the chance of a
<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">merge conflict</a>
when you submit your pull request.

Add a remote:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git remote add <var>upstream</var> git@github.com:tensorflow/docs.git</code>

# View remote repos
<code class="devsite-terminal">git remote -v</code>
origin    git@github.com:<var>username</var>/docs.git (fetch)
origin    git@github.com:<var>username</var>/docs.git (push)
<var>upstream</var>  git@github.com:tensorflow/docs.git (fetch)
<var>upstream</var>  git@github.com:tensorflow/docs.git (push)
</pre>

To update:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout master</code>
<code class="devsite-terminal">git pull <var>upstream</var> master</code>

<code class="devsite-terminal">git push</code>  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub workflow

#### 1. Create a new branch

After you update your repo from `tensorflow/docs`, create a new branch from the
local *master* branch:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout -b <var>feature-name</var></code>

<code class="devsite-terminal">git branch</code>  # List local branches
  master
* <var>feature-name</var>
</pre>

#### 2. Make changes

Edit files in your favorite editor and please follow the
[TensorFlow documentation style guide](./docs_style.md).

Commit your file change:

<pre class="prettyprint lang-bsh">
# View changes
<code class="devsite-terminal">git status</code>  # See which files have changed
<code class="devsite-terminal">git diff</code>    # See changes within files

<code class="devsite-terminal">git add <var>path/to/file.md</var></code>
<code class="devsite-terminal">git commit -m "Your meaningful commit message for the change."</code>
</pre>

Add more commits, as necessary.

#### 3. Create a pull request

Upload your local branch to your remote GitHub repo
(github.com/<var>username</var>/docs):

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

After the push completes, a message may display a URL to automatically
submit a pull request to the upstream repo. If not, go to the
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
repo—or your own repo—and GitHub will prompt you to create a pull request.

#### 4. Review

Maintainers and other contributors will review your pull request. Please
participate in the discussion and make the requested changes. When your pull
request is approved, it will be merged into the upstream TensorFlow docs repo.

Success: Your changes have been accepted to the TensorFlow documentation.

There is a separate publishing step to update
[tensorflow.org](https://www.tensorflow.org) from the GitHub repo. Typically,
changes are batched together and the site is updated on a regular cadence.

## Interactive notebooks

While it's possible to edit the notebook JSON file with GitHub's
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web-based file editor</a>,
it's not recommended since malformed JSON can corrupt the file. Make sure to
test the notebook before submitting a pull request.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>
is a hosted notebook environment that makes it easy to edit—and run—notebook
documentation. Notebooks in GitHub are loaded in Google Colab by passing the
path to the Colab URL, for example,
the notebook located in GitHub here:
<a href="https://&#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://&#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br/>
can be loaded into Google Colab at this URL:
<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>
<!-- github.com path intentionally formatted to hide from import script. -->

There is an
<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>
Chrome extension that performs this URL substitution when browsing a notebook on
GitHub. This is useful when opening a notebook in your repo fork, because the
top buttons always link to the TensorFlow Docs `master` branch.

### Notebook formatting

To create a new notebook, copy and edit the
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow
notebook template</a>.

Notebooks are stored on disk as JSON and the various notebook implementations
format the JSON differently. Diff tools and version control don't handle this
very well, so TensorFlow Docs enforces a standard notebook formatting.

The
[nbfmt.py](https://github.com/tensorflow/docs/blob/master/tools/nbfmt.py)
script applies this formatting:

```
# Run it on a single file
python nbfmt.py path/to/notebooks/example.ipynb

# Run  it on a whole directory
python nbfmt.py path/to/notebooks/
```

For the same reasons notebook output should be cleared before submission
(except in a few special cases). In Colab the "private outputs" option is used
to enforce this. Other notebook implementations do not recognize this option.
You can clear the outputs by passing `--preserve_outputs=False` to `nbfmt`:

```
python nbfmt.py --preserve_outputs=False path/to/notebooks/example.ipynb
```

### Edit in Colab

Within the Google Colab environment, double-click cells to edit text and code
blocks. Text cells use Markdown and should follow the
[TensorFlow docs style guide](./docs_style.md).

Download notebook files from Colab with *File > Download .pynb*. Commit
this file to your [local Git repo](##set_up_a_local_git_repo) and send a pull
request.

To create a new notebook, copy and edit the
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow notebook template</a>.

### Colab-GitHub workflow

Instead of downloading a notebook file and using a local Git workflow, you can
edit and update your forked GitHub repo directly from Google Colab:

1. In your forked <var>username</var>/docs repo, use the GitHub web UI to
   <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">create a new branch</a>.
2. Navigate to the notebook file to edit.
3. Open the notebook in Google Colab: use the URL swap or the *Open in Colab*
   Chrome extension.
4. Edit the notebook in Colab.
5. Commit the changes to your repo from Colab with
   *File > Save a copy in GitHub...*. The save dialog should link to the
   appropriate repo and branch. Add a meaningful commit message.
6. After saving, browse to your repo or the
   <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
   repo, GitHub should prompt you to create a pull request.
7. The pull request is reviewed by maintainers.

Success: Your changes have been accepted to the TensorFlow documentation.


## Community translations

Community translations are a great way to make TensorFlow accessible all over
the world. To update a translation, find or add a file in the
[language directory](https://github.com/tensorflow/docs/tree/master/site) that
matches the same directory structure of the `en/` directory. The English docs
are the *source-of-truth* and translations should follow these guides as close
as possible. That said, translations are written for the communities they serve.
If the English terminology, phrasing, style, or tone does not translate to
another language, please use a translation appropriate for the reader.

Note: The API reference is *not* translated for tensorflow.org.

There are language-specific docs groups that make it easier for translation
contributors to organize. Please join if you're an author, reviewer, or just
interested in building out TensorFlow.org content for the community:

* Chinese (Simplified): [docs-zh-cn@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)
* Italian: [docs-it@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-it)
* Japanese: [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)
* Korean: [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
* Russian: [docs-ru@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru)
* Turkish: [docs-tr@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-tr)

### Review notifications

All documentation updates require a review. To collaborate more efficiently with
the TensorFlow translation communities, here are some ways to keep on top of
language-specific activity:

* Join a language group listed above to receive an email for any *created* pull
  request that touches the <code><a
  href="https://github.com/tensorflow/docs/tree/master/site">site/<var>lang</var></a></code>
  directory for that language.
* Add your GitHub username to the `site/<lang>/REVIEWERS` file to get
  automatically comment-tagged in a pull request. When comment-tagged, GitHub
  will send you notifications for all changes and discussion in that pull
  request.

### Keep code up-to-date in translations

For an open source project like TensorFlow, keeping documentation up-to-date is
challenging. After talking with the community, readers of translated content
will tolerate text that is a little out-of-date, but out-of-date code is
frustrating. To make it easier to keep the code in sync, use the
[nb-code-sync](https://github.com/tensorflow/docs/blob/master/tools/nb_code_sync.py)
tool for the translated notebooks:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">./tools/nb_code_sync.py [--lang=en] site/<var>lang</var>/notebook.ipynb</code>
</pre>

This script reads the code cells of a language notebook and checks it against the
English version. After stripping the comments, it compares the code blocks and
updates the language notebook if they are different. This tool is particularly
useful with an interactive git workflow to selectively add hunks of the file to
the commit using: `git add --patch site/lang/notebook.ipynb`

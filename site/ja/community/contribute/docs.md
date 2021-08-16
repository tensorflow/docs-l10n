# TensorFlow ドキュメント作成への貢献

TensorFlow はドキュメント作成への貢献を歓迎します。ドキュメントの改善は、TensorFlow ライブラリそのものが改善につながります。 tensorflow.org にあるドキュメントは以下のカテゴリに分類されます。

- *API リファレンス* — [API リファレンスドキュメント](https://www.tensorflow.org/api_docs/)は [TensorFlow ソースコードの](https://github.com/tensorflow/tensorflow) docstring から生成されています。
- *物語風のドキュメント* —これは [チュートリアル](https://www.tensorflow.org/tutorials)、 [ガイド](https://www.tensorflow.org/guide)、さらに他の TensorFlow のコードの一部ではない執筆物です。これらドキュメントは GitHub リポジトリの [tensorflow/docs](https://github.com/tensorflow/docs) にあります。
- *コミュニティによる翻訳* —これらのガイドやチュートリアルはコミュニティが翻訳したものです。コミュニティによる翻訳活動は [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) で活発に行われています。

いくつかの [TensorFlow プロジェクト](https://github.com/tensorflow)では、ドキュメント用のソースファイルはコードの近く（多くの場合、 `docs/` の下のディレクトリ）に配置されています。貢献するには、各プロジェクトの `CONTRIBUTING.md` ファイルを参照するか、メンテナに連絡してください。

TensorFlow docs コミュニティに参加するには

- GitHub の [tensorflow/docs](https://github.com/tensorflow/docs) リポジトリをご覧ください。
- [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs) にご参加ください。

## API リファレンス

リファレンスドキュメントを更新するには、[ソースファイル](https://www.tensorflow.org/code/tensorflow/python/)を見つけて、シンボルの <a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a> を編集します。 tensorflow.org の多くの API リファレンスページには、シンボルが定義されているソースファイルへのリンクを含みます。docstrings は <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown 形式</a>をサポートしており、(ほとんどの)<a href="http://tmpvar.com/markdown.html" class="external">ビューアー（リンク先は Markdown previewer のものです）</a>でプレビューすることができます。

リファレンスドキュメントの品質と、短期集中型ドキュメント作成（訳注: doc sprints）とコミュニティに深くかかわる方法を知るには、[TensorFlow 2 API Docs アドバイス](https://docs.google.com/document/d/1e20k9CuaZ_-hp25-sSd8E8qldxKPKQR-SkwojYr_r-U/preview)を参照してください。

### バージョンとブランチ

[API リファレンス](https://www.tensorflow.org/api_docs/python/tf)は、通常最新の安定バイナリを元に作成されています。これは`pip install tensorflow`でインストールされるパッケージと一致しています。

デフォルトの TensorFlow パッケージは、メインリポジトリである <a>tensorflow/tensorflow</a> のステーブル版ブランチ <code>rX.x</code> から構築されます。リファレンスドキュメントは、<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>、 <a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>、および、<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a> のソースコードのコメントと docstring から生成されます。

TensorFlow ドキュメントの前のバージョンは、[ブランチ rX.x](https://github.com/tensorflow/docs/branches)のように TensorFlow Docs リポジトリで手に入れることができます。新しいバージョンがリリースされると、新しいブランチが追加されていきます。

### API ドキュメントの構築

注：このステップは、tensorflow.org で利用される HTML を生成するためにのみ必要です。API docstring を編集、または、プレビューするためにはこのステップでは必要はありません。

#### Python 向けリファレンス

`tensorflow_docs`パッケージには [Python API リファレンスドキュメント](https://www.tensorflow.org/api_docs/python/tf)のジェネレータが含まれます。以下のようにインストールします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

TensorFlow 2 リファレンスドキュメントを生成するには、`tensorflow/tools/docs/generate2.py`スクリプトを使用します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

注：このスクリプトは、*インストールされている* TensorFlowパッケージを利用し、ドキュメントを生成します。このスクリプトは、TensorFlow 2.xのみで使用できます。

## 解説ドキュメント

TensorFlow の[ガイド](https://www.tensorflow.org/guide)と[チュートリアル](https://www.tensorflow.org/tutorials)は、<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> ファイル、および対話型の <a href="https://jupyter.org/" class="external">Jupyter</a> notebook として執筆されています。 Jupyter notebook は<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> を利用しているブラウザ上で動作します。[tensorflow.org](https://www.tensorflow.org) の解説ドキュメントは<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> の`master`ブランチから構築されています。古いバージョンは GitHub のリリースブランチ`rX.x`で入手できます。

### 簡単な変更

The easiest way to make straightforward documentation updates to Markdown files is to use GitHub's <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web-based file editor</a>. Browse the [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) repository to find the Markdown that roughly corresponds to the <a href="https://www.tensorflow.org">tensorflow.org</a> URL structure. In the upper right corner of the file view, click the pencil icon <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> to open the file editor. Edit the file and then submit a new pull request.

### ローカル Git リポジトリを設定する

複数ファイルの編集やより複雑な更新の場合は、ローカルのGitワークフローを使用してプルリクエストを作成することをお勧めします。

注：<a href="https://git-scm.com/" class="external">Git</a>はオープンソースのバージョン管理システム (VCS)で、ソースコードの変更を追跡するために使われます。<a href="https://github.com" class="external">GitHub</a>は、gitを使用したコラボレーションを簡単にするオンラインサービスです。GitHubのアカウントを設定して利用するには、<a href="https://help.github.com" class="external">GitHubのヘルプ</a>を参照してください。

The following Git steps are only required the first time you set up a local project.

#### tensorflow/docs リポジトリをフォークする

<a href="https://github.com/tensorflow/docs" class="external">tensorflow / docs</a> GitHubページで、*フォーク*ボタンをクリックすることで<svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> あなたのGitHubアカウント上にリポジトリのコピーを作成します。フォークしたら、あなたには、上流のTensorFlowリポジトリに追随してリポジトリコピーを最新に保つ責任があります。

#### リポジトリをクローンする

*あなたの*リモートリポジトリ<var>username</var>/docsのコピーをローカルマシンにダウンロードします。これは、変更を加える作業用ディレクトリです。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 上流リポジトリを設定して、最新の状態に保つ（オプション）

ローカルリポジトリと`tensorflow/docs`の同期を維持するには、 *上流*リモートを追加して最新の変更をダウンロードします。

注意：貢献し始める*前*に、確実にローカルリポジトリを更新しておいてください。定期的に同期しておくと、プルリクエスト発行時の<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">マージコンフリクト</a>の危険性を減らすことができます。

以下のように、リモートリポジトリを追加します。

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git remote add &lt;var&gt;upstream&lt;/var&gt; git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
&lt;var&gt;upstream&lt;/var&gt;  git@github.com:tensorflow/docs.git (fetch)
&lt;var&gt;upstream&lt;/var&gt;  git@github.com:tensorflow/docs.git (push)
</pre>

更新するには

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull &lt;var&gt;upstream&lt;/var&gt; master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub ワークフロー

#### 1. 新しいブランチを作成する

`tensorflow/docs`からリポジトリを更新した後、ローカル*マスター*ブランチから新しいブランチを作成します。

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. 変更を加える

任意のエディタでファイルを編集します。このとき [TensorFlowのドキュメントスタイルガイド](./docs_style.md)に従うようにしてください。

以下のように、変更をコミットします。

<pre class="prettyprint lang-bsh"># View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

必要に応じて、さらにコミットを追加します。

#### 3. プルリクエストを作成する

以下のようにして、ローカルブランチをGitHub上のリモートリポジトリ（github.com/<var>username</var>/docs）にアップロードします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

pushが完了したら、自動的に上流リポジトリへ発行されたプルリクエストのURLが表示されるかもしれません。もし表示されなかった場合は、<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> リポジトリ—それかあなたのフォークしたリポジトリ—にアクセスすると、GitHubがプルリクエストを発行するように促してくるでしょう。

#### 4. レビューする

メンテナや他のコントリビューターがあなたのプルリクエストをレビューします。ディスカッションに参加し、必要な変更を加えてください。プルリクエストが承認されると、上流のTensorFlowのドキュメントリポジトリにマージされます。

成功：あなたの変更は TensorFlow ドキュメントに取り込まれました。

GitHubリポジトリから[tensorflow.org](https://www.tensorflow.org)を更新するには、別の公開手順があります。通常、変更はまとめて処理され、サイトは定期的に更新されます。

## 対話的 notebook

While it's possible to edit the notebook JSON file with GitHub's <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">web-based file editor</a>, it's not recommended since malformed JSON can corrupt the file. Make sure to test the notebook before submitting a pull request.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>は、 ノートブックドキュメントの編集や実行を簡単にする、ホストされたノートブック環境です。GitHubのnotebookは、ColabのURLを渡してあげることで、Google Colabから読み込まれています。例えば、GitHubのここ（<a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br> ）に置いてあるノートブックは、Google Colabの以下のURLで読み込まれています。<br><a>https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

GitHubのノートブックをブラウジングする際のこのようなURLの置き換えを行ってくれる、<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>という、Chromeの拡張機能があります。これは、画面上部のボタンがTensorFlowドキュメントの`master`ブランチに常にリンクしているので、あなたのフォークリポジトリでノートブックを開くときにとても役立ちます。

### ノートブックのフォーマット

ノートブックのフォーマットツールを使用すると、Jupyter ノートブックのソースの違いの一貫性が保たれ、簡単に確認できます。ノートブックのオーサリング環境は、ファイル出力、インデント、メタデータ、およびその他の指定されていないフィールドに関して異なるため、 `nbfmt`は、TensorFlow docs Colab ワークフローを優先して、オーバーライド可能なデフォルトを使用します。ノートブックをフォーマットするには、<a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools/" external="class"> TensorFlow docs ノートブックツール</a>をインストールし、`nbfmt`ツールを実行します。

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

TensorFlow ドキュメントプロジェクトの場合、*出力セルなし*のノートブックは実行およびテストされます。保存された*出力セルがある*ノートブックはそのまま公開されます。`nbfmt`はノートブックの状態を尊重し、`--remove_outputs`オプションを使用して出力セルを明示的に削除します。

To create a new notebook, copy and edit the <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow docs notebook template</a>.

### Google Colab での編集作業

Google Colab環境内で、セルをダブルクリックしてテキストとコードブロックを編集します。テキストセルはMarkdownを使用しており、 [TensorFlowのドキュメントスタイルガイド](./docs_style.md)に従う必要があります。

Download notebook files from Colab with *File &gt; Download .pynb*. Commit this file to your [local Git repo](##set_up_a_local_git_repo) and send a pull request.

Jupyter notebookを新規に作成するときは<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow notebookテンプレート</a>をコピーして編集してください。

### Colab と GitHub のワークフロー

Instead of downloading a notebook file and using a local Git workflow, you can edit and update your forked GitHub repo directly from Google Colab:

1. フォークした <var>username</var>/docsリポジトリで、<a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">新しいブランチを作成する</a>ためにGitHub Web UIを使います。
2.  編集する notebook ファイルに移動します。
3. notebookをGoogle Colabで開きます。このときURLの変換にはChromeの拡張機能の*Open in Colab*を使います。
4.  Colab で notebook を編集します。
5. *ファイル &gt; GitHubにコピーを保存*で変更をリポジトリにコミットします。保存ダイアログは適切なリポジトリのブランチへのリンクになっているはずです。そうしたら意味のあるコミットメッセージを入力します。
6. After saving, browse to your repo or the <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> repo, GitHub should prompt you to create a pull request.
7.  メンテナにプルリクエストをレビューしてもらいます。

成功：あなたの変更はTensorFlowドキュメントに取り込まれました。

## 翻訳

TensorFlow チームは、コミュニティやベンダーと協力して、tensorflow.org の翻訳を提供しています。ノートブックやその他の技術コンテンツの翻訳は、<a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> GitHubリポジトリにあります。プルリクエストは <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">TensorFlowGitLocalize プロジェクト</a>を介して送信してください。

英語のドキュメントは*信頼できる唯一の情報源*であり、可能な限りこれらのガイドに従って翻訳する必要がありますが、翻訳はコミュニティのためのものです。英語の用語、言い回し、スタイル、または語調が他の言語に適していない場合は、読者に適するように翻訳してください。

言語サポートは、ウェブサイトの指標、需要、コミュニティサポート、<a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">英語能力</a>、読者の好みなど、さまざまな要因により決定されます。言語のサポートにはコストがかかるため、保守されていない言語は削除されます。新しい言語のサポートは、<a class="external" href="https://blog.tensorflow.org/">TensorFlow ブログ</a>または <a class="external" href="https://twitter.com/TensorFlow">Twitter</a> で発表されます。

ご希望の言語がサポートされていない場合は、オープンソースの貢献者のためにコミュニティフォークを維持することができます。これらは tensorflow.org には公開されません。

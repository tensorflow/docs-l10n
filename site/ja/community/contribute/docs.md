# TensorFlowドキュメント作成への貢献

TensorFlow はドキュメント作成への貢献を歓迎します。ドキュメントの改善は、TensorFlow ライブラリそのものが改善につながります。 tensorflow.org にあるドキュメントは以下のカテゴリに分類されます。

- *API リファレンス*  — API リファレンスドキュメントは TensorFlow ソースコードの docstring から生成されています。
- 物語風のドキュメント —これは チュートリアル、 ガイド、さらに他の TensorFlow のコードの一部ではない執筆物です。これらドキュメントは GitHub リポジトリの [tensorflow/docs](https://github.com/tensorflow/docs) にあります。
- コミュニティによる翻訳 —これらのガイドやチュートリアルはコミュニティが翻訳したものです。コミュニティによる翻訳活動は [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) で活発に行われています。

いくつかの TensorFlow プロジェクトでは、ドキュメント用のソースファイルはコードの近く（多くの場合、` docs/ `の下のディレクトリ）に配置されています。貢献するには、各プロジェクトの `CONTRIBUTING.md `ファイルを参照するか、メンテナに連絡してください。

TensorFlow docsコミュニティに参加するのであれば、以下の2つを行ってください。

- GitHub の <a>tensorflow/docs</a> リポジトリを見る。
- [TensorFlow フォーラム](https://discuss.tensorflow.org/)の[ドキュメント](https://discuss.tensorflow.org/tag/docs)タグをフォローする。

## APIリファレンス

詳細については、[ TensorFlow API ドキュメント寄稿者ガイド](docs_ref.md)を参照してください。このガイドは<a href="https://www.python.org/dev/peps/pep-0257/" class="external">ソースファイル</a>を見つけて、シンボルの <a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a> を編集する方法を説明します。tensorflow.org の多くの API リファレンスページには、シンボルが定義されているソースファイルへのリンクを含みます。docstrings は  <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a> 形式をサポートしており、ほとんどの <a href="http://tmpvar.com/markdown.html" class="external">Markdown プレビューア</a>でプレビューすることができます。

### バージョンとブランチ

API リファレンスは、通常最新の安定バイナリを元に作成されています。これは `pip install tensorflow` でインストールされるパッケージと一致しています。

デフォルトの TensorFlow パッケージは、メインリポジトリである <a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a> のステーブル版ブランチ` rX.x` から構築されます。リファレンスドキュメントは、<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>、 <a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>、および、<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a> のソースコードのコメントと docstring から生成されます。

TensorFlow ドキュメントの前のバージョンは、ブランチ rX.xのように TensorFlow Docs リポジトリで手に入れることができます。新しいバージョンがリリースされると、新しいブランチが追加されていきます。

### APIドキュメントの構築

注：このステップは、tensorflow.org で利用される HTML を生成するためにのみ必要です。API docstring を編集、または、プレビューするためにはこのステップでは必要はありません。

#### Python向けリファレンス

 `tensorflow_docs `パッケージには Python API リファレンスドキュメントのジェネレータが含まれます。以下のようにインストールします。

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

注：このスクリプトは、インストールされている TensorFlowパッケージを利用し、ドキュメントを生成します。このスクリプトは、TensorFlow 2.xのみで使用できます。

## 解説ドキュメント

TensorFlow のガイドとチュートリアルは、<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> ファイル、および対話型の <a href="https://jupyter.org/" class="external">Jupyter notebook</a> として執筆されています。 Jupyter notebook は<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a> を利用しているブラウザ上で動作します。tensorflow.org の解説ドキュメントは <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> の`master`ブランチから構築されています。古いバージョンは GitHub のリリースブランチ`rX.x`で入手できます。

### 簡単な変更

直接的にMarkdownファイルに更新をかける最も簡単な方法は、GitHubのWEBベースのファイルエディタを使うことです。[tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en) リポジトリをブラウジングしマークダウンを探しましょう。このリポジトリは<a href="https://www.tensorflow.org">tensorflow.org</a>のURL構造によく似ています。ファイルビューの上方右側にある鉛筆のアイコンをクリックしてください。するとファイルエディタが開きます。編集して新しいプルリクエストを送ってください。

### ローカル Git リポジトリを設定する

複数ファイルの編集やより複雑な更新の場合は、ローカルのGitワークフローを使用してプルリクエストを作成することをお勧めします。

注： <a href="https://git-scm.com/" class="external">Git</a>はオープンソースのバージョン管理システム (VCS)で、ソースコードの変更を追跡するために使われます。GitHubは、gitを使用したコラボレーションを簡単にするオンラインサービスです。<a href="https://github.com" class="external">GitHub</a> のアカウントを設定して利用するには、GitHubのヘルプを参照してください。

次のGitの手順は、ローカルプロジェクトを初めてセットアップするときにのみ必要です。

#### tensorflow/docsリポジトリをフォークする

<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> GitHubページで、フォークボタンをクリックすることで あなたのGitHubアカウント上にリポジトリのコピーを作成します。フォークしたら、あなたには、上流のTensorFlowリポジトリに追随してリポジトリコピーを最新に保つ責任があります。

#### リポジトリをクローンする

あなたのリモートリポジトリusername/docsのコピーをローカルマシンにダウンロードします。これは、変更を加える作業用ディレクトリです。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 上流リポジトリを設定して、最新の状態に保つ（オプション）

ローカルリポジトリと `tensorflow/docs` の同期を維持するには、 上流リモートを追加して最新の変更をダウンロードします。

注意：貢献し始める前に、確実にローカルリポジトリを更新しておいてください。定期的に同期しておくと、プルリクエスト発行時のマージコンフリクトの危険性を減らすことができます。

以下のように、リモートリポジトリを追加します。

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

更新するには

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHubワークフロー

#### 1.新しいブランチを作成する

`tensorflow/docs` からリポジトリを更新した後、ローカルマスターブランチから新しいブランチを作成します。

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2.変更を加える

任意のエディタでファイルを編集します。このとき [TensorFlowのドキュメントスタイルガイドに従うようにしてください](./docs_style.md).。

以下のように、変更をコミットします。

<pre class="prettyprint lang-bsh"># View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

必要に応じて、さらにコミットを追加します。

#### 3.プルリクエストを作成する

以下のようにして、ローカルブランチをGitHub上のリモートリポジトリ（github.com/username/docs）にアップロードします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

pushが完了したら、自動的に上流リポジトリへ発行されたプルリクエストのURLが表示されるかもしれません。もし表示されなかった場合は、<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> リポジトリ—それかあなたのフォークしたリポジトリ—にアクセスすると、GitHubがプルリクエストを発行するように促してくるでしょう。

#### 4.レビューする

メンテナや他のコントリビューターがあなたのプルリクエストをレビューします。ディスカッションに参加し、必要な変更を加えてください。プルリクエストが承認されると、上流のTensorFlowのドキュメントリポジトリにマージされます。

成功：あなたの変更はTensorFlowドキュメントに取り込まれました。

GitHubリポジトリから [tensorflow.org](https://www.tensorflow.org) を更新するには、別の公開手順があります。通常、変更はまとめて処理され、サイトは定期的に更新されます。

## 対話的ノートブック

GitHubのWebベースのファイルエディタでnotebookのJSONファイルを編集できますが、JSONに誤りが含まれているとファイルが破損する可能性があるので、お勧めしません。プルリクエストを発行する前にnotebookのテストを確実に行うようにしてください。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>は、 ノートブックドキュメントの編集や実行を簡単にする、ホストされたノートブック環境です。GitHubのnotebookは、ColabのURLを渡してあげることで、Google Colabから読み込まれています。例えば、GitHubのここ<a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb </a>に置いてあるノートブックは、Google Colabの以下のURLで読み込まれています。<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

GitHubのノートブックをブラウジングする際のこのようなURLの置き換えを行ってくれる、<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>という、Chromeの拡張機能があります。これは、画面上部のボタンがTensorFlowドキュメントの `master` ブランチに常にリンクしているので、あなたのフォークリポジトリでノートブックを開くときにとても役立ちます。

### ノートブックのフォーマット

ノートブックのフォーマットツールを使用すると、Jupyter ノートブックのソースの違いの一貫性が保たれ、簡単に確認できます。ノートブックのオーサリング環境は、ファイル出力、インデント、メタデータ、およびその他の指定されていないフィールドに関して異なるため、 nbfmtは、TensorFlow docs Colab ワークフローを優先して、オーバーライド可能なデフォルトを使用します。ノートブックをフォーマットするには、 TensorFlow docs ノートブックツールをインストールし、nbfmtツールを実行します。

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

TensorFlow ドキュメントプロジェクトの場合、出力セルなしのノートブックは実行およびテストされます。保存された出力セルがあるノートブックはそのまま公開されます。nbfmtはノートブックの状態を尊重し、--remove_outputsオプションを使用して出力セルを明示的に削除します。

Jupyter notebookを新規に作成するときはTensorFlow notebookテンプレートをコピーして編集してください。

### Google Colabでの編集作業

Google Colab環境内で、セルをダブルクリックしてテキストとコードブロックを編集します。テキストセルはMarkdownを使用しており、 TensorFlowのドキュメントスタイルガイドに従う必要があります。

Download notebook files from Colab with ファイル &gt; Download.pynbを使ってColabからJupyter notebookファイルをダウンロードしてください。ファイルをあなたのローカルGitリポジトリにコミットして、プルリクエストを送りましょう。

Jupyter notebookを新規に作成するときはTensorFlow notebookテンプレートをコピーして編集してください。

### ColabとGitHubのワークフロー

Jupyter notebookファイルをダウンロードしてローカルGitワークフローを回す代わりに、Google ColabからあなたがフォークしたGitHubリポジトリを直接編集して、更新することができます。手順は以下のとおりです。

1. フォークした username/docsリポジトリで、新しいブランチを作成するためにGitHub Web UIを使います。
2. 編集したいJupyter notebookファイルに移動します。
3. notebookをGoogle Colabで開きます。このときURLの変換にはChromeの拡張機能のOpen in Colabを使います。
4. Colabでnotebookを編集します。
5. ファイル &gt; GitHubにコピーを保存で変更をリポジトリにコミットします。保存ダイアログは適切なリポジトリのブランチへのリンクになっているはずです。そうしたら意味のあるコミットメッセージを入力します。
6. 保存した後は、あなたのリポジトリか <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> リポジトリをブラウジングしてください。GitHub上でプルリクエストを作成するように促されるはずです。
7. メンテナにプルリクエストをレビューしてもらいます。

成功：あなたの変更はTensorFlowドキュメントに取り込まれました。

## 翻訳

TensorFlow チームは、コミュニティやベンダーと協力して、tensorflow.org の翻訳を提供しています。ノートブックやその他の技術コンテンツの翻訳は、<a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> GitHubリポジトリにあります。プルリクエストは <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">TensorFlow GitLocalize プロジェクト</a>を介して送信してください。

英語のドキュメントは信頼できる唯一の情報源であり、可能な限りこれらのガイドに従って翻訳する必要がありますが、翻訳はコミュニティのためのものです。英語の用語、言い回し、スタイル、または語調が他の言語に適していない場合は、読者に適するように翻訳してください。

言語サポートは、ウェブサイトの指標、需要、コミュニティサポート、英語能力、読者の好みなど、さまざまな要因により決定されます。言語のサポートにはコストがかかるため、保守されていない言語は削除されます。新しい言語のサポートは、<a class="external" href="https://twitter.com/TensorFlow">TensorFlow</a>  ブログまたは <a class="external" href="https://twitter.com/TensorFlow">Twitter</a>で発表されます。

ご希望の言語がサポートされていない場合は、オープンソースの貢献者のためにコミュニティフォークを維持することができます。これらは tensorflow.org には公開されません。

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

#### Python向けリファレンス

`tensorflow_docs` パッケージには
[Python API リファレンスドキュメント](https://www.tensorflow.org/api_docs/python/tf)のジェネレータが含まれます。
インストールするには以下のようにします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

TensorFlow 2 リファレンスドキュメントを生成するには、
`tensorflow/tools/docs/generate2.py` スクリプトを以下のようにして利用します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

注意：このスクリプトは
*インストールされている* TensorFlowパッケージを利用し、
TensorFlow 2.x 限定で動作します。


## 解説ドキュメント

TensorFlowの[ガイド](https://www.tensorflow.org/guide)と[チュートリアル](https://www.tensorflow.org/tutorials)は、
<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a>ファイル、および対話型の<a href="https://jupyter.org/" class="external">Jupyter</a> notebooksとして執筆されています。
Jupyter notebookは<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>を利用して、
ブラウザ上で動作します。
[tensorflow.org](https://www.tensorflow.org)の解説ドキュメントは<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>の
`master` ブランチから構築されています。
古いバージョンはGitHubのリリースブランチ `rX.x` で入手できます。

### 簡単な変更を行う

直接的にMarkdownファイルに更新をかける最も簡単な方法は、
GitHubの<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">WEBベースのファイルエディタ</a>を使うことです。
[tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)リポジトリをブラウジングしマークダウンを探しましょう。
このリポジトリは<a href="https://www.tensorflow.org">tensorflow.org</a>のURL構造によく似ています。
ファイルビューの上方右側にある鉛筆のアイコン<svg version="1.1" width="14" height="16" viewBox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path></svg>をクリックしてください。
するとファイルエディタが開きます。
編集して新しいプルリクエストを送ってください。

### ローカルGitリポジトリを準備する

複数ファイルの編集やより複雑な更新の場合は、
ローカルのGitワークフローを使用して
プルリクエストを作成することをお勧めします。

注意：<a href="https://git-scm.com/" class="external">Git</a>はオープンソースのバージョン管理システム (VCS)で、
ソースコードの変更を追跡するために使われます。
<a href="https://github.com" class="external">GitHub</a>はGitと相性の良いツールを提供するオンラインサービスです。
<a href="https://help.github.com" class="external">GitHubのヘルプ</a>を見て、GitHubのアカウントを設定して利用を開始しましょう。

Gitでの次の手順は、
ローカルプロジェクトを初めてセットアップするときにのみ必要です。

#### tensorflow/docsリポジトリをフォークする

<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> GitHubページで、
*フォーク*ボタン <svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path></svg> をクリックすることで
あなたのGitHubアカウント上にリポジトリのコピーを作成します。
フォークさせたら、あなたには上流のTensorFlowリポジトリに追随して、
リポジトリコピーを最新に保つ責任があります。

#### リポジトリをクローンする

*あなたの*リモートリポジトリ
<var>username</var>/docs
のコピーをローカルマシンにダウンロードします。
これは、変更を加える作業用ディレクトリです。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:<var>username</var>/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 上流リポジトリを設定して、最新の状態に保つ（オプション）

ローカルリポジトリと `tensorflow/docs`の同期を維持するには、
*上流*リモートリポジトリを追加して最新の変更をダウンロードします。

注意：貢献し始める*前*に、確実にローカルリポジトリを更新しておいてください。
定期的に同期しておくと、
プルリクエスト発行時の<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">マージコンフリクト</a>の危険性を減らすことができます。

以下のように、リモートリポジトリを追加します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git remote add <var>upstream</var> git@github.com:tensorflow/docs.git</code>

# リモートリポジトリを参照する。
<code class="devsite-terminal">git remote -v</code>
origin    git@github.com:<var>username</var>/docs.git (fetch)
origin    git@github.com:<var>username</var>/docs.git (push)
<var>upstream</var>  git@github.com:tensorflow/docs.git (fetch)
<var>upstream</var>  git@github.com:tensorflow/docs.git (push)
</pre>

更新をかける場合は以下のようにします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout master</code>
<code class="devsite-terminal">git pull <var>upstream</var> master</code>

<code class="devsite-terminal">git push</code>  # 変更した内容をGitHubアカウント(デフォルトはoriginブランチ)にプッシュします。
</pre>

### GitHubワークフロー

#### 1. 新しいブランチを作成する

`tensorflow/docs`からリポジトリを更新した後、
次のようにローカル *master* ブランチから新しいブランチを作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout -b <var>feature-name</var></code>

<code class="devsite-terminal">git branch</code>  # ローカルブランチのリストを表示します
  master
* <var>feature-name</var>
</pre>

#### 2. 変更を加える

お好みのエディタでファイルを編集します。
このとき[TensorFlowのドキュメントスタイルガイド](./docs_style.md)に従うようにしてください。

以下のように、変更をコミットします。

<pre class="prettyprint lang-bsh">
# 変更内容を確認する。
<code class="devsite-terminal">git status</code>  # どのファイルが変更されたかを表示します。
<code class="devsite-terminal">git diff</code>    # ファイルの変更内容を表示します。

<code class="devsite-terminal">git add <var>path/to/file.md</var></code>
<code class="devsite-terminal">git commit -m "Your meaningful commit message for the change."</code>
</pre>

必要に応じて、さらにコミットを追加します。

#### 3. プルリクエストを作成する

以下のようにして、ローカルブランチを
GitHub上のリモートリポジトリ(github.com/<var>username</var>/docs)にアップロードします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

pushが完了したら、上流リポジトリへ発行されたプルリクエストのURLが、
自動的に表示されるかもしれません。
もし表示されなかった場合は、<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> リポジトリ—
またはフォークしたリポジトリ—にアクセスすると、
GitHubがプルリクエストを発行するように促してくるでしょう。

#### 4. レビューする

メンテナや他のコントリビューターが
あなたのプルリクエストをレビューしてくれます。
議論に参加し、必要な変更を加えてください。
プルリクエストが承認されると、
上流のTensorFlowのドキュメントリポジトリにマージされます。

上手くいきましたね！
あなたの変更がTensorFlowドキュメントに受け入れられました。

GitHubリポジトリから[tensorflow.org](https://www.tensorflow.org)を更新するには、
別の公開手順があります。
通常、変更はまとめて処理され、サイトは定期的に更新されます。

## 対話的な Jupyter notebook

GitHubの <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">WEBベースのファイルエディタ</a>で
Jupyter notebookのJSONファイルが編集可能とはいえ、
誤りを含むJSONのせいでファイルがおかしくなることもあるので、
お勧めしません。
プルリクエストを発行する前に
Jupyter notebookのテストを確実に行うようにしてください。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>は、Jupyter notebookドキュメントの編集、
そして実行を簡単にしてくれる、
ホストされたJupyter notebook環境です。
GitHub上のJupyter notebookは、
ColabのURLを渡してあげることで、
Google Colab上に読み込まれています。
例えば、GitHubのここ
<a href="https://&#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://&#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br/>
に置いてあるJupyter notebookは、Google Colabの以下のURLで読み込まれています。
<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>
<!-- github.com path intentionally formatted to hide from import script. -->

GitHub上のJupyter notebookをブラウジングする際の
このようなURLの置き換えを行ってくれる、
<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>という、Chromeの拡張機能があります。
画面上部のボタンがTensorFlowドキュメントの
`master` ブランチにいつでもリンクしているので、
あなたのフォークリポジトリで Jupyter notebook を開くときにとても役立ちます。

### Jupyter notebook フォーマット

新しいJupyter notebookを作るためには、
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow Jupyter notebook テンプレート</a>をコピーして編集してください。

Jupyter notebook は JSON ファイルとしてディスクに保存されており、
様々なJupyter notebookは異なるJSONフォーマットで実装されています。
こうなると差分比較ツールやバージョン管理システムは上手く役に立たないので、
TensorFlowドキュメントは標準的なJupyter notebookのフォーマットを強制しています。

[nbfmt.py](https://github.com/tensorflow/docs/blob/master/tools/nbfmt.py) というスクリプトはこのフォーマットを適用します。

```
# 1ファイルに対して実行
python nbfmt.py path/to/notebooks/example.ipynb

# ディレクトリ全体に対して実行
python nbfmt.py path/to/notebooks/
```

同じような理由で、Jupyter notebookの成果物は
提案の前段階で問題ないものにしておくべきです
(数少ない特別な場合を除いて)。
これを強制するために、Google Colab では
"private outputs" オプションを利用しています。
他のJupyter notebook の実装はこのオプションを認識しません。
以下のように、 `nbfmt` に `--preserve_outputs=False` オプションを渡すことで、
成果物の問題を除去することができます。

```
python nbfmt.py --preserve_outputs=False path/to/notebooks/example.ipynb
```

### Google Colabでの編集作業

Google Colab環境では、セルをダブルクリックして
テキストとコードブロックを編集します。
テキストセルはMarkdownを使用しており、
[TensorFlowのドキュメントスタイルガイド](./docs_style.md)に従う必要があります。

*ファイル > .pynbをダウンロード*(訳注：2020/06/06時点では *.ipynbをダウンロード* に変更されている)を使って
ColabからJupyter notebookファイルをダウンロードしてください。
このファイルをあなたの[ローカルGitリポジトリ](##set_up_a_local_git_repo)にコミットして、
プルリクエストを送りましょう。

Jupyter notebookを新規に作成するときは
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow notebook テンプレート</a>をコピーして編集してください。

### Google Colab-GitHub間のワークフロー

Jupyter notebookファイルをダウンロードして
ローカルGitワークフローを回す代わりに、
Google ColabからあなたがフォークしたGitHubリポジトリを
直接編集して更新することができます。
以下のような手順です。

1. フォークした <var>username</var>/docs リポジトリで、
   <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">新しいブランチを作る</a>ためにGitHub Web UIを使います。
2. 編集したい Jupyter notebook ファイルに移動します。
3. Jupyter notebook を Google Colabで開きます。
   このときURLの変換には Chrome の拡張機能の
   *Open in Colab* を使います。
4. Google Colabで Jupyter notebook を編集します。
5. *ファイル > GitHub にコピーを保存*で
   変更をリポジトリにコミットします。
   保存ダイアログは適切なリポジトリの
   ブランチへのリンクになっているはずです。
   意味のあるコミットメッセージを入力します。
6. 保存した後は、あなたのリポジトリか
   <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> リポジトリをブラウジングしてください。
   GitHub上でプルリクエストを作成するように促されるはずです。
7. メンテナにプルリクエストをレビューしてもらいます。

やりました！変更はTensorFlowドキュメントに取り込まれました。

## コミュニティによる翻訳

コミュニティによる翻訳はTensorFlowを世界中に広めるための素晴らしい方法です。
翻訳をするためには、`en/` とディレクトリ構造が一致している[各言語のディレクトリ](https://github.com/tensorflow/docs/tree/master/site) 上のファイルを見つけるか追加します。
英語版ドキュメントは*真実の源泉*であり、
翻訳結果はなるべくこれらのガイドと近いものにすべきです。
とは言ったものの、翻訳は彼らが従事するコミュニティのために行われています。
英語の専門用語、言い回し、文体、語調の他言語への翻訳が難しいときは、
読み手にとって適切な翻訳を行ってください。

注意：APIリファレンスはtensorflow.orgのために翻訳*しないこと*。

翻訳を行うコントリビュータを簡単に取りまとめるための、
各言語のドキュメンテーショングループがあります。
あなたが執筆者であったりレビュアーであった場合、
また単にTensorFlow.orgのコミュニティ向けコンテンツの構築に
興味のあるだけの方もぜひ参加してください。
例えば以下のようなグループがあります。

* （簡体字）中国語: [docs-zh-cn@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)
* イタリア語: [docs-it@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-it)
* 日本語: [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)
* 韓国語: [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
* ロシア語: [docs-ru@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru)
* トルコ語: [docs-tr@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-tr)

### レビューの通知

すべてのドキュメントの更新にはレビューが必要です。
TensorFlow翻訳コミュニティとより効率的に協力するために、
言語グループ毎の活動状況を深く知るためのいくつかの方法を示します。

* 上記に挙げた言語グループに参加し、
  その言語の<code><a
  href="https://github.com/tensorflow/docs/tree/master/site">site/<var>lang</var></a></code>ディレクトリに関係する、
  *作成された*いかなるプルリクエストのメールも受け取るようにしましょう。
* プルリクエストの際に自動でコメントでタグ付けされるように、
  `site/<lang>/REVIEWERS` ファイルに GitHub ユーザー名を追加しましょう。
  そうすればプルリクエスト上の変更や議論について
  GitHub からの通知が来るようになるでしょう。

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

# TensorFlowのドキュメントに貢献する

TensorFlowはドキュメントへの貢献を歓迎します。ドキュメントへの貢献は、TensorFlowのライブラリ自体の改善につながります。tensorflow.orgにおけるドキュメントは次の種類に分かれます。

* **API レファレンス** —[API ドキュメント](https://www.tensorflow.org/api_docs/) は
  [TensorFlow ソースコード](https://github.com/tensorflow/tensorflow) のdocstringから生成されています。
* **説明書** —[チュートリアル](https://www.tensorflow.org/tutorials)、
  [ガイド](https://www.tensorflow.org/guide)などのTensorflowのコードに含まれない文書を指します。GitHubのレポジトリ
  [tensorflow/docs](https://github.com/tensorflow/docs) にあります。
* **コミュニティ翻訳** —チュートリアルやガイドはコミュニティが翻訳しています。コミュニティによる翻訳はレポジトリ
  [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) にあります。

一部の[TensorFlow プロジェクト](https://github.com/tensorflow) はドキュメントのソースファイルを別のレポジトリのコードの近く（通常 `docs/` ディレクトリ内）に置いています。貢献するには、プロジェクトの`CONTRIBUTING.md` ファイルを参照するか、プロジェクトのメンテナーにご連絡ください。

 TensorFlow ドキュメントのコミュニティに参加するには、

*  GitHub
  repository [tensorflow/docs](https://github.com/tensorflow/docs) をご覧ください。
* [docs@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs)にご参加ください。

## API レファレンス

レファレンスドキュメントを更新するには、対応する
[ソースファイル](https://www.tensorflow.org/code/tensorflow/python/)をみつけて
<a href="https://www.python.org/dev/peps/pep-0257/" class="external">docstring</a>を編集してください。
tensorflow.org上のAPIレファレンスページでは、多くの場合ソースファイルの対応する箇所へのリンクが貼ってあります。Docstrings は
<a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a>
をサポートしており、 お好きな<a href="http://tmpvar.com/markdown.html" class="external">Markdown プレビュアー</a>で（大体）プレビューすることができます。

レファレンスドキュメントの品質について、またDoc sprintsやコミュニティに参加する方法については、
[TensorFlow 2 API Docs advice](https://docs.google.com/document/d/1e20k9CuaZ_-hp25-sSd8E8qldxKPKQR-SkwojYr_r-U/preview) をご覧ください。

### バージョンとブランチ

サイトの[API レファレンス](https://www.tensorflow.org/api_docs/python/tf)のデフォルトバージョンは、最新の安定版になっています（`pip install tensorflow`でインストールできるものに対応します）。

デフォルトのTensorFlowパッケージは、メインレポジトリ
<a href="https://github.com/tensorflow/tensorflow" class="external">tensorflow/tensorflow</a>
中の安定版ブランチ `rX.x` からビルドされます。
レファレンスドキュメントは
<a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>,
<a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>,
<a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a> 用のソースコード中のコメントやdocstringから生成されます。


TensorFlowのドキュメントの以前のバージョンはTensoflowドキュメントレポジトリの[rX.x branches](https://github.com/tensorflow/docs/branches)から入手できます。[rX.x branches](https://github.com/tensorflow/docs/branches)は新バージョンのリリース時に追加されます。

### API ドキュメントをビルドする

Note: APIのdocstingのプレビューや編集を行う場合はこのステップは不要です。tensorflow.orgで使われているHTMLを生成する場合のみ必要です。

#### Python レファレンス

`tensorflow_docs` パッケージには
[Python API レファレンスドキュメント](https://www.tensorflow.org/api_docs/python/tf) を生成するツールが含まれています。パッケージは
<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

でインストールできます。TensorFlow 2 のレファレンスドキュメントを生成するには
`tensorflow/tools/docs/generate2.py` のスクリプトを用いて

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

とします。

Note: このスクリプトは **インストール済みの** TensorFlow パッケージを使ってドキュメントを生成し、TensorFlow 2.x.のみに対応しています。


## 説明書

TensorFlowの[ガイド](https://www.tensorflow.org/guide) と
[チュートリアル](https://www.tensorflow.org/tutorials) は
<a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a>
ファイルもしくはインタラクティブな
<a href="https://jupyter.org/" class="external">Jupyter</a> ノートブックで書かれています。ノートブックは
<a href="https://colab.research.google.com/notebooks/welcome.ipynb"
   class="external">Google Colaboratory</a>を使ってブラウザで動かすことができます。
[tensorflow.org](https://www.tensorflow.org) の説明書は
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
の`master`ブランチからビルドされています。以前のバージョンはGitHubの `rX.x` リリースブランチから入手できます。

### 簡単な変更

ドキュメントの簡単な更新や修正はGitHubの
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">ウェブベース ファイルエディタ</a>を使うと容易です。
Browse the [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site/en)
レポジトリから、<a href="https://www.tensorflow.org">tensorflow.org</a> のURL構造に概ね対応するMarkdownやノートブックのファイルをみつけます。 ファイルビューの右上端にある鉛筆アイコン
<svg version="1.1" width="14" height="16" viewBox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path></svg>
をクリックして、ファイルエディタを開きます。ファイルを編集して新しいプルリクエストを送信してください。

### ローカルGitレポジトリを作成する

複数のファイルの編集や複雑な更新を行う場合には、ローカルのGitワークフローを用いてプルリクエストを行うのがよいです。

Note: <a href="https://git-scm.com/" class="external">Git</a> はソースコードの変更を追跡するオープンソースのバージョン管理システムです。
<a href="https://github.com" class="external">GitHub</a>はGitを用いた共同作業ツールを提供するオンラインサービスです。GitHubアカウントを作成して始めるには
<a href="https://help.github.com" class="external">GitHub ヘルプ</a> をご覧ください。

以下のGitステップは、はじめてローカルプロジェクトを立ち上げる場合にのみ必要です。

#### tensorflow/docs レポジトリをフォーク

GitHubの<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
ページで**Fork** ボタン
<svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path></svg>
をクリックして、自分のGitHubアカウントにレポジトリのコピーを作成します。フォークを行った場合、自分でコピーしたレポジトリにはTensorFlowの公式レポジトリの更新を常に反映させておく必要があります。

#### 自分のレポジトリをクローン

 **自分の** リモートレポジトリ <var>username</var>/docs のコピーを自分のローカルマシン上にダウンロードします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:<var>username</var>/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>
`./docs` はワーキングディレクトリになり、この中で作業を行います。

#### 上流のレポジトリを加えて更新を反映させる (オプション)

`tensorflow/docs`と自分のローカルレポジトリを同期するには、最新の変更をダウンロードするための **上流**リモートブランチを追加します。

Note: 編集を行う**前に**、必ずローカルレポジトリを更新するようにしてください。定期的に上流ブランチと同期を行うと、プルリクエストを送信する際の
<a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">マージコンフリクト</a>
の可能性が減ります。

リモートブランチの追加方法
<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git remote add <var>upstream</var> git@github.com:tensorflow/docs.git</code>

# リモートレポジトリを確認
<code class="devsite-terminal">git remote -v</code>
origin    git@github.com:<var>username</var>/docs.git (fetch)
origin    git@github.com:<var>username</var>/docs.git (push)
<var>upstream</var>  git@github.com:tensorflow/docs.git (fetch)
<var>upstream</var>  git@github.com:tensorflow/docs.git (push)
</pre>

更新方法

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout master</code>
<code class="devsite-terminal">git pull <var>upstream</var> master</code>

<code class="devsite-terminal">git push</code>  # 自分のGitHubアカウント（デフォルトはorigin）に変更をプッシュする
</pre>

### GitHub ワークフロー

#### 1. 新しいブランチを作成する

`tensorflow/docs`から自分のレポジトリを更新したあとで、ローカルの**master**ブランチから新しいブランチを作成します。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git checkout -b <var>feature-name</var></code>

<code class="devsite-terminal">git branch</code>  # ローカルブランチの一覧を表示
  master
* <var>feature-name</var>
</pre>

#### 2. 変更を加える

お好みのエディタを使ってファイルを編集してください。
[TensorFlowドキュメントスタイルガイド](./docs_style.md)に従ってください。

ファイルの変更をコミットします。

<pre class="prettyprint lang-bsh">
# 変更を確認
<code class="devsite-terminal">git status</code>  # どのファイルが変更されたのかを確認
<code class="devsite-terminal">git diff</code>    # ファイル内の変更を確認する

<code class="devsite-terminal">git add <var>path/to/file.md</var></code>
<code class="devsite-terminal">git commit -m "Your meaningful commit message for the change."</code>
</pre>

必要な場合はさらにコミットを追加してください。

#### 3. プルリクエストを作成する

自分のGitHubリモートレポジトリ(github.com/<var>username</var>/docs)にローカルブランチをアップロードします。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

プッシュが完了したあとで、プルリクエストを上流レポジトリに自動的に送信したURLを含むメッセージが表示されます。表示されない場合は、
<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>
レポジトリ—もしくはご自身のレポジトリ—にアクセスしてください。GitHubがプルリクエストを作成するように誘導するはずです。

#### 4. レビュー

メンテナーやほかの貢献者がプルリクエストをレビューします。議論に参加し、必要に応じて変更を行うようにしてください。プルリクエストが承認されると、上流のTensorFlowドキュメントレポジトリにマージされます。

Success: TensorFlowドキュメントに変更が承認されました。

GitHubレポジトリから[tensorflow.org](https://www.tensorflow.org)への変更の反映は独立して行われます。通常、変更をとりまとめ、定期的にサイトを更新します。

## インタラクティブなノートブック

ノートブックのJSONファイルをGitHub上の
<a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">ウェブベースファイルエディタ</a>で編集することは可能ですが、不正なJSONがファイルを破壊することがあるため非推奨です。プルリクエストを送信する前に、必ずノートブックのテストを行ってください。

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>
はオンラインのノートブック環境で、ノートブックドキュメントを簡単に編集・実行することができます。GitHub上のノートブックはパスをColaboratoryに渡すことで読み込まれます。例えば、GitHub上の
<a href="https://   &#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://  &#103;ithub.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br/>
は、Google ColaboratoryではURL
<a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a> で読み込むことができます。


Chromeの拡張機能には、<a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a>
があります。GitHub上のノートブックを閲覧している際に、上に示したURLの置換を行います。フォークしたレポジトリでノートブックを開いているとき、一番上のボタンが常にTensorFlowドキュメントの`master`ブランチのリンクになっているため便利です。

### Colaboratoryで編集する

Google Colaboratoryの環境では、セルをダブルクリックしてテキストやコードブロックを編集します。テキストセルはMarkdown記法を用いており、[TensorFlow ドキュメントスタイルガイド](./docs_style.md)に従わなくてはなりません。

Colaboratoryからノートブックファイルを **ファイル > .ipynbをダウンロード** によってダウンロードします。このファイルを自分の[ローカルGitレポジトリ](##set_up_a_local_git_repo)にコミットしてプルリクエストを送信してください。

新しいノートブックを作成するには、
<a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow notebook template</a>をコピーしてから編集してください。

### Colaboratory-GitHub ワークフロー

ノートブックファイルをダウンロードするのではなく、フォークした自分のGitHubレポジトリをGoogle Colaboratoryから直接編集・更新することができます。

1. 自分でフォークした <var>username</var>/docs レポジトリで、GitHubのweb UIを使って
  <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">新しいブランチを作成</a>してます。
2. 編集するノートブックファイルに移動します。
3. ノートブックをGoogle Colaboratoryで開きます。上述のURLの置換か、Chrome拡張**Open in Colab**を使ってください。
4. Colaboratoryでノートブックを編集します。
5. **ファイル > GitHubにコピーを保存** を選び、Colaboratoryからレポジトリに変更をコミットします。保存ダイアログでは、適切なレポジトリ・ブランチを選択する必要があります。有益なコミットメッセージを追加します。
6. 保存後、自分のレポジトリかレポジトリ
   <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a>を開くと、GitHubがプルリクエストを作成するよう誘導します。
7. プルリクエストはメンテナーによりレビューが行われます。

Success: TensorFlowドキュメントに変更が承認されました。


## コミュニティ翻訳

コミュニティ翻訳を行うと、Tensorflowを世界中に広めることができます。翻訳を更新するには、[言語ディレクトリ](https://github.com/tensorflow/docs/tree/master/site)の`en/`以下の構造と対応するファイルを[言語ディレクトリ](https://github.com/tensorflow/docs/tree/master/site)に追加・更新してください。
英語版ドキュメントは **信頼できる情報源** であり、できる限り齟齬が生じないようにガイドを翻訳する必要があります。しかしながら、翻訳は貢献するコミュニティのためにあります。もしも英語版の用語、フレーズ、スタイル、口調がほかの言語に翻訳できない場合は、読者にとって適切な翻訳を用いてください。

Note: APIレファレンスはtensorflow.org用に翻訳され**ません**。

翻訳貢献者が集まりやすいように、言語別のドキュメントグループがあります。作者・レビュアーの方や、コミニュティーのためにTensorFlow.orgのコンテンツを作成することに興味がある方はご参加ください。

* 中国語 (簡体字): [docs-zh-cn@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)
* イタリア語: [docs-it@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-it)
* 日本語: [docs-ja@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ja)
* 韓国語: [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
* ロシア語: [docs-ru@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ru)
* トルコ語: [docs-tr@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-tr)

### レビューの通知

ドキュメントの更新にはすべてレビューが必要になります。より効率的にTensorFlowの翻訳コミニュティーと作業を行うために、言語別の活動を見逃さない方法を挙げます。

* 上に示した言語グループに参加して、 その言語のディレクトリ（<code><a
  href="https://github.com/tensorflow/docs/tree/master/site">site/<var>lang</var></a></code>）内に関わるプルリクエストの**作成**すべてについてのメールを受け取る。
* `site/<lang>/REVIEWERS`に自分のGitHubユーザーネームを加え、プルリクエストをcomment-taggedにする。comment-taggedとなった場合、GitHubはそのプルリクエストにおけるすべての変更と議論を通知します。

### 翻訳内のコードを最新に保つ

Tensorflowのようなオープニングソースのプロジェクトでは、ドキュメントを最新に保つのは一種の挑戦です。コミュニティにおける議論の後、翻訳されたコンテンツを読むとき、少し古いテキストには耐えられるかもしれませんが、古いコードは迷惑です。コードの同期を簡単にするには、翻訳されたノートブックに[nb-code-sync](https://github.com/tensorflow/docs/blob/master/tools/nb_code_sync.py)ツールを使用してください。

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">./tools/nb_code_sync.py [--lang=en] site/<var>lang</var>/notebook.ipynb</code>
</pre>

このスクリプトは、ノートブックのコードセルを読み込み、英語版と照らし合わせます。コメントを除去したあとでコードブロックを比較し、異なる場合にはノートブックを更新します。インタラクティブなGitワークフローの場合にこのツールは特に便利で、`git add --patch site/lang/notebook.ipynb`を使ってファイルの変更範囲をコミットに選択的に加えることができます。

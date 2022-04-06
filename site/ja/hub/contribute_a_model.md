<!--* freshness: { owner: 'maringeo' reviewed: '2021-11-25' review_interval: '6 months' } *-->

# プルリクエストを送信する

このページは、[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) GitHub リポジトリに Markdown ドキュメントを含むプルリクエストを送信する方法を説明しています。まず Markdown ファイルの記述方法に関する詳細を確認するには、[ドキュメントの作成ガイド](writing_documentation.md)をご覧ください。

## GitHub Actions のチェック

[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) リポジトリでは、GitHub Actions を使用して、PR のファイルの形式を検証しています。PR の検証に使用されるワークフローは [.github/workflows/contributions-validator.yml](https://github.com/tensorflow/tfhub.dev/blob/master/.github/workflows/contributions-validator.yml) に定義されています。ワークフローの外にある独自のブランチで検証ツールスクリプトを実行することは可能ですが、正しい pip パッケージ依存関係がすべてインストールされていることを確認する必要があります。

初めて貢献するコントリビューターは、[GitHub ポリシー](https://github.blog/changelog/2021-04-22-github-actions-maintainers-must-approve-first-time-contributor-workflow-runs/)に従って、レポジトリ管理者の承認を得てのみ自動チェックを実行することができます。パブリッシャーは最初の PR として、タイポを修正するかモデルのドキュメントを改善する小規模の PR を送信するか、パブリッシャーページのみを含む PR を送信することをお勧めします。それ以降の PR から、自動チェックを実行できるようになります。

重要: プルリクエストは自動チェックを通過してからのみレビューされます。

## PR を送信する

完全な Markdown ファイルは、次のいずれかの方法で、[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) のマスターブランチにプルすることができます。

### Git CLI で提供する

特定した Markdown ファイルのパスが `assets/docs/publisher/model/1.md` であるという前提で、標準の Git[Hub] の手順に従って、新たに追加されたファイルで新しいプルリクエストを作成することができます。

これにはまず、TensorFlow Hub の GitHub リポジトリをフォークし、[このフォークからプルリクエストを](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) TensorFlow Hub のマスターブランチに作成することから始まります。

以下は、フォークされたリポジトリのマスターブランチに新しいファイルを追加するために必要な、典型的な CLI の Git コマンドです。

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### GitHub GUI で提出する

もう少し簡単な提供方法として、GitHub のグラフィカルユーザーインターフェース (GUI) を利用する方法があります。GitHub では、[新規ファイル](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)や[ファイル編集](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)の PR を GUI から直接作成することができます。

1. <a>TensorFlow Hub の GitHub のページ</a>で<code>Create new file</code>ボタンを押します。
2. 適切なファイルパスを設定します。 `assets/docs/publisher/model/1.md`
3. 既存のマークダウンをコピーして貼り付けます。
4. 一番下で「Create a new branch for this commit and start a pull request（このコミットの新しいブランチを作成してプルリクエストを開始する）」を選択します。

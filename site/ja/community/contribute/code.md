# TensorFlow コードへの貢献

コントリビュータガイドのこのセクションは、損失関数の追加、テストカバレッジの改善、主要な設計変更のための RFC を作成などの貢献をはじめるのに役立ちます。TensorFlow の改善にご協力いただき、ありがとうございます。

## はじめる前に

TensorFlow プロジェクトにソースコードを提供する前に、プロジェクトの GitHub リポジトリにある`CONTRIBUTING.md` ファイルを確認してください。例として、コアの TensorFlow リポジトリの [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) ファイルをご覧ください。すべてのコードコントリビュータは、[コントリビュータライセンス契約](https://cla.developers.google.com/clas)（CLA）に署名する必要があります。

作業の重複を避けるために、重要な機能の作業を開始する前に[最新](https://github.com/tensorflow/community/tree/master/rfcs)または[提案中](https://github.com/tensorflow/community/labels/RFC%3A%20Proposed)の RFC を確認し、TensorFlow フォーラムの開発者に連絡してください（[developers@tensorflow.org](https://groups.google.com/u/1/a/tensorflow.org/g/developers)）。私たちは新しい機能の追加についてやや選択的であるため、既知の問題の解決に貢献いただくことでプロジェクトを支援していただくよう、お願いしています。

## 新しいコントリビュータのための issue

新しいコントリビュータには、TensorFlow コードベースへの最初の貢献を検索するときに、次のタグを探すことをお勧めします。新しいコントリビュータには最初に「good first issue」および「contributions welcome」プロジェクトに取り組むことを強くお勧めします。 このようなプロジェクトに取り組むことによりコントリビュータは、貢献ワークフローに慣れ、作業の中心となる開発者が新しいコントリビュータに慣れることができます。

- [good first issue](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue)
- [contributions welcome](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome)

大規模な問題や新機能への取り組みを支援するチームの採用に関心がある場合は、[developers@ group](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers) にメールでお問い合わせください。また、RFC の最新リストをご確認ください。

## コードレビュー

新機能、バグ修正、およびコードベースに対するその他の変更は、コードレビューの対象となります。

プルリクエストとしてプロジェクトに貢献されたコードのレビュー作業は、TensorFlow 開発の重要なコンポーネントです。そのため、ほかの開発者が提出したコードをレビューすることをお勧めしています。特にその機能を使用する可能性が高いのであれば、尚更です。

コードレビューでは、以下の点を検討してください。

- *これを TensorFlow に含めるべきでしょうか？* 使用される可能性が高いと考えられますか？ TensorFlow ユーザーとして、この変更を気に入って使用しますか？ この変更は TensorFlow の範囲内ですか？ 新しい機能を維持するためのコストは、そのメリットに値するでしょうか？

- *コードは TensorFlow API と一貫性がありますか？*パブリック関数、クラス、およびパラメータは適切な名前で直感的に設計されていますか？

- *ドキュメントは含まれていますか？*すべてのパブリック関数、クラス、パラメータ、戻り値の型、および格納された属性は、TensorFlow の規則に従って名付けられ、明確に文書化されていますか？新しい機能は TensorFlow のドキュメントに記載されていますか？可能な場合は例が示されていますか？ドキュメントは適切にレンダリングされますか？

- *コードは人間が読めるように書かれていますか？*冗長性は低いですか？変数名は、明確性または一貫性のために改善する必要がありますか？コメントを追加する必要がありますか？役に立たない、または無関係なコメントを削除する必要がありますか？

- *コードは効率的ですか？*より効率的に実行するために簡単に書き換えることはできますか？

- コードは以前のバージョンの TensorFlow と*下位互換性*がありますか？

- 新しいコードは他のライブラリに*新しい依存関係*を追加しますか？

## テストとテストカバレッジの改善

高品質な単体テストは、TensorFlow 開発プロセスにとって非常に重要です。そのために、Docker イメージが使用されます。 テスト関数に適切な名前が付けられていること、アルゴリズムの有効性やコードのさまざまなオプションをチェックします。

すべての新機能とバグ修正には、*適切なテストカバレッジが含まれている必要があります*。また、新しいテストケースの貢献や既存のテストの改善も歓迎します。現時点でバグが発生していない場合でも、既存のテストが完了していないことが判明した場合は、問題を報告し、可能であればプルリクエストを送信してください。

各 TensorFlow プロジェクトのテスト手順の具体的な詳細については、GitHub のプロジェクトリポジトリにある`README.md`と`CONTRIBUTING.md`ファイルをご覧ください。

*十分なテスト*を実行する上で特に注意する点：

- *すべてのパブリック関数とクラス*がテストされていますか？
- *パラメータの妥当性*、それらの値、値のタイプ、および組み合わせがテストされていますか？
- テストでは*コードが正しい*ことが検証されていますか？また、コードは*ドキュメントに記載されている*とおりのことを実行しますか？**
- 変更がバグ修正の場合、*非回帰テスト*が含まれていますか？
- テストは*継続的インテグレーションに合格*しますか？
- *コードのすべての行がテストされていますか？*そうでない場合、例外は合理的かつ明示的ですか？

問題を見つけた場合は、コントリビュータがそれらの問題を理解して解決できるように支援してください。

## エラーメッセージまたはログの改善

エラーメッセージやログを改善するための貢献を歓迎します。

## 貢献ワークフロー

コードの貢献（バグ修正、新しい開発、テストの改善）はすべて、GitHub 中心のワークフローに従います。TensorFlow 開発に参加するには、GitHub アカウントを設定し、以下を行います。

1. 作業する予定のリポジトリをフォークします。プロジェクトリポジトリページに移動し、［*フォーク*］ボタンを選択します。自分のユーザー名の下にリポジトリのコピーが作成されます。（リポジトリをフォークする方法の詳細については、[このガイド](https://help.github.com/articles/fork-a-repo/)をご覧ください。）

2. リポジトリをローカルシステムにクローンします。

    `$ git clone git@github.com:your-user-name/project-name.git`

3. 作業を保持する新しいブランチを作成します。

    `$ git checkout -b new-branch-name`

4. 新しいコードを記述し、テストを作成して実行します。

5. 変更をコミットします。

    `$ git add -A`

    `$ git commit -m "commit message here"`

6. 変更を GitHub リポジトリにプッシュします。

    `$ git push origin branch-name`

7. *プルリクエスト*（PR）を開きます。GitHub の元のプロジェクトリポジトリに移動します。最近プッシュされたブランチに関するメッセージが表示され、プルリクエストを開くかどうかが聞かれます。プロンプトに従い、*リポジトリ間で比較*して、PR を送信します。コミッターにメールが送信されます。周知させるためにメーリングリストにメールを送信することをお勧めします（詳細については、[「 PR に関する GitHub ガイド」](https://help.github.com/articles/creating-a-pull-request-from-a-fork)をご覧ください。

8. メンテナや他のコントリビュータがあなたの PR を*レビュー*します。会話に参加して、*必要な変更を追加してください*。PR が承認されると、コードがマージされます。

*次の貢献に取り組む前に*、ローカルリポジトリが最新であることを確認してください。

1. 上流をリモートに設定します。(これを行う必要があるのは、プロジェクトごとに 1 回だけです。毎回行う必要はありません。)

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. ローカルマスターブランチに切り替えます。

    `$ git checkout master`

3. 上流から変更をプルダウンします。

    `$ git pull upstream master`

4. 変更を GitHub アカウントにプッシュします。（オプションですが、良い習慣です。）

    `$ git push origin master`

5. 新しい作業を開始する場合は、新しいブランチを作成します。

    `$ git checkout -b branch-name`

追加の `git` および GitHub リソース:

- [Git ドキュメント](https://git-scm.com/documentation)
- [Git 開発ワークフロー](https://docs.scipy.org/doc/numpy/dev/development_workflow.html)
- [マージ中に発生した競合の解決](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)

## コントリビュータチェックリスト

- [貢献ガイドライン](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)を読む
- [行動規範](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md)を読む
- [コントリビュータライセンス契約（CLA）](https://cla.developers.google.com/)に署名したことを確認する
- 変更が[ガイドライン](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution)に準拠しているかどうかを確認する
- 変更が[TensorFlowコーディングスタイル](https://www.tensorflow.org/community/contribute/code_style)と一致しているかどうかを確認する
- [ユニットテストを実行する](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#running-unit-tests)

# TFDS リポジトリに貢献する

ライブラリにご関心いただきありがとうございます！これほど意気込みに満ちたコミュニティが存在することを嬉しく思っています。

## はじめに

- TFDS に新しい方は、[リクエストされたデータセット](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22dataset+request%22+sort%3Areactions-%2B1-desc)の 1 つを実装することから始めるのが最も簡単です。一番リクエストの多いデータセットに専念してください。手順については、[ガイドに従ってください](https://www.tensorflow.org/datasets/add_dataset)。
- 課題、機能リクエスト、バグなどには、新しいデータセットを追加するよりもはるかに大きなインパクトがあります。これらは TFDS コミュニティ全体にメリットをもたらすためです。[potential contribution list](https://github.com/tensorflow/datasets/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+-label%3A%22dataset+request%22+)（潜在的な貢献リスト）をご覧ください。[contribution-welcome](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) のラベルが付いたものから着手してください。これらは手始めに対応できる規模の小さな自己完結型の簡単な課題です。
- 割り当て済みであっても、しばらく更新のないバグにぜひ対応してください。
- 課題の割り当てを待つ必要はありません。対応し始める際に、コメントを残してください :)
- 課題に関心があっても対応の仕方がわからない場合は、すぐに気軽に問い合わせください。早期にフィードバックが必要な場合は、PR 草案を送信してください。
- 不要な重複作業を行わないためにも、[pending Pull Requests](https://github.com/tensorflow/datasets/pulls)（保留中のプルリクエスト）リストを確認し、作業中の課題にはコメントを残すようにしてください。

## MNIST モデルをビルドする

### リポジトリを複製する

はじめに、[Tensorflow Datasets](https://github.com/tensorflow/datasets) リポジトリを Clone してダウンロードし、ローカルにインストールします。

```sh
git clone https://github.com/tensorflow/datasets.git
cd datasets/
```

開発依存関係をインストールします。

```sh
pip install -e .  # Install minimal deps to use tensorflow_datasets
pip install -e ".[dev]"  # Install all deps required for testing and development
```

データベース固有のすべての依存関係をインストールする `pip install -e ".[tests-all]"` もあることに注意してください。

### Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/) を使って開発する場合、リポジトリには開発を支援する[事前定義済みの設定](https://github.com/tensorflow/datasets/tree/master/.vscode/settings.json)（適切なインデント設定や pylint など）がいくつか含まれています。

注意: VS Code でテスト検出を有効しても、VS Code の [#13301](https://github.com/microsoft/vscode-python/issues/13301) と [#6594](https://github.com/microsoft/vscode-python/issues/6594) のバグにより失敗する可能性があります。この問題を解決するには、テスト検出ログを確認してください。

- TensorFlow 警告メッセージが出力されている場合は、[こちらの修正](https://github.com/microsoft/vscode-python/issues/6594#issuecomment-555680813)を試してみてください。
- インストールされているはずのインポートが欠落している理由で検出が失敗する場合は、`dev` pip インストールを更新する PR を送信してください。

## PR チェックリスト

### CLA に署名する

このプロジェクトへの貢献には、コントリビューターライセンス契約（CLA）が伴います。あなた（またはあなたの従業員）が貢献への著作権を保持する場合、この契約によってプロジェクトの一環としてあなたの貢献を使用し再配布する権限を私たちに付与することができます。[https://cla.developers.google.com/](https://cla.developers.google.com/) にアクセスし、現在提出済みの契約を確認するか、新しい契約に署名してください。

一般的に CLA の提出は 1 回だけで済みます。過去に提出したことがあれば、別のプロジェクトであっても、もう一度署名する必要はありません。

### ベストプラクティスに従う

- 可読性は重要です。コードは、プログラミングのベストプラティス（重複を回避する、小さな自己完結型の関数に分解する、明示的な変数名など）に従ってください。
- 単純であるほど最適です（たとえば、実装が複数の小さな自己完結型の PR に分割されている方がレビューしやすくなります）。
- 必要に応じてテストを追加し、終了テストに合格することが推奨されます。
- [型指定の注釈](https://docs.python.org/3/library/typing.html)を追加してください。

### スタイルガイドを確認する

スタイルガイドは [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) に基づいており、これは [PEP 8 Python style guide](https://www.python.org/dev/peps/pep-0008) に従っています。新しいコードは [Black code style](https://github.com/psf/black/blob/master/docs/the_black_code_style.md) に従いますが、以下の点に注意してください。

- 行の長さ: 80
- インデントはスペース 4 個ではなく 2 個とする。
- 一重引用符 `'`

**重要:** コードに必ず `pylint` を実行し、コードのフォーマットが正しいことを確認してください。

```sh
pip install pylint --upgrade
pylint tensorflow_datasets/core/some_file.py
```

`yapf` を使ってファイルを自動フォーマットできますが、このツールは完璧ではありません。そのため、実行後にほぼ手動で修正する必要があります。

```sh
yapf tensorflow_datasets/core/some_file.py
```

`pylint` と `yapf` はいずれも `pip install -e ".[dev]"` でインストールされるはずですが、`pip install` を使って手動でインストールすることも可能です。VS Code を使用している場合は、これらのツールが UI に統合されています。

### ドキュメント文字列と型指定の注釈

クラスと関数は、ドキュメント文字列と型指定の注釈で文書化されている必要があります。ドキュメント文字列は、[Google スタイル](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)に従ってください。以下に例を示します。

```python
def function(x: List[T]) -> T:
  """One line doc should end by a dot.

  * Use `backticks` for code and tripple backticks for multi-line.
  * Use full API name (`tfds.core.DatasetBuilder` instead of `DatasetBuilder`)
  * Use `Args:`, `Returns:`, `Yields:`, `Attributes:`, `Raises:`

  Args:
    x: description

  Returns:
    y: description
  """
```

### ユニットテストを追加して実行する

新しい機能は、必ずユニットテストを使って検証してください。テストは、VS Code インターフェースを通じて実行できます。またはコマンドラインも使用可能です。たとえば、以下のようにします。

```sh
pytest -vv tensorflow_datasets/core/
```

`pytest` と `unittest`: これまで、`unittest` モジュールを使用してテストを記述してきましたが、新しいテストでは、より単純で自由度の高い最新の `pytest` を使用するようにしてください。これはほとんどの有名なライブラリ（numpy、pandas、sklearn、matplotlib、scipy、six など）で使用されているものです。pytest に精通していない場合は、[pytest ガイド](https://docs.pytest.org/en/stable/getting-started.html#getstarted)をお読みください。

DatasetBuilders のテストは特殊で、[データセットの追加ガイド](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md#test-your-dataset)に説明されています。

### PR を送信してレビューを依頼しましょう！

お疲れ様です！プルリクエストの使用に関する詳細は、[GitHub ヘルプ](https://help.github.com/articles/about-pull-requests/)をご覧ください。

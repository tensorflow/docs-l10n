# TensorFlow API ドキュメントに貢献する

## テスト可能な docstring

TensorFlow は [DocTest](https://docs.python.org/3/library/doctest.html) を使用して Python ドキュメント文字列（docstring）のコードスニペットをテストします。スニペットは、実行可能な Python コードである必要があります。テストを有効にするには、行の先頭に`>>>`（3 つの右山括弧）を追加します。例えば、以下は [array_ops.py](https://www.tensorflow.org/code/tensorflow/python/ops/array_ops.py) ソースファイルの`tf.concat`関数からの抜粋です。

```
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.
  ...

  >>> t1 = [[1, 2, 3], [4, 5, 6]]
  >>> t2 = [[7, 8, 9], [10, 11, 12]]
  >>> concat([t1, t2], 0)
  <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
  array([[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9],
         [10, 11, 12]], dtype=int32)>

  <... more description or code snippets ...>

  Args:
    values: A list of `tf.Tensor` objects or a single `tf.Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

    Returns:
      A `tf.Tensor` resulting from concatenation of the input tensors.
  """

  <code here>
```

注意: TensorFlow DocTest は TensorFlow 2 および Python 3 を使用しています。

### DocTest でコードをテスト可能にする

現在、多くの docstring はバッククォート（```）を使用してコードを識別しています。DocTest でコードをテスト可能にするには、以下のようにします。

- バッククォート（```）を削除し、各行の先頭に右山括弧（>>>）を使用します。継続する行の先頭には「...」を使用します。
- 改行を追加して、Markdown テキストから DocTest スニペットを分離し、tensorflow.org で適切にレンダリングします。

### カスタマイズ

TensorFlowは、組み込みの doctest ロジックにいくつかのカスタマイズを使用しています。

- 浮動小数点数値はテキストとして比較されません。浮動小数点数値はテキストから抽出され、`allclose`を使用して*リベラルな`atol`許容値および`rtol`許容値*と比較されます。これにより、以下が実現します。
    - ドキュメントがより明確になる - 作成者が小数点以下を含める必要は一切ありません。
    - テストがより堅牢になる - 基礎となる実装の数値の変更によって、doctest が失敗することはありません。
- 作成者が行の出力を含めた場合にのみ、出力をチェックします。これによって、通常、作成者は無関係な中間値をキャプチャして出力を防ぐ必要がないため、ドキュメントがより明確になります。

### docstring に関する考慮事項

- *全体* : doctest の目標は、ドキュメントを提供し、ドキュメントが機能することを確認することです。これは単体テストとは異なります。したがって、次を考慮するとよいでしょう。

    - 例は単純にする。
    - 長く複雑な出力は避ける。
    - 可能な限り、四捨五入した数を使用する。

- *出力形式* : スニペットの出力は、出力を生成するコードのすぐ下に位置する必要があります。また、docstring の出力は、コードが実行された後の出力と正確に一致していなければなりません。上記の例をご覧ください。また、DocTest ドキュメントの[この部分](https://docs.python.org/3/library/doctest.html#warnings)を確認してください。出力が 80 文字の行制限を超える場合は、余分な出力を新しい行に置くと DocTest がそれを認識します。例として、以下の複数行ブロックをご覧ください。

- *グローバル* : <code>tf</code>、`np`、および`os`モジュールは、TensorFlow の DocTest で常に使用可能です。

- *シンボルの使用* : DocTest では、同じファイル内で定義されたシンボルに直接アクセスできます。現在のファイルで定義されていないシンボルを使用する場合は、`xxx`の代わりに TensorFlow のパブリック API `tf.xxx`を使用してください。以下の例からも分かるように、<code>random.normal</code>は`NewLayer`に認識されないため、<code>random.normal</code>は<code>tf.random.normal</code>を介してアクセスされます。

    ```
    def NewLayer():
      “””This layer does cool stuff.

      Example usage:

      >>> x = tf.random.normal((1, 28, 28, 3))
      >>> new_layer = NewLayer(x)
      >>> new_layer
      <tf.Tensor: shape=(1, 14, 14, 3), dtype=int32, numpy=...>
      “””
    ```

- *浮動小数点数値* : TensorFlow doctest は結果の文字列から浮動小数点数値を抽出し、`np.allclose`を使用して妥当な許容値（`atol=1e-6`、`rtol=1e-6`）で比較します。こうすると、数値の問題を発生させないように過度に正確な docstring を作成する必要がなくなります。単純に、期待される値を貼り付けます。

- *非確定的な出力* : 不確実な部分には省略記号（`...`）を使用すると、DocTest はその部分文字列を無視します。

    ```
    >>> x = tf.random.normal((1,))
    >>> print(x)
    <tf.Tensor: shape=(1,), dtype=float32, numpy=..., dtype=float32)>
    ```

- *複数行ブロック* : DocTestは、単一行ステートメントと複数行ステートメントの違いについては厳密です。以下の「...」の使用方法に注意してください。

    ```
    >>> if x > 0:
    ...   print("X is positive")
    >>> model.compile(
    ...   loss="mse",
    ...   optimizer="adam")
    ```

- *例外* : 発生した例外を除いて、例外の詳細は無視されます。詳細は[こちら](https://docs.python.org/3/library/doctest.html#doctest.IGNORE_EXCEPTION_DETAIL)をご覧ください。

    ```
    >>> np_var = np.array([1, 2])
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
    ```

### ローカルマシンでテストする

docstring のコードをローカルでテストする方法は 2 つあります。

- クラス/関数/メソッドの docstring を変更するだけなら、そのファイルのパスを [tf_doctest.py](https://www.tensorflow.org/code/tensorflow/tools/docs/tf_doctest.py) に渡すとテストが可能です。例えば、以下のようにします。

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">python tf_doctest.py --file=</code>
    </pre>

    これはインストールされているバージョンの TensorFlow を使用して実行します。テストしているコードと同じコードの実行を確実にするには、以下のようにします。

    - 最新の [tf-nightly](https://pypi.org/project/tf-nightly/) の`pip install -U tf-nightly`を使用する。
    - [TensorFlow](https://github.com/tensorflow/tensorflow) のマスターブランチからの最新のプルに、プルリクエストをリベースする。

- コードとクラス/関数/メソッドの docstring を変更する場合には、[ソースから TensorFlow を構築する](../../install/source.md)必要があります。ソースから構築するようにセットアップすると、テストを実行できるようになります。

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest</code>
    </pre>

    または

    <pre class="prettyprint lang-bsh">
    <code class="devsite-terminal">bazel run //tensorflow/tools/docs:tf_doctest -- --module=ops.array_ops</code>
    </pre>

    `--module`は、`tensorflow.python`に相対的です。

# TensorFlowコードスタイルガイド

## Pythonスタイル

[PEP 8 Pythonスタイルガイド](https://www.python.org/dev/peps/pep-0008/)に従ってください。ただし、TensorFlowでは4文字ではなく2文字の半角空白文字を使用します。[Google Pythonスタイルガイド](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)に準拠し、[pylint ](https://www.pylint.org/)を使用してPythonの変更を確認してください。

### pylint

`pylint`をインストールしてTensorFlowのカスタムスタイル定義を取得します。

```bash

$ pip install pylint
$ wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc

```

`pylint`でファイルを確認します。

```bash
$ pylint --rcfile=/tmp/pylintrc myfile.py
```

### サポートされているPythonバージョン

TensorFlowはPython 3.5以降をサポートしています。 詳しくは、[「インストールガイド」](https://www.tensorflow.org/install)を参照してください。

公式およびコミュニティでサポートされているビルドについては、TensorFlow[継続的ビルドステータス](https://github.com/tensorflow/tensorflow/blob/master/README.md#continuous-build-status)を参照してください。

## C++ コードスタイル

TensorFlow C ++コードへの変更は、[Google C ++スタイルガイド](https://google.github.io/styleguide/cppguide.html)に準拠する必要があります。`clang-format`を使用して、C / C ++の変更を確認します。

Ubuntu 16以降をインストールするには、次の手順に従います。

```bash
$ apt-get install -y clang-format
```

C / C ++ファイルの形式は、次のようにして確認できます。

```bash
$ clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
$ diff <my_cc_file> /tmp/my_cc_file.cc
```

## 他の言語

- [Google Javaスタイルガイド](https://google.github.io/styleguide/javaguide.html)
- [Google JavaScriptスタイルガイド](https://google.github.io/styleguide/jsguide.html)
- [Google Shellスタイルガイド](https://google.github.io/styleguide/shell.xml)
- [Google Objective-Cスタイルガイド](https://google.github.io/styleguide/objcguide.html)

## TensorFlowの規則と特別な使用

### Python演算

TensorFlow*演算*は、指定された入力テンソルが出力テンソルを返す関数です（またはグラフ作成時にグラフにopを追加します）。

- 最初の引数は必ずテンソルで、その後に基本的なPythonパラメータが続きます。最後の引数は`name`で、デフォルト値は`None`です。
- テンソル引数は、単一のテンソルまたは反復可能なテンソルのいずれかである必要があります。つまり、「テンソルまたはテンソルのリスト」では広すぎます。 `assert_proper_iterable`を参照してください。
- テンソルを引数として受け取る演算では、`convert_to_tensor`を呼び出して、テンソル以外の入力をテンソルに変換する必要があります（C ++演算を使用している場合）。引数は、ドキュメントでは特定のdtypeの<code>Tensor</code>オブジェクトとして説明されていることに注意してください。
- それぞれのPython演算には、`name_scope`が必要です。 以下に示すように、opの名前を文字列として渡します。
- 演算には、各値のタイプと意味の両方を説明する引数および戻り値について詳しく説明するPythonコメントを記述する必要があります。可能な形状、dtype、またはランクは、説明で指定する必要があります。詳細はドキュメントを参照してください。
- より使いやすくするために、例のセクションにopの入力/出力の使用例を含めてください。
- `tf.Tensor.eval`または`tf.Session.run`を明示的に使用しないでください。たとえば、テンソル値に依存するロジックを作成するには、TensorFlow制御フローを使用します。または、eager実行が有効な場合(`tf.executing_eagerly()`)にのみ実行するように演算を制限します。

例：

```python
def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
          output_collections=(), name=None):
  """My operation that adds two tensors with given coefficients.

  Args:
    tensor_in: `Tensor`, input tensor.
    other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
    my_param: `float`, coefficient for `tensor_in`.
    other_param: `float`, coefficient for `other_tensor_in`.
    output_collections: `tuple` of `string`s, name of the collection to
                        collect result of this op.
    name: `string`, name of the operation.

  Returns:
    `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

  Example:
    >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
              output_collections=['MY_OPS'], name='add_t1t2')
    [2.3, 3.4]
  """
  with tf.name_scope(name or "my_op"):
    tensor_in = tf.convert_to_tensor(tensor_in)
    other_tensor_in = tf.convert_to_tensor(other_tensor_in)
    result = my_param * tensor_in + other_param * other_tensor_in
    tf.add_to_collection(output_collections, result)
    return result
```

使用法：

```python
output = my_op(t1, t2, my_param=0.5, other_param=0.6,
               output_collections=['MY_OPS'], name='add_t1t2')
```

# Pytype

[Pytype](https://github.com/google/pytype) は Python の静的アナライザであり、Python コードの型を確認して推測します。

## メリットと課題

Pytype を使用することには多くのメリットがあります。詳細については、https://github.com/google/pytype を参照してください。ただし、Pytype による型アノテーションの解釈や Pytype により生成されるエラーは TensorFlow Federated の可読性を下げ、不便な場合があります。

- デコレータ

Pytype は、アノテーションを付けている関数に対してアノテーションをチェックします。関数がデコレートされていると、同じアノテーションが適用されなくなる新しい関数が作成される場合があります。TensorFlow と TensorFlow Federated はどちらも、デコレートされた関数の入力と出力を大きく変換するデコレータを使用します。つまり、`@tff.tf_computation`、`@tff.tf_computation`、または `@tff.federated_computation` でデコレートされた関数は、pytype で分析すると驚くような動作をする可能性があります。

以下に例を示します。

```
def decorator(fn):

  def wrapper():
    fn()
    return 10  # Anything decorated with this decorator will return a `10`.

  return wrapper


@decorator
def foo() -> str:
  return 'string'


@decorator
def bar() -> int:  # However, this annotation is incorrect.
  return 'string'
```

関数 `foo` と `bar` の戻り型は `str` である必要があります。これは、これらの関数が文字列を返すためです。関数がデコレートされているかどうかに関係ありません。

Python デコレータの詳細については、<code>@tff.tf_computation</code> を参照してください。

- `getattr()`

Pytype は、[`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 関数を使用して属性が提供されているクラスを解析できません。TensorFlow Federatedは、`tff.Struct`、`tff.Value`、および`tff.StructType` などのクラスで `getattr()` を利用します。これらのクラスは  Pytype によって正しく分析されません。

- パターンマッチング

Pytype は、Python 3.10 より前のパターンマッチングを十分に処理しません。TensorFlow Federated は、ユーザー定義の型ガードを頻繁に使用します。つまり、パフォーマンス上の理由から [`isinstance`](https://docs.python.org/3/library/functions.html#isinstance) 以外の型ガードを使用しますが、Pytype はこれらの型ガードを解釈できません。これは、`typing.cast`を挿入するか、Pytype をローカルで無効にすることで修正できます。ただし、TensorFlow Federated の一部では、ユーザー定義型ガードが非常に多く使用されているため、これらの修正は Python コードを読みにくくします。

注意: Python 3.10 では、[ユーザー定義型ガード](https://www.python.org/dev/peps/pep-0647/)のサポートが追加されたため、Python 3.10 が Python TensorFlow Federated のサポート対象の最小バージョンになった時点で、この問題は解決されます。

## TensorFlow Federated での Pytype の使用

TensorFlow Federated は **Python アノテーションと Pytype アナライザを使用します**。ただし、Python アノテーションを使用しないか、Pytype を無効にすると、*便利なことがあります*。 Pytype をローカルで無効すると、Python コードが読みにくくなる場合は、[特定のファイルのすべての pytype チェックを無効にする](https://google.github.io/pytype/faq.html#how-do-i-disable-all-pytype-checks-for-a-particular-file)ことをお勧めします。

# トレース

[TOC]

Python 関数から [AST](compilation.md#ast) を構築するプロセスをトレースします。

TODO(b/153500547): トレースシステムの個別のコンポーネントを説明し、リンクを示してください。

## 連合コンピュテーションをトレースする

連合コンピュテーションをトレースするには、大まかに、3 つのコンポーネントがあります。

### 引数をパックする

内部的に、TFF computation にはゼロまたは 1 つの引数シカありません。[computations.federated_computation](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/api/computations.py) デコレータに指定された引数が TFF computation の引数の型シグネチャを記述します。TFF はこの情報を使用して、Python 関数の引数を 1 つの [structure.Struct](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/common_libs/structure.py) にパックする方法を決定します。

注意: `Struct` が名前付きと名前なしの両方のフィールドを受け入れるのは、`Struct` を単一のデータ構造として使用して Python の `args` と `kwargs` の両方を表現しているためです。

詳細は、「[function_utils.create_argument_unpacking_fn](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/computation/function_utils.py)」をご覧ください。

### 関数をトレースする

`federated_computation` をトレースする場合、ユーザーの関数は、各引数の代用として [value_impl.Value](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) を使用して呼び出されます。`Value` は、一般的な Python ダンダーメソッド（例: `__getattr__`）を実装することで、元の引数の型の動作をエミュレートしようとします。

さらに詳しく述べると、ちょうど 1 つの引数がある場合、トレースは次のようにして行われます。

1. 引数を表す適切な型シグネチャを使用して、[building_blocks.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) に基づいて [value_impl.ValueImpl](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/federated_context/value_impl.py) を構築します。

2. `ValueImpl` で関数を呼び出します。これにより、Python ランタイムが `ValueImpl` によって実装されるダンダーメソッドを呼び出し、それらを AST 構造として解釈します。各ダンダーメソッドは AST を構築して AST に基づく `ValueImpl` を返します。

以下に例を示します。

```python
def foo(x):
  return x[0]
```

ここでは、関数のパラメーターはタプルであり、関数の本体の 0 番目の要素が選択されています。これが Python の `__getitem__` メソッドを呼び出して、`ValueImpl` で上書きされます。最も単純なケースでは、`ValueImpl.__getitem__` の実装は [building_blocks.Selection](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) を構築して `__getitem__` の呼び出しを表現し、この新しい `Selection` に基づく `ValueImpl` を返します。

各ダンダーメソッドが `ValueImpl` を返し、オーバーライドされたダンダーメソッドの 1 つを呼び出す関数の本体にあるすべての演算をスタンプアウトするため、トレースは続行されます。

### AST を構築する

関数のトレース結果は、[building_blocks.Lambda](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) にパッケージがされます。この `parameter_name` と `parameter_type` はパックされた引数を表現するために作成された [building_block.Reference](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/core/impl/compiler/building_blocks.py) にマッピングされます。その結果生成される `Lambda` が、ユーザーの Python 関数を完全に表現する Python オブジェクトとして返されます。

## TensorFlow Computation をトレースする

TODO(b/153500547): TensorFlow computation をトレースするプロセスを説明してください。

## トレース中に例外のエラーメッセージをクリーンアップする

TFF の歴史の中で、ユーザーの計算をトレースするプロセスには、ユーザーの関数を呼び出す前に、いくつかのラッパー関数を通過させることが含まれていました。このため、以下のようなエラーメッセージが生成されるという、望ましくない効果がありました。

```
Traceback (most recent call last):
  File "<user code>.py", in user_function
    @tff.federated_computation(...)
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<tff code>.py", in tff_function
    <line of TFF code>
  File "<user code>", in user_function
    <some line of user code inside the federated_computation>
  File "<tff code>.py", tff_function
  ...
  File "<tff code>.py", tff_function
    <raise some error about something the user did wrong>
FederatedComputationWrapperTest.test_stackframes_in_errors.<locals>.DummyError
```

このトレースバックでは、ユーザーコード（実際にバグが含まれる行）を突き止めるのが非常に困難です。このため、ユーザーはこれらの問題を TFF のバグとして報告し、全体としてユーザーの業務をより困難にしていました。

現在では、TFF はさまざまな処理を通じて、これらの呼び出しスタックに余分な TFF 関数が含まれないようにしています。TFF のトレースコードでジェネレータを使用するのはこのためであり、通常以下のようなパターンで示されます。

```
# Instead of writing this:
def foo(fn, x):
  return 5 + fn(x + 1)

print(foo(user_fn, 20))

# TFF uses this pattern for its tracing code:
def foo(x):
  result = yield x + 1
  yield result + 5

fooer = foo(20)
arg = next(fooer)
result = fooer.send(user_fn(arg))
print(result)
```

このパターンによって、ユーザーのコード（上記の `user_fn`）を呼び出しスタックの最上位で呼び出すことが可能で、同時にラッパー関数の引数、出力、さらにはスレッドのコンテキストも呼び出すことが可能となっています。

このパターンの単純なバージョンは、"before" 関数と "after" 関数を用いてさらに単純化することができます<br>たとえば、上記の `foo` を以下のように置き換えることができます。

```
def foo_before(x):
  return x + 1

def foo_after(x):
  return x + 5
```

このパターンは、"before" と "after" の部分で状態を共有sるう必要がない場合に適しています。ただし、複雑な状態やコンテキストマネージャが伴うより複雑なケースでは、このように表現するのが面倒なこともあります。

```
# With the `yield` pattern:
def in_ctx(fn):
  with create_ctx():
    yield
    ... something in the context ...
  ...something after the context...
  yield

# WIth the `before` and `after` pattern:
def before():
  new_ctx = create_ctx()
  new_ctx.__enter__()
  return new_ctx

def after(ctx):
  ...something in the context...
  ctx.__exit__()
  ...something after the context...
```

コードがコンテキスト内で実行している後者の例ではあまり明確ではありません。また、より多くの状態が before と after のセクションで共有されると、さらに明確さが劣ってしまいます。

ほかにも「ユーザーエラーメッセージから TFF 関数を非表示にする」という一般的な問題に対する解決策が試されました。例外をキャッチして再表示する（スタックに呼び出し元のコードを含まずに最低位のユーザーコードのみが含まれる例外を作成できないため、失敗しました）、例がをキャッチしてトレースバックをフィルタされたトレースバックに置き換える（CPython 固有であり、Python 言語ではサポートされていない方法）、例外ハンドラーの置き換え（`sys.excepthook` が `absltest` によって使用されておらず、他のフレームワークでオーバーライドされるため失敗しました）といった解決策です。最終的に、TFF 実装の複雑さを犠牲にした上で、ジェネレータベースの反転制御が最も優れたエンドユーザーエクスペリエンスをもたらしました。

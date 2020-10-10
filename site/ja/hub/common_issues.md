<!--* freshness: { owner: 'maringeo' reviewed: '2020-09-14' review_interval: '3 months' } *-->

# 一般的な課題

このリストに該当する課題が掲載されていない場合は、新しい課題を送信する前に [github issues](https://github.com/tensorflow/hub/issues) を検索してください。

## TypeError: 'AutoTrackable' object is not callable

```python
# BAD: Raises error
embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed(['my text', 'batch'])
```

このエラーは、TF2 の `hub.load()` API を使って、TF1 Hub 形式のモジュールを読み込む際に良く発生します。正しいシグネチャを追加することで、この問題を修正できます。TF2 への移行と TF2 における TF1 Hub 形式のモデルの使用については、[TF2 の TF-Hub 移行ガイド](migration_tf2.md)をご覧ください。

```python

embed = hub.load('https://tfhub.dev/google/nnlm-en-dim128/1')
embed.signatures['default'](['my text', 'batch'])
```

## モジュールをダウンロードできません

ネットワークスタックが原因で、URL からモジュールを使用する過程でさまざまなエラーが発生することがあります。多くの場合は、これはコードを実行しているマシンに特化した問題であり、ライブラリの問題ではありません。次に、一般的なエラーを示します。

- **"EOF occurred in violation of protocol"** - この課題は、インストールされた Python バージョンが、モジュールをホストしているサーバーの TLS 要件をサポートしない場合に発生する可能性があります。特に、Python 2.7.5 では tfhub.dev ドメインのモジュールを解決できないことが知られています。**修正方法**: 新しい Python バージョンに更新してください。

- **"cannot verify tfhub.dev's certificate"** - この課題は、ネットワーク上の何かが dev gTLD として振る舞おうとしている場合に発生する可能性があります。開発者とフレームワークは、.dev が gTLD として使用される前に .dev の名前を使用してコードをテストすることがあります。**修正方法:** ".dev" ドメインの名前解決を阻止するソフトウェアを特定し、設定し直してください。

- キャッシュディレクトリ `/tmp/tfhub_modules`（またはこれに類似）への書き込みに失敗する: キャッシュディレクトリとその場所の変更方法については、[キャッシング](caching.md)をご覧ください。

上記に該当するエラーがあり、その修正方法で解決できない場合は、tar 圧縮ファイルのダウンロード URL に `?tf-hub-format=compressed` を添付するプロトコルをシミュレーションすることで、モジュールを手動でダウンロードしてみてください。tar 圧縮ファイルは、ローカルファイルに手動で解凍する必要があります。その後で、URL の代わりにローカルファイルへのパスを使用できるようになります。次に簡単な例を示します。

```bash
# Create a folder for the TF hub module.
$ mkdir /tmp/moduleA
# Download the module, and uncompress it to the destination folder. You might want to do this manually.
$ curl -L "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" | tar -zxvC /tmp/moduleA
# Test to make sure it works.
$ python
> import tensorflow_hub as hub
> hub.Module("/tmp/moduleA")
```

## 初期化済みのモジュールで推論を実行する

入力データにモジュールを何度も適用する Python プログラムを記述している場合、次のレシピを適用することができます（注意: 本番サービスでリクエストを推論するには、go/servo またはその他の拡張可能な、Python を使用しないソリューションを検討してください）。

使用事例のモデルは**初期化リクエスト**と以降の**リクエスト**（Django、Flask、カスタム HTTP サーバーなど）であると仮定すると、次のように推論をセットアップすることができます。

### TF2 SavedModels

- 初期化の部分で、次を行います。
    - TF2.0 モデルを読み込みます。

```python
import tensorflow_hub as hub

embedding_fn = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
```

- リクエストの部分で、次を行います。
    - 組み込み関数を使用して、推論を実行します。

```python
embedding_fn(["Hello world"])
```

この tf.function の呼び出しは、パフォーマンスを改善できるように最適化されています。[tf.function ガイド](https://www.tensorflow.org/guide/function)をご覧ください。

### TF1 Hub モジュール

- 初期化の部分で、次を行います。
    - **プレースホルダ**を使ってグラフを構築します。グラフのエントリポイントです。
    - セッションを初期化します。

```python
import tensorflow as tf
import tensorflow_hub as hub

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
  # We will be feeding 1D tensors of text into the graph.
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)
```

- リクエストの部分で、次を行います。
    - セッションを使用して、プレースホルダ’を通じてグラフにデータ’をフィードします。

```python
result = session.run(embedded_text, feed_dict={text_input: ["Hello world"]})
```

## モデルの dtype を変更できません（float32 から bfloat16 へなど）

TensorFlow の SavedModels（TF Hub で共有）には一定のデータ型（通常、ニューラルネットワークの重みと即時アクティベーションには float32 が使用されます）で動作する演算が含まれます。こういった演算は、SavedModel が読み込まれた後に変更することはできません（ただし、モデルのパブリッシャーは、異なるデータ型でさまざまなモデルを公開することができます）。

<!--* freshness: { owner: 'akhorlin' reviewed: '2021-11-22' } *-->

# テキストタスクの一般的な SavedModel API

このページでは、テキスト関連のタスクに使用する [TF2 SavedModel](../tf2_saved_model.md) が [Reusable SavedModel API](../reusable_saved_models.md) をどのように実装しているかを説明します。（これは、[テキストの共通シグネチャ](../common_signatures/text.md)に置き換わります。[TF1 Hub 形式](../tf1_hub_module)は使用廃止となっています。）

## 概要

**テキストの埋め込み**（テキストの密な表現またはテキスト特徴量ベクトルとも知られています）を計算する API にはいくつかあります。

- *テキスト入力からのテキストの埋め込み*に使用する API は SavedModel によって実装されており、文字列のバッチを埋め込みベクトルのバッチにマッピングします。これは非常に使いやすく、TF Hub の多くのモデルが実装している API ではありますが、TPU でモデルをファインチューニングすることはできません。

- *事前処理された入力によるテキスト埋め込み*に使用する API は、同じタスクを解決しますが、次の 2 つの独立した SavedModel によって実装されます。

    - tf.data 入力パイプライン内で実行し、文字列とその他の可変長データを数値テンソルに変換する*プリプロセッサ*
    - プリプロセッサの結果を受け入れてトレーニング可能な部分の埋め込み計算を実施する*エンコーダ*

    この分割によって、入力がトレーニングループにフィードされる前に、入力を非同期的に事前処理することができます。特に、[TPU](https://www.tensorflow.org/guide/tpu) で実行してファインチューニングを行えるエンコーダを構築することが可能になります。

- *Transformer エンコーダによるテキストの埋め込み*に使用する API は、事前処理された入力からのテキストの埋め込みに使用する API を、BERT とその他の Transformer エンコーダの特定のケースに拡張します。

    - *プリプロセッサ*は、入力テキストの 2 つ以上のセグメントからエンコーダ入力を構築するように拡張されます。
    - *Transformer エンコーダ*は各トークンのコンテキスト認識埋め込みを公開します。

モデルのドキュメントで異なる指定がない限り、各ケースのテキスト入力は UTF-8 エンコード文字列で、通常プレーンテキストです。

API に関係なく、モデルはそれぞれ、さまざまな言語の分野のテキストで、さまざまなタスクを念頭に事前にトレーニングさらえています。そのため、すべてのテキスト埋め込みモデルがあらゆる問題に適しているわけではありません。

<a name="feature-vector"></a>
<a name="text-embeddings-from-text"></a>

## テキスト入力からのテキストの埋め込み

**テキスト入力からのテキストの埋め込み**の SavedModel は、形状 `[batch_size]` の文字列テンソルにある入力のバッチを受け入れ、入力の密な表現（特徴量ベクトル）でそれらを形状 `[batch_size, dim]` の float32 テンソルにマッピングします。

### 使用例

```python
obj = hub.load("path/to/model")
text_input = ["A long sentence.",
              "single-word",
              "http://example.com"]
embeddings = obj(text_input)
```

[Reusable SavedModel API](../reusable_saved_models.md) から、トレーニングモードでモデルを実行する（ドロップアウトのため）には、キーワード引数 `obj(..., training=True)` が必要であり、その `obj` によって `.variables`、`.trainable_variables`、および `.regularization_losses` 属性が適時提供されることを思い出してください。

Keras では、このすべてを次のように処理します。

```python
embeddings = hub.KerasLayer("path/to/model", trainable=...)(text_input)
```

### 分散型トレーニング

テキスト埋め込みが、分散ストラテジーでトレーニングされるモデルの一部として使用される場合、`hub.load("path/to/model")` または `hub.KerasLayer("path/to/model", ...)` への呼び出しは、モデルの変数を分散された方法で作成するために、それぞれ DistributionStrategy スコープで発生する必要があります。たとえば、次のようにします。

```python
  with strategy.scope():
    ...
    model = hub.load("path/to/model")
    ...
```

### 例

- Colab チュートリアル「[映画レビューを使ったテキスト分類](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)」

<a name="text-embeddings-preprocessed"></a>

## 事前処理された入力によるテキストの埋め込み

**事前処理された入力によるテキスト埋め込み**は、次の 2 つの独立した SavedModel によって実装されます。

- 形状 `[batch_size]` の文字列テンソルを数値テンソルの dict にマッピングする**プリプロセッサ**
- プリプロセッサが返す dict を受け入れ、埋め込み計算のトレーニング可能な部分を実行して出力の dict を返す**エンコーダ**。キー `"default"` にある出力は、形状 `[batch_size, dim]` の float32 テンソルです。

これにより、入力パイプラインでプリプロセッサを実行できますが、より大きなモデルの一部としてエンコーダが計算した埋め込みをファインチューニングすることはできません。特に、[TPU](https://www.tensorflow.org/guide/tpu) で実行し、ファインチューニングできるエンコーダを構築することができます。

テンソルがプリプロセッサの出力に格納され、`"default"` 以外のほかのテンソルがある場合は、エンコーダの出力に格納されるという実装です。

エンコーダのドキュメントには、どのプリプロセッサをそれと使用するかが指定されている必要があります。通常、適切な選択肢は 1 つです。

### 使用例

```python
text_input = tf.constant(["A long sentence.",
                          "single-word",
                          "http://example.com"])
preprocessor = hub.load("path/to/preprocessor")  # Must match `encoder`.
encoder_inputs = preprocessor(text_input)

encoder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
embeddings = enocder_outputs["default"]
```

[Reusable SavedModel API](../reusable_saved_models.md) から、（ドロップアウトなどのために）トレーニングモードでエンコーダを実行するには、キーワード引数 `encoder(..., training=True)` が必要であり、その `encoder` によって `.variables`、`.trainable_variables`、および `.regularization_losses` 属性が適時提供されることを思い出してください。

`preprocessor` モデルには `.variables` があるかもしれませんが、さらにトレーニングされることを意図していません。事前処理はモードに依存するものではなく、`preprocessor()` に `training=...` 引数があれば、何の効果もありません。

Keras では、このすべてを次のように処理します。

```python
encoder_inputs = hub.KerasLayer("path/to/preprocessor")(text_input)
encoder_outputs = hub.KerasLayer("path/to/encoder", trainable=True)(encoder_inputs)
embeddings = encoder_outputs["default"]
```

### 分散型トレーニング

エンコーダが分散ストラテジーでトレーニングされるモデルの一部に使用されている場合、`hub.load("path/to/encoder")` または `hub.KerasLayer("path/to/encoder", ...)` はそれぞれ次の中で発生しなければなりません。

```python
  with strategy.scope():
    ...
```

これは、エンコーダ変数を分散の方法で再作成するためです。

同様に、プリプロセッサがトレーニングされたモデルの一部である場合（上記の単純の例で示すように）、分散ストラテジーのスコープで読み込まれる必要もあります。ただし、プリプロセッサが入力パイプラインで使用されている場合（`tf.data.Dataset.map()` に渡されるコーラブルで）、その読み込みは、変数（ある場合）をホスト CPU に配置するために、分散ストラテジーのスコープ外で行う必要があります。

### 例

- Colab チュートリアル「[BERT によるテキストの分類](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/classify_text_with_bert.ipynb)」

<a name="transformer-encoders"></a>

## Transformer エンコーダによるテキストの埋め込み

テキストの Transformer エンコーダは、入力シーケンスのバッチで動作します。各シーケンスはトークン化されたテキストの *n* ≥ 1 セグメントで構成されており、*n* のモデル固有の境界内にあります。BERT やその多くの拡張では、その境界は 2 であるため、1 つの引数とセグメントペアを受け入れます。

**Transformer エンコーダによるテキスト埋め込み**に使用する API は、事前処理された入力によるテキスト埋め込みに使用する API をこの設定に拡張します。

### プリプロセッサ

Transformer エンコーダによるテキスト埋め込みに使用するプリプロセッサ SavedModel は、事前処理された入力によるテキスト埋め込みに使用するプリプロセッサ SavedModel の API を実装し、単一セグメントのテキスト入力を直接エンコーダの入力にマッピングする方法を提供します。

また、プリプロセッサ SavedModel は、トークン化を行う  `tokenize` と *n* 個のトークン化されたセグメントをエンコーダの 1 つの入力シーケンスにパッキングする `tokenize` という、呼び出し可能なサブオフジェクトを提供します。それぞれのサブオブジェクトは、[Reusable SavedModel API](../reusable_saved_models.md) に準じます。

#### 使用例

2 つのセグメントのテキストに関する具体的な例として、前提（第 1 セグメント）が仮定（第 2 セグメント）を暗示しているかどうかを尋ねる文の含意関係タスクを見てみましょう。

```python
preprocessor = hub.load("path/to/preprocessor")

# Tokenize batches of both text inputs.
text_premises = tf.constant(["The quick brown fox jumped over the lazy dog.",
                             "Good day."])
tokenized_premises = preprocessor.tokenize(text_premises)
text_hypotheses = tf.constant(["The dog was lazy.",  # Implied.
                               "Axe handle!"])       # Not implied.
tokenized_hypotheses = preprocessor.tokenize(text_hypotheses)

# Pack input sequences for the Transformer encoder.
seq_length = 128
encoder_inputs = preprocessor.bert_pack_inputs(
    [tokenized_premises, tokenized_hypotheses],
    seq_length=seq_length)  # Optional argument.
```

Keras では、この計算は次のように表現されます。

```python
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_hypotheses = tokenize(text_hypotheses)
tokenized_premises = tokenize(text_premises)

bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))  # Optional argument.
encoder_inputs = bert_pack_inputs([tokenized_premises, tokenized_hypotheses])
```

#### `tokenize` の詳細

`preprocessor.tokenize()` への呼び出しは、形状 `[batch_size]` の文字列テンソルを受け入れ、[RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor) という形状 `[batch_size, ...]` のテンソルを返します。その値は int32 トークン ID で、入力文字列を表します。`batch_size` の後に *r* ≥ 1 の不規則な次元があっても、ほかの均一の次元があることはありません。

- *r*=1 である場合、形状は `[batch_size, (tokens)]` であり、各入力は単に、フラットなトークンのシーケンスにトークン化されます。
- *r*&gt;1 である場合、*r*-1 個の追加レベルのグループがあります。たとえば、[tensorflow_text.BertTokenizer](https://github.com/tensorflow/text/blob/v2.3.0/tensorflow_text/python/ops/bert_tokenizer.py#L138) は *r*=2 を使って単語ごとにトークンをグループ化し、形状 `[batch_size, (words), (tokens_per_word)]` を生み出します。これらの追加レベルがいくつ存在するか（存在する場合）、またどのグループを表現しているのかは、手元のモデルによって決まります。

ユーザーはトークン化された入力を変更し、たとえばエンコーダ入力のパッキングで適用される seq_length の制限を利用できるようにすることができます（する必要はありません）。ここで（単語の境界を尊重するために）、トークナイザ出力の追加の次元が役立てられますが、次のステップで無意味になります。

[Reusable SavedModel API](../reusable_saved_models.md) の観点では、`preprocessor.tokenize` オブジェクトに `.variables` があっても、さらにトレーニングされることを意図していません。トークン化はモードに依存するものではなく、`preprocessor.tokenize()` に `training=...` 引数があれば、何の効果もありません。

#### `bert_pack_inputs` の詳細

`preprocessor.bert_pack_inputs()` への呼び出しは、Python のトークン化された入力リスト（入力セグメントごとに個別にバッチ化）を受け入れ、Transformer エンコーダモデル用の固定長の入力シーケンスを表すテンソルの dict を返します。

トークン化された各入力は int32 の RaggedTensor で、形状は `[batch_size, ...]` です。batch_size の後の不規則な次元の番号 *r* は、1 か、`preprocessor.tokenize().` の出力と同じになります（後者は、便宜上のもので、追加の次元は、パッキングの前にフラット化されます）。

パッキングは、エンコーダーが期待するように、入力セグメントの周りに特別なトークンを追加します。`bert_pack_inputs()` 呼び出しは、元の BERT モデルとその拡張機能の多くで使用されている圧縮スキームを正確に実装しています。パッキングされたシーケンスは、1 つの start-of-sequence token トークンで始まり、トークン化されたセグメントが続いた後、1 つの end-of-segment トークンで終了します。残りの seq_length までの位置がある場合は、パディングのトークンで満たされます。

パッキングされたシーケンスが seq_length を超える場合、`bert_pack_inputs()` はそのセグメントを適切な等価サイズのプレフィクスまで切り捨て、パッキングされたシーケンスが seq_length 以内にピッタリと収まるようにします。

パッキングは、モードに依存するものではなく、`preprocessor.bert_pack_inputs()` に `training=...` 引数があれば、何の効果もありません。また、`preprocessor.bert_pack_inputs` には変数は期待されておらず、ファインチューニングもサポートしていません。

### エンコーダ

エンコーダは、事前処理された入力によるテキスト埋め込みに使用する API（上参照）と同様に、`encoder_inputs` のdict で呼び出されます。これには、[Reusable SavedModel API](../reusable_saved_models.md) からのプロビジョンも含まれます。

#### 使用例

```python
enocder = hub.load("path/to/encoder")
enocder_outputs = encoder(encoder_inputs)
```

Kera では次のように処理されます。

```python
encoder = hub.KerasLayer("path/to/encoder", trainable=True)
encoder_outputs = encoder(encoder_inputs)
```

#### 詳細

`encoder_outputs` は、次のキーを持つテンソルの dict です。

<!-- TODO(b/172561269): More guidance for models trained without poolers. -->

- `"sequence_output"`: パッキングされた入力シーケンスごとのトークンのコンテキスト認識埋め込みを持つ、形状 `[batch_size, seq_length, dim]` の float32 テンソル
- `"pooled_output"`: トレーニング可能な方法で sequence_output から取得された、各入力シーケンスの全体的な埋め込みを持つ、形状 `[batch_size, dim]` の float32 テンソル
- 事前処理された入力によるテキスト埋め込みに使用する API で必要な `"default"`。各入力シーケンスの埋め込みを持つ、形状 `[batch_size, dim]` の float32 テンソル。（これは、pooled_output の単なるエイリアスである場合があります。）

`encoder_inputs` の内容は、API の定義で厳密に要求されているものではありませんが、BERT 式の入力を使用するエンコーダでは、次の名前（[NLP Modeling Toolkit of TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/nlp) 参照）を使って、エンコーダの相互交換とプリプロセッサモデルの再利用に生じる摩擦を最小限に抑えることを推奨します。

- `"input_word_ids"`: 形状 `[batch_size, seq_length]` の int32 テンソルで、パッキングされた入力シーケンスのトークン ID を持ちます（つまり、start-of-sequence トークン、end-of-segment トークン、およびパディングを含みます）。
- `"input_mask"`: 形状 `[batch_size, seq_length]` の int32 テンソルで、パディングの前に存在するすべての入力トークンの位置に値 1、パディングトークンに値 0 があります。
- `"input_type_ids"`: 形状 `[batch_size, seq_length]` の int32 テンソルで、それぞれの位置で入力トークンを生成した入力セグメントのインデックスを持ちます。最初の入力セグメント（index 0）には、start-of-sequence トークンと end-of-segment トークンが含まれます。2 番目以降のセグメント（ある場合）には、それぞれの end-of-segment トークンが含まれます。パディングトークンのインデクスはもう一度 0 になります。

### 分散型トレーニング

分散ストラテジースコープの内外でのプリプロセッサとエンコーダオブジェクトの読み込みについては、事前処理された入力によるテキスト埋め込みに使用する API（上参照）と同じルールが適用されます。

### Examples

- Colab チュートリアル「[Solve GLUE tasks using BERT on TPU ](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/bert_glue.ipynb)」

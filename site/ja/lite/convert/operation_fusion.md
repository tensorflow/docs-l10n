# TensorFlow 演算の融合

## 概要

このページでは、TensorFlow の複合演算を TensorFlow Lite の融合演算に変換するために必要な設計とステップを説明します。このインフラストラクチャは汎用であり、TensorFlow のあらゆる複合演算を TensorFlow Lite の対応する融合演算に変換する操作をサポートしています。

このインフラストラクチャの使用例では、[こちら](https://www.tensorflow.org/lite/convert/rnn)に説明される通りに、TensorFlow RNN 演算を TensorFlow Lite に融合しています。

### 融合演算とは

![drawing](https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/images/convert/op_fusion_banner.jpg?raw=true)

TensorFlow 演算は、[tf.add](https://www.tensorflow.org/api_docs/python/tf/math/add) のような原子演算であるか、[tf.einsum](https://www.tensorflow.org/api_docs/python/tf/einsum) などのほかの原子演算から作成することができます。原子演算は、TensorFlow グラフでは単一ノードとして現れますが、複合演算は TensorFlow グラフではノードのコレクションです。複合演算を実行することは、それを構成する原子演算をそれぞれ実行することに相当します。

融合演算は、対応する複合演算内の各原子演算が実行するすべての計算を組み込んだ単一の演算です。

### 融合演算のメリット

融合演算は、根底にあるカーネル実装のパフォーマンスを最大化するために存在しており、計算全体を最適化し、メモリのフットプリントを縮小することで実現されます。特に、低レーテンシ推論ワークロードとリソース制限のあるモバイルプラットフォームにおいて非常に有益です。

融合演算は、量子化といった複雑な変換を定義するためのより高レベルのインターフェースも提供します。これがない場合、より粒度の高いレベルで実行することが不可能でなかったとしても、非常に困難となります。

TensorFlow Lite には、前述の理由により、融合演算のインスタンスが多数あります。こういった融合演算は通常、ソース TensorFlow プログラムの複合演算に対応しています。TensorFlow Lite で単一の融合演算として実装されている TensorFlow の複合演算には、単方向および双方向シーケンスLSTM、畳み込み（conv2d、バイアス加算、relu）、完全接続（matmul、バイアス加算、relu）などのさまざまな RNN 演算などの例があります。TensorFlow Lite では現在のところ、LSTM 量子化は融合 LSTM 演算にのみ実装されています。

### 融合演算の課題

複合演算を TensorFlow から TensorFlow Lite の融合演算に変換するのは、難しい問題です。これには次の理由があります。

1. 複合演算は、TensorFlow グラフで十分に定義された境界のない原子演算のセットとして表されています。このような複合演算に対応するサブグラフを（パターンマッチなどで）特定することは非常に困難です。

2. 融合 TensorFlow Lite 演算をターゲットとする TensorFlow 実装が 1 つ以上存在する可能性があります。たとえば、TensorFlow には多数の LSTM 実装（Keras、Babelfish/lingvo など）があり、それぞれが異なる原子演算で構成されてはいますが、TensorFlow Lite では同一の融合 LSTM 演算に変換される可能性があります。

そのため、融合演算の変換は、難易度が非常に高いことが証明されています。

## 複合演算から融合演算に変換する

TensorFlow 複合演算から TensorFlow Lite 融合演算への変換をおこなうための全体的なアーキテクチャは、次のようになっています。

![drawing](https://gitlocalize.com/repo/4592/ja/site/en-snapshot/lite/images/convert/op_fusion.png)

### 複合演算を `tf.function` でラッピングする

TensorFlow モデルのソースコードで、複合演算を特定し、それを [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470) 注釈を使って `tf.function` に抽象化します。[埋め込みルックアップ](#composing_ops)の例をご覧ください。この関数はインターフェースを定義し、変換ロジックにその引数が使用されます。

### 変換コードを書く

変換コードは、`implements` 注釈を使って、関数のインターフェースに従って記述されます。[埋め込みルックアップ](#fusion_code)の例をご覧ください。概念的には、変換コードによってこのインターフェースの複合実装が融合実装に置き換えられることになります。

prepare-composite-functions パスに、[変換コード](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115)をプラグインしましょう。

より高度な使用例では、融合演算のオペランドを導出するために、複合演算のオペランドの複雑な変換を実装することが可能です。例として、[Keras LSTM](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627) 変換コードをご覧ください。

### TensorFlow Lite に変換する

[TFLiteConverter.from_saved_model](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_saved_model) API を使用して、TensorFlow Lite に変換します。

## 内部動作

<a id="under_the_hood"></a>

それでは、TensorFlow Lite の融合演算への変換に高レベルの設計全体の詳細を説明します。

### TensorFlow で演算を記述する

<a id="composing_ops"></a>

`tf.function` を [experimental_implements](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/python/eager/def_function.py#L470) 関数の属性で使用することで、TensorFlow の原子演算を使用sh知恵新しい演算を明示的に記述し、その結果として形成される複合演算が実装するインターフェースを指定することができます。これは次の項目を提供するため、非常に有用です。

1. 根底の TensorFlow グラフにおける複合演算の十分に定義された境界。
2. この演算が実装するインターフェースを明示的に指定できる。`tf.function` の引数は、このインターフェースの引数に対応します。

例として、埋め込みルックアップを実装するために定義される複合演算を考察してみましょう。これは、TensorFlow Lite の融合演算にマッピングします。

```python
  @tf.function(
        experimental_implements="embedding_lookup")
    def EmbFprop(embs, ids_vec):
      """Embedding forward prop.

      Effectively, it computes:
        num = size of ids_vec
        rets = zeros([num, embedding dim])
        for i in range(num):
          rets[i, :] = embs[ids_vec[i], :]
        return rets

      Args:
        embs: The embedding matrix.
        ids_vec: A vector of int32 embedding ids.

      Returns:
        The result of embedding lookups. A matrix of shape
        [num ids in ids_vec, embedding dims].
      """
      num = tf.shape(ids_vec)[0]
      rets = inplace_ops.empty([num] + emb_shape_suf, py_utils.FPropDtype(p))

      def EmbFpropLoop(i, embs, ids_vec, rets):
        # row_id = ids_vec[i]
        row_id = tf.gather(ids_vec, i)
        # row = embs[row_id]
        row = tf.reshape(tf.gather(embs, row_id), [1] + emb_shape_suf)
        # rets[i] = row
        rets = inplace_ops.alias_inplace_update(rets, [i], row)
        return embs, ids_vec, rets

      _, _, rets = functional_ops.For(
          start=0,
          limit=num,
          delta=1,
          inputs=[embs, ids_vec, rets],
          body=EmbFpropLoop,
          rewrite_with_while=compiled)
      if len(weight_shape) > 2:
        rets = tf.reshape(rets, [num, symbolic.ToStatic(p.embedding_dim)])
      return rets
```

上記に示すように、`tf.function` を使ってモデルが複合演算を使用するようにすることで、このような演算を融合 TensorFlow Lite 演算に**特定して変換**する一般的なインフラストラクチャを構築することが可能となります。

### TensorFlow Lite コンバータを拡張する

今年前期にリリースされた TensorFlow Lite コンバータは、TensorFlow モデルを、すべての変数が対応する低数値と置き換えられた状態でグラフとしてインポートすることだけだサポートされていました。こういったグラフでは、変数が定数に変換されるようにすべての関数がインライン化されているため、演算の融合には有用ではありませんでした。

変換プロセス中に `tf.function` を `experimental_implements` 機能で利用するには、関数は、後の変換プロセスまで維持される必要があります。

そのため、複合演算の融合の使用事例をサポートするように、コンバータにおける TensorFlow モデルのインポートと変換の新しいワークフローを実装しました。具体的には、新しい機能として次の項目が追加されています。

1. TensorFlow の [SavedModel を MLIR](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/translate/import_model.cc#L3748) にインポートする
2. [融合複合演算](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L103)
3. [変数可変性分析](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/optimize_global_tensors.cc#L43)
4. [すべての読み取り専用変数の凍結](https://github.com/tensorflow/tensorflow/blob/1099faa8d6a941ef44d09ed8c372ff0ffda94112/tensorflow/compiler/mlir/tensorflow/transforms/freeze_global_tensors.cc#L44)

これにより、関数がインライン化する前、そして変数が凍結する前に複合演算を表現する関数を使用して演算の融合を実行することが可能となります。

### 演算の融合を実装する

演算の融合パスをより詳しく見てみましょう。このパスは、次のことを行います。

1. MLIR モジュールのすべての関数をループする。
2. 関数に tf._implements 属性がある場合、属性の値に基づいて、適切な演算の融合ユーティリティを呼び出す。
3. 演算の融合ユーティリティは、関数のオペランドと属性で演算（変換のインターフェースとして機能）し、関数の本文を融合演算を含む同等の関数本文に置き換える。
4. 多くの場合、置き換えられた本文には、融合演算の以外の演算が含まれる。この演算は、融合演算のオペランドを取得するために、関数のオペランドでの静的変換に対応しています。この計算はすべて定数で折り畳めるため、融合演算のみが存在するエクスポートされた Flatbuffer には存在しません。

次は、メインのワークフローを示す、このパスのコードスニペットです。

```
void PrepareCompositeFunctionsPass::ConvertTFImplements(FuncOp func,
                                                        StringAttr attr) {
  if (attr.getValue() == "embedding_lookup") {
    func.eraseBody();
    func.addEntryBlock();
    // Convert the composite embedding_lookup function body to a
    // TFLite fused embedding_lookup op.
    ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
    if (failed(convert_embedded_lookup.VerifySignature())) {
      return signalPassFailure();
    }
    convert_embedded_lookup.RewriteFunc();
  } else if (attr.getValue() == mlir::TFL::kKerasLstm) {
     func.eraseBody();
     func.addEntryBlock();
     OpBuilder builder(func.getBody());
     if (failed(ConvertKerasLSTMLayer(func, &builder))) {
       return signalPassFailure();
     }
  } else if (.....) /* Other fusions can plug in here */
}
```

次は、関数を変換インターフェースとして使用し、この複合演算を TensorFlow Lite の融合演算にマッピングする方法を示すコードスニペットです。

<a id="fusion_code"></a>

```C++
void RewriteFunc() {
    Value lookup = func_.getArgument(1);
    Value value = func_.getArgument(0);
    auto output_type = func_.getType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::ReturnOp>(func_.getLoc(), op.getResult());
  }
```

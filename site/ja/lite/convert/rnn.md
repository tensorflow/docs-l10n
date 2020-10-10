# TensorFlow RNN から TensorFlow Lite への変換

## 概要

TensorFlow Lite は、TensorFlow RNN モデルを TensorFlow Lite の融合 LSTM 演算に変換することをサポートしています。融合演算は基本的なカーネル実装のパフォーマンスを最大化し、量子化などの複雑な変換を定義するための高レベルのインターフェースを提供します。

TensorFlow には RNN API の多くのバリアントがあるため、以下の ２ つのアプローチを使用します。

1. Keras LSTM のような**標準的な TensorFlow RNN API のネイティブサポート**が提供されています（推奨されるオプション）。
2. **ユーザー定義の****&nbsp;RNN 実装**をプラグインして TensorFlow Lite に変換するための**変換インフラストラクチャへの****インターフェース**を提供します。Lingvo の [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) および [ LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) RNN インターフェースを使用した変換など、変換にすぐに使える例がいくつか提供されています。

## コンバーター API

この機能は TensorFlow 2.3 リリースの一部です。[tf-nightly](https://pypi.org/project/tf-nightly/) pip または head からも入手できます。

この変換機能は、SavedModel を介して、または Keras モデルから直接 TensorFlow Lite に変換するときに使用できます。使用例を参照してください。

### SavedModel から変換

<a id="from_saved_model"></a>

```
# build a saved model. Here concrete_function is the exported function
# corresponding to the TensorFlow model containing one or more
# Keras LSTM layers.
saved_model, saved_model_dir = build_saved_model_lstm(...)
saved_model.save(saved_model_dir, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
```

### Keras モデルから変換

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## 使用例

Keras LSTM から TensorFlow Lite への変換に関する [Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb) では、TensorFlow Lite インタープリタを用いたエンドツーエンドの使用法が説明されています。

## TensorFlow RNN API サポート対象

<a id="rnn_apis"></a>

### Keras LSTM 変換 (推薦)

Keras LSTM から TensorFlow Lite への標準の変換はサポートされています。この詳細については、[こちら](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627)から [Keras LSTM インターフェース](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;"></span>および変換ロジックを参照してください。

また、Keras 演算の定義に関する TensorFlow Lite の LSTM コントラクトに注意してください。

1. **入力**テンソルの次元 0 はバッチサイズです。
2. **&nbsp;recurrent_weight** テンソルの次元 0 は出力の数です。
3. **重み**および **recurrent_kernel** テンソルは転置されます。
4. 転置された重み、転置された recurrent_kernel および**バイアス**テンソルは、次元 0 に沿って 4 つの等しいサイズのテンソル(**入力ゲート、忘却ゲート、セル、および出力ゲート**)に分割されます。

#### Keras LSTM バリアント

##### 時間優先

ユーザーはオプションで時間優先を選択できます。Keras LSTM は、関数定義属性に時間優先属性を追加します。単方向シーケンス LSTM の場合、単純に unidirecional_sequence_lstm の[時間優先属性](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902)にマッピングできます。

##### 双方向性 LSTM

双方向 LSTM は、2 つの Keras LSTM レイヤーを使用して実装できます。1 つはフォワード用、もう1 つはバックワード用です。例は[こちら](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/wrappers.py#L382)を参照してください。go_backward 属性が表示されたら、それをバックワード LSTM として認識し、フォワードおよびバックワード LSTM をグループ化します。**これは今後の取り組みです。**現在、これにより TensorFlow Lite モデルに 2 つの UnidirectionalSequenceLSTM 演算が作成されます。

### ユーザー定義の LSTM 変換の例

TensorFlow Lite は、ユーザー定義の LSTM 実装を変換する方法も提供します。ここでは、Lingvo の LSTM を実装方法の例として使用しています。詳細については、[lingvo.LSTMCellSimple インターフェース](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)と変換ロジックを[こちら](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130)から参照してください。[こちら](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137)にもう 1 つの[lingvo.LayerNormalizedLSTMCellSimple インターフェース](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173)とその変換ロジックの例が提供されています。

## TensorFlow Lite に「自分の TensorFlow RNN を持ち込む」

ユーザーの RNN インターフェースが標準でサポートされているものと異なる場合、いくつかのオプションがあります。

**オプション 1:** RNN インターフェースを Keras RNN インターフェースに適合させるために TensorFlow Python でアダプターコードを記述する。これは、生成された RNN インターフェースの関数に[ tf_implements アノテーション](https://github.com/tensorflow/community/pull/113)が付いた tf.function であり、Keras LSTM レイヤーにより生成されたものと同じです。この後、Keras LSTM で使用されているものと同じ変換 API が機能します。

**オプション 2:** 上記が不可能な場合 (たとえば、Keras LSTM に、TensorFlow Lite に融合された LSTM 演算レイヤーなどにより現在公開されている一部の機能がない場合)、カスタム変換コードを記述して TensorFlow Lite コンバーターを拡張し、それを[こちら](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115)の prepare-composite-functions MLIR-pass にプラグインします。関数のインターフェースは API コントラクトのように扱う必要があり、融合 TensorFlow Lite LSTM 演算に変換するために必要な引数 (入力、バイアス、重み、投影、レイヤーの正規化など) を含める必要があります。この関数に引数として渡されるテンソルは、既知のランク (つまり、MLIR の RankTensorType) を持つことが推奨されます。これにより、これらのテンソルを RankTensorType と見なすことができる変換コードを簡単に記述でき、融合した TensorFlow Lite オペレーターのオペランドに対応するランク付けされたテンソルに変換することが簡単になります。

このような変換フローの完全な例は、Lingvo の LSTMCellSimple からTensorFlow Lite への変換です。

Lingvo の LSTMCellSimple は、[こちら](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)で定義されています。この LSTM セルでトレーニングされたモデルは、次のように TensorFlow Lite に変換できます。

1. LSTMCellSimple のすべての使用を適切にラベル付けされた tf_implements の注釈を使用して tf.function でラップします (たとえば、ここでは lingvo.LSTMCellSimple が適切な注釈名になります)。生成される tf.function が、変換コードで予期される関数のインターフェースと一致することを確認してください。これは、注釈を追加するモデル作成者と変換コードの間の契約です。

2. prepare-composite-functions パスを拡張して、カスタムコンポジットオペレーションを TensorFlow Lite 融合 LSTM 演算変換にプラグインします。[LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130) 変換コードを参照してください。

    変換契約:

3. **重み**と**射影**テンソルは転置されます。

4. **{input, recurrent}** から **{cell, input gate, forget gate, output gate}** は、転置された重みテンソルをスライスすることによって抽出されます。

5. **{bias}** から **{cell, input gate, forget gate, output gate}** はバイアステンソルをスライスすることによって抽出されます。

6. **射影**は転置射影テンソルをスライスすることにより抽出されます。

7. [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137) にも同様の変換が記述されています。

8. 定義されているすべての [MLIR パス](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57)や、TensorFlow Lite フラットバッファへの最終エクスポートを含む、TensorFlow Lite 変換インフラストラクチャの残りの部分は再利用できます。

## 既知の問題/制限

1. 在、ステートレス Keras LSTM の変換のみがサポートされています (Keras のデフォルトの動作)。ステートフル Keras LSTM の変換は今後開発されます。
2. 基礎となるステートレス Keras LSTM レイヤーを使用してステートフルな Keras LSTM レイヤーをモデル化し、ユーザープログラムで明示的に状態を管理することは可能です。このような TensorFlow プログラムは、ここで説明されている機能を使用して TensorFlow Lite に変換できます。
3. 双方向 LSTM は現在、TensorFlow Lite の 2 つの UnidirectionalSequenceLSTM 演算としてモデル化されています。これは、単一の BidirectionalSequenceLSTM 演算に置き換えられます。

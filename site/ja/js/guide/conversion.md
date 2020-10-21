# モデル変換

TensorFlow.js には、ブラウザですぐに使用できる事前トレーニング済みのさまざまなモデルが付属しています。これらは[モデルリポジトリ](https://github.com/tensorflow/tfjs-models)から入手できます。ウェブアプリケーションで使用する TensorFlow モデルを他の場所で見つけた場合、または作成した場合は、TensorFlow.js が提供するモデル[コンバータ](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)を利用できます。TensorFlow.js コンバータには以下の 2 つの要素があります。

1. TensorFlow.js で使用するために Keras および TensorFlow モデルを変換するコマンドラインユーティリティ。
2. TensorFlow.js を使用してブラウザでモデルを読み込み、実行するための API。

## モデルの変換

TensorFlow.js コンバータは、以下のモデル形式で動作します。

**SavedModel**: これは、TensorFlow モデルが保存されるデフォルトの形式です。SavedModel 形式についてのドキュメントは、[こちら](https://www.tensorflow.org/guide/saved_model)をご覧ください。

**Keras model**: Keras モデルは通常、HDF5 ファイルとして保存されます。Keras モデルの保存についての詳細は、[こちら](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state)をご覧ください。

**TensorFlow Hub モジュール**: これらは、モデルを共有および検出するためのプラットフォームである TensorFlow Hub での配布用にパッケージ化されたモデルです。モデルライブラリは[こちら](https://tfhub.dev/)をご覧ください。

変換するモデルのタイプに応じて、異なる引数をコンバータに渡す必要があります。たとえば、`model.h5`という名前の Keras モデルを`tmp/`ディレクトリに保存したとします。TensorFlow.js コンバータを使用してモデルを変換するには、次のコマンドを実行します。

```
$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model
```

これにより、`/tmp/model.h5`のモデルが変換され、`model.json`ファイルとバイナリウェイトファイルを`tmp/tfjs_model/`ディレクトリに出力します。

さまざまなモデル形式に対応するコマンドライン引数の詳細については、TensorFlow.js コンバータ [README](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) をご覧ください。

変換プロセス中に、モデルグラフを走査して、各演算が TensorFlow.js でサポートされていることを確認します。その場合は、ブラウザが使用できる形式にグラフを書き込みます。モデルを Web 上で提供できるように重みを 4MB のファイルにシャーディングすることにより最適化します (ブラウザでキャッシュできるようにするため)。また、オープンソースの [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) プロジェクトを使用して、モデルグラフ自体を簡略化することも試みます。グラフを簡略化するは、隣接する演算の折り畳みや一般的なサブグラフの削除などを実行します。これらの変更は、モデルの出力には影響しません。さらに最適化するために、モデルを特定のバイトサイズに量子化するようコンバータに指示する引数を渡すことができます。量子化は、より少ないビットで重みを表現することによりモデルサイズを縮小する手法です。ユーザーは、量子化後もモデルが許容可能な精度を維持するように注意する必要があります。

変換中にサポートされていない演算が発生した場合、プロセスは失敗し、演算名が出力されます。[GitHub](https://github.com/tensorflow/tfjs/issues) での問題がありましたら、通知してください。ユーザーの皆様からの要求に応じて新しい演算を実装します。

### ベストプラクティス

変換中にモデルを最適化するためにあらゆる努力を払っていますが、多くの場合、モデルのパフォーマンスを保証する最善の方法は、リソースに制約のある環境を考慮してモデルを構築することです。そのために、過度に複雑なアーキテクチャーを避け、可能な場合はパラメータ (重み) の数を最小限に抑えることをお勧めします。

## モデルの実行

モデルが正常に変換されると、一連の重みファイルとモデルトポロジファイルが作成されます。TensorFlow.js は、これらのモデルアセットをフェッチしてブラウザで推論を実行するために使用できるモデル読み込み API を提供します。

変換された TensorFlow SavedModel または TensorFlow Hub モジュールの API は次のようになります。

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

変換された Keras モデルは次のとおりです。

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

`tf.loadGraphModel` API は`tf.FrozenModel`を返します。これは、パラメータが固定されており、新しいデータでモデルを微調整できないことを意味します。`tf.loadLayersModel` API は、トレーニング可能な tf.Model を返します。tf.Model をトレーニングする方法については、[トレーニングモデルガイド](train_models.md)を参照してください。

変換後、推論を数回実行し、モデルの速度をベンチマークすることをお勧めします。このためには、以下のスタンドアロンのベンチマークページをご利用ください。https://tensorflow.github.io/tfjs/e2e/benchmarks/local-benchmark/index.html。お気づきになったかもしれませんが、最初のウォームアップ実行からの測定値が破棄されています。これは、テクスチャの作成とシェーダーのコンパイルのオーバーヘッドにより、一般的にモデルの最初の推論がその後の推論よりも数倍遅くなるためです。

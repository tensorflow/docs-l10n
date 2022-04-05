# よくある質問

質問に対する回答がここで見つからない場合は、トピックに関する詳細なドキュメントを確認するか、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues)を提出してください。

## モデル変換

#### TensorFlow から TensorFlow Lite への変換では、どの形式がサポートされていますか？

サポートされている形式は、[こちら](../convert/index.md#python_api)にリストされています。

#### TensorFlow Lite に実装されていない演算があるのはなぜですか？

TFLite を軽量に維持するため、TFLite では特定の TF 演算子のみがサポートされています（[allowlist](op_select_allowlist.md) を参照）。

#### モデルを変換できない場合があるのはなぜですか？

TensorFlow Lite 演算の数は、TensorFlow の演算よりも少ないため、変換できないモデルがある場合があります。一部の一般的なエラーは、[こちら](../convert/index.md#conversion-errors)にリストされています。

サポートされていない演算や制御フロー演算に関連しない変換の問題については、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)を検索するか、[新しい課題](https://github.com/tensorflow/tensorflow/issues)を提出してください。

#### TensorFlow Lite モデルが元の TensorFlow モデルと同じように動作することをどのようにテストしますか？

同じ入力（テストデータまたはランダム入力）による TensorFlow モデルと TensorFlow Lite モデルの出力を比較するのが最善のテスト方法です。[こちら](inference.md#load-and-run-a-model-in-python)をご覧ください。

#### GraphDef プロトコルバッファーの入力/出力を決定するにはどうすればよいですか？

`.pb`ファイルからグラフを検査する最も簡単な方法は、機械学習モデルのオープンソースビューアである [Netron](https://github.com/lutzroeder/netron) を使用することです。

Netron がグラフを開けられない場合は、[summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs) ツールを試してみてください。

summary_graph ツールでエラーが発生した場合は、[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) を使用して GraphDef を視覚化し、グラフで入力と出力を確認できます。`.pb`ファイルを視覚化するには、次のような[`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py)スクリプトを使用します。

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### `.tflite`ファイルを検査するにはどうすればよいですか？

[Netron](https://github.com/lutzroeder/netron) は、TensorFlow Lite モデルを視覚化する最も簡単な方法です。

Netron が TensorFlow Lite モデルを開けられない場合は、リポジトリにある [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py) スクリプトを試してみてください。

TF 2.5 以降のバージョンを使用している場合

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

それ以外の場合は、Bazel でこのスクリプトを実行できます

- [TensorFlow リポジトリをクローンする](https://www.tensorflow.org/install/source)
- `visualize.py`スクリプトを Bazel で実行する

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## 最適化

#### 変換した TensorFlow Lite モデルのサイズを縮小するにはどうすればよいですか？

TensorFlow Lite に変換する際に[トレーニング後の量子化](../performance/post_training_quantization.md)を使用すると、モデルのサイズを縮小できます。トレーニング後の量子化では、重みを浮動小数点から 8 ビットの精度に量子化し、実行時にそれらを逆量子化して浮動小数点計算を実行します。ただし、これは精度に影響する可能性があるので注意してください。

モデルの再トレーニングが可能な場合は、[量子化認識トレーニング](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)を検討してください。ただし、量子化認識トレーニングは、畳み込みニューラルネットワークアーキテクチャのサブセットでのみ使用できることに注意してください。

さまざまな最適化手法の詳細については、[モデルの最適化](../performance/model_optimization.md)をご覧ください。

#### 機械学習タスク用に TensorFlow Lite のパフォーマンスを最適化するにはどうすればよいですか？

TensorFlow Lite のパフォーマンスを最適化する高レベルのプロセスは、次のようになります。

- *モデルがタスクに適していることを確認します。*画像の分類については、[ホステッドモデルのリスト](hosted_models.md)を参照してください。
- *スレッド数を微調整します。*多くの TensorFlow Lite 演算子はマルチスレッドカーネルをサポートしています。これを行うには、[C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345) で`SetNumThreads()`を使用します。ただし、スレッドを増やすと、環境によってパフォーマンスが変動します。
- *ハードウェアアクセラレータを使用します。*TensorFlow Lite は、デリゲートを使用した特定のハードウェアのモデルアクセラレーションをサポートします。サポートされているアクセラレータと、デバイス上のモデルでそれらを使用する方法については、[デリゲート](../performance/delegates.md)ガイドを参照してください。
- *(高度) プロファイルモデル。*Tensorflow Lite [ベンチマークツール](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)には、演算子ごとの統計を表示できる組み込みのプロファイラがあります。特定のプラットフォームで演算子のパフォーマンスを最適化する方法をご存じの場合は、[カスタム演算子](ops_custom.md)を実装できます。

パフォーマンス最適化の詳細については、[ベストプラクティス](../performance/best_practices.md)をご覧ください。

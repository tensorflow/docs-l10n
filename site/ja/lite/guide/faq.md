# よくある質問

質問に対する回答がここで見つからない場合は、トピックに関する詳細なドキュメントを確認するか、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues)を提出してください。

## モデル変換

#### TensorFlow から TensorFlow Lite への変換では、どの形式がサポートされていますか？

TensorFlow Lite コンバータは、次の形式をサポートしています。

- SavedModels: [TFLiteConverter.from_saved_model](../convert/python_api.md#exporting_a_savedmodel_)
- [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) により生成された Frozen GraphDefs: [TFLiteConverter.from_frozen_graph](../convert/python_api.md#exporting_a_graphdef_from_file_)
- tf.keras HDF5 モデル: [TFLiteConverter.from_keras_model_file](../convert/python_api.md#exporting_a_tfkeras_file_)
- tf.Session: [TFLiteConverter.from_session](../convert/python_api.md#exporting_a_graphdef_from_tfsession_)

互換性の問題を早期に検出するために、[Python コンバータ](../convert/python_api.md)をモデルパイプラインに統合することをお勧めします。

#### モデルを変換できない場合があるのはなぜですか？

TensorFlow Lite の演算の数は TensorFlow の演算の数よりも少ないため、一部の推論モデルは変換できない場合があります。実装されていない演算については、[演算子のサポートがない場合](faq.md#why-are-some-operations-not-implemented-in-tensorflow-lite)に関する質問をご覧ください。サポートされていない演算子には、埋め込みと LSTM/RNN が含まれます。LSTM/RNN を備えたモデルの場合、試験的な API である [OpHint](https://www.tensorflow.org/api_docs/python/tf/lite/OpHint) を試して変換してみることもできます。現在、制御フロー演算 (スイッチ、マージなど) を備えたモデルは変換できませんが、Tensorflow Lite では制御フローのサポートの追加に取り組んでいます。[GitHub 課題](https://github.com/tensorflow/tensorflow/issues/28485)をご覧ください。

サポートされていない演算や制御フロー演算に関連しない変換の問題については、[GitHub 課題](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)を検索するか、[新しい課題](https://github.com/tensorflow/tensorflow/issues)を提出してください。

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

- [Clone the TensorFlow repository](https://www.tensorflow.org/install/source)
- `visualize.py`スクリプトを Bazel で実行する

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## モデルと演算

#### TensorFlow Lite に実装されていない演算があるのはなぜですか？

TensorFlow Lite を軽量に保つために、コンバータでは特定の演算のみが使用されています。TensorFlow Lite で現在サポートされている演算のリストは[互換性ガイド](ops_compatibility.md)で提供されています。

特定の演算 (または同等の演算) がリストに表示されていない場合は、その演算が優先されていない可能性があります。チームは、GitHub [課題 #21526](https://github.com/tensorflow/tensorflow/issues/21526) の新しい演算のリクエストを追跡します。リクエストがまだ対応されていない場合は、コメントを書いてください。

その間、[カスタム演算子](ops_custom.md)を実装するか、サポートされている演算子のみを含む別のモデルを使用してみてください。バイナリサイズが制約にならない場合は、TensorFlow Lite の [Select TensorFlow 演算子](ops_select.md)を使用してみてください。

#### TensorFlow Lite モデルが元の TensorFlow モデルと同じように動作することをどのようにテストしますか？

The best way to test the behavior of a TensorFlow Lite model is to use our API with test data and compare the outputs to TensorFlow for the same inputs. Take a look at our [Python Interpreter example](../convert/python_api.md) that generates random data to feed to the interpreter.

## 最適化

#### 変換した TensorFlow Lite モデルのサイズを縮小するにはどうすればよいですか？

[Post-training quantization](../performance/post_training_quantization.md) can be used during conversion to TensorFlow Lite to reduce the size of the model. Post-training quantization quantizes weights to 8-bits of precision from floating-point and dequantizes them during runtime to perform floating point computations. However, note that this could have some accuracy implications.

If retraining the model is an option, consider [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize). However, note that quantization-aware training is only available for a subset of convolutional neural network architectures.

For a deeper understanding of different optimization methods, look at [Model optimization](../performance/model_optimization.md).

#### 機械学習タスク用に TensorFlow Lite のパフォーマンスを最適化するにはどうすればよいですか？

The high-level process to optimize TensorFlow Lite performance looks something like this:

- *モデルがタスクに適していることを確認します。*画像の分類については、[ホステッドモデルのリスト](hosted_models.md)を参照してください。
- *スレッド数を微調整します。*多くの TensorFlow Lite 演算子はマルチスレッドカーネルをサポートしています。これを行うには、[C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345) で`SetNumThreads()`を使用します。ただし、スレッドを増やすと、環境によってパフォーマンスが変動します。
- *ハードウェアアクセラレータを使用します。*TensorFlow Lite は、デリゲートを使用した特定のハードウェアのモデルアクセラレーションをサポートします。サポートされているアクセラレータと、デバイス上のモデルでそれらを使用する方法については、[デリゲート](../performance/delegates.md)ガイドを参照してください。
- *(Advanced) Profile Model.* The Tensorflow Lite [benchmarking tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) has a built-in profiler that can show per-operator statistics. If you know how you can optimize an operator’s performance for your specific platform, you can implement a [custom operator](ops_custom.md).

For a more in-depth discussion on how to optimize performance, take a look at [Best Practices](../performance/best_practices.md).

# Quickstart for Linux-based devices with Python

TensorFlow Lite と Python の使用は、[Raspberry Pi](https://www.raspberrypi.org/){:.external} や [Edge TPU を使用した Coral デバイス](https://coral.withgoogle.com/){:.external} などの Linux ベースの組み込みデバイスに最適です。

This page shows how you can start running TensorFlow Lite models with Python in just a few minutes. All you need is a TensorFlow model [converted to TensorFlow Lite](../models/convert/). (If you don't have a model converted yet, you can experiment using the model provided with the example linked below.)

## TensorFlow Lite ランタイムパッケージについて

TensorFlow Lite モデルを Python で素早く実行できるようにするには、TensorFlow パッケージ全体の代わりに TensorFlow Lite インタプリタのみをインストールすることができます。この簡略化された Python パッケージは `tflite_runtime` と呼ばれています。

この `tflite_runtime` のパッケージは、`TensorFlow` のフルパッケージのわずか一部のサイズで、TensorFlow Lite で推論を実行するために最小限必要なコードのみが含まれます。含まれているのは、<a></a>[`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python クラスのみです。パッケージサイズが小さいため、`.tflite` モデルの実行のみが必要であり、大規模な TensorFlow ライブラリによるディスクスペースの浪費を避ける場合に理想的と言えます。

Note: If you need access to other Python APIs, such as the [TensorFlow Lite Converter](../models/convert/), you must install the [full TensorFlow package](https://www.tensorflow.org/install/). For example, the [Select TF ops] (https://www.tensorflow.org/lite/guide/ops_select) are not included in the `tflite_runtime` package. If your models have any dependencies to the Select TF ops, you need to use the full TensorFlow package instead.

## Python 向け TensorFlow Lite のインストール

Linux では pip を使用してインストールできます。

<pre class="devsite-terminal devsite-click-to-copy">python3 -m pip install tflite-runtime
</pre>

## サポートされているプラットフォーム

`tflite-runtime` Python ホイールは、事前に構築された状態で以下のプラットフォームに提供されています。

- Linux armv7l（例: Raspberry Pi 2、3、4、および Raspberry Pi OS 32 ビットを実行する Zero 2）
- Linux aarch64（例: Debian ARM64 を実行する Raspberry Pi 3、4）
- Linux x86_64

他のプラットフォームで TensorFlow Lite モデルを実行する場合は、[完全な TensorFlow パッケージ](https://www.tensorflow.org/install/)を使用するか、[ソースから tflite-runtime パッケージを構築します](build_cmake_pip.md)。

Coral Edge TPU で TensorFlow を使用している場合は、適切な [Coral セットアップドキュメント](https://coral.ai/docs/setup)に従う必要があります。

注意: Debian パッケージ `python3-tflite-runtime` の更新サポートは終了しています。最新の Debian パッケージは TF バージョン 2.5 用であり、[以前の手順](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python)でインストール可能です。

注意: Windows と macOS では、事前構築済みの `tflite-runtime` ホイールのリリースが終了しています。これらのプラットフォームでは、[完全な TensorFlow パッケージ](https://www.tensorflow.org/install/)を使用するか、[ソースから tflite-runtime パッケージを構築](build_cmake_pip.md)してください。

## tflite_runtime を使用して推論を実行する

そのため、`tensorflow` モジュールから `Interpreter` をインポートする代わりに、`tflite_runtime` からインポートする必要があります。

For example, after you install the package above, copy and run the [`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/) file. It will (probably) fail because you don't have the `tensorflow` library installed. To fix it, edit this line of the file:

```python
import tensorflow as tf
```

上記を次のように編集します。

```python
import tflite_runtime.interpreter as tflite
```

また、次の行も変更します。

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

上記を次のように変更します。

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

もう一度 `label_image.py` を実行してください。そうです！TensorFlow Lite モデルを実行できるようになりました。

## 今後の学習

- `Interpreter` API の詳細については、[Python でモデルを読み込んで実行する](inference.md#load-and-run-a-model-in-python)をお読みください。

- If you have a Raspberry Pi, check out a [video series](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe) about how to run object detection on Raspberry Pi using TensorFlow Lite.

- Coral ML アクセラレータを使用している場合は、[GitHub の Coral サンプル](https://github.com/google-coral/tflite/tree/master/python/examples)をご覧ください。

- To convert other TensorFlow models to TensorFlow Lite, read about the [TensorFlow Lite Converter](../models/convert/).

- `tflite_runtime` ホイールをビルドする場合は、[TensorFlow Lite Python ホイールパッケージをビルドする](build_cmake_pip.md)をお読みください。

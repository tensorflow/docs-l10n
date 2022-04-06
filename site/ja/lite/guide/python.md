# Python クイックスタート

TensorFlow Lite と Python の使用は、[Raspberry Pi](https://www.raspberrypi.org/){:.external} や [Edge TPU を使用した Coral デバイス](https://coral.withgoogle.com/){:.external} などの Linux ベースの組み込みデバイスに最適です。

このページでは、Python で TensorFlow Lite モデルをすぐに実行できるようにする方法を説明します。必要なのは、[TensorFlow Lite に変換された](../convert/) TensorFlow モデルのみです。（変換済みのモデルがまだ用意されていない場合は、以下にリンクされた例で使用されているモデルを使って実験できます。）

## TensorFlow Lite ランタイムパッケージについて

TensorFlow Lite モデルを Python で素早く実行できるようにするには、TensorFlow パッケージ全体の代わりに TensorFlow Lite インタプリタのみをインストールすることができます。この簡略化された Python パッケージは `tflite_runtime` と呼ばれています。

この `tflite_runtime` のパッケージは、`TensorFlow` のフルパッケージのわずか一部のサイズで、TensorFlow Lite で推論を実行するために最小限必要なコードのみが含まれます。含まれているのは、<a></a>[`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python クラスのみです。パッケージサイズが小さいため、`.tflite` モデルの実行のみが必要であり、大規模な TensorFlow ライブラリによるディスクスペースの浪費を避ける場合に理想的と言えます。

Note: If you need access to other Python APIs, such as the [TensorFlow Lite Converter](../convert/), you must install the [full TensorFlow package](https://www.tensorflow.org/install/). For example, the [Select TF ops] (https://www.tensorflow.org/lite/guide/ops_select) are not included in the `tflite_runtime` package. If your models have any dependencies to the Select TF ops, you need to use the full TensorFlow package instead.

## Python 向け TensorFlow Lite のインストール

You can install on Linux with pip:

<pre class="devsite-terminal devsite-click-to-copy">python3 -m pip install tflite-runtime
</pre>

## Supported platforms

The `tflite-runtime` Python wheels are pre-built and provided for these platforms:

- Linux armv7l (e.g. Raspberry Pi 2, 3, 4 and Zero 2 running Raspberry Pi OS 32-bit)
- Linux aarch64 (e.g. Raspberry Pi 3, 4 running Debian ARM64)
- Linux x86_64

If you want to run TensorFlow Lite models on other platforms, you should either use the [full TensorFlow package](https://www.tensorflow.org/install/), or [build the tflite-runtime package from source](build_cmake_pip.md).

If you're using TensorFlow with the Coral Edge TPU, you should instead follow the appropriate [Coral setup documentation](https://coral.ai/docs/setup).

Note: We no longer update the Debian package `python3-tflite-runtime`. The latest Debian package is for TF version 2.5, which you can install by following [these older instructions](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python).

Note: We no longer release pre-built `tflite-runtime` wheels for Windows and macOS. For these platforms, you should use the [full TensorFlow package](https://www.tensorflow.org/install/), or [build the tflite-runtime package from source](build_cmake_pip.md).

## tflite_runtime を使用して推論を実行する

そのため、`tensorflow` モジュールから `Interpreter` をインポートする代わりに、`tflite_runtime` からインポートする必要があります。

たとえば、上記のパッケージをインストールした後に、[`label_image.py`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/) ファイルをコピーして実行するとします。`tensorflow` ライブラリがインストールされていないため、この操作は失敗するでしょう。これを修正するには、ファイルの次の行を編集します。

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

- To convert other TensorFlow models to TensorFlow Lite, read about the [TensorFlow Lite Converter](../convert/).

- `tflite_runtime` ホイールをビルドする場合は、[TensorFlow Lite Python ホイールパッケージをビルドする](build_cmake_pip.md)をお読みください。

# Python クイックスタート

TensorFlow Lite と Python の使用は、[Raspberry Pi](https://www.raspberrypi.org/){:.external} や [Edge TPU を使用した Coral デバイス](https://coral.withgoogle.com/){:.external} などの Linux ベースの組み込みデバイスに最適です。

このページでは、Python で TensorFlow Lite モデルをすぐに実行できるようにする方法を説明します。必要なのは、[TensorFlow Lite に変換された](../convert/) TensorFlow モデルのみです。（変換済みのモデルがまだ用意されていない場合は、以下にリンクされた例で使用されているモデルを使って実験できます。）

## TensorFlow Lite ランタイムパッケージについて

TensorFlow Lite モデルを Python で素早く実行できるようにするには、TensorFlow パッケージ全体の代わりに TensorFlow Lite インタプリタのみをインストールすることができます。この簡略化された Python パッケージは `tflite_runtime` と呼ばれています。

この `tflite_runtime` のパッケージは、`TensorFlow` のフルパッケージのわずか一部のサイズで、TensorFlow Lite で推論を実行するために最小限必要なコードのみが含まれます。含まれているのは、<a></a>[`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python クラスのみです。パッケージサイズが小さいため、`.tflite` モデルの実行のみが必要であり、大規模な TensorFlow ライブラリによるディスクスペースの浪費を避ける場合に理想的と言えます。

注意: [TensorFlow Lite Converter](../convert/) などの他の Python API にアクセスする必要がある場合、[完全な TensorFlow パッケージ](https://www.tensorflow.org/install/)をインストールする必要があります。`tflite_runtime` パッケージには、Select TF 演算（https://www.tensorflow.org/lite/guide/ops_select）などが含まれていません。モデルに Select TF 演算への依存関係が含まれる場合は、代わりに完全な TensorFlow パッケージを使用する必要があります。

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

- Raspberry Pi を使用している場合は、TensorFlow Lite を使用して Rasberry Pi でオブジェクト検出を実行する方法を説明した[動画シリーズ](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe)をご覧ください。

- Coral ML アクセラレータを使用している場合は、[GitHub の Coral サンプル](https://github.com/google-coral/tflite/tree/master/python/examples)をご覧ください。

- 他の TensorFlow モデルを TensorFlow Lite に変換するには、[TensorFlow Lite コンバーター](../convert/) についてお読みください。

- `tflite_runtime` ホイールをビルドする場合は、[TensorFlow Lite Python ホイールパッケージをビルドする](build_cmake_pip.md)をお読みください。

# Python クイックスタート

TensorFlow Lite と Python の使用は、[Raspberry Pi](https://www.raspberrypi.org/){:.external} や [Edge TPU を使用した Coral デバイス](https://coral.withgoogle.com/){:.external} などの Linux ベースの組み込みデバイスに最適です。

このページでは、Python で TensorFlow Lite モデルをすぐに実行できるようにする方法を説明します。必要なのは、[TensorFlow Lite に変換された](../convert/) TensorFlow モデルのみです。（変換済みのモデルがまだ用意されていない場合は、以下にリンクされた例で使用されているモデルを使って実験できます。）

## TensorFlow Lite ランタイムパッケージについて

TensorFlow Lite モデルを Python で素早く実行できるようにするには、TensorFlow パッケージ全体の代わりに TensorFlow Lite インタプリタのみをインストールすることができます。この簡略化された Python パッケージは `tflite_runtime` と呼ばれています。

この `tflite_runtime` のパッケージは、`TensorFlow` のフルパッケージのわずか一部のサイズで、TensorFlow Lite で推論を実行するために最小限必要なコードのみが含まれます。含まれているのは、<a></a>[`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python クラスのみです。パッケージサイズが小さいため、`.tflite` モデルの実行のみが必要であり、大規模な TensorFlow ライブラリによるディスクスペースの浪費を避ける場合に理想的と言えます。

注意: [TensorFlow Lite コンバータ](../convert/python_api.md)などのほかの Python API にアクセスする必要がある場合は、[TensorFlow のフルパッケージ](https://www.tensorflow.org/install/)をインストールする必要があります。

## Python 向け TensorFlow Lite のインストール

Debian Linux または Debian の派生物（Raspberry Pi OS を含む）を実行している場合は、Debian パッケージリポジトリからインストールする必要があります。そのためには、新しいリポジトリリストとキーをシステムに追加してから、次のようにインストールします。

<pre class="devsite-terminal">echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
&lt;code class="devsite-terminal"
&gt;GL_CODE_5&lt;/code&gt;&lt;code class="devsite-terminal"
&gt;GL_CODE_6&lt;/code&gt;&lt;code class="devsite-terminal"
&gt;GL_CODE_7&lt;/code&gt;
</pre>

他のすべてのシステムでは、pip を使用してインストールできます。

<pre class="devsite-terminal devsite-click-to-copy">pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

そのため、`tensorflow` モジュールから `Interpreter` をインポートする代わりに、`tflite_runtime` からインポートする必要があります。

注：Debian Linux を実行していて、pip を使用して `tflite_runtime` をインストールする場合、Debian パッケージとしてインストールした TF Lite に依存する他のソフトウェア（[Coral libraries](https://coral.ai/software/) など）を使用すると、ランタイムエラーが発生する可能性があります。`tflite_runtime` を pip でアンインストールしてから、上記の `apt-get` コマンドで再インストールすると、修正できます。

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

`Interpreter` API の詳細については、[Python でモデルを読み込んで実行する](inference.md#load-and-run-a-model-in-python)をお読みください。

Raspberry Pi を使用している場合は、Pi Camera と TensorFlow Lite を使った画像分類を実行する[classify_picamera.py の例](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) をお試しください。

Coral ML アクセラレータを使用している場合は、[GitHub の Coral サンプル](https://github.com/google-coral/tflite/tree/master/python/examples)をご覧ください。

ほかの TensorFlow モデルを TensorFlow Lite に変換するには、[TensorFlow Lite コンバータ](../convert/) についてお読みください。

`tflite_runtime` ホイールをビルドする場合は、[TensorFlow Lite Python ホイールパッケージをビルドする](build_cmake_pip.md)をお読みください。

# Python クイックスタート

TensorFlow Lite と Python の使用は、[Raspberry Pi](https://www.raspberrypi.org/){:.external} や [Edge TPU を使用した Coral デバイス](https://coral.withgoogle.com/){:.external} などの Linux ベースの組み込みデバイスに最適です。

このページでは、Python で TensorFlow Lite モデルをすぐに実行できるようにする方法を説明します。必要なのは、[TensorFlow Lite に変換された](../convert/) TensorFlow モデルのみです。（変換済みのモデルがまだ用意されていない場合は、以下にリンクされた例で使用されているモデルを使って実験できます。）

## TensorFlow Lite インタープリタのみをインストールする

TensorFlow Lite モデルを Python で素早く実行できるようにするには、TensorFlow パッケージ全体の代わりに TensorFlow Lite インタープリタのみをインストールすることができます。

このインタープリタのみのパッケージは、TensorFlow のフルパッケージのわずか一部のサイズで、TensorFlow Lite で推論を実行するために最小限必要なコードのみが含まれます。含まれているのは、[`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python クラスのみです。パッケージサイズが小さいため、`.tflite` モデルの実行のみが必要であり、大規模な TensorFlow ライブラリによるディスクスペースの浪費を避ける場合に理想的と言えます。

注意: [TensorFlow Lite コンバータ](../convert/python_api.md)などのほかの Python API にアクセスする必要がある場合は、[TensorFlow のフルパッケージ](https://www.tensorflow.org/install/)をインストールする必要があります。

インストールするには、`pip3 install` を実行して、次のテーブルに示される適切な Python wheel の URL を渡します。

たとえば、Raspbian Buster（Python 3.7 を使用）を実行している Raspberry Pi があるとした場合、次のように Python wheel をインストールします。

<pre class="devsite-terminal devsite-click-to-copy">pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
</pre>

<table>
<tr>
<th>プラットフォーム</th>
<th>Python</th>
<th>URL</th>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux（ARM 32）</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_armv7l.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux（ARM 64）</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="4">Linux（x86-64）</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.8</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">macOS 10.14</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Windows 10</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl</td>
</tr>
</table>

## tflite_runtime を使用して推論を実行する

このインタープリタのみのパッケージと TensorFlow フルパッケージを区別するために（両方をインストール可）、上記の wheel で提供される Python モジュールは、`tflite_runtime` と名付けられています。

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

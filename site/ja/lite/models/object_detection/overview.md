# 物体検出

物体検出モデルは、画像またはビデオストリームに存在する既知の物体のセットを識別し、画像内のそれらの位置に関する情報を提供します。

たとえば、以下の<a href="#get_started">サンプルアプリ</a>のスクリーンショットは、2 つのオブジェクトがどのように認識され、それらの位置に注釈が付けられていることを示しています。

 <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/models/images/detection.png?raw=true">

Note: (1) To integrate an existing model, try [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector). (2) To customize a model, try [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker).

## はじめに

モバイルアプリで物体検出を使用する方法については、<a href="#example_applications_and_guides">サンプルアプリとガイド</a>をご覧になることをお勧めします。

Android または iOS 以外のプラットフォームを使用する場合、または、すでに <a href="https://www.tensorflow.org/api_docs/python/tf/lite">TensorFlow Lite API</a> に精通している場合は、物体検出スターターモデルと付随するラベルをダウンロードしてください。

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">Download starter model with Metadata</a>

For more information about Metadata and associated fields (eg: `labels.txt`) see <a href="../../models/convert/metadata#read_the_metadata_from_models">Read the metadata from models</a>

独自のタスク用にカスタム検出モデルをトレーニングする場合は、<a href="#model-customization">モデルのカスタマイズ</a>をご覧ください。

次の使用例では、別のタイプのモデルを使用する必要があります。

<ul>
  <li>画像が表す可能性が最も高い 1 つのラベルを予測する場合（<a href="../image_classification/overview.md">画像分類</a>を参照）</li>
  <li>画像の構成（被写体と背景など）を予測する場合（<a href="../segmentation/overview.md">セグメンテーション</a>を参照）</li>
</ul>

### サンプルアプリとガイド

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。

#### Android

TensorFlow Lite Task Library のそのまま簡単に使用できる API を利用して、わずか数行のコードで[物体検出モデルを統合する](../../inference_with_metadata/task_library/object_detector)ことができます。また、TensorFlow Lite Interpreter Java API を使用して、[独自のカスタム推論パイプライン](../../guide/inference#load_and_run_a_model_in_java)を構築することもできます。

以下の Android の例では、両方のメソッドをそれぞれ [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_task_api) および [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter) として実装しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Android の例を見る</a>

#### iOS

[TensorFlow Lite Interpreter Swift API](../../guide/inference#load_and_run_a_model_in_swift) を使用してモデルを統合できます。以下の iOS の例を参照してください。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">View iOS example</a>

## モデルの説明

このセクションでは、[TensorFlow Object Detection API](https://arxiv.org/abs/1512.02325) から TensorFlowLite に変換された [Single-Shot Detector](https://github.com/tensorflow/models/blob/master/research/object_detection/) モデルの署名について説明します。

物体検出モデルは、物体の複数のクラスの存在と位置を検出するようにトレーニングされています。たとえば、さまざまな果物を含む画像、果物の種類を示す*ラベル*（リンゴ、バナナ、イチゴなど）、および各物体が画像のどこにあるかを指定するデータでモデルをトレーニングすることができます。

その後、モデルに画像を提供すると、検出した物体のリスト、各物体を含む境界矩形の場所、および検出に対する信頼度を示すスコアを出力します。

### 入力署名

モデルは画像を入力として受け取ります。

画像は 300x300 ピクセルで、ピクセルごとに 3 つのチャネル （赤、青、緑）があるとします。これは、270,000 バイト値 (300x300x3) のフラット化されたバッファーとしてモデルにフィードする必要があります。モデルは<a href="../../performance/post_training_quantization.md">量子化</a>されている場合は、各値は 0〜255 間の値を表す 1 バイトである必要があります。

Android でこの前処理を行う方法については[サンプルアプリコード](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)をご覧ください。

### 出力署名

モデルは、インデックス 0〜4 にマップされた 4 つの配列を出力します。配列 0、1、2 は`N`個の検出されたオブジェクトを表し、各配列の 1 つの要素は各オブジェクトに対応します。

<table>
  <thead>
    <tr>
      <th>インデックス</th>
      <th>名称</th>
      <th>説明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>位置</td>
      <td>[N][4]の多次元配列 浮動小数点値（0~1） 、内部配列[上,左,下,右]は、境界矩形を示します。</td>
    </tr>
    <tr>
      <td>1</td>
      <td>クラス</td>
      <td>N 個の整数の配列（浮動小数点値として出力）。それぞれがラベルファイルからのクラスラベルのインデックスを示します。</td>
    </tr>
    <tr>
      <td>2</td>
      <td>スコア</td>
      <td>0～1 の間の N 個の浮動小数点値の配列。クラスが検出された確率を表します。</td>
    </tr>
    <tr>
      <td>3</td>
      <td>検出数</td>
      <td>N の整数値</td>
    </tr>
  </tbody>
</table>

注意：結果の数（上記の場合は 10）は、検出モデルをTensorFlowLite にエクスポートするときに設定されるパラメータです。詳細については、<a href="#model-customization">モデルのカスタマイズ</a>をご覧ください。

リンゴ、バナナ、イチゴを検出するようにモデルがトレーニングされているとします。画像を渡すと、設定した数の検出結果が出力されます。この例では 5 です。

<table style="width: 60%;">
  <thead>
    <tr>
      <th>クラス</th>
      <th>スコア</th>
      <th>位置</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>リンゴ</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>バナナ</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>イチゴ</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td>バナナ</td>
      <td>0.23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>リンゴ</td>
      <td>0.11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### 信頼度スコア

これらの結果を解釈するには、検出された各物体のスコアと場所を確認します。0〜1 の数値のスコアは、物体が正確に検出されたという信頼度を示し、数値が 1 に近いほど、モデルの信頼性が高くなります。

アプリケーションによってはカットオフしきい値を決定し、それを下回る値を破棄できます。この例では、妥当なカットオフスコアが 0.5 です（この場合、検出が有効である確率は 50％ です）。この場合、配列の最後の 2 つの物体の信頼スコアは 0.5 未満であるため無視します。

<table style="width: 60%;">
  <thead>
    <tr>
      <th>クラス</th>
      <th>スコア</th>
      <th>位置</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>リンゴ</td>
      <td>0.92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>バナナ</td>
      <td>0.88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>イチゴ</td>
      <td>0.87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">バナナ</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">リンゴ</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0.11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

適切なカットオフ値は、偽陽性（誤って識別された物体、またはそうでない場合に誤って物体として識別される画像の領域）または偽陰性（信頼性が低いために見落とされた本物の物体）のバランスに基づく必要があります。

たとえば、次の画像では、ナシ（モデルが検出するようにトレーニングされた物体ではない）が「人」として誤って識別されました。これは、適切なカットオフを選択することで無視できる誤検知の例です。この場合、0.6（または 60％）のカットオフは、誤検知を適切に除外します。

<img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/models/object_detection/images/android_apple_banana.png?raw=true" alt="Screenshot of Android example">

#### 位置

モデルは、検出されたオブジェクトごとに、その位置を囲む境界矩形を表す 4 つの数値の配列を返します。スターターモデルの場合、数値は次のように並べられます。

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>上,</td>
      <td>左,</td>
      <td>下,</td>
      <td>右</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

上の値は、画像の上端からの矩形の上端までの距離をピクセル単位で表します。左の値は、入力画像の左からの左端の距離を表します。他の値は、同様に下端と右端を表します。

注意: 物体検出モデルは、特定のサイズの入力画像を受け入れます。これは、デバイスのカメラでキャプチャされた未加工画像のサイズとは異なる可能性が高く、モデルの入力サイズに合わせて未加工画像をトリミングおよびスケーリングするコードを記述する必要があります（例は<a href="#get_started">サンプルアプリケーション</a>をご覧ください）。<br><br>モデルによって出力されたピクセル値は、トリミングおよびスケーリングされた画像内の位置を参照するため、正しく解釈するには、生の画像に合わせてスケーリングする必要があります。

## パフォーマンスベンチマーク

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">スターターモデル</a>のパフォーマンスベンチマークの数値は、[ここで説明する](https://www.tensorflow.org/lite/performance/benchmarks)ツールで生成されます。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>モデルサイズ</th>
      <th>デバイス</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">
      <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
    </td>
    <td rowspan="3">       27 Mb     </td>
    <td>Pixel 3 (Android 10)</td>
    <td>22ms</td>
    <td>46ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20ms</td>
    <td>29ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>7.6ms</td>
    <td>11ms**</td>
  </tr>
</table>

* 4 threads used.

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

## モデルのカスタマイズ

### 事前トレーニング済みモデル

[Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models) には、さまざまなレイテンシと精度の特性を備えたモバイル向けに最適化された検出モデルがあります。それぞれ、次のセクションで説明する入力署名および出力署名に従います。

Most of the download zips contain a `model.tflite` file. If there isn't one, a TensorFlow Lite flatbuffer can be generated using [these instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). SSD models from the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) can also be converted to TensorFlow Lite using the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md). It is important to note that detection models cannot be converted directly using the [TensorFlow Lite Converter](../../models/convert), since they require an intermediate step of generating a mobile-friendly source model. The scripts linked above perform this step.

[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) と [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) の両方のエクスポートスクリプトには、より多くの出力オブジェクトや、より低速でより正確な後処理を可能にするパラメータがあります。サポートされている引数の完全なリストを表示するには、スクリプトで`--help`を使用してください。

> 現在、デバイス上の推論は SSD モデルでのみ最適化されています。CenterNet や EfficientDet などの他のアーキテクチャに対するサポートの改善は研究されています。

### カスタマイズするモデルを選択するには

各モデルには、独自の精度（mAP 値で定量化）とレイテンシ特性があります。ユースケースと対象となるハードウェアに最適なモデルを選択する必要があります。たとえば、[Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) モデルは、Pixel 4 上の Google Edge TPU での推論に最適です。

利用可能な最も効率的なオプションを選択するには[ベンチマークツール](https://www.tensorflow.org/lite/performance/measurement)を使用してモデルを評価します。

## カスタムデータを使用するモデルのファインチューニング

提供される事前トレーニング済みモデルは、90 クラスの物体を検出するようにトレーニングされています。クラスの完全なリストについては、<a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">モデルメタデータ</a>のラベルファイルをご覧ください。

You can use a technique known as transfer learning to re-train a model to recognize classes not in the original set. For example, you could re-train the model to detect multiple types of vegetable, despite there only being one vegetable in the original training data. To do this, you will need a set of training images for each of the new labels you wish to train. The recommended way is to use [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) library which simplifies the process of training a TensorFlow Lite model using custom dataset, with a few lines of codes. It uses transfer learning to reduce the amount of required training data and time. You can also learn from [Few-shot detection Colab](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb) as an example of fine-tuning a pre-trained model with few examples.

より大きなデータセットでファインチューニングするには、TensorFlow Object Detection API を使用して独自のモデルをトレーニングするためのガイド、[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md) と [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md) をご覧ください。トレーニングが完了すると、[TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)、[TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) の手順で、TFLite に適した形式に変換できます。

# 画像分類


<img src="../images/image.png" class="attempt-right">

画像の内容を特定するタスクは、*画像分類*と呼ばれます。画像分類モデルは、画像のさまざまなクラスを認識するようにトレーニングされます。たとえば、ウサギ、ハムスター、イヌという 3 つの異なる種類の動物を表す写真を認識するようにモデルをトレーニングできます。TensorFlow Lite は最適化されたトレーニング済みモデルを提供しており、モバイルアプリケーションにデプロイできます。TensorFlow を使用した画像分類の詳細については、[こちら](https://www.tensorflow.org/tutorials/images/classification)を参照してください。

次の画像は、Android での画像分類モデルの出力を示します。


<img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/models/image_classification/images/android_banana.png?raw=true" alt="Screenshot of Android example" class="">

注意: (1) 既存のモデルを統合するには、[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier) を試してください。(2) モデルをカスタマイズするには、[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification) を試してください。

## はじめに

TensorFlow Lite を初めて使用する場合、Android または iOS を使用する場合は、以下のサンプルアプリをご覧ください。

[TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_classifier) のそのまま簡単に使用できる API を利用して、わずか数行のコードで画像分類モデルを統合できます。また、[TensorFlow Lite Support Library](../../inference_with_metadata/lite_support) を使用して、独自のカスタム推論パイプラインを構築することもできます。

以下の Android の例では、両方のメソッドをそれぞれ [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) および [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support) として実装しています。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android の例を見る</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS の例を見る</a>

Android/iOS 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、スターターモデルと追加ファイル (該当する場合) をダウンロードしてください。

スターターモデルをダウンロードする

## モデルの説明

### 使い方

トレーニング中、画像分類モデルには画像とそれに関連付けられた*ラベル*が提供されます。各ラベルは、モデルが認識することを学習する個別の概念またはクラスの名前です。

十分なトレーニングデータ (一般的には、ラベルごとに数百または数千個の画像) が入力されると、画像分類モデルは学習して、新しい画像がトレーニングされたクラスのいずれかに属するかどうかを予測できます。この予測プロセスは*推論*と呼ばれます。すでに[転移学習](https://www.tensorflow.org/tutorials/images/transfer_learning)を使用して、既存のモデルを利用することで、新しい画像のクラスを特定することもできます。転移学習には、大量のトレーニングデータセットは必要ありません。

その後、モデルへの入力として新しい画像を提供すると、トレーニングされた動物の各種類を表す画像の確率が出力されます。出力例は次のようになります。

<table style="width: 40%;">
  <thead>
    <tr>
      <th>動物の種類</th>
      <th>確率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ウサギ</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>ハムスター</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">イヌ</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

出力の各数値は、トレーニングデータのラベルに対応しています。出力をモデルがトレーニングされた 3 つのラベルに関連付けると、モデルが画像が犬を表すことが確率高いことを予測したことがわかります。

すべての確率 (ウサギ、ハムスター、イヌ) の合計が 1 であることにお気づきかもしれません。これは、複数のクラスを持つモデルの一般的なタイプの出力です (詳細については、<a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a> を参照してください)。

注: 画像分類は、画像がモデルがトレーニングされた 1 つ以上のクラスを表す確率のみを予測できます。画像内の物体が何であるか特定したり、物体の位置を知ることはできません。画像内の物体とその位置を特定する必要がある場合は、<a href="../object_detection/overview">物体検出</a>モデルを使用する必要があります。

<h4>不確実な結果</h4>

確率の合計は常に 1 になるため、画像がモデルがトレーニングされたどのクラスにも属していると確信して認識されない場合、1 つの値が大幅に大きくならずラベル全体に確率が分布することがあります。

たとえば、次の結果は不確実な結果を示している可能性があります。


<table style="width: 40%;">   <thead>     <tr>       <th>ラベル</th>       <th>確率</th>     </tr>   </thead>   <tbody>     <tr>       <td>ウザギ</td>       <td>0.31</td>     </tr>     <tr>       <td>ハムスター</td>       <td>0.35</td>     </tr>     <tr>       <td>イヌ</td>       <td>0.34</td>     </tr>   </tbody> </table> モデルが頻繁に曖昧な結果を返す場合は、別のより正確なモデルが必要になることがあります。

<h3>モデルアーキテクチャの選択</h3>

TensorFlow Lite には、さまざまな画像分類モデルが用意されており、すべて元のデータセットでトレーニング済みです。MobileNet、Inception、NASNet などのモデルアーキテクチャは、<a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a> で提供されています。ユースケースに最適なモデルを選択するには、個別のアーキテクチャと、各種モデル間のトレードオフをある程度検討する必要があります。これらのモデルトレードオフの一部は、パフォーマンス、精度、モデルサイズなどのメトリックに基づいています。たとえば、医療画像アプリで低速でも精度の高いモデルが必要なときに、バーコードスキャナを構築するための高速なモデルが必要になる場合があります。

提供される<a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">画像分類モデル</a>は、さまざまなサイズの入力を許可します。一部のモデルでは、これはファイル名に示されています。たとえば、Mobilenet_V1_1.0_224 モデルは、224x224 ピクセルの入力を受け入れます。すべてのモデルで、ピクセルごとに 3 つのカラーチャネル (赤、緑、青) が必要です。量子化モデルはチャネルごとに 1 バイトを必要とし、浮動小数点モデルはチャネルごとに 4 バイトを必要とします。<a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md">Android</a> と <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios/EXPLORE_THE_CODE.md">iOS</a> コードサンプルは、フルサイズのカメラ画像を各モデルに必要な形式に処理する方法を示しています。

<h3>使用と制限</h3>

TensorFlow Lite 画像分類モデルは、単一ラベル分類に役立ちます。つまり、画像が表す可能性が最も高い単一ラベルを予測します。これらは 1000 クラスの画像を認識するようにトレーニングされています。クラスの完全なリストについては、<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">モデル zip</a> のラベルファイルをご覧ください。

新しいクラスを認識するようにモデルをトレーニングする場合は、<a href="#customize_model">モデルのカスタマイズ</a>をご覧ください。

次の使用例では、別のタイプのモデルを使用する必要があります。

<ul>
  <li>画像内の 1 つ以上の物体の種類と位置を予測する場合（<a href="../object_detection/overview">物体検出</a>を参照）</li>
  <li>画像の構成 (被写体と背景) を予測する場合 (<a href="../segmentation/overview">セグメンテーション</a>を参照)</li>
</ul>

ターゲットデバイスでスターターモデルを実行してから、さまざまなモデルを試し、パフォーマンス、精度、モデルサイズの最適なバランスを見つけてください。

<h3>モデルをカスタマイズする</h3>

提供される事前トレーニング済みモデルは、1000 クラスの画像を検出するようにトレーニングされています。クラスの完全なリストについては、<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">モデル zip</a> のラベルファイルをご覧ください。

元のセットにないクラスを認識するようにモデルを再トレーニングするには転移学習を使用できます。たとえば、元のトレーニングデータには木がない場合でも、モデルを再トレーニングすると複数の種類の木を検出できます。これを行うには、トレーニングする新しいラベルごとに一連のトレーニング画像が必要です。

<a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a> で転移学習を実行する方法については、<a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Recognize flowers with TensorFlow</a> codelab を参照してください。

<h2>パフォーマンスベンチマーク</h2>

モデルパフォーマンスは、モデルが特定のハードウェアで推論を実行するのにかかる時間に基づいて測定されます。この時間が短いほど、モデルが高速になります。

必要とされるパフォーマンスは、アプリによって異なります。パフォーマンスは、次のフレームに対してドローが実行される前にリアルタイムで各フレームを分析することが重要な場合があるリアルタイムビデオなどのアプリでは重要になる場合があります。(たとえば、推論は 30fps ビデオストリームでリアルタイム推論を実行するために 33ms よりも高速でなければなりません)。

TensorFlow Lite 量子化 MobileNet モデルのパフォーマンスは、3.7ms から 80.3ms の範囲です。

パフォーマンスベンチマークの数値は、<a href="https://www.tensorflow.org/lite/performance/benchmarks">ベンチマークツール</a>で生成されます。

<table>
  <thead>
    <tr>
      <th>モデル名</th>
      <th>モデルサイズ</th>
      <th>デバイス</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3"><a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a></td>
    <td rowspan="3">       4.3 Mb     </td>
    <td>Pixel 3 (Android 10)</td>
    <td>6ms</td>
    <td>13ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>3.3ms</td>
    <td>5ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td></td>
    <td>11ms**</td>
  </tr>
</table>

* 4 threads used.

** 最高のパフォーマンス結果を得るために、iPhone では 2 つのスレッドを使用。

### モデルの精度

精度はモデルが画像を正確に分類する頻度として測定します。たとえば、指定された精度が 60％ のモデルでは、平均して 60％ の場合、画像が正しく分類されることが期待できます。

モデル関連性精度メトリックは、Top-1 と Top-5 です。Top-1 は、モデルの出力において、最も高い確率で、正しいラベルが表示される頻度を示します。Top-5 は、モデルの出力において、5 番目に高い確率で正しいラベルが表示される頻度を示します。

TensorFlow Lite 量子化 MobileNet モデルの上位 5 の精度の範囲は 64.4〜89.9％ です。

### モデルサイズ

The size of a model on-disk varies with its performance and accuracy. Size may be important for mobile development (where it might impact app download sizes) or when working with hardware (where available storage might be limited).

TensorFlow Lite 量子化 MobileNet モデルのサイズは、0.5〜3.4 Mb です。

## その他の資料とリソース

画像分類に関連する概念の詳細については、次のリソースを使用してください。

- [TensorFlow を使用した画像分類](https://www.tensorflow.org/tutorials/images/classification)
- [CNN を使用した画像分類](https://www.tensorflow.org/tutorials/images/cnn)
- [転移学習](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [データ拡張](https://www.tensorflow.org/tutorials/images/data_augmentation)

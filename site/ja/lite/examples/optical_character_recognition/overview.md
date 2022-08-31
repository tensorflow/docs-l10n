# 光学文字認識 (OCR)

光学文字認識 (OCR) は、コンピュータビジョンと機械学習手法を使用して、画像から文字を認識するプロセスです。このリファレンスアプリでは、TensorFlow Lite を使用して OCR を実行する方法についてデモで説明します。[テキスト検出モデル](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1)[と](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1)[テキスト認識モデル](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)を組み合わせて OCR パイプラインとして使用し、テキスト文字を認識します。

## Get started

<img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ja/lite/examples/optical_character_recognition/images/screenshot.gif?raw=true" class="">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example application that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Android example</a>

Android 以外のプラットフォームを使用する場合、または、すでに [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite) に精通している場合は、[TF Hub](https://tfhub.dev/) からモデルをダウンロードできます。

## How it works

一般的に、OCR タスクは 2 つのステージに分けられます。まず、テキスト検出モデルを使用して、テキストの可能性がある部分の周辺でバウンディングボックスを検出します。次に、処理されたバウンディングボックスをテキスト認識モデルに入力し、バウンディングボックス内の特定の文字を判定します (テキスト認識の前に、Non-Maximal Suppression、透視変換なども実行する必要があります)。このケースでは、両方のモデルが TensorFlow Lite Hub のモデルで、FP16 量子化モデルです。

## Performance benchmarks

Performance benchmark numbers are generated with the tool described [here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>デバイス</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">テキスト検出</a>
</td>
    <td>45.9 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181.93ms*</td>
     <td>89.77ms*</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">テキスト認識</a>
</td>
    <td>16.8 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338.33ms*</td>
     <td>N/A**</td>
  </tr>
</table>

* 4 threads used.

** このモデルでは、実行するために TensorFlow 演算が必要であるため、GPU デリゲートを使用できません。

## 入力

テキスト検出モデルでは、(1, 320, 320, 3) の 4-D `float32` テンソルを入力できます。

テキスト認識モデルでは、(1, 31, 200, 1) の 4-D `float32` テンソルを入力できます。

## 出力

テキスト検出モデルは、形状 (1, 80, 80, 5) の 4-D `float32` テンソルをバウンディングボックスとして返し、形状 (1,80, 80, 5) の 4-D `float32` テンソルを検出スコアとして返します。

テキスト認識モデルは、形状 (1, 48) の 2-D `float32` テンソルをアルファベットリスト '0123456789abcdefghijklmnopqrstuvwxyz' へのマッピングインデックスとして返します。

## Limitations

- 現在の[テキスト認識モデル](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)は、英語の文字と数字の合成データを使用してトレーニングされているため、英語のみがサポートされています。

- モデルはそのままでは OCR 向けに十分な一般性がありません (低い光条件でスマートフォンカメラで撮影されたランダム画像のような状態)。

このため、TensorFlow Lite で OCR を実行する方法を示す目的で、3 つの Google 製品ロゴを選びました。すぐに使用できる本番グレードの OCR 製品を探している場合は、[Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition) を検討してください。ML Kit の基盤では TFLite が使用されていますが、ほとんどの OCR ユースケースに十分対応できます。ただし、TFLite で独自の OCR ソリューションを構築する場合もあります。次にその例を示します。

- 独自のテキスト検出/認識 TFLite モデルがあり、それを使用したい。
- 特殊なビジネス要件 (例: 上下逆さまのテキストを認識する) があり、OCR パイプラインをカスタマイズする必要がある。
- ML Kit で対応していない言語をサポートしたい。
- ターゲットユーザーデバイスに必ずしも Google Play サービスがインストールされていない。

## References

- OpenCV テキスト検出/認識の例: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- コミュニティコントリビュータによる OCR TFLite コミュニティプロジェクト: https://github.com/tulasiram58827/ocr_tflite
- OpenCV テキスト検出: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- OpenCV を使用したディープラーニングベースのテキスト検出: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

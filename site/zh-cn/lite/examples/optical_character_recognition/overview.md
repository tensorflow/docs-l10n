# 光学字符识别 (OCR)

光学字符识别 (OCR) 是利用计算机视觉和机器学习技术从图像中识别字符的过程。此参考应用演示了如何使用 TensorFlow Lite 进行 OCR。它使用[文本检测模型](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1)和[文本识别模型](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)的组合作为识别文本字符的 OCR 流水线。

## Get started

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

If you are new to TensorFlow Lite and are working with Android, we recommend exploring the following example application that can help you get started.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Android 示例</a>

如果您使用的不是 Android 平台，或者您已经熟悉 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)，则可以从 [TF Hub](https://tfhub.dev/) 下载模型。

## How it works

OCR 任务通常分为两个阶段。首先，我们使用文本检测模型来检测可能的文本周围的边界框。其次，我们将处理后的边界框送入文本识别模型，以确定边界框内的特定字符（在文本识别之前，我们还需要进行非最大抑制、透视变换等）。在我们的示例中，这两个模型都来自 TensorFlow Hub，它们都是 FP16 量化模型。

## Performance benchmarks

Performance benchmark numbers are generated with the tool described [here](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Model size </th>
      <th>设备</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">文本检测</a>
</td>
    <td>45.9 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181.93ms*</td>
     <td>89.77ms*</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">文本识别</a>
</td>
    <td>16.8 Mb</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338.33ms*</td>
     <td>N/A**</td>
  </tr>
</table>

* 4 threads used.

** 此模型无法使用 GPU 委托，因为我们需要 TensorFlow 算子来运行该模型

## 输入

文本检测模型接受形状为 (1, 320, 320, 3) 的四维 `float32` 张量作为输入。

文本识别模型接受形状为 (1, 31, 200, 1) 的四维 `float32` 张量作为输入。

## 输出

文本检测模型返回形状为 (1, 80, 80, 5) 的四维 `float32` 张量作为边界框，并返回形状为 (1,80, 80, 5) 的四维 `float32` 张量作为检测分数。

文本识别模型返回形状为 (1, 48) 的二维 `float32` 张量，作为到字母列表 '0123456789abcdefghijklmnopqrstuvwxyz' 的映射索引。

## Limitations

- 当前的[文本识别模型](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2)使用英文字母和数字的合成数据进行训练，因此只支持英语。

- 这些模型还不够通用，无法随意进行 OCR（比如，在光线较弱的情况下，由智能手机摄像头拍摄的随机图像）。

因此，我们选择了 3 个 Google 产品徽标，只是为了演示如何使用 TensorFlow Lite 进行 OCR。如果您正在寻找现成的生产级 OCR 产品，您应该考虑 [Google ML Kit](https://developers.google.com/ml-kit/vision/text-recognition)。在底层使用 TFLite 的 ML Kit 对于大多数 OCR 用例来说应该足够了，但在某些情况下，您可能希望使用 TFLite 构建自己的 OCR 解决方案。以下是一些示例：

- 您希望使用自己的文本检测/识别 TFLite 模型
- 您有特殊的业务需求（即识别颠倒的文本），并且需要自定义 OCR 流水线
- 您想要支持 ML Kit 未涵盖的语言
- 您的目标用户设备不一定安装了 Google Play 服务

## References

- OpenCV 文本检测/识别示例：https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- 社区贡献者的 OCR TFLite 社区项目：<br>https://github.com/tulasiram58827/ocr_tflite
- OpenCV 文本检测：https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- 基于深度学习的 OpenCV 文本检测：https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

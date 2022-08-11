# 姿态预测


<img src="../images/pose.png" class="attempt-right">

*PoseNet* 能够通过预测图像或视频中人体的关键位置进行姿态的预测。

## 开始使用

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite">下载此模块</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android">Android 示例</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">iOS 示例</a>

如果您熟悉 [TensorFlow Lite API](https://www.tensorflow.org/api_docs/python/tf/lite)，请下载入门 MoveNet 姿态预测模型和支持文件。

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet">下载入门模型</a>

如果你想在 Web 浏览器上尝试姿态预测，请查看 <a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet">TensorFlow JS Demo</a>。

## 工作原理

### 使用案例

为了达到清晰的目的，该算法只是对图像中的人简单的预测身体关键位置所在，而不会去辨别此人是谁。

姿态预测模型会将处理后的相机图像作为输入，并输出有关关键点的信息。检测到的关键点由部位 ID 索引，置信度分数介于 0.0 和 1.0 之间。置信度分数表示该位置存在关键点的概率。

我们提供了两个 TensorFlow Lite 姿态预测模型的参考实现：

- MoveNet：最先进的姿态预测模型，有两个版本可供选择：Lightning 和 Thunder。在以下部分可以看到这两者之间的对比。
- PoseNet：2017 年发布的上一代姿态预测模型。

姿态预测模型检测到的各种身体关节如下表所示：

<table style="width: 30%;">
  <thead>
    <tr>
      <th>ID</th>
      <th>部位</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>鼻子</td>
    </tr>
    <tr>
      <td>1</td>
      <td>左眼</td>
    </tr>
    <tr>
      <td>2</td>
      <td>右眼</td>
    </tr>
    <tr>
      <td>3</td>
      <td>左耳</td>
    </tr>
    <tr>
      <td>4</td>
      <td>右耳</td>
    </tr>
    <tr>
      <td>5</td>
      <td>左肩</td>
    </tr>
    <tr>
      <td>6</td>
      <td>右肩</td>
    </tr>
    <tr>
      <td>7</td>
      <td>左肘</td>
    </tr>
    <tr>
      <td>8</td>
      <td>右肘</td>
    </tr>
    <tr>
      <td>9</td>
      <td>左腕</td>
    </tr>
    <tr>
      <td>10</td>
      <td>右腕</td>
    </tr>
    <tr>
      <td>11</td>
      <td>左胯</td>
    </tr>
    <tr>
      <td>12</td>
      <td>右胯</td>
    </tr>
    <tr>
      <td>13</td>
      <td>左膝</td>
    </tr>
    <tr>
      <td>14</td>
      <td>右膝</td>
    </tr>
    <tr>
      <td>15</td>
      <td>左踝</td>
    </tr>
    <tr>
      <td>16</td>
      <td>右踝</td>
    </tr>
  </tbody>
</table>

输出示例如下所示：

<img alt="Output stride and heatmap resolution" src="https://storage.googleapis.com/download.tensorflow.org/example_images/movenet_demo.gif" class="">

## 性能基准

MoveNet 有两种版本：

- MoveNet.Lightning 比 Thunder 版更小、更快，但准确率较低。它可以在当下的智能手机上实时运行。
- MoveNet.Thunder 是更准确的版本，但比 Lightning 版更大、更慢。对于需要更高准确率的用例，它非常有用。

MoveNet 在各种数据集上的表现都优于 PoseNet，尤其是在包含健身动作的图像上。因此，我们建议使用 MoveNet 而不是 PoseNet。

性能基准数值使用[此处介绍的](../../performance/measurement)工具生成。准确率 (MAP) 数值在 [COCO 数据集](https://cocodataset.org/#home)的子集上测得，在该数据集中，我们筛选并裁剪了每个图像，使其仅包含一个人。

<table>
<thead>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">大小 (MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">延迟 (ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - CPU 4 线程</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - CPU 4 线程</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder（FP16 量化）</a>
</td>
    <td>12.6MB</td>
    <td>72.0</td>
    <td>155ms</td>
    <td>45ms</td>
    <td>594ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder（INT8 量化）</a>
</td>
    <td>7.1MB</td>
    <td>68.9</td>
    <td>100ms</td>
    <td>52ms</td>
    <td>251ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning（FP16 量化）</a>
</td>
    <td>4.8MB</td>
    <td>63.0</td>
    <td>60ms</td>
    <td>25ms</td>
    <td>186ms</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning（INT8 量化）</a>
</td>
    <td>2.9MB</td>
    <td>57.4</td>
    <td>52ms</td>
    <td>28ms</td>
    <td>95ms</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet（MobileNetV1 主干，FP32）</a>
</td>
    <td>13.3MB</td>
    <td>45.6</td>
    <td>80ms</td>
    <td>40ms</td>
    <td>338ms</td>
  </tr>
</tbody>
</table>

## 补充阅读和资源

- 请查看这篇[博文](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)，了解更多使用 MoveNet 和 TensorFlow Lite 进行姿态预测的信息。
- 请查看这篇[博文](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)，了解更多关于 Web 姿态预测的信息。
- 请查看此[教程](https://www.tensorflow.org/hub/tutorials/movenet)，了解如何使用 TensorFlow Hub 的模型在 Python 上运行 MoveNet。
- Coral/EdgeTPU 可以加快姿态预测在边缘设备上的运行速度。有关更多详细信息，请参阅 [EdgeTPU 优化模型](https://coral.ai/models/pose-estimation/)。
- 请在[此处](https://arxiv.org/abs/1803.08225)阅读 PoseNet 论文。

另外，请查看以下姿态预测的用例。

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li><a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a></li>
  <li><a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a></li>
  <li><a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a></li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">增强现实</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">交互动画</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Gait 分析</a></li>
</ul>

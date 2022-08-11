# 强化学习

与使用强化学习进行训练，并使用 TensorFlow Lite 进行部署的代理进行棋盘游戏对战。

## 开始

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

如果您是 TensorFlow Lite 新用户，并且使用的是 Android 平台，我们建议您研究以下可以帮助您入门的示例应用。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android">Android 示例</a>

如果您使用的不是 Android 平台，或者您已经熟悉 TensorFlow Lite API，则可以下载我们训练好的模型。

<a class="button button-primary" href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike_tf.tflite">下载模型</a>

## 工作原理

该模型是为一个游戏代理构建的，用于游玩一款名为 'Plane Strike' 的小型棋盘游戏。有关此游戏及其规则的快速介绍，请参阅此[自述文件](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android)。

在该应用的用户界面底层，我们构建了一个与人类玩家对战的代理。该代理是一个 3 层 MLP，将棋盘状态作为输入，并输出每个棋盘格（共 64 个）的预测分数。该模型使用策略梯度（加强）进行训练，您可以在[此处](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml)找到训练代码。在对代理进行训练后，我们将模型转换为 TFLite 并将其部署在 Android 应用中。

在 Android 应用的实际游戏中，当轮到代理采取行动时，代理会查看人类玩家的棋盘状态（底部的棋盘），其中包含有关之前成功和不成功的攻击（命中和未命中）的信息，并使用训练好的模型预测下一次攻击的位置，这样它就可以在人类玩家之前完成游戏。

## 性能基准

性能基准数值使用[此处所述](https://www.tensorflow.org/lite/performance/benchmarks)工具生成。

<table>
  <thead>
    <tr>
      <th>模型名称</th>
      <th>模型大小</th>
      <th>设备</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike.tflite">策略梯度</a>
</td>
    <td rowspan="2">       84 Kb</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0.01ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0.01ms*</td>
  </tr>
</table>

* 使用 1 个线程。

## 输入

该模型接受形状为 (1, 8, 8) 的三维 `float32` 张量作为输入。

## 输出

该模型返回形状为 (1,64) 的二维 `float32` 张量，作为 64 个可能的打击位置中每个位置的预测分数。

## 训练您自己的模型

您可以通过更改[训练代码](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml) 中的 `BOARD_SIZE` 参数来针对较大/较小的棋盘来训练您自己的模型。

# TensorBoard：可视化学习

您使用 TensorFlow 进行的计算——如训练一个大型深度神经网络，可能会是十分复杂和令人困惑的。为使 TensorFlow 程序更加易于理解、调试与优化，我们提供了一套称作 TensorBoard 的可视化工具。您可以使用 TensorBoard 来可视化您的 TensorFlow 流图，绘制图执行的量化指标，以及展示通过流图的其他数据，如图像。当 TensorBoard 被完全配置时，将如下图所示：

![MNIST TensorBoard](https://tensorflow.google.cn/images/mnist_tensorboard.png "MNIST TensorBoard")

<div class="video-wrapper">
  <iframe class="devsite-embedded-youtube-video" data-video-id="eBbEDRsCmv4"
          data-autohide="1" data-showinfo="0" frameborder="0" allowfullscreen>
  </iframe>
</div>

该 30 分钟的教程旨在助您开启简单的 TensorBoard 使用之旅。该视频假设您对 TensorFlow 有基本了解。

还有其他资源！[TensorBoard GitHub](https://github.com/tensorflow/tensorboard) 提供了更多关于使用 TensorBoard 内部各个仪表板的信息，包括提示、技巧以及调试信息。

## 设置

[安装 TensorFlow](../install/)。通过 pip 安装 TensorFlow 将同时自动安装 TensorBoard。 


## 序列化数据

TensorBoard 通过读取 TensorFlow 事件（events）文件进行工作。事件文件包含您在运行 TensorFlow 时可生成的摘要（summary）数据。以下是 TensorBoard 中摘要数据的一般生命周期。


首先，创建您要从中收集数据的 TensorFlow 图，并决定您要使用 `tf.summary` 操作进行标记的节点。

例如，假设您正在训练一个用于识别 MNIST 数字的卷积神经网络。您希望记录学习率随时间的变换情况以及目标函数的变化情况。分别在输出学习率和损失的节点上附加 `tf.summary.scalar` 操作来收集上述信息。然后给每个 `scalar_summary` 都赋予一个有意义的标签，如 `'learning rate'` 或 `'loss function'`。

也许您还想可视化来自特定层的激活分布,或者梯度与权重的分布。通过分别在梯度输出节点及保存权重的变量上附加 `tf.summary.histogram` 操作来收集这些信息。

TensorFlow 中的操作只有在您运行它们，或运行依赖这些节点输出的节点时才会执行。我们刚刚创建的摘要节点是您数据流图的外围节点：当前运行的操作都不依赖于它们。所以，为了生成摘要我们需要运行所有这些摘要节点。手动管理它们很繁琐，因此请使用 `tf.summary.merge_all` 将它们组合为一个生成摘要数据的单一操作。

然后，您可以运行合并摘要的操作，这将在给定步骤生成一个序列化的 `Summary` protobuf 对象，其中包含所有的摘要数据。最后，为了将这些摘要数据写入硬盘，请将该摘要的 protobuf 对象传递给 `tf.summary.FileWriter`。

`FileWriter` 的构造函数中带有一个日志目录——该目录非常重要，它是用于写入所有事件（event）的目录。`FileWriter`可选择性地在其构造函数中使用 `Graph`。若该构造函数接收到一个 `Graph` 对象，TensorBoard 将可视化您的流图及张量维度信息。这将能让您更好了解图中所流动的内容：请参阅[张量维度信息](../guide/graph_viz.md#tensor-shape-information)。

现在您已经修改了流图并且实例化了一个 `FileWriter`，就可以开始运行网络了！如果需要，您可以在每一步（step）中运行合并的摘要操作，并记录大量的训练数据。不过，这可能产生超出您需求的数据量。因此您可以考虑每 `n` 步运行一次合并的摘要操作。

下面的代码示例修改自[简单 MNIST 教程](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py)，我们在其中加入了一些摘要操作，并每十步运行一次。如果您运行该示例，并执行 `tensorboard --logdir=/tmp/tensorflow/mnist`，您将能够看到统计数据，如训练期间权重或准确率（accuracy）的变化情况。下方的是代码摘录，完整源码请参阅[此处](https://tensorflow.google.cn/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)


```python
def variable_summaries(var):
  """将大量摘要附加到 Tensor 上（为了 TensorBoard 可视化）"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """用于创建简单神经网络层的可复用代码。

  本段代码执行矩阵乘法，偏置加法以及使用 ReLU 进行非线性化操作。同时还设定了命名范围使结果图易于阅读，并且增加了一些摘要操作。
  """
  # 增加命名范围以确保图中网络层的逻辑分组。
  with tf.name_scope(layer_name):
    # 此变量将保存网络层的权重状态
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# 尚未应用 softmax 激活函数，请参阅下文。
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # 交叉熵的原始公式,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # 可能在数值上不稳定。
  #
  # 因此这里我们在 nn_layer 的原始 logit 输出上使用
  # tf.losses.sparse_softmax_cross_entropy，然后对 batch 做平均化。
  with tf.name_scope('total'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# 合并所有摘要并且将它们写入到
# /tmp/tensorflow/mnist/logs/mnist_with_summaries （默认情况下）
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
tf.global_variables_initializer().run()
```

初始化 `FileWriters` 后，我们需要在训练和测试模型时向 `FileWriters` 添加摘要。

```python
# 训练模型以及写入摘要。
# 每 10 步计算一次测试集的准确率，并且记录测试摘要。
# 其他的所有步在训练集上运行 train_step，以及添加训练摘要。

def feed_dict(train):
  """构造 Tensorflow feed_dict：将数据映射到 Tensor 占位符上。"""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # 记录摘要与测试集准确率
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # 记录训练集摘要并训练模型
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
```

现在，您已经准备好使用 TensorBoard 可视化这些数据。


## 启动 TensorBoard


为启动 TensorBoard，使用以下命令（或者 `python -m
tensorboard.main` ）

```bash
tensorboard --logdir=path/to/log-directory
```

其中 `logdir` 指向 `FileWriter` 序列化其数据的目录。若该 `logdir` 囊括含有来自不同运行的序列化数据的子目录，TensorBoard 将可视化来自所有运行的数据。TensorBoard运行后，将您的浏览器导航到 `localhost:6006` 以查看 TensorBoard。

当查看 TensorBoard 时，您会在右上角看到导航选项卡。每一个选项卡代表一组可被可视化的序列化数据。

有关如何使用 *graph* 选项卡可视化您的流图，请参阅[TensorBoard：流图可视化](graphs.md)。

更多 TensorBoard 使用信息，请参阅[TensorBoard GitHub](https://github.com/tensorflow/tensorboard)。

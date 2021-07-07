# 高效的 TensorFlow 2

TensorFlow 2.0 中进行了多处更改，旨在提高 TensorFlow 用户的效率。TensorFlow 2.0 移除了[冗余的 API](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)，使 API 更加一致（[统一 RNN](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md)、[统一优化器](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md)），并可通过 [Eager Execution](https://www.tensorflow.org/guide/eager) 与 Python 运行时更好地集成。

许多 [RFC](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr) 已经解释了 TensorFlow 2.0 中的变化。本文对 TensorFlow 2.0 中的开发进行了展望，并假设您对 TensorFlow 1.x 有一定的了解。

## 重大更改的简要总结

### API 清理

许多 API 在 TF 2.0 中[消失或发生移动](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)。一些重大更改包括移除 `tf.app`、`tf.flags` 和 `tf.logging`，转而采用现在开源的 [absl-py](https://github.com/abseil/abseil-py)，重新安置了 `tf.contrib` 中的项目，并清理了主要的 `tf.*` 命名空间，将不常用的函数移动到像 `tf.math` 这样的子包中。一些 API 已替换为 2.0 版本等效项：`tf.summary`、`tf.keras.metrics` 和 `tf.keras.optimizers`。自动应用这些重命名的最简单方法是使用 [v2 升级脚本](upgrade.md)。

### Eager Execution

TensorFlow 1.X 要求用户通过进行 `tf.*` API 调用手动将[抽象语法树](https://en.wikipedia.org/wiki/Abstract_syntax_tree)（计算图）拼接在一起。随后，它要求用户通过将一组输出张量和输入张量传递给 `session.run()` 调用来手动编译抽象语法树。TensorFlow 2.0 默认采用 Eager Execution（像 Python 通常做的那样），在 2.0 中，计算图和会话应当像实现细节一样。

Eager Execution 一个值得注意的地方是不再需要 `tf.control_dependencies()`，因为所有代码行均按顺序执行（在 `tf.function` 中，带副作用的代码按编写顺序执行）。

### 没有更多全局变量

TensorFlow 1.X 严重依赖隐式全局命名空间。当您调用 `tf.Variable()` 时，它会被放入默认计算图并保留在其中，即使您已失去指向它的 Python 变量的踪迹。随后，您可以恢复该 `tf.Variable()`，但前提是您知道它在创建时的名称。如果您无法控制变量的创建，这就很难做到。结果，各种机制激增，试图帮助用户再次找到他们的变量，并寻找框架来查找用户创建的变量：变量范围、全局集合、辅助方法（如 `tf.get_global_step()`、`tf.global_variables_initializer()`）、隐式计算所有可训练变量梯度的优化器，等等。TensorFlow 2.0 取消了所有这些机制（[变量 2.0 RFC](https://github.com/tensorflow/community/pull/11)），转而支持默认机制：跟踪您的变量！如果您失去了 `tf.Variable` 的踪迹，则会对它进行垃圾收集。

跟踪变量的要求为用户带来了一些额外工作，但借助 Keras 对象（见下文），可最大程度减少负担。

### 函数，而非会话

`session.run()` 调用几乎就像一个函数调用：指定输入和要调用的函数，然后返回一组输出。在 TensorFlow 2.0 中，您可以使用 `tf.function()` 装饰 Python 函数，将其标记为进行 JIT 编译，这样 TensorFlow 便可将其作为单个计算图运行（[函数 2.0 RFC](https://github.com/tensorflow/community/pull/20)）。这种机制可让 TensorFlow 2.0 获得计算图模式的所有好处：

- 性能：可以优化函数（节点剪枝、内核融合等）
- 可移植性：可以导出/重新导入函数 ([SavedModel 2.0 RFC](https://github.com/tensorflow/community/pull/34))，从而允许用户重用和共享模块化 TensorFlow 函数。

```python
# TensorFlow 1.X
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# TensorFlow 2.0
outputs = f(input)
```

凭借自由穿插 Python 和 TensorFlow 代码的能力，用户能够充分利用 Python 的表达性。但是，可移植的 TensorFlow 可在没有 Python 解释器的上下文（例如移动端、C++ 和 JavaScript）中执行。为帮助用户避免在添加 `@tf.function` 时重写代码，[AutoGraph](function.ipynb) 会将 Python 构造的一个子集转换成其 TensorFlow 等效项：

- `for`/`while` -&gt; `tf.while_loop`（支持 `break` 和 `continue`）
- `if` -&gt; `tf.cond`
- `for _ in dataset` -&gt; `dataset.reduce`

AutoGraph 支持控制流的任意嵌套，这样便有可能高效而简洁地实现许多复杂的 ML 程序，例如序列模型、强化学习、自定义训练循环等。

## 惯用 TensorFlow 2.0 的建议

### 将代码重构为更小的函数

TensorFlow 1.X 中常见的使用模式是“kitchen sink”策略，在这种策略中，所有可能计算的并集都已预先安排好，然后通过 `session.run()` 对所选张量进行评估。在 TensorFlow 2.0 中，用户应将代码重构为按需调用的更小函数。通常，没有必要使用 `tf.function` 来装饰这些小函数；只需使用 `tf.function` 装饰高级计算，例如，一个训练步骤或模型的前向传递。

### 使用 Keras 层和模型管理变量

Keras 模型和层提供了方便的 `variables` 和 `trainable_variables` 属性，它们以递归方式收集所有因变量。这样便可以轻松地在使用变量的地方对它们进行本地管理。

对比如下：

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

# You still have to manage w_i and b_i, and their shapes are defined far away from the code.
```

Keras 版本如下：

```python
# Each layer can be called, with a signature equivalent to linear(x)
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

Keras 层/模型继承自 `tf.train.Checkpointable` 并与 `@tf.function` 集成，这样便可以从 Keras 对象直接导出 SavedModel 或为其添加检查点。您不必使用 Keras 的 `.fit()` API 来利用这些集成。

下面是一个迁移学习示例，演示了 Keras 如何简化收集相关变量子集的工作。假设您正在训练一个拥有共享主干的多头模型：

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

# Train on primary dataset
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path1(x, training=True)
    loss = loss_fn_head1(prediction, y)
  # Simultaneously optimize trunk and head1 weights.
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# Fine-tune second head, reusing the trunk
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prediction = path2(x, training=True)
    loss = loss_fn_head2(prediction, y)
  # Only optimize head2 weights, not trunk weights
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# You can publish just the trunk computation for other people to reuse.
tf.saved_model.save(trunk, output_path)
```

### 结合 tf.data.Datasets 和 @tf.function

在迭代适合装入内存的训练数据时，可以随意使用常规 Python 迭代。除此之外，`tf.data.Dataset` 是从磁盘流式传输训练数据的最佳方式。数据集[可迭代（但不是迭代器）](https://docs.python.org/3/glossary.html#term-iterable)，就像其他 Python 迭代器在 Eager 模式下工作一样。您可以通过将代码包装在 `tf.function()` 中来充分利用数据集异步预提取/流式传输功能，此代码将 Python 迭代替换为使用 AutoGraph 的等效计算图运算。

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      prediction = model(x, training=True)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

如果您使用 Keras `.fit()` API，则不必担心数据集迭代。

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

### 借助 Python 控制流充分利用 AutoGraph

AutoGraph 提供了一种将依赖于数据的控制流转换为计算图模式等效项（如 `tf.cond` 和 `tf.while_loop`）的方法。

数据依赖控制流出现的一个常见位置是序列模型。`tf.keras.layers.RNN` 包装一个 RNN 单元，允许您以静态或动态方式展开递归。为了演示，您可以重新实现动态展开，如下所示：

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

有关 AutoGraph 功能的更详细概述，请参阅[指南](./function.ipynb)。

### 使用 tf.metrics 聚合数据并使用 tf.summary 记录这些数据

要记录摘要，请使用 `tf.summary.(scalar|histogram|...)` 并使用上下文管理器将其重定向到编写器。（如果您省略上下文管理器，则不会发生任何事情。）与 TF 1.x 不同，摘要直接发送给编写器；没有单独的“合并”运算，也没有单独的 `add_summary()` 调用，这意味着必须在调用点提供 `step` 值。

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

要在将数据记录为摘要之前对其进行聚合，请使用 `tf.metrics`。指标是有状态的：它们累积值并在您调用 `.result()` 时返回累积结果。可以使用 `.reset_states()` 清除累积值。

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  loss = loss_fn(model(test_x, training=False), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

通过将 TensorBoard 指向摘要日志目录来显示生成的摘要：

```
tensorboard --logdir /tmp/summaries
```

### 调试时使用 tf.config.experimental_run_functions_eagerly()

在 TensorFlow 2.0 中，Eager Execution 使您可以分步运行代码以检查形状、数据类型和值。某些 API（如 `tf.function`、`tf.keras` 等）设计为使用计算图执行来提高性能和可移植性。调试时，请使用 `tf.config.experimental_run_functions_eagerly(True)`，以便在此代码内使用 Eager Execution。

例如：

```python
@tf.function
def f(x):
  if x > 0:
    import pdb
    pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)
f(tf.constant(1))
```

```
>>> f()
-> x = x + 1
(Pdb) l
  6  	@tf.function
  7  	def f(x):
  8  	  if x > 0:
  9  	    import pdb
 10  	    pdb.set_trace()
 11  ->	    x = x + 1
 12  	  return x
 13
 14  	tf.config.experimental_run_functions_eagerly(True)
 15  	f(tf.constant(1))
[EOF]
```

这也可以在 Keras 模型和其他支持 Eager Execution 的 API 中使用：

```
class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      import pdb
      pdb.set_trace()
      return input_data // 2


tf.config.experimental_run_functions_eagerly(True)
model = CustomModel()
model(tf.constant([-2, -4]))
```

```
>>> call()
-> return input_data // 2
(Pdb) l
 10  	    if tf.reduce_mean(input_data) > 0:
 11  	      return input_data
 12  	    else:
 13  	      import pdb
 14  	      pdb.set_trace()
 15  ->	      return input_data // 2
 16
 17
 18  	tf.config.experimental_run_functions_eagerly(True)
 19  	model = CustomModel()
 20  	model(tf.constant([-2, -4]))
```

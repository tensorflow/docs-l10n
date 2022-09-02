# 已知问题

使用 XLA 进行编译可以大幅提高程序的性能，但 TensorFlow 互操作具有许多已知问题。

## 不同设备上的 `tf.Variable`

*错误消息*：`INVALID_ARGUMENT: Trying to access resource <Variable> (defined @ <Loc>) located in device CPU:0 from device GPU:0`

XLA 聚类只在一台设备上运行，它无法读取或写入位于不同设备上的 `tf.Variable`。通常，此错误消息表明变量一开始就没有放置在正确的设备上。错误消息应当精确地指定有问题的变量的位置。

注：`int32` 类型的 `tf.Variable` 始终置于主机上，而不能置于 GPU 上。作为一种变通方法，可以使用 `int64`。

## 不支持 TensorArray TF/XLA 相互转换

*错误消息*：`Support for TensorList crossing the XLA/TF boundary is not implemented`。

XLA 支持 `tf.TensorArray`。但是，尚未实现 TF 与 XLA 表示之间的*相互转换*。如果在已编译的块内使用 `TensorArray`，而在外部采用导数，通常会出现此错误。

解决方法：编译采用导数的最外层作用域。

## TensorFlow while 循环需要存在界限（或禁用反向传播）

*错误消息*：`XLA compilation requires a fixed tensor list size. Set the max number of elements. This could also happen if you're using a TensorArray in a while loop that does not have its maximum_iteration set, you can fix this by setting maximum_iteration to a suitable value`。

使用 `tf.while_loop` 创建的 TF while [循环](https://www.tensorflow.org/api_docs/python/tf/while_loop)通过在 `TensorArray` 中累积所有中间结果来支持反向传播，但 XLA 仅支持有界限的 `TensorArray`。

*解决方法*：所有编译的 while 循环都需要将 `maximum_iterations` 参数设置为编译时已知的常量值，或者使用 `back_prop=False` 禁用反向传播。

## 不支持动态 `tf.TensorArray`

对 `tf.TensorArray(..., dynamic_size=True)` 的写入无法用 XLA 编译，因为当数组超出原始边界时，此类写入需要未知数量的重新分配。

解决方法：为您的数组提供一个静态已知的界限。

## 随机数生成忽略 TF 种子

XLA 目前会忽略随机运算的 TF 种子。这会影响有状态 TF 随机运算，例如 `tf.random.normal` 或 `tf.nn.dropout`。XLA 的行为就像在同一进程的每次运行时都使用新的唯一种子为编译提供种子（该进程的第一次运行将始终产生相同的结果）。

*变通方法*：直接使用[推荐的 RNG](https://www.tensorflow.org/guide/random_numbers#stateless_rngs)，例如 `tf.random.stateless_uniform` 或 `tf.random.Generator`。

## 不支持作为归纳变量函数的“必须是常量”输入

*错误消息*：`XLA compilation requires that operator arguments that represent shapes or dimensions be evaluated to concrete values at compile time. This error means that a shape or dimension argument could not be evaluated at compile time, usually because the value of the argument depends on a parameter to the computation, on a variable, or on a stateful operation such as a random number generator`。

XLA 要求在编译时知道某些值，例如归约运算的归约轴或转置维度。考虑这种情况，归约轴被定义为 `tf.range` 的归纳变量的函数：如果不展开整个循环，就不可能静态地解决它，这可能不是用户所希望的。

*解决方法*：展开循环，例如通过将 `tf.range` 转换为 Python `range`。

注：上述错误消息并非此问题独有，也可能由于其他限制或错误而出现。

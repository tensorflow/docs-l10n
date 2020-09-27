# XLA 中的别名

本文档介绍 XLA 的别名 API：构建 XLA 程序时，您可以在输入和输出缓冲区之间指定所需的别名。

## 在编译时定义别名

例如，考虑一个简单的 HLO 模块，仅对其输入增加 `1`：

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

此模块将分配两个 4 字节缓冲区：一个用于输入 `%p`，一个用于输出 `%out`。

但是，通常需要执行就地更新（例如，如果在生成表达式的前端中，输入变量在计算之后便不再有效，如增量 `p++`）。

为了有效地执行此类更新，您可以指定输入别名：

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

该格式可指定将整个输出（由 `{}` 标记）别名化为输入参数 `0`。

请参阅 [`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) API 以编程方式指定别名。

## 在运行时定义别名

上一步骤中定义的别名在*编译*期间指定。在执行期间，您可以选择是否使用 [`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h) API 来实际捐献缓冲区。

程序的输入缓冲区包装在 [`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h) 内，它转而包含 `MaybeOwningDeviceMemory` 树。如果内存被指定为*拥有*（缓冲区的所有权被传递给 XLA 运行时），则实际上已捐献缓冲区，并按照编译时别名 API 的请求执行了就地更新。

但是，如果运行时*不*捐献在编译时定义别名的缓冲区，则会启动*复制保护*：将分配一个额外的缓冲区 `O`，并将要定义别名的输入缓冲区 `P` 中的内容复制到 `O` 中（这样一来，程序就会按照运行时已捐献缓冲区 `O` 的情况有效地执行）。

## 前端互操作性

### TF/XLA

在使用 XLA 编译的 TensorFlow 程序集群中，所有资源变量更新都在编译时定义别名（在运行时定义别名取决于是否有其他任何内容引用了资源变量张量）。

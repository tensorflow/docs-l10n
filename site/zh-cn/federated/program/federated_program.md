# 联合程序

本文档适用于对联合程序概念的简要概览感兴趣的任何人。它假定您了解 TensorFlow Federated，尤其是其类型系统。

有关联合程序的更多信息，请参阅：

- [API 文档](https://www.tensorflow.org/federated/api_docs/python/tff/program)
- [示例](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
- [开发者指南](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[目录]

## 什么是联合程序？

**联合程序**是在联合环境中执行计算和其他处理逻辑的程序。

更具体地说，是一种**联合的程序**：

- 执行[计算](#computations)
- 使用[程序逻辑](#program-logic)
- 使用[平台特定的组件](#platform-specific-components)
- [与平台无关的组件](#platform-agnostic-components)
- [程序](#program)设置的给定[参数](#parameters)
- 和[客户](#customer)设置的[参数](#parameters)
- 当[客户](#customer)运行[程序](#program)时
- 并且可以[具体化](#materialize)[平台存储空间](#platform storage)中的数据以：
    - 在 Python 逻辑中使用
    - 实现[容错](#fault tolerance)
- 以及将数据[发布](#release)到[客户存储空间](#customer storage)

通过定义这些[概念](#concepts)和抽象，可以描述联合程序各[组件](#components)之间的关系，并允许这些组件由不同的[角色](#roles)拥有和创作。这种解耦允许开发者使用与其他联合程序共享的组件来创作联合程序，通常这意味着在许多不同的平台上执行相同的程序逻辑。

TFF 的联合程序库 ([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)) 定义了创建联合程序所需的抽象，并且提供了[与平台无关的组件](#platform-agnostic-components)。

## 组件

TFF 联合程序库的**组件**经过专门设计，可以由不同的[角色](#roles)拥有和创作。

注：这是组件的简要概览，请参阅 [tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program) 了解特定 API 的文档。

### 程序

**程序**是一个 Python 二进制文件，它用于：

1. 定义[参数](#parameters)（例如标志）
2. 构建[平台特定的组件](#platform-specific-components)和[与平台无关的组件](#platform-agnostic-components)
3. 在联合上下文中使用[程序逻辑](#program_logic)执行[计算](#computations)

例如：

```python
# Parameters set by the customer.
flags.DEFINE_string('output_dir', None, 'The output path.')

def main() -> None:

  # Parameters set by the program.
  total_rounds = 10
  num_clients = 3

  # Construct the platform-specific components.
  context = tff.program.NativeFederatedContext(...)
  data_source = tff.program.DatasetDataSource(...)

  # Construct the platform-agnostic components.
  summary_dir = os.path.join(FLAGS.output_dir, 'summary')
  metrics_manager = tff.program.GroupingReleaseManager([
      tff.program.LoggingReleaseManager(),
      tff.program.TensorBoardReleaseManager(summary_dir),
  ])
  program_state_dir = os.path.join(..., 'program_state')
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  # Define the computations.
  initialize = ...
  train = ...

  # Execute the computations using program logic.
  tff.framework.set_default_context(context)
  asyncio.run(
      train_federated_model(
          initialize=initialize,
          train=train,
          data_source=data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          metrics_manager=metrics_manager,
          program_state_manager=program_state_manager,
      )
  )
```

### 参数

**参数**是[程序](#program)的输入，这些输入可由[客户](#customer)设置（如果它们作为标志公开），或者它们可由程序设置。在上面的示例中，`output_dir` 是[客户](#customer)设置的参数，`total_rounds` 和 `num_clients` 是程序设置的参数。

### 平台特定的组件

**平台特定的组件**是由实现 TFF 联合程序库定义的抽象接口的[平台](#platform)提供的组件。

### 与平台无关的组件

**与平台无关的组件**是由实现 TFF 联合程序库定义的抽象接口的[库](#library)（例如 TFF）提供的组件。

### 计算

**计算**是抽象接口 [`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation) 的实现。

例如，在 TFF 平台中，您可以使用 [`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation) 或 [`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation) 装饰器来创建 [`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation)：

如需了解详情，请参阅[计算的生命](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md)。

### 程序逻辑

**程序逻辑**是一个 Python 函数，它可用作输入：

- [客户](#customer)和[程序](#program)设置的[参数](#parameters)
- [平台特定的组件](#platform-specific-components)
- [与平台无关的组件](#platform-agnostic-components)
- [计算](#computations)

并执行一些运算，通常包括：

- 执行[计算](#computations)
- 执行 Python 逻辑
- [具体化](#materialize)[平台存储](#platform storage)中的数据，以便：
    - 在 Python 逻辑中使用
    - 实现[容错](#fault tolerance)

并可能产生一些输出，通常包括：

- 将数据作为[指标](#release)[发布](#metrics)到[客户存储空间customer storage](#customer storage)

例如：

```python
async def program_logic(
    initialize: tff.Computation,
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    metrics_manager: tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int
    ],
) -> None:
  state = initialize()
  start_round = 1

  data_iterator = data_source.iterator()
  for round_number in range(1, total_rounds + 1):
    train_data = data_iterator.select(num_clients)
    state, metrics = train(state, train_data)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

## 角色

讨论联合程序时，有三个**角色**需要定义：[客户](#customer)、[平台](#platform)和[库](#library)。每个角色都拥有并创作一些用于创建联合程序的[组件](#components)。但是，单个实体或组可以担当多个角色。

### 客户

**客户**通常：

- 拥有[客户存储空间](#customer-storage)
- 启动[程序](#program)

但可能：

- 创作[程序](#program)
- 实现[平台](#platform)的任何功能

### 平台

**平台**通常：

- 拥有[平台存储空间](#platform-storage)
- 创作[平台特定的组件](#platform-specific-components)

但可能：

- 创作[程序](#program)
- 实现[库](#library)的任何功能

### 库

**库**通常：

- 创作[与平台无关的组件](#platform-agnostic-components)
- 创作[计算](#computations)
- 创作[程序逻辑](#program-logic)

## 概念

在讨论联合程序时，有一些**概念**需要定义。

### 客户存储

**客户存储**是[客户](#customer)具有读写访问权限且[平台](#platform)具有写入访问权限的存储。

### 平台存储空间

**平台存储空间**是只有[平台](#platform)具有读写访问权限的存储空间。

### 发布

**发布**一个值可使该值供[客户存储空间](#customer-storage)使用（例如，将值发布到信息中心、记录值或将值写入磁盘）。

### 具体化

**具体化**一个值引用会使引用的值可供[程序](#program)使用。通常需要具体化值引用来[发布](#release)该值或使[程序逻辑](#program-logic)具有[容错](#fault-tolerance)能力。

### 容错

**容错**是[程序逻辑](#program-logic)在执行计算时从错误中恢复的能力。例如，如果成功训练了 100 个轮次中的前 90 个轮次，随后遇到错误，那么程序逻辑能否从第 91 个轮次开始恢复训练，或者是否需要在第 1 个轮次重新开始训练？

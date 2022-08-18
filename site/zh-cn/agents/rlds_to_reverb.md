# TF-Agents 中的 RLDS 到 Reverb 实用工具

[RLDS](https://github.com/google-research/rlds) 到 [Reverb](https://github.com/deepmind/reverb) 实用工具是 TF Agents 中的一个工具，用于从 RLDS 读取片段，将片段转换为轨迹，并将轨迹推送到 Reverb。

### RLDS 数据集

RLDS（强化学习数据集）是一个工具生态系统，用于在包括强化学习 (RL)、从演示中学习、离线强化学习或模仿学习在内的顺序决策环境中存储、检索和操作片段数据。

每个步骤都有以下字段（有时还包括步骤元数据的额外字段）。作为示例，我们使用 D4RL 数据集 [half-cheetah/v0-expert](https://www.tensorflow.org/datasets/catalog/d4rl_mujoco_halfcheetah#d4rl_mujoco_halfcheetahv0-expert_default_config) 中的规范

- **'action'**: `TensorSpec(shape = (6,), dtype = tf.float32, name = None)`

- **'discount'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)`

- **'is_first'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_last'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'is_terminal'**: `TensorSpec(shape = (), dtype = tf.bool, name = None)`

- **'observation'**: `TensorSpec(shape = (17,), dtype = tf.float32, name = None)`

- **'reward'**: `TensorSpec(shape = (), dtype = tf.float32, name = None)}, TensorShape([]))`

## RLDS 到 TF-Agents 实用工具的 API

### 从数据集创建轨迹规范

创建用于初始化 Reverb 服务器和 Reverb 回放缓冲区的数据规范。

```
def create_trajectory_data_spec(rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
```

为可以使用作为输入提供的 `rlds_data` 创建的相应轨迹数据集创建数据规范。此数据规范对于初始化 Reverb 服务器和 Reverb 回放缓冲区是必需的。

**参数**：

- `rlds_data` ：RLDS 数据集是 RLDS 片段的 `tf.data.Dataset` ，其中每个片段包含 RLDS 步骤的 `tf.data.Dataset` 以及可选的片段元数据。 RLDS 步骤是一个张量字典，包含 `is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal` 和 `discount`（有时还包含步骤元数据）。

**返回**：

- 可用于创建轨迹数据集的轨迹规范，其中 `rlds_data` 作为输入提供。

**引发**：

- `ValueError` ：如果 `rlds_data` 中不存在 RLDS 步骤。

### 将 RLDS 数据转换为 TF Agents 轨迹

将 RLDS 数据转换为轨迹数据集。目前，我们仅支持转换为两步轨迹。

```
def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset,
    policy_info_fn: _PolicyFnType = None) -> tf.data.Dataset:
```

将提供的 `rlds_data` 转换为 TF Agents 轨迹数据集，方法是将其展平并转换为批次，然后转换为相邻 RLDS 步骤的重叠对的元组。

在末尾用类型为 `first` 的步骤填充 RLDS 数据，以确保使用最后一个片段的最后一个步骤创建的轨迹具有有效的下一步类型。

**参数**：

- `rlds_data`：RLDS 数据集是 RLDS 片段的 `tf.data.Dataset`，其中每个片段包含 RLDS 步骤的 `tf.data.Dataset`（以及可选的片段元数据）。RLDS 步骤是一个张量字典，包含 `is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal` 和 `discount`（以及可选的步骤元数据）。
- `policy_info_fn`：一个可选函数，用于创建一些在生成 TF-Agents 轨迹时使用的 policy.info。

**返回**：

- 类型为 `tf.data.Dataset` 的数据集，其元素是与 `rlds_data` 中提供的 RLDS 步骤相对应的 TF Agents 轨迹。

**引发**：

- `ValueError` ：如果 `rlds_data` 中不存在 RLDS 步骤。

- `InvalidArgumentError` ：如果提供的 RLDS 数据集包含以下片段：

    - 错误结束，即没有在最后一步结束。
    - 错误终止，即终止步骤不是最后一步。
    - 错误开始，即第一步未跟在最后一步之后。请注意，最后一个片段的最后一步在函数中处理，用户不需要确保最后一个片段的最后一步之后是第一步。

### 将 RLDS 数据推送到 Reverb

将 RLDS 数据作为 TF Agents 轨迹推送到 Reverb 服务器。Reverb 观察器必须在调用接口之前实例化并作为参数提供。

```
def push_rlds_to_reverb(rlds_data: tf.data.Dataset, reverb_observer: Union[
    reverb_utils.ReverbAddEpisodeObserver,
    reverb_utils.ReverbAddTrajectoryObserver],
    policy_info_fn: _PolicyFnType = None) -> int:
```

将提供的 `rlds_data` 转换为 TF Agents 轨迹后，使用 `reverb_observer` 将其推送到 Reverb 服务器。

请注意，用于初始化回放缓冲区和用于创建 `reverb_observer` 的 Reverb 服务器的数据规范必须与 `rlds_data` 的数据规范匹配。

**参数**：

- `rlds_data`：RLDS 数据集是 RLDS 片段的 `tf.data.Dataset`，其中每个片段包含 RLDS 步骤的 `tf.data.Dataset`（以及可选的片段元数据）。RLDS 步骤是一个张量字典，包含 `is_first`、`is_last`、`observation`、`action`、`reward`、`is_terminal` 和 `discount`（以及可选的步骤元数据）。
- `reverb_observer`：用于将轨迹数据写入 Reverb 的 Reverb 观察器。
- `policy_info_fn`：一个可选函数，用于创建一些在生成 TF-Agents 轨迹时使用的 policy.info。

**返回**：

- 表示成功推送到 RLDS 的轨迹数的 `int`。

**引发**：

- `ValueError` ：如果 `rlds_data` 中不存在 RLDS 步骤。

- `ValueError`：如果用于初始化回放缓冲区和用于创建 `reverb_observer` 的 Reverb 服务器的数据规范与可以使用 `rlds_data` 创建的轨迹数据集的数据规范不匹配。

- `InvalidArgumentError`：如果提供的 RLDS 数据集具有以下片段：

    - 错误结束，即没有在最后一步结束。
    - 错误终止，即终止步骤不是最后一步。
    - 错误开始，即第一步未跟在最后一步之后。请注意，最后一个片段的最后一步在函数中处理，用户不需要确保最后一个片段的最后一步之后是第一步。

## RLDS 步骤如何映射到 TF Agents 轨迹

以下序列是时间步骤 t、t+1 和 t+2 的 RLDS 步骤。每个步骤都包含一个观测值 (o)、操作 (a)、奖励 (r) 和折扣 (d)。同一步骤的元素分组在括号中。

```
(o_t, a_t, r_t, d_t), (o_t+1, a_t+1, r_t+1, d_t+1), (o_t+2, a_t+2, r_t+2, d_t+2)
```

在 RLDS 中，

- `o_t` 对应于时间 t 的观测值

- `a_t` 对应于时间 t 的操作

- `r_t` 对应于在观测值 `o_t` 处执行操作而获得的奖励

- `d_t` 对应于应用于奖励 `r_t` 的折扣

```
Step 1 =  o_0, a_0, r_0, d_0, is_first = true, is_last = false, is_terminal = false
```

```
Step 2 =  o_1, a_1, r_1,d_1, is_first = False, is_last = false, is_terminal = false
```

…

```
Step n =  o_t, a_t, r_t, d_t, is_first = False, is_last = false, is_terminal = false
```

```
Step n+1 =   o_t+1, a_t+1, r_t+1, d_t+1, is_first = False, is_last = true, is_terminal = false
```

当 `is_terminal = True` 时，观测值对应于最终状态，因此奖励、折扣和操作没有意义。根据环境的不同，最终的观测值也可能没有意义。

如果片段以 `is_terminal = False` 的步骤结束，则意味着该片段已被截断。在这种情况下，根据环境，操作、奖励和折扣也可能为空。

![RLDS 步骤到 TF-Agents 轨迹](images/rlds/rlds_step_to_trajectory.png)

### 转换过程

#### 展平数据集

RLDS 数据集是由片段组成的数据集，而片段又是 RLDS 步骤的数据集。它首先被展平为步骤数据集。

![展平 RLDS](images/rlds/flatten_rlds.png)

#### 创建相邻步骤的重叠对

然后对展平的 RLDS 数据集进行批处理并转换为相邻 RLDS 步骤重叠对的数据集。

![RLDS 到重叠对](images/rlds/rlds_to_pairs.png)

#### 转换为 TF-Agents 轨迹

然后将数据集转换为 TF-Agents 轨迹。

![RLDS 对到 TF-Agents 轨迹](images/rlds/pairs_to_trajectories.png)

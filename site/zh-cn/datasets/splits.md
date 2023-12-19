# 拆分和切片

所有 TFDS 数据集都提供了不同的数据拆分（例如 `'train'`、`'test'`），可以在[目录](https://www.tensorflow.org/datasets/catalog/overview)中进行探索。除了 `all`（它是一个保留术语，表示所有拆分的并集，见下文），任何字母字符串都可以用作拆分名称。

除了“官方”数据集拆分之外，TFDS 还允许选择拆分的切片和各种组合。

## Slicing API

切片指令通过 `split=` kwarg 在 `tfds.load` 或`tfds.DatasetBuilder.as_dataset` 中指定。

```python
ds = tfds.load('my_dataset', split='train[:75%]')
```

```python
builder = tfds.builder('my_dataset')
ds = builder.as_dataset(split='test+train[:75%]')
```

拆分可以是：

- **普通拆分名称**（字符串，例如 `'train'`、`'test'`…）：所选拆分中的所有样本。
- **切片**：切片与 [python 切片表示法](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)具有相同的语义。切片可以是：
    - **绝对**（`'train[123:450]'`、`train[:4000]`）：（请参阅下方注释了解有关读取顺序的注意事项）
    - **百分比** (`'train[:75%]'`、`'train[25%:75%]'`)：将完整数据分成均匀的切片。如果数据不能被整除，则某些百分比可能包含附加样本。支持小数百分比。
    - **分片**（`train[:4shard]`、`train[4shard]`）：选择请求的分片中的所有样本。（请参阅 `info.splits['train'].num_shards` 以获取拆分的分片数）
- **拆分联合**（`'train+test'`、`'train[:25%]+test'`）：拆分将交错在一起。
- **完整数据集** (`'all'`)：`'all'` 是一个与所有拆分的联合对应的特殊拆分名称（相当于 `'train+test+...'`）。
- **拆分列表** (`['train', 'test']`)：分别返回多个 `tf.data.Dataset`：

```python
# Returns both train and test split separately
train_ds, test_ds = tfds.load('mnist', split=['train', 'test[:50%]'])
```

注：由于分片是[交错的](https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#interleave)，不能保证子拆分之间的顺序一致。换句话说，先后读取 `test[0:100]` 和 `test[100:200]` 可能会产生与读取 `test[:200]` 时顺序不同的样本。请参阅[确定性指南](https://www.tensorflow.org/datasets/determinism#determinism_when_reading)了解 TFDS 读取样本的顺序。

## `tfds.even_splits` 和多主机训练

`tfds.even_splits` 可以生成大小相同的非重叠子拆分列表。

```python
# Divide the dataset into 3 even parts, each containing 1/3 of the data
split0, split1, split2 = tfds.even_splits('train', n=3)

ds = tfds.load('my_dataset', split=split2)
```

这在分布式设置中训练时特别有用，其中每个主机都应接收原始数据的一个切片。

借助 `Jax`，使用 `tfds.split_for_jax_process` 甚至可以进一步简化：

```python
split = tfds.split_for_jax_process('train', drop_remainder=True)
ds = tfds.load('my_dataset', split=split)
```

`tfds.split_for_jax_process` 是一个简单的别名：

```python
# The current `process_index` loads only `1 / process_count` of the data.
splits = tfds.even_splits('train', n=jax.process_count(), drop_remainder=True)
split = splits[jax.process_index()]
```

`tfds.even_splits`、`tfds.split_for_jax_process` 接受任何拆分值作为输入（例如 `'train[75%:]+test'`）

## 切片和元数据

可以使用[数据集信息](https://www.tensorflow.org/datasets/overview#access_the_dataset_metadata)获取有关拆分/子拆分（`num_examples`、`file_instructions`…）的附加信息：

```python
builder = tfds.builder('my_dataset')
builder.info.splits['train'].num_examples  # 10_000
builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
builder.info.splits.keys()  # ['train', 'test']
```

## 交叉验证

使用字符串 API 的 10 折交叉验证示例：

```python
vals_ds = tfds.load('mnist', split=[
    f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
])
trains_ds = tfds.load('mnist', split=[
    f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
])
```

每个验证数据集将是 10%：`[0%:10%]`、`[10%:20%]`、…、`[90%:100%]`。并且每个训练数据集都将是互补的 90%：`[10%:100%]`（相应的验证集为 `[0%:10%]`）、`[0%:10%]

- [20%:100%]（验证集为 `[10%:20%]``）…

## `tfds.core.ReadInstruction` 和舍入

可以将拆分作为 `tfds.core.ReadInstruction` 而不是 `str` 传递：

例如，`split = 'train[50%:75%] + test'` 等价于：

```python
split = (
    tfds.core.ReadInstruction(
        'train',
        from_=50,
        to=75,
        unit='%',
    )
    + tfds.core.ReadInstruction('test')
)
ds = tfds.load('my_dataset', split=split)
```

`unit` 可以是：

- `abs`：绝对切片
- `%`：百分比切片
- `shard`：分片切片

`tfds.ReadInstruction` 也有一个舍入参数。如果数据集中的样本数量不能被整除：

- `rounding='closest'`（默认）：剩余的样本会在百分比中分布，因此某些百分比可能包含附加样本。
- `rounding='pct1_dropremainder'`：剩余的样本会被丢弃，但这可保证所有百分比均包含完全相同数量的样本（例如：`len(5%) == 5 * len(1%)`）。

### 重现性和确定性

在生成期间，对于给定的数据集版本，TFDS 可保证样本在磁盘上确定性地打乱顺序。因此，生成数据集两次（在 2 台不同的计算机上）不会改变样本顺序。

同样，subsplit API 将始终选择相同的样本 `set`，无论平台、架构等如何。这意味着 `set('train[:20%]') == set('train[:10%]') + set('train[10%:20%]')`。

但是，读取样本的顺序可能**不**是确定性的。这取决于其他参数（例如，无论 `shuffle_files=True` 与否）。

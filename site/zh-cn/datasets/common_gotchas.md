# 常见实现陷阱

本页介绍了实现新数据集时的常见实现问题。

## 应避免使用旧版 `SplitGenerator`

旧的 `tfds.core.SplitGenerator` API 已弃用。

```python
def _split_generator(...):
  return [
      tfds.core.SplitGenerator(name='train', gen_kwargs={'path': train_path}),
      tfds.core.SplitGenerator(name='test', gen_kwargs={'path': test_path}),
  ]
```

应替换为：

```python
def _split_generator(...):
  return {
      'train': self._generate_examples(path=train_path),
      'test': self._generate_examples(path=test_path),
  }
```

**理由**：新的 API 不那么冗长而且更加明确。旧 API 将在未来版本中移除。

## 新数据集应位于独立的文件夹中

向 `tensorflow_datasets/` 仓库中添加数据集时，请确保遵循数据集即文件夹结构（所有校验和、虚拟数据、实现代码均位于独立的文件夹中）。

- 旧数据集（不良）：`<category>/<ds_name>.py`
- 新数据集（良好）：`<category>/<ds_name>/<ds_name>.py`

使用 [TFDS CLI](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset)（`tfds new`，或对于 Google 员工为 `gtfds new`）生成模板。

**理由**：旧结构需要校验和、假数据的绝对路径，并且在许多地方分发数据集文件。这样便导致在 TFDS 仓库之外实现数据集变得更加困难。为了保持一致性，现在应当在任何地方使用新结构。

## 描述列表应格式化为 markdown

`DatasetInfo.description` `str` 被格式化为 markdown。Markdown 列表在第一项之前需要一个空行：

```python
_DESCRIPTION = """
Some text.
                      # << Empty line here !!!
1. Item 1
2. Item 1
3. Item 1
                      # << Empty line here !!!
Some other text.
"""
```

**理由**：格式错误的描述会在我们的目录文档中创建可视化工件。如果没有空行，上面的文本将呈现为：

Some text. 1. Item 1 2. Item 1 3. Item 1 Some other text

## 忘记了 ClassLabel 名称

使用 `tfds.features.ClassLabel` 时，尝试使用 `names=` 或 `names_file=`（而不是 `num_classes=10`）提供人类可读的标签 `str`。

```python
features = {
    'label': tfds.features.ClassLabel(names=['dog', 'cat', ...]),
}
```

**理由**：许多地方都使用了人类可读的标签：

- 允许直接在 `_generate_examples` 中产生 `str`：`yield {'label': 'dog'}`
- 在 `info.features['label'].names` 等用户中公开（转换方法 `.str2int('dog')`... 也可用）
- 在[可视化实用工具](https://www.tensorflow.org/datasets/overview#tfdsas_dataframe) `tfds.show_examples`、`tfds.as_dataframe` 中使用

## 忘记了图片形状

使用 `tfds.features.Image`、`tfds.features.Video` 时，如果图片具有静态形状，则应明确指定：

```python
features = {
    'image': tfds.features.Image(shape=(256, 256, 3)),
}
```

**理由**：它允许静态形状推断（例如 `ds.element_spec['image'].shape`），这是批处理所必需的（要批处理未知形状的图片，需要先调整它们的大小）。

## 首选更具体的类型而不是 `tfds.features.Tensor`

如果可能，首选更具体的类型 `tfds.features.ClassLabel`、`tfds.features.BBoxFeatures`，而不是通用的 `tfds.features.Tensor`。

**理由**：除了在语义上更准确之外，具体特征还为用户提供了额外的元数据并可被工具检测到。

## 全局空间中的延迟导入

不应从全局空间调用延迟导入。例如，下面的代码是错误的：

```python
tfds.lazy_imports.apache_beam # << Error: Import beam in the global scope

def f() -> beam.Map:
  ...
```

**理由**：在全局范围内使用延迟导入会为所有 tf​​ds 用户导入模块，这违背了延迟导入的目的。

## 动态计算训练/测试拆分

如果数据集不提供官方拆分，TFDS 也不应提供。应避免以下情况：

```python
_TRAIN_TEST_RATIO = 0.7

def _split_generator():
  ids = list(range(num_examples))
  np.random.RandomState(seed).shuffle(ids)

  # Split train/test
  train_ids = ids[_TRAIN_TEST_RATIO * num_examples:]
  test_ids = ids[:_TRAIN_TEST_RATIO * num_examples]
  return {
      'train': self._generate_examples(train_ids),
      'test': self._generate_examples(test_ids),
  }
```

**理由**：TFDS 尝试提供尽可能接近原始数据的数据集。应当使用 [sub-split API](https://www.tensorflow.org/datasets/splits) 来让用户动态创建他们想要的子拆分：

```python
ds_train, ds_test = tfds.load(..., split=['train[:80%]', 'train[80%:]'])
```

## Python 风格指南

### 首选使用 pathlib API

最好使用 [pathlib API](https://docs.python.org/3/library/pathlib.html) 而不是 `tf.io.gfile` API。所有 `dl_manager` 方法都返回与 GCS、S3 等兼容的类似路径库的对象

```python
path = dl_manager.download_and_extract('http://some-website/my_data.zip')

json_path = path / 'data/file.json'

json.loads(json_path.read_text())
```

**理由**：pathlib API 是一个移除了样板的面向现代对象的文件 API。使用 `.read_text()` / `.read_bytes()` 还可以保证文件正确关闭。

### 如果方法不使用 `self`，它应当是一个函数

如果一个类方法不使用 `self`，它应当是一个简单的函数（定义在类之外）。

**理由**：它向读者明确表明该函数没有副作用，也没有隐藏的输入/输出：

```python
x = f(y)  # Clear inputs/outputs

x = self.f(y)  # Does f depend on additional hidden variables ? Is it stateful ?
```

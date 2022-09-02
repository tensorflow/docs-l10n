# 为 TFDS 仓库做出贡献

感谢您对我们库的关注！我们很高兴拥有这样一个积极主动的社区。

## 开始

- 如果您是 TFDS 的新用户，最简单的入门方式是实现我们的[请求数据集](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22dataset+request%22+sort%3Areactions-%2B1-desc)之一，重点关注呼声最高的数据集。请[按照我们的指南](https://www.tensorflow.org/datasets/add_dataset)获取说明。
- 议题、功能请求、错误等比添加新数据集的影响要大得多，因为它们会使整个 TFDS 社区受益。请参阅[潜在贡献列表](https://github.com/tensorflow/datasets/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+-label%3A%22dataset+request%22+)。从标有 [contribution-welcome](https://github.com/tensorflow/datasets/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) 的议题开始，这些都是容易上手的独立小议题。
- 不要犹豫接管已经分配但有一段时间没有更新的错误。
- 无需将议题分配给您。当您开始处理议题时，只需对它发表评论 :)
- 如果您对某个议题感兴趣但不知道如何开始，请随时寻求帮助。如果您需要早期反馈，请发送草案 PR。
- 为避免不必要的重复工作，请检查[待处理的拉取请求](https://github.com/tensorflow/datasets/pulls)列表，并对您正在处理的议题发表评论。

## 设置

### 克隆仓库

首先，克隆或下载 [Tensorflow Datasets](https://github.com/tensorflow/datasets) 仓库并在本地安装仓库。

```sh
git clone https://github.com/tensorflow/datasets.git
cd datasets/
```

安装开发依赖项：

```sh
pip install -e .  # Install minimal deps to use tensorflow_datasets
pip install -e ".[dev]"  # Install all deps required for testing and development
```

请注意，还需要通过 `pip install -e ".[tests-all]"` 来安装所有特定于数据集的依赖项。

### Visual Studio Code

在使用 [Visual Studio Code](https://code.visualstudio.com/) 进行开发时，我们的仓库附带了一些[预定义设置](https://github.com/tensorflow/datasets/tree/master/.vscode/settings.json)来帮助开发（正确的缩进、pylint…）。

注：由于某些 VS Code 错误 [#13301](https://github.com/microsoft/vscode-python/issues/13301) 和 [#6594](https://github.com/microsoft/vscode-python/issues/6594)，在 VS Code 中启用测试发现可能会失败。要解决这些议题，可以查看测试发现日志：

- 如果您遇到了一些 TensorFlow 警告消息，请尝试[此修正](https://github.com/microsoft/vscode-python/issues/6594#issuecomment-555680813)。
- 如果由于缺少应该安装的导入而导致发现失败，请发送 PR 以更新 `dev` pip 安装。

## PR 核对清单

### 签署 CLA

对此项目的贡献必须随附贡献者许可协议 (CLA)。您（或您的雇主）保留您贡献的版权；这样做的目的是允许我们在项目中使用和重新分配您的贡献。前往 [https://cla.developers.google.com/](https://cla.developers.google.com/) 查看您当前归档的协议或签署新协议。

您通常只需要提交一次 CLA，因此如果您已经提交了一份（即使是针对不同的项目），您可能不需要再次提交。

### 遵循最佳做法

- 可读性非常重要。代码应遵循最佳编程做法（避免重复，分解成小的独立函数，使用显式变量名称…）
- 越简单越好（例如，将实现分成多个较小的独立 PR，这样更容易审核）。
- 需要时添加测试，现有测试应当能够通过。
- 添加[输入注解](https://docs.python.org/3/library/typing.html)

### 检查风格指南

我们的风格基于 [Google Python 风格指南](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)，后者基于 [PEP 8 Python 风格指南](https://www.python.org/dev/peps/pep-0008)。新代码应努力遵循 [Black 代码风格](https://github.com/psf/black/blob/master/docs/the_black_code_style.md)，但应具有：

- 行长度：80
- 2 个而不是 4 个空格缩进。
- 单引号 `'`

**重要提示**：确保在您的代码上运行 `pylint` 以检查您的代码格式是否正确：

```sh
pip install pylint --upgrade
pylint tensorflow_datasets/core/some_file.py
```

您可以尝试 `yapf` 来自动格式化文件，但该工具并不完美，因此您可能必须在之后手动应用修正。

```sh
yapf tensorflow_datasets/core/some_file.py
```

`pylint` 和 `yapf` 都应使用 `pip install -e ".[dev]"` 安装，但也可以使用 `pip install` 手动安装。如果您使用的是 VS Code，那么这些工具应当集成在 UI 中。

### 文档字符串和输入注解

类和函数应当用文档字符串和输入注解来记录。文档字符串应遵循 [Google 风格](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)。例如：

```python
def function(x: List[T]) -> T:
  """One line doc should end by a dot.

  * Use `backticks` for code and tripple backticks for multi-line.
  * Use full API name (`tfds.core.DatasetBuilder` instead of `DatasetBuilder`)
  * Use `Args:`, `Returns:`, `Yields:`, `Attributes:`, `Raises:`

  Args:
    x: description

  Returns:
    y: description
  """
```

### 添加并运行单元测试

确保使用单元测试对新功能进行测试。您可以通过 VS Code 界面或命令行运行测试。例如：

```sh
pytest -vv tensorflow_datasets/core/
```

`pytest` 与 `unittest`：我们一直使用 `unittest` 模块来编写测试。新测试最好使用 `pytest`，它更简单、灵活、现代，并且被大多数著名的库（numpy、pandas、sklearn、matplotlib、scipy、6…）使用。如果您不熟悉 pytest，可以阅读 [pytest 指南](https://docs.pytest.org/en/stable/getting-started.html#getstarted)。

DatasetBuilder 的测试是特殊的，记录在[添加数据集指南](https://github.com/tensorflow/datasets/blob/master/docs/add_dataset.md#test-your-dataset)中。

### 发送 PR 以供审核！

恭喜！如需详细了解如何使用拉取请求，请参阅 [GitHub 帮助](https://help.github.com/articles/about-pull-requests/)。

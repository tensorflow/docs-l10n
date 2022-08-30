# TensorFlow Agents

**使用 TensorFlow 进行强化学习**

通过提供经充分测试的可修改、可拓展的模块化组件，Agents使得设计、实现和测试新的回归算法更加容易。它实现了快速的代码迭代，同时保证了良好的测试集成和基准。

要开始使用，我们建议您先阅读我们的[教程](/tutorials)之一。

## 安装

TF-Agents发布有测试版和稳定版。有关发行列表，请阅读 <a href="#Releases">发行</a> 部分。以下命令包含从 [pypi.org](https://pypi.org)以及GitHub克隆安装TF-Agents的稳定版和测试版。

### 稳定版本

运行以下命令安装最新的稳定版本。[tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents) 上提供了版本的 API 文档。

```shell
$ pip install --user tf-agents[reverb]

# Use this tag get the matching examples and colabs.
$ git clone https://github.com/tensorflow/agents.git
$ cd agents
$ git checkout v0.13.0
```

如果您希望将 TF-Agents 与在 pip 依赖项检查中被标记为不兼容的 TensorFlow 或 [Reverb](https://github.com/deepmind/reverb) 版本一起安装，请使用以下模式并自行承担风险。

```shell
$ pip install --user tensorflow
$ pip install --user dm-reverb
$ pip install --user tf-agents
```

如果要将 TF-Agents 与 TensorFlow 1.15 或 2.0 一起使用，请安装版本 0.3.0：

```shell
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
$ pip install tensorflow-probability==0.8.0
$ pip install tf-agents==0.3.0
```

### Nightly 版本

Nightly 版本包含较新的功能，但可能不如带版本号的版本稳定。Nightly 版本以 `tf-agents-nightly` 方式推送。我们建议安装 Nightly 版本的 TensorFlow (`tf-nightly`) 和 TensorFlow (`tfp-nightly`) ，因为它们是经过测试的 TF-Agents Nightly 版本。

克隆仓库后，可以通过运行 `pip install -e .[tests]` 来安装依赖项。TensorFlow 需要独立安装：`pip install --user tf-nightly`。

```shell
# `--force-reinstall helps guarantee the right versions.
$ pip install --user --force-reinstall tf-nightly
$ pip install --user --force-reinstall tfp-nightly
$ pip install --user --force-reinstall dm-reverb-nightly

# Installing with the `--upgrade` flag ensures you'll get the latest version.
$ pip install --user --upgrade tf-agents-nightly
```

### 从 GitHub

克隆存储库后，可以通过运行 `pip install -e .[tests]` 来安装依赖项。TensorFlow 需要独立安装：`pip install --user tf-nightly`。

<a id="Contributing"></a>

## 贡献

我们渴望与您合作！有关如何贡献的指南，请参阅 [`CONTRIBUTING.md`](https://github.com/tensorflow/agents/blob/master/CONTRIBUTING.md)。此项目遵循 TensorFlow 的[行为准则](https://github.com/tensorflow/agents/blob/master/CODE_OF_CONDUCT.md)。参与，即表示您同意遵循此准则。

<a id="Releases"></a>

## 版本

TF Agents 包含稳定版本和 Nightly 版本。Nightly 版本通常表现良好，但是由于上游库在不断变化，因此可能存在一些问题。下表列出了测试每个 TF Agents 版本所用的 TensorFlow 版本，以帮助可能被限制为特定版本 TensorFlow 的用户。

版本 | 分支/标签 | TensorFlow 版本
--- | --- | ---
Nightly 版本 | [master](https://github.com/tensorflow/agents) | tf-nightly
0.13.0 | [v0.13.0](https://github.com/tensorflow/agents/tree/v0.13.0) | 2.9.0
0.12.0 | [v0.12.0](https://github.com/tensorflow/agents/tree/v0.12.0) | 2.8.0
0.9.0 | [v0.11.0](https://github.com/tensorflow/agents/tree/v0.11.0) | 2.7.0
0.8.0 | [v0.10.0](https://github.com/tensorflow/agents/tree/v0.10.0) | 2.5.0
0.9.0 | [v0.9.0](https://github.com/tensorflow/agents/tree/v0.9.0) | 2.6.0
0.8.0 | [v0.8.0](https://github.com/tensorflow/agents/tree/v0.8.0) | 2.5.0
0.7.1 | [v0.7.1](https://github.com/tensorflow/agents/tree/v0.7.1) | 2.4.0
0.6.0 | [v0.6.0](https://github.com/tensorflow/agents/tree/v0.6.0) | 2.3.0
0.5.0 | [v0.5.0](https://github.com/tensorflow/agents/tree/v0.5.0) | 2.2.0
0.4.0 | [v0.4.0](https://github.com/tensorflow/agents/tree/v0.4.0) | 2.1.0
0.3.0 | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 和 2.0.0

<a id="Principles"></a>

## 原则

该项目坚持 [Google's AI principles](https://github.com/tensorflow/agents/blob/master/PRINCIPLES.md). 参与、使用或为该项目做出贡献的过程，您需要遵守这些原则。

<a id="Citation"></a>

## 引用

如果您使用此代码，请按如下方式引用：

```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = {Sergio Guadarrama and Anoop Korattikara and Oscar Ramirez and
     Pablo Castro and Ethan Holly and Sam Fishman and Ke Wang and
     Ekaterina Gonina and Neal Wu and Efi Kokiopoulou and Luciano Sbaiz and
     Jamie Smith and Gábor Bartók and Jesse Berent and Chris Harris and
     Vincent Vanhoucke and Eugene Brevdo},
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```

<div align="center">   <img src="https://tensorflow.google.cn/images/SIGAddons.png" class=""><br><br> </div>

---

# TensorFlow Addons

**TensorFlow Addons** 是一个符合完善的 API 模式但实现了核心 TensorFlow 中未提供的新功能的贡献仓库。TensorFlow 原生支持大量算子、层、指标、损失和优化器。但是，在像机器学习一样的快速发展领域中，有许多有趣的新开发成果无法集成到核心 TensorFlow 中（因为它们的广泛适用性尚不明确，或者主要由社区的较小子集使用）。

## 安装

#### 稳定版本

要安装最新版本，请运行以下命令：

```
pip install tensorflow-addons
```

要使用插件：

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Nightly 版本

此外，TensorFlow Addons 的 pip 软件包 `tfa-nightly` 下还有 Nightly 版本，该软件包是针对 TensorFlow 的最新稳定版本构建的。Nightly 版本包含较新的功能，但可能不如带版本号的版本稳定。

```
pip install tfa-nightly
```

#### 从源安装

您也可以从源安装。这需要 [Bazel](https://bazel.build/) 构建系统。

```
git clone https://github.com/tensorflow/addons.git
cd addons

# If building GPU Ops (Requires CUDA 10.0 and CuDNN 7)
export TF_NEED_CUDA=1
export CUDA_HOME="/path/to/cuda10" (default: /usr/local/cuda)
export CUDNN_INSTALL_PATH="/path/to/cudnn" (default: /usr/lib/x86_64-linux-gnu)

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## 核心理念

#### 子软件包中的标准化 API

用户体验和项目可维护性是 TF-Addons 中的核心理念。为了实现这些目标，我们要求添加的插件符合核心 TensorFlow 中完善的 API 模式。

#### GPU/CPU 自定义运算

TensorFlow Addons 的主要好处是提供预编译运算。如果未找到 CUDA 10 安装，则运算将自动退回到 CPU 实现。

#### 代理维护权

Addons 设计为区分子软件包和子模块，以便对相关组件具有专业知识和浓厚兴趣的用户可以维护它们。

仅在做出实质性贡献后才会授予子软件包维护权，目的是限制拥有写入权限的用户数量。贡献可以是问题关闭、错误修复、文档、新代码或优化现有代码。授予子模块维护权的门槛较低，因为它不包括对仓库的写入权限。

有关此主题的更多信息，请参阅 [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md)。

#### 子软件包的定期评估

鉴于此仓库的性质，随着时间的推移，子软件包和子模块对社区的用处可能越来越少。为了保持仓库的可持续性，我们将每半年对代码执行一次审核，以确保所有内容仍属于仓库。这种审核的促成因素包括：

1. 活跃维护者数量
2. OSS 使用量
3. 归因于代码的问题或错误数量
4. 如果现在有更好的解决方案

TensorFlow Addons 中的功能可以分为三类：

- **建议**：完善的 API；鼓励使用。
- **不鼓励**：已有更好的选择。保留 API 是出于历史原因；或者 API 需要维护，并且处于弃用的等待时间。
- **弃用**：使用风险自负；随时会删除。

这三个组之间的状态更改为：建议 <-> 不鼓励 -> 弃用。

一个 API 从被标记为弃用到被删除之间的间隔为 90 天。理由如下：

1. 如果 TensorFlow Addons 每月发布一次，则在删除 API 之前将有 2-3 个版本。版本说明可能会给用户提供足够的警告。

2. 90 天为维护人员提供了充足的时间来修复其代码。

## 贡献

TF-Addons 是一个社区主导的开源项目。因此，该项目取决于公共贡献、错误修复和文档。有关如何贡献的指南，请参阅[贡献准则](https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md)。此项目遵循 [TensorFlow 的行为准则](https://github.com/tensorflow/addons/blob/master/CODE_OF_CONDUCT.md)。参与，即表示您同意遵守此准则。

## 社区

- [公开邮寄名单](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
- [SIG 每月会议记录](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    - 加入我们的邮寄名单，并接收参加会议的日历邀请

## 许可证

[Apache License 2.0](LICENSE)

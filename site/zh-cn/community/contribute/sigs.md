# 为 TensorFlow 特殊兴趣小组 (SIG) 做出贡献

TensorFlow 特殊兴趣小组 (TF SIG) 负责组织社区对 TensorFlow 生态系统的重要部分做出贡献。SIG 领导和成员共同致力于构建和支持重要的 TensorFlow 用例。

SIG 由包括行业协作者和[机器学习 Google 开发者专家](https://developers.google.com/community/experts) (ML GDE) 在内的开源社区成员领导。TensorFlow 的成功在很大程度上归功于他们的辛勤工作和贡献。

我们鼓励您加入致力于您最关心的 TensorFlow 生态系统领域的 SIG。并非所有 SIG 都具有相同的能量水平、范围广度或治理模式 – 如需了解详情，请浏览我们的 [SIG 章程](https://github.com/tensorflow/community/tree/master/sigs)。在 [TensorFlow Forum](https://discuss.tensorflow.org/c/special-interest-groups/8) 上与 SIG 领导和成员保持联系，您可以在论坛中订阅喜爱的[标签](https://discuss.tensorflow.org/tags)并详细了解定期 SIG 会议。

## SIG Addons

SIG Addons 负责构建和维护一个社区贡献仓库，这些贡献符合完善的 API 模式，但实现了核心 TensorFlow 中未提供的新功能。

TensorFlow 原生支持大量算子、层、指标、损失、优化器等内容。然而，在像机器学习这样快速发展的领域中，有许多新的开发内容无法集成到核心 TensorFlow 当中（因为它们的广泛适用性尚不明确，或者它们主要是被社区中的一小部分人群使用）。SIG Addons 使用户能够以可持续的方式向 TensorFlow 生态系统引入新的扩展程序。

<a class="button button-primary" href="https://github.com/tensorflow/addons">在 GitHub 上查看 SIG Addons</a> <a class="button" href="https://github.com/tensorflow/addons/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/addons/11">在 Forum 上讨论</a>

## SIG Build

SIG Build 负责改进并扩展 TensorFlow 构建过程。SIG Build 维护着一个仓库，其中展示了由社区贡献并供社区使用的资源、指南、工具和构建。

<a class="button button-primary" href="https://github.com/tensorflow/build">在 GitHub 上查看 SIG Build</a> <a class="button" href="https://github.com/tensorflow/build/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/build">在 Forum 上讨论</a>

## SIG IO

SIG IO 负责维护 TensorFlow I/O，这是 TensorFlow 内置支持中不可用的文件系统和文件格式的集合。

<a class="button button-primary" href="https://github.com/tensorflow/io">在 GitHub 上查看 SIG IO</a> <a class="button" href="https://github.com/tensorflow/io/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/io">在 Forum 上讨论</a>

## SIG JVM

SIG JVM 负责维护 TF Java 绑定，使用户可以使用 JVM 来构建、训练和运行机器学习模型。

Java 和诸如 Scala 或 Kotlin 的其他 JVM 语言在世界各地不同规模的企业中受到广泛使用，这使得 TensorFlow 成为大规模采用机器学习的战略选择。

<a class="button button-primary" href="https://github.com/tensorflow/java">在 GitHub 上查看 SIG JVM</a> <a class="button" href="https://github.com/tensorflow/java/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/jvm">在 Forum 上讨论</a>

## SIG Models

SIG Models 专注于为 TensorFlow 2 中最先进的模型实现做出贡献，以及分享使用 TensorFlow 2 进行最先进研究的最佳做法。各个子小组负责不同的机器学习应用（视觉、NLP 等）。

SIG Models 会主办围绕 [TensorFlow Model Garden](https://github.com/tensorflow/models) 和 [TensorFlow Hub](https://tfhub.dev) 的讨论和协作。参阅下文以了解如何在 GitHub 上做出贡献，或在 Forum 上讨论[研究和模型](https://discuss.tensorflow.org/c/research-models/26)。

<a class="button button-primary" href="https://github.com/tensorflow/models">在 GitHub 上查看 TensorFlow Model Garden</a> <a class="button" href="https://github.com/tensorflow/models/blob/master/CONTRIBUTING.md">贡献</a>

<a class="button button-primary" href="https://github.com/tensorflow/hub">在 GitHub 上查看 TensorFlow Hub</a> <a class="button" href="https://github.com/tensorflow/hub/blob/master/CONTRIBUTING.md">贡献</a>

## SIG Micro

SIG Micro 负责讨论和分享 [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) 的更新，TensorFlow Lite for Microcontrollers 是 TensorFlow Lite 的一个端口，旨在实现在 DSP、微控制器和其他内存有限的设备上运行机器学习模型。

<a class="button button-primary" href="https://github.com/tensorflow/tflite-micro">在 GitHub 上查看 TensorFlow Lite Micro</a> <a class="button" href="https://github.com/tensorflow/tflite-micro/blob/main/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/micro">在 Forum 上讨论</a>

## SIG MLIR

SIG MLIR 负责维护适用于 TensorFlow、XLA 和 TF Lite 的 [MLIR](https://mlir.llvm.org/) 方言与实用工具，提供可应用于 TensorFlow 计算图和代码生成的高性能编译器和优化技术。它们的首要目标是创建通用中间表示 (IR)，以降低开发新硬件的成本，并为现有 TensorFlow 用户提高易用性。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir">在 GitHub 上查看 SIG MLIR</a> <a class="button" href="https://mlir.llvm.org/">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/mlir">在 Forum 上讨论</a>

## SIG Networking

SIG Networking 负责维护 TensorFlow Networking 仓库，该仓库存储着对核心 TensorFlow 和相关实用工具的针对特定平台的网络扩展程序。

<a class="button button-primary" href="https://github.com/tensorflow/networking">在 GitHub 上查看 SIG Networking</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/networking">在 Forum 上讨论</a>

## SIG Recommenders

SIG Recommenders 负责维护一系列由社区贡献和维护的基于 TensorFlow 的大型推荐系统相关项目。这些贡献是对 [TensorFlow Core](https://www.tensorflow.org/overview) 和 [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) 的补充。

<a class="button button-primary" href="https://github.com/tensorflow/recommenders-addons">在 GitHub 上查看 SIG Recommenders</a> <a class="button" href="https://github.com/tensorflow/recommenders-addons/blob/master/CONTRIBUTING.md/">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/recommenders">在 Forum 上讨论</a>

## SIG Rust

SIG Rust 负责维护 TensorFlow 的惯用 Rust 语言绑定。

<a class="button button-primary" href="https://github.com/tensorflow/rust/blob/master/CONTRIBUTING.md">在 GitHub 上查看 SIG Rust</a> <a class="button" href="https://github.com/tensorflow/rust/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/rust">在 Forum 上讨论</a>

## SIG TensorBoard

SIG TensorBoard 负责促进围绕 [TensorBoard](https://www.tensorflow.org/tensorboard) 的讨论，TensorBoard 是一套用于检查、调试和优化 TensorFlow 程序的工具。

<a class="button button-primary" href="https://github.com/tensorflow/tensorboard">在 GitHub 上查看 TensorBoard</a> <a class="button" href="https://github.com/tensorflow/tensorboard/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/tensorboard/">在 Forum 上讨论</a>

## SIG TF.js

SIG TF.js 负责促进社区为 [TensorFlow.js](https://www.tensorflow.org/js) 贡献组件，并通过 SIG 提供项目支持。

<a class="button button-primary" href="https://github.com/tensorflow/tfjs">在 GitHub 上查看 TensorFlow.js</a> <a class="button" href="https://github.com/tensorflow/tfjs/blob/master/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/tfjs/">在 Forum 上讨论</a>

## SIG TFX-Addons

SIG TFX-Addons 负责促进共享定制和新增内容，以满足生产型机器学习的需求、拓展视野并帮助向新的方向推动 [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) 和机器学习社区。

<a class="button button-primary" href="https://github.com/tensorflow/tfx-addons">在 GitHub 上查看 SIG TFX-Addons</a> <a class="button" href="https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md">贡献</a> <a class="button" href="https://discuss.tensorflow.org/c/special-interest-groups/tfx-addons/">在 Forum 上讨论</a>

## New SIGs

没有找到自己所需的内容？如果您认为有新的 TensorFlow SIG 亟待开发，请阅读 [SIG 手册](https://www.tensorflow.org/community/sig_playbook)并按照说明向我们的贡献者社区提出建议。

# 编排 TFX 流水线

## 本地编排器

本地编排器是一个简单的编排，包含在 TFX Python 软件包中。它在本地环境中以单个进程运行流水线。它可以为开发和调试提供快速迭代，但不适合大型生产工作负载。请将 [Vertex Pipelines](/tfx/guide/vertex) 或 [Kubeflow Pipelines](/tfx/guide/kubeflow) 用于生产用例。

尝试在 Colab 中运行的 [TFX 教程](/tfx/tutorials/tfx/penguin_simple)，了解如何使用本地编排器。

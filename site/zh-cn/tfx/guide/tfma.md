# 使用 TensorFlow Model Analysis 改善模型质量

## 简介

在开发过程中调整模型时，您需要检查所做的更改是否会改善模型质量。不过，仅检查准确率可能还不够。例如，如果您有一个问题的分类器，其中 95% 的实例是正相关的，那么您也许能够通过始终预测为正相关来提高准确率，但是这样做将无法获得一个非常稳健的分类器。

## 概述

TensorFlow Model Analysis 的目标是为 TFX 中的模型评估提供一种机制。TensorFlow Model Analysis 允许您在 TFX 流水线中执行模型评估，并在 Jupyter 笔记本中查看结果指标和绘图。具体来说，它可以提供：

- 根据整个训练和保留数据集计算的指标，以及第二天的评估
- 随时间跟踪指标
- 在不同特征切片上的模型质量性能
- [模型验证](../model_analysis/model_validations)，用于确保模型的性能保持一致

## 后续步骤

阅读我们的 [TFMA 教程](../tutorials/model_analysis/tfma_basic)。

查看我们的 [GitHub](https://github.com/tensorflow/model-analysis) 页面了解有关支持的[指标与图表](../model_analysis/metrics)以及关联的笔记本[可视化效果](../model_analysis/visualizations)的详细信息。

参阅[安装](../model_analysis/install)和[使用入门](../model_analysis/get_started)指南，了解有关在独立流水线中进行[设置](../model_analysis/setup)的信息和示例。回想一下，TFMA 也在 TFC 的 [Evaluator](evaluator.md) 组件中使用，因此，这些资源也有助于开始使用 TFX。

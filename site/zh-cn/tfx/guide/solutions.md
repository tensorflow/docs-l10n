# TFX 云解决方案

想了解如何应用 TFX 来构建满足您需求的解决方案吗？这些深入的文章和指南可能会有所帮助！

注：这些文章讨论了完整的解决方案，TFX 在其中为关键部分，而非唯一部分。实际部署几乎总是如此。因此，您自己实施这些解决方案需要的不仅仅是 TFX。主要目标是让您深入了解其他人如何实施所满足的要求可能与您相类似的解决方案，而不是充当 TFX 获批应用程序的清单或列表。

## 用于近实时项目匹配的机器学习系统架构

使用本文档了解学习并提供项目嵌入的机器学习 (ML) 解决方案的架构。嵌入可以帮助您了解客户认为哪些项目相似，从而使您能够在应用程序中提供实时“相似项目”建议。此解决方案向您展示如何找出数据集中的相似歌曲，然后使用此信息进行歌曲推荐。<a href="https://cloud.google.com/solutions/real-time-item-matching" class="external" target="_blank">阅读更多</a>

## 机器学习的数据预处理：选项和建议

这篇由两部分组成的文章探讨了机器学习 (ML) 的数据工程和特征工程的主题。第一部分讨论在 Google Cloud 上的机器学习流水线中预处理数据的最佳实践。本文重点介绍使用 TensorFlow 和开源 TensorFlow Transform (tf.Transform) 库来准备数据、训练模型以及为模型提供预测服务。此部分重点介绍了为机器学习预处理数据的挑战，并说明在 Google Cloud 上有效执行数据转换的选项和场景。<a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt1" class="external" target="_blank">第 1 部分</a><a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt2" class="external" target="_blank">第 2 部分</a>

## 使用 TFX、Kubeflow Pipelines 和 Cloud Build 的 MLOps 架构

本文档描述了使用 TensorFlow Extended (TFX) 库的机器学习 (ML) 系统的整体架构。它还讨论了如何使用 Cloud Build 和 Kubeflow Pipelines 为 ML 系统设置持续集成 (CI)、持续交付 (CD) 和持续训练 (CT)。<a href="https://cloud.google.com/solutions/machine-learning/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build" class="external" target="_blank">阅读更多</a>

## MLOps：机器学习中的持续交付和自动化流水线

本文档讨论了用于为机器学习 (ML) 系统实施和自动化持续集成 (CI)、持续交付 (CD) 和持续训练 (CT) 的技术。数据科学和 ML 正在成为解决复杂现实问题、使行业转型和在所有领域创造价值的核心能力。<a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" class="external" target="_blank">阅读更多</a>

## 在 Google Cloud 上设置 MLOps 环境

本参考指南概述了 Google Cloud 上机器学习操作 (MLOps) 环境的架构。在 GitHub 中，**该指南随附练习实验室**，以引导您完成此处描述的预置和配置环境的过程。几乎所有行业都在加快采用机器学习 (ML) 的步伐。从 ML 中获取价值的一个关键挑战是创建有效部署和操作 ML 系统的方法。本指南适用于机器学习 (ML) 和 DevOps 工程师。<a href="https://cloud.google.com/solutions/machine-learning/setting-up-an-mlops-environment" class="external" target="_blank">阅读更多</a>

## MLOps 基础的关键要求

AI 驱动的组织正在使用数据和机器学习来解决他们最困难的问题，并开始坐收渔翁之利。

麦肯锡全球研究院表示*：“到 2025 年，在其创造价值的工作流程中完全吸收 AI 的公司将实现 +120% 的现金流增长，从而在 2030 年的世界经济中占据主导地位*。”

但现在并不容易。机器学习 (ML) 系统有一项特殊能力，那就是在管理不善时产生技术债务。<a href="https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation" class="external" target="_blank">阅读更多</a>

## 如何使用 Scikit-Learn 在云中创建和部署模型卡

机器学习模型现在被用于完成许多具有挑战性的任务。凭借其巨大的潜力，ML 模型还引发了有关其使用、构造和局限性的问题。记录这些问题的答案有助于理顺思路并达成共识。为了帮助推进这些目标，Google 推出了模型卡。<a href="https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn" class="external" target="_blank">阅读更多</a>

## 使用 TensorFlow Data Validation 大规模分析和验证数据以进行机器学习

本文档讨论如何在实验期间使用 TensorFlow Data Validation (TFDV) 库进行数据探索和描述性分析。数据科学家和机器学习 (ML) 工程师可以在生产 ML 系统中使用 TFDV 来验证用于连续训练 (CT) 流水线的数据，并检测接收到的用于预测服务的数据中的偏差和异常值。它包括**练习实验室**。<a href="https://cloud.google.com/solutions/machine-learning/analyzing-and-validating-data-at-scale-for-ml-using-tfx" class="external" target="_blank">阅读更多</a>

# 协作优化

<sub>由 Arm ML Tooling 维护</sub>

本文档概述了用于组合各种技术来优化部署用机器学习模型的实验性 API。

## 概述

协作优化是一个包含各种技术的拱型流程，旨在产生在部署时展现出目标特征（如推断速度、模型大小和准确率）最佳平衡的模型。

协作优化的理念建立在个别技术的基础上，通过逐个应用这些技术来实现累积的优化效果。可以实现以下优化的各种组合：

- [权重剪枝](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a)

- [权重聚类](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)

- 量化

    - [训练后量化](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
    - [量化感知训练](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html) (QAT)

试图将这些技术链接在一起时出现的问题是，应用一种技术通常会破坏先前技术的结果，因而会损害同时应用所有这些技术的整体效益；例如，聚类不会保留剪枝 API 引入的稀疏性。为了解决这个问题，我们引入了以下实验性的协作优化技术：

- [稀疏性保留聚类](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)
- [稀疏性保留量化感知训练](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example) (PQAT)
- [聚类保留量化感知训练](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example) (CQAT)
- [稀疏性和聚类保留量化感知训练](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example)

这些技术提供了多个可用于压缩机器学习模型和在推断时利用硬件加速的部署路径。下图展示了几个可以在搜索具有所需部署特征的模型时进行探索的部署路径，其中的叶节点是部署就绪模型，意味着它们被部分或完全量化并采用 tflite 格式。填充为绿色的步骤表示需要重新训练/微调，协作优化步骤用红色虚线边框突出显示。在给定节点用于获取模型的技术在相应的标签中加以指示。

![协作优化](images/collaborative_optimization.png "协同优化")

上图省略了直接仅量化（训练后或 QAT）部署路径。

理念是在上述部署树的第三层实现完全优化的模型；但是，任何其他层的优化也可能被证明是令人满意的，并且实现了所需的推断延迟/准确性权衡，在这种情况下，就不需要进一步优化。推荐的训练过程是反复检查适用于目标部署场景的部署树的各个层，看模型是否满足推断延迟要求，如果不满足，则使用相应的协作优化技术进一步压缩模型并在需要时重复此操作直至模型完全优化（剪枝、聚类和量化）。

下图显示了通过协作优化流水线的样本权重内核的密度图。

![协作优化密度图](images/collaborative_optimization_dist.png "collaborative optimization density plot")

结果是一个减少了唯一值并具有大量稀疏权重的量化部署模型，权重量取决于训练时指定的目标稀疏度。除了显著的模型压缩优势，特定的硬件支持还可以利用这些稀疏的聚类模型来大幅减少推断延迟。

## 结果

下面是我们在试验 PQAT 和 CQAT 协作优化路径时获得的一些准确率和压缩结果。

### 稀疏性保留量化感知训练 (PQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Pruned Model (50% sparsity)</th><th>QAT Model</th><th>PQAT Model</th></tr>
 <tr><td>DS-CNN-L</td><td>FP32 Top1 Accuracy</td><td><b>95.23%</b></td><td>94.80%</td><td>(Fake INT8) 94.721%</td><td>(Fake INT8) 94.128%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.48%</td><td><b>93.80%</b></td><td>94.72%</td><td><b>94.13%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>528,128 → 434,879 (17.66%)</td><td>528,128 → 334,154 (36.73%)</td><td>512,224 → 403,261 (21.27%)</td><td>512,032 → 303,997 (40.63%)</td></tr>
 <tr><td>Mobilenet_v1-224</td><td>FP32 Top 1 Accuracy</td><td><b>70.99%</b></td><td>70.11%</td><td>(Fake INT8) 70.67%</td><td>(Fake INT8) 70.29%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.37%</td><td><b>67.82%</b></td><td>70.67%</td><td><b>70.29%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,520 → 3,880,331 (16.83%)</td><td>4,665,520 → 2,939,734 (37.00%)</td><td>4,569,416 → 3,808,781 (16.65%)</td><td>4,569,416 → 2,869,600 (37.20%)</td></tr>
</table>
</figure>

### 聚类保留量化感知训练 (CQAT)

<figure>
<table class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Items</th><th>Baseline</th><th>Clustered Model</th><th>QAT Model</th><th>CQAT Model</th></tr>
 <tr><td>Mobilenet_v1 on CIFAR-10</td><td>FP32 Top1 Accuracy</td><td><b>94.88%</b></td><td>94.48%</td><td>(Fake INT8) 94.80%</td><td>(Fake INT8) 94.60%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>94.65%</td><td><b>94.41%</b></td><td>94.77%</td><td><b>94.52%</b></td></tr>
 <tr><td> </td><td>Size</td><td>3.00 MB</td><td>2.00 MB</td><td>2.84 MB</td><td>1.94 MB</td></tr>
 <tr><td>Mobilenet_v1 on ImageNet</td><td>FP32 Top 1 Accuracy</td><td><b>71.07%</b></td><td>65.30%</td><td>(Fake INT8) 70.39%</td><td>(Fake INT8) 65.35%</td></tr>
 <tr><td> </td><td>INT8 full integer quantization</td><td>69.34%</td><td><b>60.60%</b></td><td>70.35%</td><td><b>65.42%</b></td></tr>
 <tr><td> </td><td>Compression</td><td>4,665,568 → 3,886,277 (16.7%)</td><td>4,665,568 → 3,035,752 (34.9%)</td><td>4,569,416 → 3,804,871 (16.7%)</td><td>4,569,472 → 2,912,655 (36.25%)</td></tr>
</table>
</figure>

### 按通道聚类的模型的 CQAT 和 PCQAT 结果

下面的结果是使用[按通道聚类](https://www.tensorflow.org/model_optimization/guide/clustering)技术获得的。这些结果表明，如果模型的卷积层按通道进行聚类，则模型准确度更高。如果您的模型有许多卷积层，那么我们建议按通道聚类。压缩比保持不变，但模型准确度会更高。在我们的实验中，模型优化流水线为“聚类 -&gt; 聚类保留 QAT -&gt; 训练后量化，int8”。

<figure>
<table  class="tableizer-table">
<tr class="tableizer-firstrow"><th>Model</th><th>Clustered -> CQAT, int8 quantized</th><th>Clustered per channel -> CQAT, int8 quantized</th>
 <tr><td>DS-CNN-L</td><td>95.949%</td><td> 96.44%</td></tr>
 <tr><td>MobileNet-V2</td><td>71.538%</td><td>72.638%</td></tr>
 <tr><td>MobileNet-V2 (pruned)</td><td>71.45%</td><td>71.901%</td></tr>
</table>
</figure>

## 示例

有关这里介绍的协作优化技术的端到端示例，请参阅 [CQAT](https://www.tensorflow.org/model_optimization/guide/combine/cqat_example)、[PQAT](https://www.tensorflow.org/model_optimization/guide/combine/pqat_example)、[稀疏性保留聚类](https://www.tensorflow.org/model_optimization/guide/combine/sparse_clustering_example)和 [PCQAT](https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example) 示例笔记本。

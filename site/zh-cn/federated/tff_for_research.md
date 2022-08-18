# 使用 TFF 进行联合学习研究

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## 概述

TFF 是一个可扩展的强大框架，通过在实际代理数据集上模拟联合计算来进行联合学习 (FL) 研究。本页面描述了与研究模拟相关的主要概念和组件，以及在 TFF 中进行各种研究的详细指南。

## TFF 中研究代码的典型结构

在 TFF 中实现的研究 FL 模拟通常包括三种主要的逻辑类型。

1. Individual pieces of TensorFlow code, typically `tf.function`s, that encapsulate logic that runs in a single location (e.g., on clients or on a server). This code is typically written and tested without any `tff.*` references, and can be re-used outside of TFF. For example, the [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) is implemented at this level.

2. TensorFlow Federated orchestration logic, which binds together the individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s and then orchestrating them using abstractions like `tff.federated_broadcast` and `tff.federated_mean` inside a `tff.federated_computation`. See, for example, this [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. An outer driver script that simulates the control logic of a production FL system, selecting simulated clients from a dataset and then executing federated computations defined in 2. on those clients. For example, [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).

## 联合学习数据集

TensorFlow Federated [托管了多个数据集](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)，这些数据集代表可以通过联合学习解决的实际问题的特征。

注：这些数据集也可以被任意基于 Python 的 ML 框架（如 Numpy 数组）使用，如 [ClientData API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData) 中所述。

数据集包括：

- [**StackOverflow**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) A realistic text dataset for language modeling or supervised learning tasks, with 342,477 unique users with 135,818,730 examples (sentences) in the training set.

- [**Federated EMNIST**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)。EMNIST 字符和数字数据集的联合预处理，其中每个客户端对应一个不同的编写器。完整的训练集包含 3400 个用户和来自 62 个标签的 671,585 个样本。

- [**Shakespeare**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)。基于威廉·莎士比亚全集的较小的字符级文本数据集。该数据集由 715 个用户（莎士比亚戏剧中的角色）组成，其中每个样本对应给定戏剧中的角色所说的一组连续台词。

- [**CIFAR-100**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data) A federated partitioning of the CIFAR-100 dataset across 500 training clients and 100 test clients. Each client has 100 unique examples. The partitioning is done in a way to create more realistic heterogeneity between clients. For more details, see the [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data).

- [**Google Landmark v2 数据集。**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data)该数据集由各种世界地标的照片组成，图像按摄影师分组以实现数据的联合分区。提供两种形式的数据集：较小的数据集包括 233 个客户端和 23080 个图像，较大的数据集包括 1262 个客户端和 164172 个图像。

- [**CelebA。**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data)名人面部样本（图像和面部属性）数据集。该联合数据集将每个名人的样本组合在一起形成一个客户端。共有 9343 个客户端，每个客户端至少包含 5 个样本。该数据集可以按客户端或按样本分为训练组和测试组。

- [**iNaturalist。**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data)一个由不种物种的照片组成的数据集。该数据集包含 1203 个物种的 120300 个图像。提供七种形式的数据集。其中一种按摄影师分组，包含 9257 个客户端。其余数据集按拍摄照片的地理位置分组。这六种数据集包含 11 - 3606 个客户端。

## 高性能模拟

虽然 FL *模拟*的时钟时间不是评估算法的相关指标（因为模拟硬件不代表真实的 FL 部署环境），但是快速运行 FL 模拟的能力对于提高研究效率至关重要。因此，TFF 投入了大量资源来提供高性能的单机和多机运行时。相关文档正在编写中，但现在您可以参阅[使用 Kubernetes 进行高性能模拟](https://www.tensorflow.org/federated/tutorials/high_performance_simulation_with_kubernetes)教程、有关[使用加速器进行 TFF 模拟](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators)的说明，以及有关[设置 GCP 上的 TFF 模拟](https://www.tensorflow.org/federated/gcp_setup)的说明。默认情况下，高性能 TFF 运行时处于启用状态。

## 针对不同研究领域的 TFF

### 联合优化算法

在 TFF 中，根据所需自定义程度的不同，可以采用不同的方法对联合优化算法进行研究。

A minimal stand-alone implementation of the [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg). The code includes [TF functions](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py) for local computation, [TFF computations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py) for orchestration, and a [driver script](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py) on the EMNIST dataset as an example. These files can easily be adapted for customized applciations and algorithmic changes following detailed instructions in the [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py). This implementation allows for more sophisticated optimization techniques, including the use of different optimizers on both the server and client. Other federated learning algorithms, including federated k-means clustering, can be found [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/algorithms/).

### 模型更新压缩

模型更新的有损压缩可以降低通信成本，进而减少总体训练时间。

要复制最近的[论文](https://arxiv.org/abs/2201.02664)，请参阅[本研究项目](https://github.com/google-research/federated/tree/master/compressed_communication)。要实现自定义压缩算法，请参阅基线项目中的 [comparison_methods](https://github.com/google-research/federated/tree/master/compressed_communication/aggregators/comparison_methods) 作为示例，如果您尚不熟悉，请参阅 [TFF 聚合器教程](https://www.tensorflow.org/federated/tutorials/custom_aggregators)。

### 差分隐私

TFF 可与 [TensorFlow 隐私](https://github.com/tensorflow/privacy)库互操作，以研究新的算法，从而对使用差分隐私的模型进行联合训练。有关使用[基本 DP-FedAvg 算法](https://arxiv.org/abs/1710.06963)和[扩展程序](https://arxiv.org/abs/1812.06210)进行 DP 训练的示例，请参阅[此实验驱动程序](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py)。

如果要实现自定义 DP 算法并将其应用于联合平均算法的聚合更新，可以实现一个新的 DP 均值算法作为 [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) 的子类，然后使用查询实例创建 `tff.aggregators.DifferentiallyPrivateFactory`。实现 [DP-FTRL 算法](https://arxiv.org/abs/2103.00039)的示例可以在[此处](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)找到。

联合 GAN（[如下](#generative_adversarial_networks)所述）是 TFF 项目的另一个示例，它实现了用户级别的差分隐私（例如[此处的代码](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)）。

### 鲁棒性和攻击

TFF 还可以用于模拟联合学习系统上的针对性攻击以及 *[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)* 中所考虑的基于差分隐私的防御。这是通过使用潜在的恶意客户端构建迭代过程来实现的（请参阅 [`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/attacked_fedavg.py#L412)）。[targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack) 目录中包含更多详细信息。

- 新的攻击算法可以通过编写客户端更新函数来实现，该函数是 Tensorflow 函数。有关示例请参阅 [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460)。
- 新的防御可通过自定义 ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)（聚合客户端输出以获得全局更新）来实现。

有关模拟的示例脚本，请参阅 [`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py)。

### 生成对抗网络

GAN 提供了一种有趣的[联合编排模式](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L266-L316)，这种模式看上去和标准的联合平均略有不同。它们涉及两种不同的网络（生成器和判别器），每种网络使用自己的优化步骤进行训练。

TFF 可用于研究 GAN 的联合训练。例如，[最近研究工作](https://arxiv.org/abs/1911.06679)中展示的 DP-FedAvg-GAN 算法就是[在 TFF 中实现](https://github.com/tensorflow/federated/tree/main/federated_research/gans)的。此研究工作演示了将联合学习、生成模型和[差分隐私](#differential_privacy)相结合的有效性。

### 个性化

联合学习设置中的个性化是一个活跃的研究领域。个性化的目的是为不同的用户提供不同的推理模型。此问题可能有不同的解决方法。

One approach is to let each client fine-tune a single global model (trained using federated learning) with their local data. This approach has connections to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An example of this approach is given in [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/emnist_p13n_main.py). To explore and compare different personalization strategies, you can:

- Define a personalization strategy by implementing a `tf.function` that starts from an initial model, trains and evaluates a personalized model using each client's local datasets. An example is given by [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/personalization/p13n_utils.py).

- 定义一个 `OrderedDict`，将策略名称映射到相应的个性化策略，并将其用作 [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval) 中的 `personalize_fn_dict` 参数。

Another approach is to avoid training a fully global model by training part of a model entirely locally. An instantiation of this approach is described in [this blog post](https://ai.googleblog.com/2021/12/a-scalable-approach-for-partially-local.html). This approach is also connected to meta learning, see [this paper](https://arxiv.org/abs/2102.03448). To explore partially local federated learning, you can:

- Check out the [tutorial](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization) for a complete code example applying Federated Reconstruction and [follow-up exercises](https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization#further_explorations).

- Create a partially local training process using [`tff.learning.reconstruction.build_training_process`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/reconstruction/build_training_process), modifying `dataset_split_fn` to customize process behavior.

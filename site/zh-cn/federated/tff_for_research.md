# 使用 TFF 进行联合学习研究

<!-- Note that some section headings are used as deep links into the document.
     If you update those section headings, please make sure you also update
     any links to the section. -->

## 概述

TFF 是一个可扩展的强大框架，通过在实际代理数据集上模拟联合计算来进行联合学习 (FL) 研究。本页面描述了与研究模拟相关的主要概念和组件，以及在 TFF 中进行各种研究的详细指南。

## TFF 中研究代码的典型结构

在 TFF 中实现的研究 FL 模拟通常包括三种主要的逻辑类型。

1. Individual pieces of TensorFlow code, typically `tf.function`s, that encapsulate logic that runs in a single location (e.g., on clients or on a server). This code is typically written and tested without any `tff.*` references, and can be re-used outside of TFF. For example, the [client training loop in Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222) is implemented at this level.

2. TensorFlow Federated orchestration logic, which binds together the individual `tf.function`s from 1. by wrapping them as `tff.tf_computation`s and then orchestrating them using abstractions like `tff.federated_broadcast` and `tff.federated_mean` inside a `tff.federated_computation`. See, for example, this [orchestration for Federated Averaging](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tff.py#L112-L140).

3. An outer driver script that simulates the control logic of a production FL system, selecting simulated clients from a dataset and then executing federated computations defined in 2. on those clients. For example, [a Federated EMNIST experiment driver](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py).

## 联合学习数据集

TensorFlow Federated [托管了多个数据集](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets)，这些数据集代表可以通过联合学习解决的实际问题的特征。

注：这些数据集也可以被任意基于 Python 的 ML 框架（如 Numpy 数组）使用，如 [ClientData API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData) 中所述。

数据集包括：

- [**StackOverflow**.](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) A realistic text dataset for language modeling or supervised learning tasks, with 342,477 unique users with 135,818,730 examples (sentences) in the training set.

- [**Federated EMNIST**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)EMNIST 字符和数字数据集的联合预处理，其中每个客户端对应一个不同的编写器。完整的训练集包含 3400 个用户和来自 62 个标签的 671,585 个样本。

- [**Shakespeare**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)基于威廉·莎士比亚全集的较小的字符级文本数据集。该数据集由 715 个用户（莎士比亚戏剧中的角色）组成，其中每个样本对应给定戏剧中的角色所说的一组连续台词。

- [**CIFAR-100**。](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)CIFAR-100 数据集在 500 个训练客户端和 100 个测试客户端上的联合分区。 每个客户端都有 100 个唯一样本。 分区的完成方式是在客户端之间创建更实际的异构性。 有关更多详细信息，请参阅 [API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)。

- [**Google Landmark v2 dataset**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/gldv2/load_data) The dataset consists of photos of various world landmarks, with images grouped by photographer to achieve a federated partitioning of the data. Two flavors of dataset are available: a smaller dataset with 233 clients and 23080 images, and a larger dataset with 1262 clients and 164172 images.

- [**CelebA**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/celeba/load_data) A dataset of examples (image and facial attributes) of celebrity faces. The federated dataset has each celebrity's examples grouped together to form a client. There are 9343 clients, each with at least 5 examples. The dataset can be split into train and test groups either by clients or by examples.

- [**iNaturalist**](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/inaturalist/load_data) A dataset consists of photos of various species. The dataset contains 120,300 images for 1,203 species. Seven flavors of the dataset are available. One of them is grouped by the photographer and it consists of 9257 clients. The rest of the datasets are grouped by the geo location where the photo was taken. These six flavors of the dataset consists of 11 - 3,606 clients.

## 高性能模拟

While the wall-clock time of an FL *simulation* is not a relevant metric for evaluating algorithms (as simulation hardware isn't representative of real FL deployment environments), being able to run FL simulations quickly is critical for research productivity. Hence, TFF has invested heavily in providing high-performance single and multi-machine runtimes. Documentation is under development, but for now see the [High-performance simulations with TFF](https://www.tensorflow.org/federated/tutorials/simulations) tutorial, instructions on [TFF simulations with accelerators](https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators), and instructions on [setting up simulations with TFF on GCP](https://www.tensorflow.org/federated/gcp_setup). The high-performance TFF runtime is enabled by default.

## 针对不同研究领域的 TFF

### 联合优化算法

在 TFF 中，根据所需自定义程度的不同，可以采用不同的方法对联合优化算法进行研究。

A minimal stand-alone implementation of the [Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided [here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg). The code includes [TF functions](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tf.py) for local computation, [TFF computations](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/simple_fedavg_tff.py) for orchestration, and a [driver script](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/emnist_fedavg_main.py) on the EMNIST dataset as an example. These files can easily be adapted for customized applciations and algorithmic changes following detailed instructions in the [README](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/simple_fedavg/README.md).

A more general implementation of Federated Averaging can be found [here](https://github.com/google-research/federated/blob/master/optimization/fed_avg_schedule.py). This implementation allows for more sophisticated optimization techniques, including learning rate scheduling and the use of different optimizers on both the server and client. Code that applies this generalized Federated Averaging to various tasks and federated datasets can be found [here](https://github.com/google-research/federated/blob/master/optimization).

### 模型和更新压缩

TFF 使用 [tensor_encoding](https://github.com/tensorflow/model-optimization/tree/master/tensorflow_model_optimization/python/core/internal/tensor_encoding) API 启用有损压缩算法，来降低服务器和客户端之间的通信成本。有关[使用联合平均算法](https://arxiv.org/abs/1812.07210)对服务器到客户端和客户端到服务器的压缩进行训练的示例，请参阅[此实验](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/compression/run_experiment.py)。

要实现自定义压缩算法并将其应用于训练循环，您可以进行以下操作：

1. Implement a new compression algorithm as a subclass of [`EncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L75) or its more general variant, [`AdaptiveEncodingStageInterface`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/encoding_stage.py#L274) following [this example](https://github.com/google-research/federated/blob/master/compression/sparsity.py).
2. Construct your new [`Encoder`](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/internal/tensor_encoding/core/core_encoder.py#L38) and specialize it for [model broadcast](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L118) or [model update averaging](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L144).
3. 使用这些对象来构建整个[训练计算](https://github.com/google-research/federated/blob/master/compression/run_experiment.py#L247)。

### 差分隐私

TFF 可与 [TensorFlow 隐私](https://github.com/tensorflow/privacy)库互操作，以研究新的算法，从而对使用差分隐私的模型进行联合训练。有关使用[基本 DP-FedAvg 算法](https://arxiv.org/abs/1710.06963)和[扩展程序](https://arxiv.org/abs/1812.06210)进行 DP 训练的示例，请参阅[此实验驱动程序](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/differential_privacy/stackoverflow/run_federated.py)。

If you want to implement a custom DP algorithm and apply it to the aggregate updates of federated averaging, you can implement a new DP mean algorithm as a subclass of [`tensorflow_privacy.DPQuery`](https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/dp_query.py#L54) and create a `tff.aggregators.DifferentiallyPrivateFactory` with an instance of your query. An example of implementing the [DP-FTRL algorithm](https://arxiv.org/abs/2103.00039) can be found [here](https://github.com/google-research/federated/blob/master/dp_ftrl/dp_fedavg.py)

联合 GAN（[如下](#generative_adversarial_networks)所述）是 TFF 项目的另一个示例，它实现了用户级别的差分隐私（例如[此处的代码](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L293)）。

### 鲁棒性和攻击

TFF 还可以用于模拟联合学习系统上的针对性攻击以及 *[Can You Really Back door Federated Learning?](https://arxiv.org/abs/1911.07963)* 中所考虑的基于差分隐私的防御。这是通过使用潜在的恶意客户端构建迭代过程来实现的（请参阅 [`build_federated_averaging_process_attacked`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/attacked_fedavg.py#L412)）。[targeted_attack](https://github.com/tensorflow/federated/tree/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack) 目录中包含更多详细信息。

- 新的攻击算法可以通过编写客户端更新函数来实现，该函数是 Tensorflow 函数。有关示例请参阅 [`ClientProjectBoost`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/federated_research/targeted_attack/attacked_fedavg.py#L460)。
- 新的防御可通过自定义 ['tff.utils.StatefulAggregateFn'](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/core/utils/computation_utils.py#L103)（聚合客户端输出以获得全局更新）来实现。

有关模拟的示例脚本，请参阅 [`emnist_with_targeted_attack.py`](https://github.com/tensorflow/federated/blob/6477a3dba6e7d852191bfd733f651fad84b82eab/tensorflow_federated/python/research/targeted_attack/emnist_with_targeted_attack.py)。

### 生成对抗网络

GAN 提供了一种有趣的[联合编排模式](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/gans/tff_gans.py#L266-L316)，这种模式看上去和标准的联合平均略有不同。它们涉及两种不同的网络（生成器和判别器），每种网络使用自己的优化步骤进行训练。

TFF can be used for research on federated training of GANs. For example, the DP-FedAvg-GAN algorithm presented in [recent work](https://arxiv.org/abs/1911.06679) is [implemented in TFF](https://github.com/tensorflow/federated/tree/main/federated_research/gans). This work demonstrates the effectiveness of combining federated learning, generative models, and [differential privacy](#differential_privacy).

### 个性化

联合学习设置中的个性化是一个活跃的研究领域。个性化的目的是为不同的用户提供不同的推理模型。此问题可能有不同的解决方法。

One approach is to let each client fine-tune a single global model (trained using federated learning) with their local data. This approach has connections to meta-learning, see, e.g., [this paper](https://arxiv.org/abs/1909.12488). An example of this approach is given in [`emnist_p13n_main.py`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/personalization/emnist_p13n_main.py). To explore and compare different personalization strategies, you can:

- Define a personalization strategy by implementing a `tf.function` that starts from an initial model, trains and evaluates a personalized model using each client's local datasets. An example is given by [`build_personalize_fn`](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/examples/personalization/p13n_utils.py).

- 定义一个 `OrderedDict`，将策略名称映射到相应的个性化策略，并将其用作 [`tff.learning.build_personalization_eval`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_personalization_eval) 中的 `personalize_fn_dict` 参数。

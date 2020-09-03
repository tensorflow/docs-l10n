# TensorFlow 白皮书

本文档介绍有关 TensorFlow 的白皮书。

## 异构分布式系统上的大规模机器学习

[访问此白皮书](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)。

**摘要：**TensorFlow 是一种用于表达机器学习算法的接口以及一种用于执行此类算法的实现。使用 TensorFlow 表达的计算可以在从移动设备（如手机和平板电脑）到大规模分布式系统（包括数百台机器和数千个计算设备，例如 GPU 卡）的各种异构系统上执行，而无需任何更改或只需极少的更改。这种系统非常灵活，可用于表达各种各样的算法（包括用于深度神经网络模型的训练和推理算法），并且已用于进行研究以及将机器学习系统部署到计算机科学等十几个领域中，包括语音识别、计算机视觉、机器人、信息检索、自然语言处理、地理信息提取和计算药物发现。本文介绍了 TensorFlow 接口以及我们在 Google 上构建的该接口的实现。2015 年 11 月，TensorFlow API 和参考实现已在 Apache 2.0 许可下以开源软件包的形式发布，可在 www.tensorflow.org 上获得。

### 以 BibTeX 格式

如果您在研究中使用 TensorFlow 并想引用 TensorFlow 系统，我们建议您引用此白皮书。

<pre>@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{\i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
</pre>

或以文本形式：

<pre>Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
</pre>

## TensorFlow：用于大规模机器学习的系统

[访问此白皮书](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)。

**摘要：**TensorFlow 是一种在大规模和异构环境中运行的机器学习系统。TensorFlow 使用数据流计算图来表示计算、共享状态以及使该状态发生突变的运算。它在集群中的许多机器之间以及一台机器中的多个计算设备之间映射数据流计算图的节点，这些计算设备包括多核 CPU、通用 GPU 和称为张量处理单元 (TPU) 的定制设计 ASIC。这种架构为应用开发者提供了很高的灵活性：在之前的“参数服务器”设计中，共享状态的管理已内置到系统中，TensorFlow 使开发者能够试验新颖的优化和训练算法。TensorFlow 支持各种应用，专注于在深度神经网络上进行训练和推理。一些 Google 服务在生产中使用 TensorFlow，我们已将其作为开源项目发布，并且它已广泛用于机器学习研究。在本白皮书中，我们介绍了 TensorFlow 数据流模型，并演示了 TensorFlow 在多种实际应用中实现的出色性能。

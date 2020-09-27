# Apache Beam 和 TFX

[Apache Beam](https://beam.apache.org/) 提供了一个框架，用于运行在各种执行引擎上运行的数据批处理和流处理作业。一些 TFX 库使用 Beam 运行任务，实现了跨计算集群的高度可扩展性。Beam 包含对各种执行引擎或“运行程序”的支持，其中包括在单个计算节点上运行的直接运行程序，这对于开发、测试或小型部署而言非常实用。Beam 提供了一个抽象层，使 TFX 无需修改代码便可在任何支持的运行程序上运行。TFX 使用 Beam Python API，因此仅适用于 Python API 支持的运行程序。

## 部署和可扩展性

随着工作负载要求的增加，Beam 可以扩展到跨大型计算集群的超大型部署。它的可扩展性仅受限于底层运行程序的可扩展性。大型部署中的运行程序通常将部署到诸如 Kubernetes 或 Apache Mesos 之类的容器编排系统中，实现应用部署、扩展和管理的自动化。

有关 Apache Beam 的更多信息，请参阅 [Apache Beam](https://beam.apache.org/) 文档。

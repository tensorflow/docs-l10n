# 编排 TFX 流水线

## Apache Beam

一些 TFX 组件依赖于 [Beam](beam.md) 进行分布式数据处理。此外，TFX 可以使用 Apache Beam 来编排和执行流水线 DAG。Beam 编排器使用不同于组件数据处理所用的 [BeamRunner](https://beam.apache.org/documentation/runners/capability-matrix/)。在默认 [DirectRunner](https://beam.apache.org/documentation/runners/direct/) 设置下，Beam 编排器可用于本地调试，而不会产生额外的 Airflow 或 Kubeflow 依赖项，这有助于简化系统配置。

有关详细信息，请参阅 [Beam 上的 TFX 样本](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py)。

# 编排 TFX 流水线

## 自定义编排器

TFX 被设计为可移植到多个环境和编排框架中。除了 TFX 支持的默认编排器（[Airflow](airflow.md)、[Beam](beam_orchestrator.md) 和 [Kubeflow](kubeflow.md)）之外，开发者可以创建自定义编排器或添加其他编排器。

所有编排器都必须从 [TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py) 继承。TFX 编排器接受逻辑流水线对象，该对象包含流水线参数、组件和 DAG，并负责根据 DAG 定义的依赖关系调度 TFX 流水线的组件。

例如，我们来看看如何使用 [ComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/component_launcher.py) 创建自定义编排器。ComponentLauncher 已经可以处理单个组件的驱动程序、执行器和发布程序。新的编排器只需根据 DAG 调度 ComponentLaunchers。这里提供了一个简单的编排器 [LocalDagRunner] (https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/local/local_dag_runner.py)，该编排器会按照 DAG 的拓扑顺序逐一运行组件。

此编排器可在 Python DSL 中使用：

```python
def _create_pipeline(...) -> dsl.Pipeline:
  ...
  return dsl.Pipeline(...)

if __name__ == '__main__':
  orchestration.LocalDagRunner().run(_create_pipeline(...))
```

要运行上面的 Python DSL 文件（假设它名为 dsl.py），只需运行以下代码：

```bash
python dsl.py
```

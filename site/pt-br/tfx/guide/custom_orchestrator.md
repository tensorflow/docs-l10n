# Orquestração de pipelines TFX

## Orquestrador personalizado

O TFX foi projetado para ser portável para múltiplos ambientes e frameworks de orquestração. Os desenvolvedores podem criar orquestradores personalizados ou incluir orquestradores adicionais além dos orquestradores padrão que já são suportados pelo TFX, que são [Local](local_orchestrator.md), [Vertex AI](vertex.md), [Airflow](airflow.md) e [Kubeflow](kubeflow.md).

Todos os orquestradores precisam herdar de [TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py). Os orquestradores TFX recebem o objeto lógico do pipeline, que contém argumentos do pipeline, componentes e DAG, e são responsáveis ​​por agendar componentes do pipeline TFX com base nas dependências definidas pelo DAG.

Por exemplo, vejamos como criar um orquestrador personalizado com [BaseComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/launcher/base_component_launcher.py). O BaseComponentLauncher já cuida do driver, da execução e publicação de um único componente. O novo orquestrador só precisa agendar ComponentLaunchers com base no DAG. Um orquestrador simples é fornecido como [LocalDagRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/local/local_dag_runner.py), que executa os componentes um por um na ordem topológica do DAG.

Este orquestrador pode ser usado na DSL Python:

```python
def _create_pipeline(...) -> dsl.Pipeline:
  ...
  return dsl.Pipeline(...)

if __name__ == '__main__':
  orchestration.LocalDagRunner().run(_create_pipeline(...))
```

Para executar o arquivo Python DSL acima (assumindo que seu nome seja dsl.py), basta fazer o seguinte:

```bash
import direct_runner
from tfx.orchestration import pipeline

def _create_pipeline(...) -> pipeline.Pipeline:
  ...
  return pipeline.Pipeline(...)

if __name__ == '__main__':
  direct_runner.DirectDagRunner().run(_create_pipeline(...))
```

# Cómo orquestar canalizaciones de TFX

## Orquestador personalizado

TFX se diseñó para que pueda adaptarse a varios entornos y marcos de orquestación. Los desarrolladores pueden crear orquestadores personalizados o agregar orquestadores adicionales además de los orquestadores predeterminados compatibles con TFX, a saber, [Local](local_orchestrator.md), [Vertex AI](vertex.md), [Airflow](airflow.md) y [Kubeflow](kubeflow.md).

Todos los orquestadores deben heredar de [TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py). Los orquestadores de TFX toman el objeto de canalización lógica, que contiene argumentos de canalización, componentes y DAG, y son responsables de programar los componentes de la canalización de TFX en función de las dependencias definidas por el DAG.

Por ejemplo, veamos cómo crear un orquestador personalizado con [BaseComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/launcher/base_component_launcher.py). BaseComponentLauncher ya maneja el controlador, el ejecutor y el editor de un único componente. El nuevo orquestador solo tiene que programar ComponentLaunchers según el DAG. Se proporciona un orquestador simple como [LocalDagRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/local/local_dag_runner.py), que ejecuta los componentes uno por uno en el orden topológico del DAG.

Este orquestador se puede utilizar en el DSL de Python:

```python
def _create_pipeline(...) -> dsl.Pipeline:
  ...
  return dsl.Pipeline(...)

if __name__ == '__main__':
  orchestration.LocalDagRunner().run(_create_pipeline(...))
```

Para ejecutar el archivo de DSL de Python mencionado anteriormente (suponiendo que se llame dsl.py), simplemente haga lo siguiente:

```bash
python dsl.py
```

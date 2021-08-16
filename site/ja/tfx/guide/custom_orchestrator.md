# TFX パイプラインのオーケストレーション

## カスタムオーケストレータ

TFX は、複数の環境とオーケストレーションフレームワークに移植できるように設計されており、開発者はカスタムオーケストレータを作成するか、TFX がサポートしているデフォルトのオーケストレータ（[Airflow](airflow.md)、[Beam](beam_orchestrator.md)、および [Kubeflow](kubeflow.md)）のほかにオーケストレータを追加することができます。

すべてのオーケストレータは [TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py) を継承している必要があります。TFX オーケストレータは論理パイプラインオブジェクトを取ります。このオブジェクトにはパイプライン引数、コンポーネント、および DAG が含まれており、TFX パイプラインのコンポーネントを DAG が定義する依存関係に基づいてスケジューリングを管理します。

例として、[ComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/component_launcher.py) でカスタムオーケストレータを作成する方法を見てみましょう。ComponentLauncher はすでに 1 つのコンポーネントのドライバ、executor、およびパブリッシャを処理するため、新しいオーケストレータでは DAG に基づいて ComponentLauncher をスケジューリングすることだけが必要です。単純なオーケストレータは、LocalDagRunner（https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/local/local_dag_runner.py）として提供されており、DAG のトポロジー順でコンポーネントを 1 つずつ実行します。

このオーケストレータは、Python DSL で次のように使用することができます。

```python
def _create_pipeline(...) -> dsl.Pipeline:
  ...
  return dsl.Pipeline(...)

if __name__ == '__main__':
  orchestration.LocalDagRunner().run(_create_pipeline(...))
```

上記の Python DSL ファイルを実行するには（名前を dsl.py とした場合）、単純に次のようにします。

```bash
python dsl.py
```

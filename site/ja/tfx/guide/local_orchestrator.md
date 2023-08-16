# TFX パイプラインのオーケストレーション

## ローカルオーケストレータ

ローカルオーケストレータは、TFX Python パッケージに含まれる単純なオーケストレータです。単一プロセスのローカル環境でパイプラインを実行します。開発やデバッグで高速イテレーションを実行できますが、大規模な本番ワークロードには適していません。本番でのユースケースには、[Vertex Pipelines](/tfx/guide/vertex) または [Kubeflow Pipelines](/tfx/guide/kubeflow) を使用してください。

ローカルオーケストレータの使用方法を学習するには、Colab で実行できる [TFX チュートリアル](/tfx/tutorials/tfx/penguin_simple)をお試しください。

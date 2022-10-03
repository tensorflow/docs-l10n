# TFX パイプラインのオーケストレーション

## Apache Beam

いくつかの TFX コンポーネントは、分散データ処理を [Beam](beam.md) に任せています。また、TFX は Apache Beam を使用して、パイプライン DAG のオーケストレーションと実行を行うことも可能です。Beam オーケストレータはコンポーネントのデータ処理に使用するものとは異なる [BeamRunner](https://beam.apache.org/documentation/runners/capability-matrix/) を使用します。デフォルトの [DirectRunner](https://beam.apache.org/documentation/runners/direct/) がセットアップされれば、追加の Airflow や Kuberflow 依存関係が発生することなくローカルデバッグに Beam オーケストレータを使用できるため、システム構成を簡略化できます。

詳細は、[Beam での TFX の例](https://blog.tensorflow.org/2020/03/tensorflow-extended-tfx-using-apache-beam-large-scale-data-processing.html)をご覧ください。

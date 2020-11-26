# TFX 파이프라인 조정하기

## Apache Beam

여러 TFX 구성 요소가 분산 데이터 처리를 위해 [Beam](beam.md)을 사용합니다. 또한, TFX는 Apache Beam을 사용하여 파이프라인 DAG를 오케스트레이션하고 실행할 수 있습니다. Beam 오케스트레이터는 구성 요소 데이터 처리에 사용되는 것과는 다른 [BeamRunner](https://beam.apache.org/documentation/runners/capability-matrix/)를 사용합니다. 기본 [DirectRunner](https://beam.apache.org/documentation/runners/direct/) 설정을 사용하면 추가 Airflow 또는 Kubeflow 종속성을 발생시키지 않고 Beam 오케스트레이터를 로컬 디버깅에 사용할 수 있으므로 시스템 구성이 간소화됩니다.

자세한 내용은 [Beam에서 TFX 예](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py)를 참조하세요.

# TFX 파이프라인 조정하기

## 사용자 정의 오케스트레이터

TFX는 여러 환경 및 오케스트레이션 프레임워크로 이식할 수 있도록 설계되었습니다. 개발자는 TFX에서 지원하는 기본 오케스트레이터, 즉 [Airflow](airflow.md), [Beam](beam_orchestrator.md) 및 [Kubeflow](kubeflow.md) 외에 사용자 정의 오케스트레이터를 만들거나 별도의 오케스트레이터를 추가할 수 있습니다.

모든 오케스트레이터는 [TfxRunner](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/tfx_runner.py)에서 상속해야 합니다. TFX 오케스트레이터는 파이프라인 인수, 구성 요소 및 DAG를 포함하는 논리적 파이프라인 객체를 입력으로 받고 DAG에서 정의한 종속성을 기반으로 TFX 파이프라인의 구성 요소를 스케줄링합니다.

예를 들어, [ComponentLauncher](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/component_launcher.py)를 사용하여 사용자 정의 오케스트레이터를 만드는 방법을 살펴보겠습니다. ComponentLauncher는 이미 단일 구성 요소의 드라이버, 실행기 및 게시자를 처리합니다. 새 오케스트레이터는 DAG를 기반으로 ComponentLauncher를 스케줄링하기만 하면 됩니다. 다음 예제는 DAG의 토폴로지 순서에 따라 구성 요소를 하나씩 실행하는 간단한 장난감 오케스트레이터를 보여줍니다.

```python
import datetime

from tfx.orchestration import component_launcher
from tfx.orchestration import data_types
from tfx.orchestration import tfx_runner

class DirectDagRunner(tfx_runner.TfxRunner):
  """Tfx direct DAG runner."""

  def run(self, pipeline):
    """Directly run components in topological order."""
    # Run id is needed for each run.
    pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    # pipeline.components are in topological order already.
    for component in pipeline.components:
      component_launcher.ComponentLauncher(
          component=component,
          pipeline_info=pipeline.pipeline_info,
          driver_args=data_types.DriverArgs(
              enable_cache=pipeline.enable_cache),
          metadata_connection_config=pipeline.metadata_connection_config,
          additional_pipeline_args=pipeline.additional_pipeline_args
      ).launch()
```

위의 오케스트레이터는 Python DSL에서 사용할 수 있습니다.

```python
import direct_runner
from tfx.orchestration import pipeline

def _create_pipeline(...) -> pipeline.Pipeline:
  ...
  return pipeline.Pipeline(...)

if __name__ == '__main__':
  direct_runner.DirectDagRunner().run(_create_pipeline(...))
```

위의 Python DSL 파일(이름이 dsl.py라고 가정)을 실행하려면 간단히 다음을 수행합니다.

```bash
python dsl.py
```

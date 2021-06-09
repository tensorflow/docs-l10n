# Fairness Indicators 대시보드를 사용하여 모델 평가하기 [베타]

![공정성 지표](./images/fairness-indicators.png)

TensorBoard Fairness Indicators를 사용하면 *이진* 및 *다중 클래스* 분류자에서 일반적으로 식별되는 공정성 메트릭을 쉽게 계산할 수 있습니다. 플러그인을 사용하면 실행에 대한 공정성 평가를 시각화하고 그룹 간에 성능을 쉽게 비교할 수 있습니다.

특히 TensorBoard Fairness Indicators를 사용하면 정의된 전체 사용자 그룹에서 세분화된 모델 성능을 평가하고 시각화할 수 있습니다. 신뢰 구간과 여러 임계값을 사용한 평가에서 결과에 대한 확신을 높이세요.

공정성 문제를 평가하기 위한 기존의 많은 도구는 대규모 데이터세트 및 모델에서 제대로 동작하지 않습니다. Google에서는 수십억 명의 사용자가 있는 시스템에서 동작할 수 있는 도구를 확보하는 것이 중요합니다. Fairness Indicators를 이용하면 TensorBoard 환경 또는 [Colab](https://github.com/tensorflow/fairness-indicators)에서 어떤 크기의 사용 사례든 평가할 수 있습니다.

## 요구 사항

TensorBoard Fairness Indicators를 설치하려면 다음을 실행합니다.

```
python3 -m virtualenv ~/tensorboard_demo
source ~/tensorboard_demo/bin/activate
pip install --upgrade pip
pip install fairness_indicators
pip install tensorboard-plugin-fairness-indicators
```

## 데모

TensorBoard에서 Fairness Indicators를 테스트하려면 Google 클라우드 플랫폼에서 TensorFlow 모델 분석의 평가 결과 샘플(eval_config.json, 메트릭 및 플롯 파일)과 `demo.py` 유틸리티를 다운로드할 수 있습니다(다음 명령을 사용하여 [여기](https://console.cloud.google.com/storage/browser/tensorboard_plugin_fairness_indicators/)에서 다운로드).

```
pip install gsutil
gsutil cp -r gs://tensorboard_plugin_fairness_indicators/ .
```

다운로드한 파일이 있는 디렉토리로 이동합니다.

```
cd tensorboard_plugin_fairness_indicators
```

이 평가 데이터는 Tensorflow 모델 분석의 [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py) 라이브러리를 사용하여 계산된 [Civil Comments 데이터세트](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)를 기반으로 합니다. 여기에는 참조를 위한 TensorBoard 요약 데이터 샘플 파일도 포함되어 있습니다.

`demo.py` 유틸리티는 TensorBoard 요약 데이터 파일을 작성합니다. 그러면 TensorBoard에서 이 파일을 읽어 Fairness Indicators 대시보드를 렌더링합니다(요약 데이터 파일에 대한 자세한 내용은 [TensorBoard 튜토리얼](https://github.com/tensorflow/tensorboard/blob/master/README.md) 참조).

다음은 `demo.py` 유틸리티와 함께 사용되는 플래그입니다.

- `--logdir`: TensorBoard가 요약을 작성하는 디렉토리
- `--eval_result_output_dir`: TFMA(마지막 단계에서 다운로드됨)에서 평가한 평가 결과를 포함한 디렉토리

`demo.py` 유틸리티를 실행하여 로그 디렉토리에 요약 결과를 작성합니다.

`python demo.py --logdir=. --eval_result_output_dir=.`

TensorBoard를 실행합니다.

참고: 이 데모의 경우, 다운로드한 모든 파일이 들어 있는 동일한 디렉토리에서 TensorBoard를 실행하세요.

`tensorboard --logdir=.`

로컬 인스턴스가 시작됩니다. 로컬 인스턴스가 시작되면 단말기에 링크가 표시됩니다. 브라우저에서 링크를 열어 Fairness Indicators 대시보드를 봅니다.

### 데모 Colab

[Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TensorBoard_Plugin_Example_Colab.ipynb)에는 모델을 훈련 및 평가하고 TensorBoard에서 공정성 평가 결과를 시각화하기 위한 엔드 투 엔드 데모가 포함되어 있습니다.

## 사용법

고유 데이터와 평가에 Fairness Indicators를 사용하려면:

1. [model_eval_lib](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/api/model_eval_lib.py)에 있는 `tensorflow_model_analysis.run_model_analysis` 또는 `tensorflow_model_analysis.ExtractEvaluateAndWriteResult` API를 사용하여 새 모델을 훈련하고 평가합니다. 방법을 보여주는 코드 조각은 [여기](https://github.com/tensorflow/fairness-indicators)에서 Fairness Indicators colab을 참조하세요.

2. `tensorboard_plugin_fairness_indicators.summary_v2` API를 사용하여 Fairness Indicators 요약을 작성합니다.

    ```
    writer = tf.summary.create_file_writer(<logdir>)
    with writer.as_default():
        summary_v2.FairnessIndicators(<eval_result_dir>, step=1)
    writer.close()
    ```

3. TensorBoard를 실행합니다.

    - `tensorboard --logdir=<logdir>`
    - 대시보드 왼쪽에 있는 드롭다운에서 새 평가 실행을 선택하여 결과를 시각화합니다.

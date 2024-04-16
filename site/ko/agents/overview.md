# TensorFlow 에이전트

**TensorFlow를 이용한 강화 학습(Reinforcement Learning)**

에이전트는 수정 및 확장 할 수있는 잘 테스트 된 모듈 식 구성 요소를 제공하여 새로운 RL 알고리즘을보다 쉽게 설계, 구현 및 테스트 할 수 있습니다. 우수한 테스트 통합 및 벤치마킹으로 빠른 코드 반복이 가능합니다.

시작하려면 [튜토리얼](/tutorials) 중 하나를 확인하는 것이 좋습니다.

## 설치

TF-Agent는 야간 및 안정적인 빌드를 게시합니다. 릴리즈의 목록은 읽기 <a href="#Releases">릴리즈</a> 섹션을 참조합니다. 아래 명령을 통해 TF-Agent 야간 및 안정적인 버전을 [pypi.org](https://pypi.org)와 GitHub 클론에서 설치하는 방법을 설명합니다.

> 경고: 매우 일반적인 Reverb(리플레이 버퍼)를 사용하는 경우 TF-Agents는 Linux에서만 작동합니다.

> 참고: Python 3.11은 파이게임 2.1.3 이상을 필요로 합니다.

### 안정적인 빌드

가장 최근의 안정적인 릴리스를 설치하려면 아래 명령을 실행하세요. 릴리스에 대한 API 문서는 [tensorflow.org](https://www.tensorflow.org/agents/api_docs/python/tf_agents)에 있습니다.

```shell
$ pip install --user tf-agents[reverb]

# Use this tag get the matching examples and colabs.
$ git clone https://github.com/tensorflow/agents.git
$ cd agents
$ git checkout v0.9.0
```

pip 종속성 검사로 호환되지 않는 것으로 플래그가 지정된 Tensorflow 또는 [Reverb](https://github.com/deepmind/reverb) 버전의 TF-Agent를 설치하려면 자신의 책임 하에 아래의 다음 패턴을 사용하세요.

```shell
$ pip install --user tensorflow
$ pip install --user dm-reverb
$ pip install --user tf-agents
```

TensorFlow 1.15 또는 2.0과 함께 TF-Agents를 사용하려면 버전 0.3.0을 설치합니다.

```shell
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
$ pip install tensorflow-probability==0.8.0
$ pip install tf-agents==0.3.0
```

### 야간 빌드

야간 빌드에는 새로운 기능이 포함되어 있지만, 버전 릴리스보다 안정성이 떨어질 수 있습니다. 야간 빌드는 `tf-agents-nightly`로 푸시됩니다. 야간 버전의 TensorFlow (`tf-nightly{/ code1}) 및 TensorFlow Probability (>tfp-nightly`)는 야간 TF-Agent 버전과 비교하여 설치하는 것이 좋습니다.

야간 빌드 버전을 설치하려면 다음을 실행하십시오.

```shell
# `--force-reinstall helps guarantee the right versions.
$ pip install --user --force-reinstall tf-nightly
$ pip install --user --force-reinstall tfp-nightly
$ pip install --user --force-reinstall dm-reverb-nightly

# Installing with the `--upgrade` flag ensures you'll get the latest version.
$ pip install --user --upgrade tf-agents-nightly
```

### GitHub에서 복제하기

리포지토리를 복제한 후 `pip install -e .[tests]`를 실행하여 종속성을 설치할 수 있습니다. TensorFlow는 독립적으로 설치해야 합니다. `pip install --user tf-nightly`

<a id="Contributing"></a>

## 기여하기

당사는 여러분과 협력하길 원합니다! 기여 방법에 대한 지침은 [`CONTRIBUTING.md`](https://github.com/tensorflow/agents/blob/master/CONTRIBUTING.md)를 참조하십시오. 이 프로젝트는 TensorFlow의 [행동 강령](https://github.com/tensorflow/agents/blob/master/CODE_OF_CONDUCT.md)을 준수합니다. 참여할 때는 해당 행동 강령을 준수해야 합니다.

<a id="Releases"></a>

## 릴리즈

TF Agent에는 안정적인 나이틀리 릴리스가 있습니다. 나이틀리 릴리스는 대체적으로 훌륭하지만 유동적인 업스트림 라이브러리로 인해 문제가 발생할 수 있습니다. 아래 표에는 각 TF Agent 릴리스와 일치하는 TensorFlow 버전이 나와 있습니다. 관심 받는 릴리스 버전:

- 0.16.0은 Python 3.11을 지원하는 첫 번째 버전입니다.
- 0.15.0은 Python 3.7과 호환되는 마지막 릴리스입니다.
- 1.19 미만의 numpy를 사용하는 경우 TF Agents 0.15.0 이하 버전을 사용합니다.
- 0.9.0은 Python 3.6과 호환되는 마지막 릴리스입니다.
- 0.3.0은 Python 2.x와 호환되는 마지막 릴리스입니다.

릴리즈 | 분기/태그 | TensorFlow 버전 | dm-reverb Version
--- | --- | --- | ---
야간 | [master](https://github.com/tensorflow/agents) | tf-nightly | dm-reverb-nightly
0.17.0 | [v0.17.0](https://github.com/tensorflow/agents/tree/v0.17.0) | 2.13.0 | 0.12.0
0.16.0 | [v0.16.0](https://github.com/tensorflow/agents/tree/v0.16.0) | 2.12.0 | 0.11.0
0.15.0 | [v0.15.0](https://github.com/tensorflow/agents/tree/v0.15.0) | 2.11.0 | 0.10.0
0.14.0 | [v0.14.0](https://github.com/tensorflow/agents/tree/v0.14.0) | 2.10.0 | 0.9.0
0.13.0 | [v0.13.0](https://github.com/tensorflow/agents/tree/v0.13.0) | 2.9.0 | 0.8.0
0.12.0 | [v0.12.0](https://github.com/tensorflow/agents/tree/v0.12.0) | 2.8.0 | 0.7.0
0.11.0 | [v0.11.0](https://github.com/tensorflow/agents/tree/v0.11.0) | 2.7.0 | 0.6.0
0.10.0 | [v0.10.0](https://github.com/tensorflow/agents/tree/v0.10.0) | 2.6.0 |
0.9.0 | [v0.9.0](https://github.com/tensorflow/agents/tree/v0.9.0) | 2.6.0 |
0.8.0 | [v0.8.0](https://github.com/tensorflow/agents/tree/v0.8.0) | 2.5.0 |
0.7.1 | [v0.7.1](https://github.com/tensorflow/agents/tree/v0.7.1) | 2.4.0 |
0.6.0 | [v0.6.0](https://github.com/tensorflow/agents/tree/v0.6.0) | 2.3.0 |
0.5.0 | [v0.5.0](https://github.com/tensorflow/agents/tree/v0.5.0) | 2.2.0 |
0.4.0 | [v0.4.0](https://github.com/tensorflow/agents/tree/v0.4.0) | 2.1.0 |
0.3.0 | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 and 2.0.0. |

<a id="Principles"></a>

## 원칙

이 프로젝트는 [Google의 AI 원칙](https://github.com/tensorflow/agents/blob/master/PRINCIPLES.md)을 준수합니다. 이 프로젝트에 참여, 사용 또는 기여함으로써 사용자는 이러한 원칙을 준수해야 합니다.

<a id="Citation"></a>

## 인용

이 코드를 사용하는 경우, 다음과 같이 인용하세요.

```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = {Sergio Guadarrama and Anoop Korattikara and Oscar Ramirez and
     Pablo Castro and Ethan Holly and Sam Fishman and Ke Wang and
     Ekaterina Gonina and Neal Wu and Efi Kokiopoulou and Luciano Sbaiz and
     Jamie Smith and Gábor Bartók and Jesse Berent and Chris Harris and
     Vincent Vanhoucke and Eugene Brevdo},
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```

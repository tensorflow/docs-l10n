# Citing TensorFlow

TensorFlow publishes a DOI for the open-source code base using Zenodo.org: [10.5281/zenodo.4724125](https://doi.org/10.5281/zenodo.4724125)

[백서에 액세스하세요.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

## 이기종 분산 시스템에 대한 대규모 머신러닝

[Access this white paper.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**개요:** TensorFlow는 머신러닝 알고리즘을 표현하기 위한 인터페이스이자 이러한 알고리즘을 실행하기 위한 구현입니다. TensorFlow를 사용하여 표현된 계산은 전화 및 태블릿과 같은 모바일 기기에서 수백 대 머신과 GPU 카드와 같은 수천 대의 계산 기기로 이루어진 대규모 분산 시스템에 이르는 광범위한 이기종 시스템에서 거의 변경하지 않거나 전혀 변경하지 않고 실행될 수 있습니다. 이 시스템은 유연하고, 심층 신경망 모델을 위한 훈련 및 추론 알고리즘을 포함하여 다양한 알고리즘을 표현하는 데 사용될 수 있으며, 음성 인식, 컴퓨터 비전, 로봇 공학, 정보 검색, 자연어 처리, 지리 정보 추출 및 계산 약물 발견을 포함한 여러 컴퓨터 과학 및 기타 분야에 걸쳐 연구를 수행하고 머신러닝 시스템을 운영 환경에 배포하는 데 사용되었습니다. 본 백서에는 TensorFlow 인터페이스와 Google에서 구축한 해당 인터페이스의 구현에 대한 설명이 제공되어 있습니다. TensorFlow API 및 참조 구현은 2015년 11월 Apache 2.0 라이선스에 따라 오픈 소스 패키지로 출시되었으며 www.tensorflow.org에서 제공됩니다.

### BibTeX 형식

연구에 TensorFlow를 사용하고 TensorFlow 시스템을 인용하려면, 다음 백서를 인용하세요.

<pre>@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
</pre>

또는 텍스트 형식으로:

<pre>Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
</pre>

## TensorFlow: 대규모 머신러닝 시스템

[Access this white paper.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**개요:** TensorFlow는 대규모 및 이기종 환경에서 작동하는 머신러닝 시스템입니다. TensorFlow는 데이터 흐름 그래프를 사용하여 계산, 공유 상태 및 해당 상태를 변경하는 연산을 나타냅니다. 클러스터의 여러 머신과 멀티 코어 CPU, 범용 GPU 및 Tensor Processing Units(TPU)로 알려진 맞춤 설계 ASIC를 포함한 여러 계산 기기의 머신 내에서 데이터 흐름 그래프의 노드를 매핑합니다. 이 아키텍처는 애플리케이션 개발자에게 유연성을 제공합니다. 이전의 "매개변수 서버" 설계에서는 공유 상태 관리 기능이 시스템에 내장되어 있으나, 개발자는 TensorFlow를 통해 새로운 최적화 및 훈련 알고리즘을 실험할 수 있습니다. TensorFlow는 심층 신경망에서 훈련 및 추론에 중점을 둔 다양한 애플리케이션을 지원합니다. 일부 Google 서비스는 운영 환경에서 TensorFlow를 사용하고 오픈 소스 프로젝트로 출시했으며, 머신러닝 연구에 광범위하게 사용했습니다. 이 백서에서는 TensorFlow 데이터 흐름 모델을 설명하고 여러 실제 애플리케이션에서 TensorFlow가 달성한 강력한 성능을 보여줍니다.

# TensorFlow 백서

이 문서에서는 TensorFlow에 대한 백서를 확인합니다.

## 이기종 분산 시스템에 대한 대규모 머신러닝

[백서에 액세스하세요.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**개요:** TensorFlow는 머신러닝 알고리즘을 표현하기 위한 인터페이스이자 이러한 알고리즘을 실행하기 위한 구현입니다. TensorFlow를 사용하여 표현된 계산은 전화 및 태블릿과 같은 모바일 기기에서 수백 대 머신과 GPU 카드와 같은 수천 대의 계산 기기로 이루어진 대규모 시스템에 이르는 광범위한 이기종 시스템에서 거의 또는 전혀 변경 없이 실행될 수 있습니다. 이 시스템은 유연하고, 심층 신경망 모델을 위한 훈련 및 추론 알고리즘을 포함하여 다양한 알고리즘을 표현하는 데 사용될 수 있으며, 음성 인식, 컴퓨터 비전, 로봇 공학, 정보 검색, 자연어 처리, 지리 정보 추출 및 계산 약물 발견을 포함한 여러 컴퓨터 과학 및 기타 분야에 걸쳐 연구를 수행하고 머신러닝 시스템을 운영 환경에 배포하는 데 사용되었습니다. 본 백서는 TensorFlow 인터페이스와 Google에서 구축한 해당 인터페이스의 구현에 대해 설명합니다. TensorFlow API 및 참조 구현은 2015년 11월 Apache 2.0 라이선스에 따라 오픈 소스 패키지로 출시되었으며 www.tensorflow.org에서 제공됩니다.

### BibTeX 형식

연구에 TensorFlow를 사용하고 TensorFlow 시스템을 인용하려면, 다음 백서를 인용하세요.

<pre>@misc{tensorflow2015-whitepaper,<br>title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},<br>url={https://www.tensorflow.org/},<br>note={Software available from tensorflow.org},<br>author={<br>    Mart\'{\i}n~Abadi and<br>    Ashish~Agarwal and<br>    Paul~Barham and<br>    Eugene~Brevdo and<br>    Zhifeng~Chen and<br>    Craig~Citro and<br>    Greg~S.~Corrado and<br>    Andy~Davis and<br>    Jeffrey~Dean and<br>    Matthieu~Devin and<br>    Sanjay~Ghemawat and<br>    Ian~Goodfellow and<br>    Andrew~Harp and<br>    Geoffrey~Irving and<br>    Michael~Isard and<br>    Yangqing Jia and<br>    Rafal~Jozefowicz and<br>    Lukasz~Kaiser and<br>    Manjunath~Kudlur and<br>    Josh~Levenberg and<br>    Dandelion~Man\'{e} and<br>    Rajat~Monga and<br>    Sherry~Moore and<br>    Derek~Murray and<br>    Chris~Olah and<br>    Mike~Schuster and<br>    Jonathon~Shlens and<br>    Benoit~Steiner and<br>    Ilya~Sutskever and<br>    Kunal~Talwar and<br>    Paul~Tucker and<br>    Vincent~Vanhoucke and<br>    Vijay~Vasudevan and<br>    Fernanda~Vi\'{e}gas and<br>    Oriol~Vinyals and<br>    Pete~Warden and<br>    Martin~Wattenberg and<br>    Martin~Wicke and<br>    Yuan~Yu and<br>    Xiaoqiang~Zheng},<br>  year={2015},<br>}</pre>

또는 텍스트 형식으로 인용하세요.

<pre>Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,<br>Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,<br>Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,<br>Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,<br>Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,<br>Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,<br>Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,<br>Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,<br>Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,<br>Yuan Yu, and Xiaoqiang Zheng.<br>TensorFlow: Large-scale machine learning on heterogeneous systems,<br>2015. Software available from tensorflow.org.</pre>

## TensorFlow: 대규모 머신러닝 시스템

[백서에 액세스하세요.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**개요:** TensorFlow는 대규모 및 이기종 환경에서 작동하는 머신러닝 시스템입니다. TensorFlow는 데이터 흐름 그래프를 사용하여 계산, 공유 상태 및 해당 상태를 변경하는 연산을 나타냅니다. 클러스터의 여러 머신과 멀티 코어 CPU, 범용 GPU 및 Tensor Processing Units(TPU)로 알려진 맞춤 설계 ASIC를 포함한 여러 계산 기기의 머신 내에서 데이터 흐름 그래프의 노드를 매핑합니다. 이 아키텍처는 애플리케이션 개발자에게 유연성을 제공합니다. 이전의 "매개변수 서버" 설계에서는 공유 상태의 관리가 시스템에 내장되어 있으나, TensorFlow에서는 개발자가 새로운 최적화 및 훈련 알고리즘을 실험할 수 있습니다. TensorFlow는 심층 신경망에서 훈련 및 추론에 중점을 둔 다양한 애플리케이션을 지원합니다. 일부 Google 서비스는 운영 환경에서 TensorFlow를 사용하고 오픈 소스 프로젝트로 출시했으며, 머신러닝 연구에 광범위하게 사용했습니다. 이 백서에서는 TensorFlow 데이터 흐름 모델을 설명하고 여러 실제 애플리케이션에서 TensorFlow의 강력한 성능을 보여줍니다.

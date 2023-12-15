# Citando o TensorFlow

O TensorFlow publica um DOI para a base de código aberto usando Zenodo.org: [10.5281/zenodo.4724125](https://doi.org/10.5281/zenodo.4724125)

Os documentos técnicos do TensorFlow são listados para citação abaixo.

## Aprendizado de máquina em grande escala em sistemas distribuídos heterogêneos

[Acesse este documento técnico.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**Resumo:** O TensorFlow é uma interface para expressar algoritmos de aprendizado de máquina e uma implementação para executar esses algoritmos. Uma computação expressa usando o TensorFlow pode ser executada com pouca ou nenhuma alteração em uma grande variedade de sistemas heterogêneos, desde dispositivos móveis, como smartphones e tablets, até sistemas distribuídos em grande escala, com centenas de máquinas e milhares de dispositivos computacionais, como placas GPU. O sistema é flexível e pode ser usado para expressar uma ampla gama de algoritmos, incluindo os de treinamento e inferência para modelos de redes neurais profundas, e tem sido usado para conduzir pesquisas e implantar sistemas de aprendizado de máquina para produção em mais de uma dúzia de áreas da ciência da computação e outros campos, incluindo reconhecimento de fala, visão computacional, robótica, recuperação de informações, processamento de linguagem natural, extração de informações geográficas e descoberta computacional de medicamentos. Esta documentação descreve a interface do TensorFlow e uma implementação dessa interface que criamos no Google. A API do TensorFlow e uma implementação de referência foram publicadas como um pacote de código aberto sob a licença Apache 2.0 em novembro de 2015, que está disponível em www.tensorflow.org.

### No formato BibTeX

Se você usa o TensorFlow na sua pesquisa acadêmica e quer citar o sistema TensorFlow, recomendamos citar este documento.

<pre>
@misc{tensorflow2015-whitepaper,
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

Ou em formato de texto:

<pre>
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
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

## TensorFlow: um sistema para o aprendizado de máquina em grande escala

[Acesse este documento técnico.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**Resumo:** O TensorFlow é um sistema de aprendizado de máquina que opera em grande escala e em ambientes heterogêneos. O TensorFlow usa grafos de fluxo de dados para representar computações, estado compartilhado e operações que transformam esse estado. Ele mapeia os nós de um grafo de fluxo de dados em várias máquinas de um cluster e dentro de uma máquina em vários dispositivos computacionais, incluindo CPUs multicore, GPUs de uso geral e ASICs personalizados conhecidos como Unidades de Processamento de Tensor (TPUs). Essa arquitetura dá flexibilidade ao desenvolvedor do aplicativo: enquanto o gerenciamento de estado compartilhado é integrado ao sistema nos designs de "servidores parametrizados" anteriores, o TensorFlow permite que os desenvolvedores testem novas otimizações e algoritmos de treinamento. O TensorFlow oferece suporte a uma variedade de aplicativos, com foco no treinamento e na inferência de redes neurais profundas. Vários serviços do Google usam o TensorFlow em produção, que foi lançado como um projeto de código aberto e se tornou amplamente usado em pesquisas com aprendizado de máquina. Neste documento, descrevemos o modelo de fluxo de dados do TensorFlow e mostramos o desempenho convincente do TensorFlow em várias aplicações no mundo real.

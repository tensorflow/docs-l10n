# Classificação de vídeo

<img src="../images/video.png" class="attempt-right">

*Classificação de vídeo* é a tarefa de aprendizado de máquina para identificar o que um vídeo representa. Um modelo de classificação de vídeo é treinado com um dataset de vídeos que contém um conjunto de classes únicas, como ações ou movimentos diferentes. O modelo recebe quadros de vídeo como entrada e gera como saída a probabilidade de cada classe estar representada no vídeo.

Os modelos de classificação de vídeo e de classificação de imagem usam imagens como entrada para prever as probabilidades de elas pertencerem a classes predefinidas. Entretanto, um modelo de classificação de vídeo também processa as relações espaciais e temporais entre quadros adjacentes para reconhecer ações em um vídeo.

Por exemplo, um modelo de *reconhecimento de ações em vídeos* pode ser treinado para identificar ações humanas, como correr, bater palmas ou acenar. A imagem abaixo mostra a saída de um modelo de classificação de vídeo no Android.

<img src="https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/push-up-classification.gif" alt="Screenshot of Android example">

## Como começar

Se você estiver usando outra plataforma que não o Android ou Raspberry Pi ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), baixe o modelo inicial de classificação de vídeo e os arquivos de suporte. Você também pode criar seu próprio pipeline de inferência personalizado usando a [TensorFlow Lite Support Library](../../inference_with_metadata/lite_support) (biblioteca de suporte do TensorFlow Lite).

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/movinet/a0/stream/kinetics-600/classification/tflite/int8/1">Baixe o modelo inicial com metadados</a>

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou Raspberry PI, confira os exemplos de aplicativo abaixo que podem te ajudar a começar.

### Android

O aplicativo para Android usa a câmera traseira do Android para classificação contínua de vídeo. A inferência é feita usando a [TensorFlow Lite Java API](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/package-summary) (API Java do TensorFlow Lite). O aplicativo de demonstração classifica quadros e exibe as classificações previstas em tempo real.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/android">Exemplo do Android</a>

### Raspberry Pi

O exemplo de Raspberry usa o TensorFlow Lite com o Python para fazer classificação contínua de vídeo. Conecte o Raspberry Pi a uma câmera, como a Pi Camera, para fazer classificação de vídeo em tempo real. Para ver os resultados da câmera, conecte um monitor ao Raspberry Pi e use SSH para acessar o shell do Pi (para evitar conectar um teclado ao Pi).

Antes de começar, [configure](https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) seu Raspberry Pi com o Raspberry Pi OS (atualizado para o Buster, preferencialmente).

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/raspberry_pi%20">Exemplo com o Raspberry Pi</a>

## Descrição do modelo

As Redes de Vídeo Móveis ([MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet)) são uma família de modelos eficientes de classificação de vídeo otimizados para dispositivos móveis. As MoViNets têm exatidão e eficiência de última geração em diversos datasets de reconhecimento de ações em vídeos, sendo, portanto, muito adequadas para tarefas de *reconhecimento de ações em vídeos*.

Existem três variantes do modelo [MoviNet](https://tfhub.dev/s?deployment-format=lite&q=movinet) para o TensorFlow Lite: [MoviNet-A0](https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification), [MoviNet-A1](https://tfhub.dev/tensorflow/movinet/a1/stream/kinetics-600/classification) e [MoviNet-A2](https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification). Essas variantes foram treinadas com o dataset [Kinetics-600](https://arxiv.org/abs/1808.01340) para reconhecer 600 ações humanas diferentes. *MoviNet-A0* é a variante menor, mais rápida e menos exata. *MoviNet-A2* é a variante maior, mais lenta e mais exata. *MoviNet-A1* fica entre a A0 e a A2.

### Como funciona

Durante o treinamento, um modelo de classificação de vídeo recebe vídeos e seus respectivos *rótulos*. Cada rótulo é o nome de um conceito distinto, ou classe, que o modelo aprenderá a reconhecer. Para *reconhecimento de ações em vídeos*, os vídeos são de ações humanas, e os rótulos são suas respectivas ações.

O modelo de classificação de vídeo pode aprender a prever se novos vídeos pertencem a alguma das classes fornecidas durante o treinamento. Esse processo é chamado de *inferência*. Você também pode usar o [aprendizado por transferência](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb) para identificar novas classes de vídeos usando um modelo preexistente.

É um modelo de streaming que recebe um vídeo continuamente e responde em tempo real. À medida que o modelo recebe um streaming de vídeo, identifica se alguma das classes do dataset de treinamento está representada nele. Para cada quadro, o modelo retorna essas classes, junto com a probabilidade de o vídeo representar a classe. Confira abaixo um exemplo de saída em um determinado momento:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Ação</th>
      <th>Probabilidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dançando quadrilha</td>
      <td>0,02</td>
    </tr>
    <tr>
      <td>Enfiando uma agulha</td>
      <td>0,08</td>
    </tr>
    <tr>
      <td>Girando os dedos</td>
      <td>0,23</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Acenando com a mão</td>
      <td style="background-color: #fcb66d;">0,67</td>
    </tr>
  </tbody>
</table>

Cada ação na saída corresponde a um rótulo nos dados de treinamento. A probabilidade indica a chance de a ação estar sendo exibida no vídeo.

### Entradas do modelo

O modelo recebe um streaming de quadros de vídeo RGB como entrada. O tamanho do vídeo de entrada é flexível, mas, idealmente, deve coincidir com a resolução e a taxa de quadros usadas no treinamento do modelo:

- **MoviNet-A0**: 172 x 172 a 5 qps
- **MoviNet-A1**: 172 x 172 a 5 qps
- **MoviNet-A2**: 224 x 224 a 5 qps

Espera-se que os vídeos de entrada tenham valores de cores no intervalo de 0 a 1, seguindo as [convenções comuns de imagens usadas como entrada](https://www.tensorflow.org/hub/common_signatures/images#input).

Internamente, o modelo também analisa o contexto de cada quadro usando as informações coletadas nos quadros anteriores. Isso é feito pegando os estados internos da saída do modelo e alimentando-os de volta no modelo para os próximos quadros.

### Saídas do modelo

O modelo retorna uma série de rótulos e suas pontuações correspondentes. As pontuações são valores logits que representam a previsão para cada classe. Essas pontuações podem ser convertidas em probabilidades usando a função softmax (`tf.nn.softmax`).

```python
    exp_logits = np.exp(np.squeeze(logits, axis=0))
    probabilities = exp_logits / np.sum(exp_logits)
```

Internamente, a saída do modelo também inclui os estados internos do modelo e alimenta-os de volta no modelo para os próximos quadros.

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a [ferramenta de referencial](https://www.tensorflow.org/lite/performance/measurement). As MoviNets têm suporte somente a CPUs.

O desempenho do modelo é mensurado pela quantidade de tempo que demora para executar a inferência em um determinado hardware. Uma quantidade de tempo menor implica um modelo mais rápido. A exatidão é mensurada pela frequência de classificação correta de uma classe no vídeo.

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho</th>
      <th>Exatidão*</th>
      <th>Dispositivo</th>
      <th>CPU**</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2"> MoviNet-A0 (quantizado em inteiro)</td>
    <td rowspan="2">       3,1 MB</td>
    <td rowspan="2">65%</td>
    <td>Pixel 4</td>
    <td>5 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>11 ms</td>
  </tr>
    <tr>
    <td rowspan="2"> MoviNet-A1 (quantizado em inteiro)</td>
    <td rowspan="2">       4,5 MB</td>
    <td rowspan="2">70%</td>
    <td>Pixel 4</td>
    <td>8 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>19 ms</td>
  </tr>
      <tr>
    <td rowspan="2"> MoviNet-A2 (quantizado em inteiro)</td>
    <td rowspan="2">       5,1 MB</td>
    <td rowspan="2">72%</td>
    <td>Pixel 4</td>
    <td>15 ms</td>
  </tr>
   <tr>
    <td>Pixel 3</td>
    <td>36 ms</td>
  </tr>
</table>

* Exatidão Top-1 mensurada para o dataset [Kinetics-600](https://arxiv.org/abs/1808.01340).

** Latência mensurada ao executar em uma CPU com 1 thread.

## Personalização do modelo

Os modelos pré-treinados foram treinados para reconhecer 600 ações humanas do dataset [Kinetics-600](https://arxiv.org/abs/1808.01340). Você pode usar o aprendizado por transferência para treinar novamente um modelo para reconhecer ações humanas que não estão presentes no dataset original. Para fazer isso, você precisa de um conjunto de vídeos de treinamento para cada uma das novas ações que deseja incorporar ao modelo.

Confira mais informações sobre como fazer ajustes finos com dados personalizados no [repositório das MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet) e no [tutorial sobre MoViNets](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb).

## Leituras e recursos adicionais

Confira os recursos abaixo para saber mais sobre os conceitos discutidos nesta página.

- [Repositório das MoViNets](https://github.com/tensorflow/models/tree/master/official/projects/movinet)
- [Artigo sobre MoViNets](https://arxiv.org/abs/2103.11511)
- [Modelos MoViNet pré-treinados](https://tfhub.dev/s?deployment-format=lite&q=movinet)
- [Tutorial sobre MoViNets](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb)
- [Datasets Kinetics](https://deepmind.com/research/open-source/kinetics)

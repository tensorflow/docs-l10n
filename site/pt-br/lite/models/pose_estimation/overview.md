# Estimativa de pose

<img src="../images/pose.png" class="attempt-right">

Estimativa de pose é a tarefa de usar um modelo de aprendizado de máquina para estimar a pose de uma pessoa em uma imagem ou vídeo. Isso é feito estimando-se as localizações espaciais de juntas importantes do corpo (pontos-chave).

## Como começar

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou iOS, confira os exemplos de aplicativo abaixo que podem te ajudar a começar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/android">Exemplo do Android</a> <a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/ios">Exemplo do iOS</a>

Se você já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), baixe o modelo inicial de estimativa de pose MoveNet e os arquivos de suporte.

<a class="button button-primary" href="https://tfhub.dev/s?q=movenet">Baixar modelo inicial</a>

Se você quiser testar a estimativa de pose em um navegador, confira a <a href="https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet">Demonstração em JS do TensorFlow</a>.

## Descrição do modelo

### Como funciona

A estimativa de pose refere-se a técnicas de visão computacional que detectam figuras humanas em imagens e vídeos para determinar, por exemplo, onde o cotovelo de uma pessoa aparece em uma imagem. É importante saber que a estimativa de pose estima apenas onde estão as juntas importantes do corpo e não reconhece quem está em uma imagem ou vídeo.

Os modelos de estimativa de pose recebem uma imagem de câmera pré-processada como entrada e geram como saída informações sobre os pontos-chave. Os pontos-chave detectados são indexados por um ID da parte do corpo, com uma pontuação de confiança entre 0,0 e 1,0. A pontuação de confiança indica a probabilidade de um ponto-chave existir nessa posição.

Fornecemos a implementação de referência de dois modelos de estimativa de pose do TensorFlow Lite:

- MoveNet: modelo moderno de estimativa de pose em duas versões: Lighting e Thunder. Confira o comparativo entre as duas na seção abaixo.
- PoseNet: modelo de estimativa de pose da geração anterior, lançado em 2017.

As diversas juntas do corpo detectadas pelo modelo de estimativa de pose estão tabuladas abaixo:

<table style="width: 30%;">
  <thead>
    <tr>
      <th>ID</th>
      <th>Parte do corpo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>nose (nariz)</td>
    </tr>
    <tr>
      <td>1</td>
      <td>leftEye (olho esquerdo)</td>
    </tr>
    <tr>
      <td>2</td>
      <td>rightEye (olho direito)</td>
    </tr>
    <tr>
      <td>3</td>
      <td>leftEar (orelha esquerda)</td>
    </tr>
    <tr>
      <td>4</td>
      <td>rightEar (orelha direita)</td>
    </tr>
    <tr>
      <td>5</td>
      <td>leftShoulder (ombro esquerdo)</td>
    </tr>
    <tr>
      <td>6</td>
      <td>rightShoulder (ombro direito)</td>
    </tr>
    <tr>
      <td>7</td>
      <td>leftElbow (cotovelo esquerdo)</td>
    </tr>
    <tr>
      <td>8</td>
      <td>rightElbow (cotovelo direito)</td>
    </tr>
    <tr>
      <td>9</td>
      <td>leftWrist (punho esquerdo)</td>
    </tr>
    <tr>
      <td>10</td>
      <td>rightWrist (punho direito)</td>
    </tr>
    <tr>
      <td>11</td>
      <td>leftHip (lado esquerdo do quadril)</td>
    </tr>
    <tr>
      <td>12</td>
      <td>rightHip (lado direito do quadril)</td>
    </tr>
    <tr>
      <td>13</td>
      <td>leftKnee (joelho esquerdo)</td>
    </tr>
    <tr>
      <td>14</td>
      <td>rightKnee (joelho direito)</td>
    </tr>
    <tr>
      <td>15</td>
      <td>leftAnkle (tornozelo esquerdo)</td>
    </tr>
    <tr>
      <td>16</td>
      <td>rightAnkle (tornozelo direito)</td>
    </tr>
  </tbody>
</table>

Veja um exemplo de saída abaixo:

<img src="https://storage.googleapis.com/download.tensorflow.org/example_images/movenet_demo.gif" alt="Animation showing pose estimation">

## Referenciais de desempenho

MoveNet está disponível em duas versões:

- MoveNet.Lightning é menor, mas menos exata do que a versão Thunder. Pode ser executada em tempo real em smartphones modernos.
- MoveNet.Thunder é a versão mais exata e lenta do que a versão Lightning. É útil para casos de uso que exigem maior exatidão.

O MoveNet tem desempenho melhor do que o PoseNet para diversos datasets, especialmente com imagens de exercícios físicos. Portanto, recomendamos usar o MoveNet em vez do PoseNet.

Os números de referencial de desempenho são gerados com a ferramenta [descrita aqui](../../performance/measurement). Os números de exatidão (mAP) são mensurados em um subconjunto do [dataset COCO](https://cocodataset.org/#home), em que filtramos e cortamos cada imagem para que contenha somente uma pessoa.

<table>
<thead>
  <tr>
    <th rowspan="2">Modelo</th>
    <th rowspan="2">Tamanho (em MB)</th>
    <th rowspan="2">mAP</th>
    <th colspan="3">Latência (em ms)</th>
  </tr>
  <tr>
    <td>Pixel 5 - 4 threads de CPU</td>
    <td>Pixel 5 - GPU</td>
    <td>Raspberry Pi 4 - 4 threads de CPU</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4">MoveNet.Thunder (quantizado em FP16)</a>
</td>
    <td>12,6</td>
    <td>72,0</td>
    <td>155</td>
    <td>45</td>
    <td>594</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4">MoveNet.Thunder (quantizado em INT8)</a>
</td>
    <td>7,1</td>
    <td>68,9</td>
    <td>100</td>
    <td>52</td>
    <td>251</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4">MoveNet.Lightning (quantizado em FP16)</a>
</td>
    <td>4,8</td>
    <td>63,0</td>
    <td>60</td>
    <td>25</td>
    <td>186</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4">MoveNet.Lightning (quantizado em INT8)</a>
</td>
    <td>2,9</td>
    <td>57,4</td>
    <td>52</td>
    <td>28</td>
    <td>95</td>
  </tr>
  <tr>
    <td>       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite">PoseNet(MobileNetV1 backbone, FP32)</a>
</td>
    <td>13,3</td>
    <td>45,6</td>
    <td>80</td>
    <td>40</td>
    <td>338</td>
  </tr>
</tbody>
</table>

## Leituras e recursos adicionais

- Confira esta [postagem no blog](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) para saber mais sobre a estimativa de pose usando o MoveNet e o TensorFlow Lite.
- Confira esta [postagem no blog](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) para saber mais sobre a estimativa de pose na web.
- Confira este [tutorial](https://www.tensorflow.org/hub/tutorials/movenet) para saber mais sobre como executar o MoveNet no Python usando um modelo do TensorFlow Hub.
- O Coral/EdgeTPU pode deixar a execução da estimativa de pose muito mais rápida em dispositivos de borda. Confira mais detalhes em [Modelos otimizados para EdgeTPU](https://coral.ai/models/pose-estimation/).
- Leia o artigo do PoseNet [aqui](https://arxiv.org/abs/1803.08225)

Além disso, confira estes casos de uso de estimativa de pose:

<ul>
  <li><a href="https://vimeo.com/128375543">‘PomPom Mirror’</a></li>
  <li>
<a href="https://youtu.be/I5__9hq-yas">Amazing Art Installation Turns You Into A Bird | Chris Milk "The Treachery of Sanctuary"</a> (Instalação de arte incrível transforma você em um pássaro | Chris Milk "A Traição do Santuário")</li>
  <li>
<a href="https://vimeo.com/34824490">Puppet Parade - Interactive Kinect Puppets</a> (Desfile de marionetes – Marionetes interativos no Kinect)</li>
  <li>
<a href="https://vimeo.com/2892576">Messa di Voce (Performance), Excerpts</a> [Messa di Voce (Desempenho), Trechos]</li>
  <li><a href="https://www.instagram.com/p/BbkKLiegrTR/">Realidade aumentada</a></li>
  <li><a href="https://www.instagram.com/p/Bg1EgOihgyh/">Animação interativa</a></li>
  <li><a href="https://www.runnersneed.com/expert-advice/gear-guides/gait-analysis.html">Análise do modo de andar</a></li>
</ul>

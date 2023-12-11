# Aprendizado por reforço

Jogue um jogo de tabuleiro contra um agente, que é treinado usando o aprendizado por reforço e implantado com o TensorFlow Lite.

## Como começar

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android, recomendamos conferir o exemplo de aplicativo abaixo que pode te ajudar a começar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android">Exemplo do Android</a>

Se você estiver usando outra plataforma que não o Android ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), pode baixar nosso modelo treinado.

<a class="button button-primary" href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike_tf.tflite">Baixar modelo</a>

## Como funciona

O modelo é criado para o agente jogar um pequeno jogo de tabuleiro chamado "Plane Strike" (Ataque de Avião). Confira uma breve introdução a esse jogo e suas regras no arquivo [README](https://github.com/tensorflow/examples/tree/master/lite/examples/reinforcement_learning/android).

Por baixo da interface gráfica do aplicativo, criamos um agente que joga contra o jogador humano. O agente é um MLP de 3 camadas que recebe o estado do tabuleiro como entrada e gera como saída a pontuação prevista para cada uma das 64 possíveis células do tabuleiro. O modelo é treinado usando o Policy Gradient (gradiente de política) REINFORCE (REFORÇO), e você pode conferir o código de treinamento [aqui](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml). Após treinar o agente, convertemos o modelo em TF Lite e o implantamos no aplicativo para Android.

Durante o jogo em si no aplicativo para Android, quando é o turno do agente, ele analisa o estado do tabuleiro do jogador humano (o tabuleiro na parte inferior), que contém informações sobre ataques anteriores bem-sucedidos e malsucedidos (acertos e erros), e usa o modelo treinado para prever onde atacar em seguida para que consiga terminar o jogo antes do jogador humano.

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="2">       <a href="https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/android/app/src/main/assets/planestrike.tflite">Policy Gradient</a> (Gradiente de Política)</td>
    <td rowspan="2">       84 KB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>0,01 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>0,01 ms*</td>
  </tr>
</table>

* 1 thread usado.

## Entradas

O modelo recebe um Tensor `float32` tridimensional de formato (1, 8, 8) como o estado do tabuleiro.

## Saídas

O modelo retorna um Tensor `float32` bidimensional de formato (1, 64) como as pontuações previstas para cada uma das 64 possíveis posições de ataque.

## Treine seu próprio modelo

Você pode treinar seu próprio modelo para um tabuleiro maior/menor alterando o parâmetro `BOARD_SIZE` (tamanho do tabuleiro) no [código de treinamento](https://github.com/tensorflow/examples/blob/master/lite/examples/reinforcement_learning/ml).

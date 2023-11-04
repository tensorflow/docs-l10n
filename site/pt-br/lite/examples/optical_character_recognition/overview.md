# Reconhecimento ótico de caracteres (OCR)

O reconhecimento ótico de caracteres (OCR, na sigla em inglês) é o processo de reconhecimento de caracteres em imagens usando técnicas de visão computacional e aprendizado de máquina. Este aplicativo de referência demonstra como usar o TensorFlow Lite para fazer reconhecimento ótico de caracteres. Ele usa uma combinação de um [modelo de detecção de texto](https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1) e um [modelo de reconhecimento de texto](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2) como um pipeline de OCR para reconhecer caracteres de texto.

## Como começar

<img src="images/screenshot.gif" class="attempt-right" style="max-width: 300px">

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android, recomendamos conferir o exemplo de aplicativo abaixo que pode te ajudar a começar.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/optical_character_recognition/android">Exemplo do Android</a>

Se você estiver usando outra plataforma que não o Android ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), pode baixar os modelos no [TF Hub](https://tfhub.dev/).

## Como funciona

As tarefas de reconhecimento ótico de caracteres são divididas em duas fases. Primeiro, usamos um modelo de detecção de texto para detectar os retângulos limítrofes ao redor de possíveis textos. Depois, alimentamos os retângulos limítrofes processados em um modelo de reconhecimento de texto para determinar os caracteres específicos dentro dos retângulos limítrofes (também precisamos fazer supressão não máxima, transformação de perspectiva, etc. antes do reconhecimento do texto). Em nosso caso, os dois modelos são do TensorFlow Hub e também são modelos quantizados em FP16.

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>CPU</th>
      <th>GPU</th>
    </tr>
  </thead>
  <tr>
    <td>       <a href="https://tfhub.dev/sayakpaul/lite-model/east-text-detector/fp16/1">Detecção de texto</a>
</td>
    <td>45,9 MB</td>
     <td>Pixel 4 (Android 10)</td>
     <td>181,93 ms*</td>
     <td>89,77 ms*</td>
  </tr>
  <tr>
    <td>       <a href="https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2">Reconhecimento de texto</a>
</td>
    <td>16,8 MB</td>
     <td>Pixel 4 (Android 10)</td>
     <td>338,33 ms*</td>
     <td>N.D.**</td>
  </tr>
</table>

* 4 threads usados.

** este modelo não pode usar delegado de GPU, pois precisamos das operações do TensorFlow para executá-lo.

## Entradas

O modelo de detecção de texto recebe um Tensor `float32` de 4 dimensões de formato (1, 320, 320, 3) como entrada.

O modelo de reconhecimento de texto recebe um Tensor `float32` de 4 dimensões de formato (1, 31, 200, 1) como entrada.

## Saídas

O modelo de detecção de texto retorna um Tensor `float32` de 4 dimensões de formato (1, 80, 80, 5) como o retângulo limítrofe e um Tensor `float32` de 4 dimensões de formato (1,80, 80, 5) como uma pontuação de detecção.

O modelo de reconhecimento de texto retorna um Tensor `float32` bidimensional de formato (1, 48) como os índices de mapeamento para a lista do alfabeto '0123456789abcdefghijklmnopqrstuvwxyz'

## Limitações

- O [modelo de reconhecimento de texto](https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/float16/2) atual é treinado usando dados sintéticos com letras e números do idioma inglês, portanto só há suporte ao inglês.

- Os modelos não são gerais o suficiente para fazer reconhecimento ótico de caracteres em áreas externas (digamos, imagens aleatórias feitas por uma câmera de smartphone com pouca iluminação).

Escolhemos três logotipos de produtos do Google para demonstrar como fazer o reconhecimento ótico de caracteres usando o TensorFlow Lite. Se você está procurando um produto de OCR profissional pronto para uso, considere o [ML Kit do Google](https://developers.google.com/ml-kit/vision/text-recognition). O ML Kit, que usa o TF Lite por baixo dos panos, deve ser suficiente para a maioria dos casos de uso de OCR, mas, para alguns casos, você vai querer criar sua própria solução de OCR com o TF Lite. Confira alguns exemplos:

- Você tem seus próprios modelos de reconhecimento/detecção de texto do TF Lite que deseja usar.
- Você tem requisitos de negócio especiais (por exemplo, reconhecer textos de cabeça para baixo) e precisa personalizar o pipeline de OCR.
- Você deseja oferecer suporte a idiomas não disponíveis no ML Kit.
- Os dispositivos dos usuários não têm necessariamente os serviços do Google Play instalados.

## Referências

- Exemplo de reconhecimento/detecção de texto do OpenCV: https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
- Projeto de OCR da comunidade do TF Lite feito por contribuidores da comunidade: https://github.com/tulasiram58827/ocr_tflite
- Detecção de texto do OpenCV: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
- Aprendizado profundo baseado em detecção de texto usando o OpenCV: https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

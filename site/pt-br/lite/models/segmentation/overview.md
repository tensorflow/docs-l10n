# Segmentação

A segmentação de imagens é o processo de particionar uma imagem digital em diversos segmentos (conjuntos de pixels, também conhecidos como objetos da imagem). O objetivo da segmentação é simplificar e/ou alterar a representação de uma imagem em algo que tenha mais significado e seja mais fácil de analisar.

A imagem abaixo mostra a saída do modelo de segmentação de imagens no Android. O modelo cria uma máscara sobre os objetos-alvo com alta exatidão.

<img src="images/segmentation.gif" class="attempt-right">

Observação, para integrar com um modelo existente, experimente a [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter) (biblioteca de tarefas do TensorFlow Lite).

## Como começar

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou iOS, recomendamos conferir os exemplos de aplicativo abaixo que podem te ajudar a começar.

Você pode usar a API integrada da [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/image_segmenter) para integrar modelos de segmentação de imagens com somente algumas linhas de código. Além disso, pode integrar o modelo usando a [TensorFlow Lite Interpreter Java API](../../guide/inference#load_and_run_a_model_in_java) (API Java Interpreter do TensorFlow Lite).

O exemplo do Android abaixo demonstra a implementação dos dois métodos como a [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_task_api) e a [lib_interpreter](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android">Ver exemplo do Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios">Ver exemplo do iOS</a>

Se você estiver usando outra plataforma que não o Android ou iOS ou se já conhecer bem as <a href="https://www.tensorflow.org/api_docs/python/tf/lite">APIs do TensorFlow Lite</a>, pode baixar nosso modelo inicial de segmentação de imagens.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Baixar modelo inicial</a>

## Descrição do modelo

O *DeepLab* é um modelo moderno de aprendizado profundo para segmentação semântica de imagens, em que o objetivo é atribuir rótulos semânticos (como pessoa, cachorro e gato) a cada pixel da imagem de entrada.

### Como funciona

A segmentação semântica de imagens prevê se cada pixel de uma imagem está associado a uma determinada classe. Isso é diferente da <a href="../object_detection/overview.md">detecção de objetos</a>, que detecta objetos em regiões retangulares, e da <a href="../image_classification/overview.md">classificação de imagens</a>, que classifica a imagem de forma geral.

A implementação atual inclui os seguintes recursos:

<ol>
  <li>DeepLabv1: usamos convolução atrous (dilatada) para controlar explicitamente a resolução de computação das respostas de características em redes neurais convolucionais profundas.</li>
  <li>DeepLabv2: usamos Atrous Spatial Pyramid Pooling (ASPP) para segmentar de forma robusta os objetos em diversas escalas com filtros com diversas taxas de amostragem e campos de visão efetivos.</li>
  <li>DeepLabv3: ampliamos o módulo ASPP com característica no nível de imagem [5, 6] para capturar informações de alcance maior. Além disso, incluímos a normalização de parâmetros [7] para realizar o treinamento. Especificamente, aplicamos a convolução atrous para extrair características de saídas em diferentes strides de saída durante o treinamento e a avaliação, o que permite treinar a normalização de lotes com stride de saída = 16 e manter o desempenho alto com stride de saída = 8 durante a avaliação.</li>
  <li>DeepLabv3+: estendemos o DeepLabv3 para incluir um módulo decodificador simples, mas eficaz, para refinar os resultados da segmentação, especialmente ao longo das fronteiras entre objetos. Além disso, nessa estrutura encoder-decoder, é possível controlar arbitrariamente a resolução das características extraídas pelo encoder por meio de convolução atrous para fazer uma contrapartida entre runtime e precisão.</li>
</ol>

## Referenciais de desempenho

Os referenciais de desempenho são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>GPU</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite">Deeplab v3</a>
</td>
    <td rowspan="3">       2,7 MB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>16 ms</td>
    <td>37 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20 ms</td>
    <td>23 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>16 ms</td>
    <td>25 ms**</td>
  </tr>
</table>

* 4 threads usados.

** 2 threads usados no iPhone para o resultado com maior desempenho.

## Leituras e recursos adicionais

<ul>
  <li><a href="https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html">Segmentação semântica de imagens com o DeepLab no TensorFlow</a></li>
  <li><a href="https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7">Agora, o TensorFlow Lite está mais rápido com GPUs em dispositivos móveis (prévia para desenvolvedores)</a></li>
  <li><a href="https://github.com/tensorflow/models/tree/master/research/deeplab">DeepLab: Rotulagem profunda para segmentação semântica de imagens</a></li>
</ul>

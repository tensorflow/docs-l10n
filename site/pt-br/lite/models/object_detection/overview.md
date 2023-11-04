# Detecção de objetos

Dada uma imagem ou um stream de vídeo, um modelo de detecção de objetos pode identificar quais objetos dentre um conjunto de objetos podem estar presentes e fornecer informações sobre suas posições na imagem.

Por exemplo, esta captura de tela do <a href="#get_started">aplicativo de exemplo</a> mostra como dois objetos foram reconhecidos e suas posições anotadas:

<img src="images/android_apple_banana.png" width="30%" alt="Screenshot of Android example">

Observação: (1) para integrar um modelo existente, use a [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/object_detector) (biblioteca de tarefas do TensorFlow Lite). (2) Para personalizar um modelo, use o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) (criador de modelos do TensorFlow Lite).

## Como começar

Para aprender a usar detecção de objetos em um aplicativo para dispositivos móveis, confira os <a href="#example_applications_and_guides">Exemplos de aplicativos e guias</a>.

Se você estiver usando outra plataforma que não o Android ou iOS ou se já conhecer bem as <a href="https://www.tensorflow.org/api_docs/python/tf/lite">APIs do TensorFlow Lite</a>, pode baixar nosso modelo inicial de detecção de objetos e os respectivos rótulos.

<a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">Baixar modelo inicial com metadados</a>

Para mais informações sobre os metadados e arquivos associados (por exemplo, `labels.txt`), confira o artigo <a href="../../models/convert/metadata#read_the_metadata_from_models">Leia metadados de modelos</a>.

Se você quiser treinar um modelo de detecção personalizado para sua própria tarefa, confira <a href="#model-customization">Personalização do modelo</a>.

Para os casos de uso abaixo, você deve usar um tipo de modelo diferente:

<ul>
  <li>Prever qual rótulo a imagem representa com a maior probabilidade (confira <a href="../image_classification/overview.md">Classificação de imagens</a>).</li>
  <li>Prever a composição de uma imagem, por exemplo, objeto versus plano de fundo (confira <a href="../segmentation/overview.md">Segmentação</a>).</li>
</ul>

### Exemplos de aplicativos e guias

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou iOS, recomendamos conferir os exemplos de aplicativo abaixo que podem te ajudar a começar.

#### Android

Você pode usar a API integrada da [TensorFlow Lite Task Library](../../inference_with_metadata/task_library/object_detector) (biblioteca de tarefas do TensorFlow Lite) para integrar modelos de detecção de objetos com somente algumas linhas de código. Além disso, pode criar seu próprio pipeline de inferência personalizado usando a [TensorFlow Lite Interpreter Java Library](../../guide/inference#load_and_run_a_model_in_java) (API Java Interpreter do TensorFlow Lite).

O exemplo do Android abaixo demonstra a implementação dos dois métodos como a [biblioteca de tarefas ](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services) e a [API do interpretador](https://github.com/tensorflow/examples/tree/eb925e460f761f5ed643d17f0c449e040ac2ac45/lite/examples/object_detection/android/lib_interpreter), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Ver exemplo do Android</a>

#### iOS

Você pode integrar o modelo usando a [TensorFlow Lite Interpreter Swift Library](../../guide/inference#load_and_run_a_model_in_swift) (API Swift Interpreter do TensorFlow Lite). Confira o exemplo do iOS abaixo.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios">Ver exemplo do iOS</a>

## Descrição do modelo

Esta seção descreve a assinatura de modelos para [detecção de imagem única](https://arxiv.org/abs/1512.02325) convertidos para o TensorFlow Lite usando a [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/) (API de detecção de objetos do TensorFlow).

Um modelo de detecção de objetos é treinado para detectar a presença e a localização de diversas classes de objetos. Por exemplo: um modelo pode ser treinado com imagens que contenham diversos pedaços de frutas junto com um *rótulo* que especifique a classe de fruta que os pedaços representam (por exemplo, uma maçã, banana ou morango), além de dados especificando onde cada objeto aparece na imagem.

Quando uma imagem é fornecida posteriormente para o modelo, será gerada como saída uma lista dos objetos detectados, a localização de um retângulo limítrofe que contém cada objeto e uma pontuação que indica a confiança de que a detecção está correta.

### Assinatura da entrada

O modelo recebe uma imagem como entrada.

Vamos supor que a imagem esperada tenha 300x300 pixels com três canais (vermelho, azul e verde) por pixel. Ela deve ser alimentada no modelo como um buffer dimensionado de valores de 270 mil bytes (300x300x3). Se o modelo for <a href="../../performance/post_training_quantization.md">quantizado</a>, cada valor deve ser um único byte representando um valor entre 0 e 255.

Confira o [código do aplicativo de exemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) para entender como fazer esse pré-processamento no Android.

### Assinatura da saída

O modelo gera como saída quatro arrays mapeados para os índices 0-4. Os arrays 0, 1 e 2 descrevem `N` objetos detectados, com um elemento em cada array correspondente a cada objeto.

<table>
  <thead>
    <tr>
      <th>Índice</th>
      <th>Nome</th>
      <th>Descrição</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Localizações</td>
      <td>Array multidimensional de [N][4] valores em ponto flutuante entre 0 e 1; os arrays internos representam os retângulos limítrofes na forma [superior, esquerda, inferior, direita]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Classes</td>
      <td>Array de N inteiros (saída como valores em ponto flutuante), cada um indicando o índice de um rótulo de classe do arquivo de rótulos</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Pontuações</td>
      <td>Array de N valores em ponto flutuante entre 0 e 1 representando a probabilidade de uma classe ter sido detectada</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Número de detecções</td>
      <td>Valor inteiro de N</td>
    </tr>
  </tbody>
</table>

Observação: o número de resultados (10 no caso acima) é um parâmetro definido ao exportar o modelo de detecção para o TensorFlow Lite. Confira mais detalhes em <a href="#model-customization">Personalização do modelo</a>.

Por exemplo, imagine um modelo que tenha sido treinado para detectar maçãs, bananas e morangos. Ao receber uma imagem, ele vai gerar como saída um determinado número de resultados de detecção – 5, neste exemplo.

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Classe</th>
      <th>Pontuação</th>
      <th>Localização</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Maçã</td>
      <td>0,92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0,88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Morango</td>
      <td>0,87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0,23</td>
      <td>[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td>Maçã</td>
      <td>0,11</td>
      <td>[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

#### Confiança da pontuação

Para interpretar esses resultados, confira a pontuação e a localização de cada objeto detectado. A pontuação é um número entre 0 e 1 que indica a confiança de o objeto ter sido detectado corretamente. Quanto mais perto de 1, maior a confiança do modelo.

Dependendo do seu aplicativo, você pode definir um limiar de corte abaixo do qual os resultados de detecção serão descartados. No exemplo atual, um ponto de corte razoável é uma pontuação de 0,5 (ou seja, 50% de probabilidade de a detecção ter sido válida). Nesse caso, os últimos dois objetos no array seriam ignorados, pois as pontuações de confiança estão abaixo de 0,5:

<table style="width: 60%;">
  <thead>
    <tr>
      <th>Classe</th>
      <th>Pontuação</th>
      <th>Localização</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Maçã</td>
      <td>0,92</td>
      <td>[18, 21, 57, 63]</td>
    </tr>
    <tr>
      <td>Banana</td>
      <td>0,88</td>
      <td>[100, 30, 180, 150]</td>
    </tr>
    <tr>
      <td>Morango</td>
      <td>0,87</td>
      <td>[7, 82, 89, 163]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Banana</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0,23</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[42, 66, 57, 83]</td>
    </tr>
    <tr>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">Maçã</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">0,11</td>
      <td style="background-color: #e9cecc; text-decoration-line: line-through;">[6, 42, 31, 58]</td>
    </tr>
  </tbody>
</table>

O ponto de corte que você deve usar depende se você tolera mais ou menos falsos positivos (objetos identificados incorretamente ou áreas da imagem identificadas incorretamente como objetos) ou falsos negativos (objetos genuínos que não foram detectados porque a confiança foi baixa).

Por exemplo, na imagem abaixo, uma pera (que não é um objeto que o modelo foi treinado para detectar) foi identificado incorretamente como uma "pessoa". Esse é um exemplo de falso positivo que poderia ser ignorado selecionando um ponto de corte apropriado. Neste caso, um ponto de corte igual a 0,6 (60%) excluiria o falso positivo.

<img src="images/false_positive.png" width="30%" alt="Screenshot of Android example showing a false positive">

#### Localização

Para cada objeto detectado, o modelo retorna um array com quatro números que representam um retângulo limítrofe ao redor da posição. Para o modelo inicial fornecido, os números estão na seguinte ordem:

<table style="width: 50%; margin: 0 auto;">
  <tbody>
    <tr style="border-top: none;">
      <td>[</td>
      <td>superior,</td>
      <td>esquerda,</td>
      <td>inferior,</td>
      <td>direita</td>
      <td>]</td>
    </tr>
  </tbody>
</table>

O valor superior representa a distância entre a borda superior do retângulo e o topo da imagem, em pixels. O valor esquerda representa a distância entre a borda esquerda e o canto esquerdo da imagem de entrada. De maneira similar, os outros valores representam as bordas inferior e direita.

Observação: os modelos de detecção de objetos recebem como entrada imagens de um tamanho específico. Provavelmente, esse tamanho é diferente do tamanho da imagem não tratada capturada pela câmera do seu dispositivo, e você precisará escrever código para recortar e redimensionar a imagem não tratada para que fique adequada ao tamanho de entrada do modelo (confira exemplos nos <a href="#get_started">aplicativos de exemplo</a>).<br><br>Os valores de pixel gerados como saída pelo módulo indicam a posição na imagem cortada e redimensionada, então você precisa dimensioná-los para que fiquem adequados à imagem não tratada para poder interpretá-los corretamente.

## Referenciais de desempenho

Os referenciais de desempenho para nosso <a class="button button-primary" href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">modelo inicial</a> são gerados com a ferramenta [descrita aqui](https://www.tensorflow.org/lite/performance/benchmarks).

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
    <td rowspan="3">       <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">COCO SSD MobileNet v1</a>
</td>
    <td rowspan="3">       27 MB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>22 ms</td>
    <td>46 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>20 ms</td>
    <td>29 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td>7,6 ms</td>
    <td>11 ms**</td>
  </tr>
</table>

* 4 threads usados.

** 2 threads usados no iPhone para o resultado com maior desempenho.

## Personalização do modelo

### Modelos pré-treinados

Modelos de detecção otimizados para dispositivos móveis com uma variedade de características de latência e precisão estão disponíveis no [Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models). Cada um deles segue as assinaturas de entrada e saída descritas nas próximas seções.

A maioria dos arquivos zip para download contêm um arquivo `model.tflite`. Se não houver um, o flatbuffer do TensorFlow Lite pode ser gerado seguindo [estas instruções](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). Além disso, modelos para SSD do [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) podem ser convertidos para o TensorFlow Lite seguindo [estas instruções](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md). É importante salientar que os modelos de detecção não podem ser convertidos diretamente usando o [TensorFlow Lite Converter](../../models/convert), já que exigem uma etapa intermediária de geração de um modelo fonte adequado para dispositivos móveis. Os scripts indicados acima realizam essa etapa.

Os scripts de importação para [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md) e [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md) têm parâmetros que possibilitam um número maior de objetos de saída ou um pós-processamento mais lento e mais exato. Utilize `--help` nos scripts para ver uma lista completa dos argumentos permitidos.

> No momento, a inferência em dispositivos é otimizada somente em modelos para SSD. Estamos avaliando um suporte melhor a outras arquiteturas, como CenterNet e EfficientDet.

### Como escolher um modelo a ser personalizado?

Cada modelo vem com sua própria precisão (quantificada pelo valor mAP) e características de latência. Você deve escolher o modelo mais adequado para seu caso de uso e hardware desejado. Por exemplo, os modelos [Edge TPU](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models) são ideais para inferência no Edge TPU do Google no Pixel 4.

Use nossa [ferramenta de referenciais](https://www.tensorflow.org/lite/performance/measurement) para avaliar os modelos e escolher a opção mais eficiente disponível.

## Ajustes finos de modelos com dados personalizados

Os modelos pré-treinados fornecidos foram treinados para detectar 90 classes de objetos. Para ver a lista completa de classes, confira o arquivo labels (rótulos) nos <a href="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite">metadados do modelo</a>.

Você pode usar a técnica aprendizado por transferência para treinar novamente um modelo a fim de reconhecer classes que não estão no dataset original. Por exemplo: você pode treinar novamente o modelo para detectar diversos tipos de vegetais, apesar de haver somente um vegetal nos dados de treinamento originais. Para fazer isso, será necessário um conjunto de imagens de treinamento para cada um dos novos rótulos que você deseja treinar. É recomendável usar a biblioteca [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) (criador de modelos do TensorFlow Lite), que simplifica o processo de treinamento de um modelo do TensorFlow Lite usando um dataset personalizado com algumas linhas de código. Essa biblioteca usa aprendizado por transferência para reduzir a quantidade de dados de treinamento necessários e o tempo gasto. Confira o [Colab de detecção de poucas imagens](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tflite.ipynb) para ver um exemplo de ajustes finos de um modelo pré-treinado com alguns exemplos.

Para fazer ajustes finos de datasets maiores, confira estes guias para treinar seus próprios modelos com a TensorFlow Object Detection API (API de detecção de objetos do TensorFlow): [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md). Após o treinamento, eles podem ser convertidos para o formato TF Lite; confira as instruções aqui: [TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md), [TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md).

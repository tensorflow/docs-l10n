# Classificação de imagens

<img src="../images/image.png" class="attempt-right">

A tarefa de identificar o que uma imagem representa é chamada de *classificação de imagem*. Um modelo desse tipo é treinado para reconhecer diversas classes de imagem. Por exemplo, é possível treinar um modelo para reconhecer fotos que representem três tipos de animais diferentes: coelhos, hamsters e cachorros. O TensorFlow Lite conta com modelos pré-treinados e otimizados que podem ser implantados em seus aplicativos móveis. Saiba mais sobre o uso do TensorFlow para classificação de imagens [aqui](https://www.tensorflow.org/tutorials/images/classification).

A imagem abaixo mostra a saída do modelo de classificação de imagens no Android.

<img src="images/android_banana.png" width="30%" alt="Screenshot of Android example">

Observação: (1) para integrar um modelo existente, use a [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier) (biblioteca de tarefas do TensorFlow Lite). (2) Para personalizar um modelo, use o [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification) (criador de modelos do TensorFlow Lite).

## Como começar

Se você estiver apenas começando a usar o TensorFlow Lite e estiver trabalhando com Android ou iOS, recomendamos conferir os exemplos de aplicativo abaixo que podem te ajudar a começar.

Você pode usar a API integrada da [TensorFlow Lite Task Library](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/inference_with_metadata/task_library/image_classifier.md) (biblioteca de tarefas do TensorFlow Lite) para integrar modelos de classificação de imagens com somente algumas linhas de código. Além disso, pode criar seu próprio pipeline de inferência personalizado usando a [TensorFlow Lite Support Library](../../inference_with_metadata/lite_support) (biblioteca de suporte do TensorFlow Lite).

O exemplo do Android abaixo demonstra a implementação dos dois métodos como [lib_task_api](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_task_api) e [lib_support](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support), respectivamente.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Ver exemplo do Android</a>

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Ver exemplo do iOS</a>

Se você estiver usando outra plataforma que não o Android/iOS ou se já conhecer bem as [APIs do TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite), baixe o modelo inicial e os arquivos de suporte (se aplicável).

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Baixar modelo inicial</a>

## Descrição do modelo

### Como funciona

Durante o treinamento, um modelo de classificação de imagens é alimentado com imagens e seus respectivos *rótulos*. Cada rótulo é o nome de um conceito distinto, ou classe, que o modelo aprenderá a reconhecer.

Com dados de treinamento suficientes (geralmente, centenas ou milhares de imagens por rótulo), um modelo de classificação de imagens pode aprender a prever se novas imagens pertencem a uma das classes do treinamento. Esse processo de previsão é chamado de *inferência*. Também é possível usar [aprendizado por transferência](https://www.tensorflow.org/tutorials/images/transfer_learning) para identificar novas classes de imagens usando um modelo pré-existente. O aprendizado por transferência não requer um dataset de treinamento muito grande.

Ao fornecer uma nova imagem como entrada ao modelo, ele gerará como saída as probabilidades da imagem representar cada tipo de animal do treinamento. Confira um exemplo de saída:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Tipo de animal</th>
      <th>Probabilidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Coelho</td>
      <td>0,07</td>
    </tr>
    <tr>
      <td>Hamster</td>
      <td>0,02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Cachorro</td>
      <td style="background-color: #fcb66d;">0,91</td>
    </tr>
  </tbody>
</table>

Cada número na saída corresponde a um rótulo nos dados de treinamento. Ao associar a saída aos três rótulos com os quais o modelo foi treinado, você pode ver que o modelo previu uma alta probabilidade de a imagem representar um cachorro.

Note que a soma de todas as probabilidades (para coelho, hamster e cachorro) é igual a 1. Esse é um tipo de saída comum para modelos com múltiplas classes (confira mais informações em <a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a>).

Observação: o modelo de classificação de imagens só pode indicar a probabilidade de uma imagem representar uma ou mais classes com as quais o modelo foi treinado. Ele não pode indicar a posição ou identidade de objetos na imagem. Se você precisar identificar objetos ou suas posições nas imagens, deverá usar um modelo de <a href="../object_detection/overview">detecção de objetos</a>.

<h4>Resultados ambíguos</h4>

Como a soma das probabilidades de saída sempre será igual a 1, se uma imagem não for reconhecida com confiança como pertencente a uma das classes com as quais o modelo foi treinado, poderá haver uma distribuição de probabilidades uniforme, sem que um valor seja muito maior que os outros.

Por exemplo, a tabela abaixo poderá indicar um resultado ambíguo:


<table style="width: 40%;">   <thead>     <tr>       <th>Rótulo</th>       <th>Probabilidade</th>     </tr>   </thead>   <tbody>     <tr>       <td>Coelho</td>       <td>0,31</td>     </tr>     <tr>       <td>Hamster</td>       <td>0,35</td>     </tr>     <tr>       <td>Cachorro</td>       <td>0,34</td>     </tr>   </tbody> </table> Se o seu modelo retornar resultados ambíguos com frequência, talvez você precisa de outro modelo mais preciso.

<h3>Escolha de uma arquitetura do modelo</h3>

O TensorFlow Lite conta com diversos modelos de classificação de imagens treinados com o dataset original. Algumas arquiteturas de modelos, como MobileNet, Inception e NASNet, estão disponíveis no <a href="https://tfhub.dev/s?deployment-format=lite">TensorFlow Hub</a>. Para escolher o melhor modelo para seu caso de uso, você precisa considerar as arquiteturas específicas, bem como algumas das contrapartidas entre os diversos modelos. Algumas dessas contrapartidas são baseadas em métricas como desempenho, exatidão e tamanho do modelo. Por exemplo, talvez você precise de um modelo mais rápido para criar um leitor de código de barras, mas talvez prefira um modelo mais lento e mais exato para um aplicativo de exames de imagens.

Os <a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">modelos de classificação de imagens</a> fornecidos aceitam diversos tamanhos de entrada. Para alguns modelos, isso é indicado no nome do arquivo. Por exemplo: o modelo Mobilenet_V1_1.0_224 aceita uma entrada de 224x224 pixels. Todos os modelos exigem três canais de cor por pixel (vermelho, verde e azul). Os modelos quantizados exigem 1 byte por canal, e os modelos de ponto flutuante exigem 4 bytes por canal. Os exemplos de código para <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android_java">Android</a> e <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS</a> demonstram como processar imagens de câmera com tamanho total para o formato exigido por cada modelo.

<h3>Usos e limitações</h3>

Os modelos de classificação de imagens do TensorFlow Lite são úteis para classificação com um único rótulo, ou seja, prever qual rótulo a imagem representa com a maior probabilidade. Esses modelos são treinados para reconhecer 1.000 classes de imagens. Confira a lista completa de classes no arquivo labels (rótulos) do <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">arquivo zip do modelo</a>.

Se você quiser treinar um modelo para reconhecer novas classes, confira <a href="#customize_model">Personalize o modelo</a>.

Para os casos de uso abaixo, você deve usar um tipo de modelo diferente:

<ul>
  <li>Prever o tipo e a posição de um ou mais objetos em uma imagem (confira <a href="../object_detection/overview">Detecção de objetos</a>).</li>
  <li>Prever a composição de uma imagem, por exemplo, objeto versus plano de fundo (confira <a href="../segmentation/overview">Segmentação</a>).</li>
</ul>

Depois que o modelo inicial estiver sendo executado no dispositivo desejado, você pode testar diferentes modelos para encontrar o equilíbrio ideal entre desempenho, exatidão e tamanho do modelo.

<h3>Personalize o modelo</h3>

Os modelos pré-treinados fornecidos são treinados para reconhecer 1.000 classes de imagens. Confira a lista completa de classes no arquivo labels (rótulos) do <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">arquivo zip do modelo</a>.

Você também pode usar aprendizado por transferência para treinar novamente um modelo para reconhecer classes que não estão presentes no dataset original. Por exemplo: você pode treinar novamente o modelo para distinguir entre diferentes espécies de árvores, apesar de não haver árvores nos dados do treinamento original. Para fazer isso, você precisa de um dataset de imagens de treinamento para cada um dos novos rótulos que deseja treinar.

Saiba como fazer o aprendizado por transferência com o <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TFLite Model Maker</a> (criador de modelos do TF Lite) ou no Codelab <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/index.html#0">Reconheça flores com o TensorFlow</a>.

<h2>Referenciais de desempenho</h2>

O desempenho do modelo é mensurado por meio da quantidade de tempo que um modelo demora para executar a inferência em um determinado hardware. Quanto menor o tempo, mais rápido o modelo é.

O desempenho necessário depende do aplicativo. O desempenho pode ser importante para diversos aplicativos, como vídeo em tempo real, em que pode ser importante analisar cada quadro antes do próximo quadro ser exibido (por exemplo: a inferência precisa ser mais rápida do que 33 ms para fazer inferência em tempo real em uma transmissão de vídeo de 30 qps).

O desempenho dos modelos MobileNet quantizados do TensorFlow Lite varia de 3,7 ms a 80,3 ms.

Os referenciais de desempenho são gerados com a <a href="https://www.tensorflow.org/lite/performance/benchmarks">ferramenta de referencial</a>.

<table>
  <thead>
    <tr>
      <th>Nome do modelo</th>
      <th>Tamanho do modelo</th>
      <th>Dispositivo</th>
      <th>NNAPI</th>
      <th>CPU</th>
    </tr>
  </thead>
  <tr>
    <td rowspan="3">       <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Mobilenet_V1_1.0_224_quant</a>
</td>
    <td rowspan="3">       4,3 MB</td>
    <td>Pixel 3 (Android 10)</td>
    <td>6 ms</td>
    <td>13 ms*</td>
  </tr>
   <tr>
     <td>Pixel 4 (Android 10)</td>
    <td>3,3 ms</td>
    <td>5 ms*</td>
  </tr>
   <tr>
     <td>iPhone XS (iOS 12.4.1)</td>
     <td></td>
    <td>11 ms**</td>
  </tr>
</table>

* 4 threads usados.

** 2 threads usados no iPhone para o resultado com maior desempenho.

### Exatidão do modelo

A exatidão é medida de acordo com a frequência em que o modelo classifica corretamente uma imagem. Por exemplo: espera-se que um modelo com uma exatidão de 60% classifique uma imagem corretamente 60% das vezes, em média.

As métricas de exatidão mais relevantes são Top-1 e Top-5. Top-1 indica a frequência em que o rótulo correto aparece como o rótulo com a maior probabilidade na saída do modelo. Top-5 indica a frequência em que o rótulo correto aparece nas 5 maiores probabilidades na saída do modelo.

A exatidão Top-5 dos modelos MobileNet quantizados do TensorFlow Lite varia de 64,4% e 89,9%.

### Tamanho do modelo

O tamanho de um modelo no disco varia de acordo com seu desempenho e exatidão. O tamanho pode ser importante para desenvolvimento para dispositivos móveis (pois pode impactar o tamanho de download do aplicativo) ou ao trabalhar com hardwares (em que o armazenamento disponível pode ser limitado).

O tamanho dos modelos MobileNet quantizados do TensorFlow Lite varia de 0,5 a 3,4 MB.

## Leituras e recursos adicionais

Confira os recursos abaixo para saber mais sobre os conceitos relacionados à classificação de imagens:

- [Classificação de imagens usando o TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
- [Classificação de imagens usando CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [Aprendizado por transferência](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Ampliação de dados](https://www.tensorflow.org/tutorials/images/data_augmentation)

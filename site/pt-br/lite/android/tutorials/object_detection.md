# Detecção de objetos com o Android

Este tutorial mostra como criar um aplicativo Android usando o TensorFlow Lite para detectar objetos continuamente em frames capturados por um dispositivo com câmera. Esse aplicativo foi feito para um dispositivo Android físico. Se você estiver atualizando um projeto existente, pode usar o código de amostra como referência e pular para as instruções sobre [como modificar seu projeto](#add_dependencies).

![Demonstração animada de detecção de objeto](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## Visão geral da detecção de objetos

A *detecção de objetos* é a tarefa de aprendizado de máquina que identifica a presença e o local de várias classes de objetos em uma imagem. Um modelo de detecção de objetos é treinado com um dataset que contém um conjunto de objetos conhecidos.

O modelo treinado recebe frames de imagens como entrada e tenta categorizar os itens nas imagens a partir do conjunto de classes conhecidas que foi treinado para reconhecer. Para cada frame de imagem, o modelo de detecção de objetos gera uma lista de objetos que detecta, o local de uma caixa delimitadora para cada objeto e uma pontuação que indica a confiança do objeto ser classificado corretamente.

## Modelos e dataset

Este tutorial usa modelos que foram treinados usando o [dataset COCO](http://cocodataset.org/). COCO é um dataset de detecção de objetos de grande escala que contém 330 mil imagens, 1,5 milhão de instâncias de objetos e 80 categorias de objetos.

Você pode usar um dos seguintes modelos pré-treinados:

- [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) *[Recomendado]*: um modelo de detecção de objetos leve com um extrator de características BiFPN, preditor de caixa compartilhado e perda focal. A mAP (precisão média) para o dataset de validação COCO 2017 é 25,69%.

- [EfficientDet-Lite1](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite1/detection/metadata/1): um modelo de detecção de objetos EfficientDet de tamanho médio. A mAP para o dataset de validação COCO 2017 é 30,55%.

- [EfficientDet-Lite2](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1): um modelo maior de detecção de objetos EfficientDet. A mAP para o dataset de validação COCO 2017 é 33,97%.

- [MobileNetV1-SSD](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2): um modelo extremamente leve e otimizado para a detecção de objetos com o TensorFlow Lite. A mAP para o dataset de validação COCO 2017 é 21%.

Para este tutorial, o modelo *EfficientDet-Lite0* apresenta um bom equilíbrio entre tamanho e exatidão.

O download, a extração e a colocação dos modelos na pasta de recursos são gerenciados automaticamente pelo arquivo `download.gradle`, que é executado no tempo de build. Você não precisa baixar modelos TFLite manualmente no projeto.

## Configure e execute o exemplo

Para configurar o aplicativo de detecção de objetos, baixe a amostra do [GitHub](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) e a execute usando o [Android Studio](https://developer.android.com/studio/). As seções a seguir deste tutorial exploram as seções relevantes do código de exemplo, para você aplicá-las aos seus próprios aplicativos Android.

### Requisitos do sistema

- **[Android Studio](https://developer.android.com/studio/index.html)**, versão 2021.1.1 (Bumblebee) ou mais recente.
- SDK do Android, versão 31 ou mais recente.
- Dispositivo Android com uma versão mínima de SO do SDK 24 (Android 7.0 - Nougat) com o modo desenvolvedor ativado.

Observação: este exemplo usa a câmera, então execute-o em um dispositivo Android físico.

### Obtenha o código de exemplo

Crie uma cópia local do código de exemplo. Você usará esse código para criar um projeto no Android Studio e executar o aplicativo de exemplo.

Para clonar e configurar o código de exemplo:

1. Clone o repositório git
    <pre class="devsite-click-to-copy">
        git clone https://github.com/tensorflow/examples.git
        </pre>
2. Opcionalmente, configure sua instância git para usar o sparse checkout e ter somente os arquivos para o aplicativo de exemplo de detecção de objetos:
    <pre class="devsite-click-to-copy">
        cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android
        </pre>

### Importe e execute o projeto

Crie um projeto a partir do código de exemplo baixado, compile e depois execute esse projeto.

Para importar e compilar o projeto do código de exemplo:

1. Inicie o [Android Studio](https://developer.android.com/studio).
2. No Android Studio, selecione **File &gt; New &gt; Import Project** (Arquivo &gt; Novo &gt; Importar projeto).
3. Acesse o diretório do código de exemplo com o arquivo build.gradle (`.../examples/lite/examples/object_detection/android/build.gradle`) e selecione esse diretório.
4. Se o Android Studio solicitar o Gradle Sync, selecione OK.
5. Garanta que o dispositivo Android esteja conectado ao seu computador e que o modo desenvolvedor esteja ativado. Clique na seta `Run` verde.

Se você selecionar o diretório correto, o Android Studio cria e compila um novo projeto. Esse processo pode levar alguns minutos, dependendo da velocidade do seu computador e se você usou o Android Studio para outros projetos. Quando o build for concluído, o Android Studio exibirá uma mensagem `BUILD SUCCESSFUL` no painel de status **Build Output**.

Observação: o código de exemplo foi criado com o Android Studio 4.2.2, mas funciona com versões mais antigas do Studio. Se você estiver usando uma versão mais antiga do Android Studio, pode tentar ajustar o número da versão do plugin Android para que o build seja concluído, em vez de fazer upgrade do Studio.

**Opcional:** para corrigir erros de build, atualize a versão do plugin Android:

1. Abra o arquivo build.gradle no diretório do projeto.

2. Mude a versão das ferramentas Android da seguinte maneira:

    ```
    // from: classpath
    'com.android.tools.build:gradle:4.2.2'
    // to: classpath
    'com.android.tools.build:gradle:4.1.2'
    ```

3. Sincronize o projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

Para executar o projeto:

1. No Android Studio, execute o projeto ao selecionar **Run &gt; Run…**.
2. Selecione um dispositivo Android conectado com uma câmera para testar o aplicativo.

As próximas seções mostram as modificações necessárias no projeto existente para adicionar essa funcionalidade ao seu próprio aplicativo, usando esse aplicativo de exemplo como um ponto de referência.

## Adicione as dependências do projeto {:#add_dependencies}

No seu próprio aplicativo, você precisa adicionar as dependências do projeto para executar os modelos de aprendizado de máquina do TensorFlow e acessar funções utilitárias que convertem dados como imagens em um formato de dados de tensor que pode ser processado pelo modelo que você está usando.

O aplicativo de exemplo usa a [Biblioteca Task para visão](../../inference_with_metadata/task_library/overview#supported_tasks) do TensorFlow Lite para permitir a execução do modelo de aprendizado de máquina de detecção de objetos. As instruções a seguir explicam como adicionar as dependências de biblioteca necessárias para o seu próprio projeto de aplicativo Android.

As instruções a seguir explicam como adicionar as dependências de projeto e módulo necessárias ao seu próprio projeto de aplicativo Android.

Para adicionar dependências de módulo:

1. No módulo que usa o TensorFlow Lite, atualize o arquivo `build.gradle` para que inclua as seguintes dependências. No código de exemplo, esse arquivo está localizado aqui: `...examples/lite/examples/object_detection/android/app/build.gradle` ([código de referência](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/build.gradle))

    ```
    dependencies {
      ...
      implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
      // Import the GPU delegate plugin Library for GPU inference
      implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0'
    }
    ```

    O projeto precisa incluir a Biblioteca Task para visão (`tensorflow-lite-task-vision`). A biblioteca da unidade de processamento gráfico (GPU) (`tensorflow-lite-gpu-delegate-plugin`) fornece a infraestrutura para executar o aplicativo na GPU e o Delegado (`tensorflow-lite-gpu`) oferece a lista de compatibilidade.

2. No Android Studio, sincronize as dependências do projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

## Inicialize o modelo de ML

No seu aplicativo Android, você precisa inicializar o modelo de aprendizado de máquina do TensorFlow com parâmetros antes de realizar previsões com o modelo. Esses parâmetros de inicialização são os mesmos em todos os modelos de detecção de objetos e podem incluir configurações como limites de exatidão mínima para previsões.

Um modelo do TensorFlow Lite inclui um arquivo `.tflite` com o código do modelo e, geralmente, um arquivo de rótulos com o nome das classes previstas pelo modelo. No caso da detecção de objetos, as classes são objetos como pessoa, cachorro, gato ou carro.

Este exemplo baixa vários modelos especificados em `download_models.gradle`, e a classe `ObjectDetectorHelper` fornece um seletor para os modelos:

```
val modelName =
  when (currentModel) {
    MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
    MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
    MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
    MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
    else -> "mobilenetv1.tflite"
  }
```

Ponto importante: os modelos devem ser armazenados no diretório `src/main/assets` do seu projeto de desenvolvimento. A TensorFlow Lite Task Library verifica automaticamente esse diretório quando você especifica o nome de arquivo do modelo.

Para inicializar o modelo no seu aplicativo:

1. Adicione um arquivo de modelo `.tflite` ao diretório `src/main/assets` do seu projeto de desenvolvimento, como: [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1).

2. Defina uma variável estática para o nome de arquivo do seu modelo. No aplicativo de exemplo, defina a variável `modelName` como `MODEL_EFFICIENTDETV0` para usar o modelo de detecção EfficientDet-Lite0.

3. Defina as opções para o modelo, como o limite de previsão, o tamanho do conjunto de resultados e, opcionalmente, delegados de aceleração de hardware:

    ```
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setScoreThreshold(threshold)
        .setMaxResults(maxResults)
    ```

4. Use as configurações desse objeto para construir um objeto [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) do TensorFlow Lite que contém o modelo:

    ```
    objectDetector =
      ObjectDetector.createFromFileAndOptions(
        context, modelName, optionsBuilder.build())
    ```

O `setupObjectDetector` configura os seguintes parâmetros do modelo:

- Limite de detecção
- Número máximo de resultados de detecção
- Número de threads de processamento para usar (`BaseOptions.builder().setNumThreads(numThreads)`)
- Próprio modelo (`modelName`)
- Objeto ObjectDetector (`objectDetector`)

### Configure o acelerador de hardware

Ao inicializar um modelo do TensorFlow Lite no seu aplicativo, você pode usar as características da aceleração de hardware para acelerar os cálculos de previsão do modelo.

Os *delegados* do TensorFlow Lite são módulos de software que aceleram a execução de modelos de aprendizado de máquina usando hardware de processamento especializado em um dispositivo móvel, como Unidades de processamento gráfico (GPUs), Unidades de processamento de tensor (TPUs) e Processadores de sinal digital (DSPs). O uso de delegados para executar modelos do TensorFlow Lite é recomendado, mas não obrigatório.

O detector de objetos é inicializado usando as configurações atuais no thread que está o usando. Você pode usar a CPU e os delegados [NNAPI](../../android/delegates/nnapi) com detectores criados no thread principal e usados em um thread em segundo plano, mas o thread que inicializou o detector precisa usar o delegado de GPU.

Os delegados são definidos na função `ObjectDetectionHelper.setupObjectDetector()`:

```
when (currentDelegate) {
    DELEGATE_CPU -> {
        // Default
    }
    DELEGATE_GPU -> {
        if (CompatibilityList().isDelegateSupportedOnThisDevice) {
            baseOptionsBuilder.useGpu()
        } else {
            objectDetectorListener?.onError("GPU is not supported on this device")
        }
    }
    DELEGATE_NNAPI -> {
        baseOptionsBuilder.useNnapi()
    }
}
```

Para mais informações sobre como usar delegados de aceleração de hardware com o TensorFlow Lite, confira os [Delegados do TensorFlow Lite](../../performance/delegates).

## Prepare os dados para o modelo

No seu aplicativo Android, seu código fornece dados ao modelo para interpretação ao transformar dados existentes, como frames de imagem, em um formato de dados de Tensor que pode ser processado pelo modelo. Os dados em um Tensor passados a um modelo precisam ter dimensões específicas, ou um formato, que correspondam ao formato dos dados usados para treinar o modelo.

O modelo [EfficientDet-Lite0](https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1) usado nesse código de exemplo aceita Tensores que representam imagens com uma dimensão de 320 x 320 e três canais (vermelho, azul e verde) por pixel. Cada valor no tensor é um único byte entre 0 e 255. Então, para realizar previsões com novas imagens, seu aplicativo precisa transformar esses dados de imagem em objetos de dados de Tensor desse tamanho e formato. A API TensorFlow Lite Task Library Vision do TensorFlow Lite faz a transformação dos dados para você.

O aplicativo usa um objeto [`ImageAnalysis`](https://developer.android.com/training/camerax/analyze) para extrair as imagens da câmera. Esse objeto chama a função `detectObject` com bitmap da câmera. Os dados são automaticamente redimensionados e girados pelo `ImageProcessor`, para que atendam aos requisitos de dados de imagem do modelo. Em seguida, a imagem é traduzida em um objeto [`TensorImage`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/support/image/TensorImage).

Para preparar os dados do subsistema da câmera para serem processados pelo modelo de ML:

1. Crie um objeto `ImageAnalysis` para extrair imagens no formato necessário:

    ```
    imageAnalyzer =
        ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            ...
    ```

2. Conecte o analisador ao subsistema da câmera e crie um buffer de bitmap para conter os dados recebidos da câmera:

    ```
    .also {
      it.setAnalyzer(cameraExecutor) {
        image -> if (!::bitmapBuffer.isInitialized)
        { bitmapBuffer = Bitmap.createBitmap( image.width, image.height,
        Bitmap.Config.ARGB_8888 ) } detectObjects(image)
        }
      }
    ```

3. Extraia os dados de imagem específicos de que o modelo precisa e passe as informações de rotação da imagem:

    ```
    private fun detectObjects(image: ImageProxy) {
      //Copy out RGB bits to the shared bitmap buffer
      image.use {bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
      }
    ```

4. Conclua quaisquer transformações de dados finais e adicione os dados de imagem a um objeto `TensorImage`, conforme mostrado no método `ObjectDetectorHelper.detect()` do aplicativo de exemplo:

    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()
    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

Observação: ao extrair as informações de imagem do subsistema da câmera Android, garanta que a imagem esteja no formato RGB. Esse formato é exigido pela classe ImageProcessor do TensorFlow Lite, que você usará para preparar a imagem para a análise pelo modelo. Se a imagem de formato RGB tiver um canal alfa, esses dados de transparência são ignorados.

## Realize previsões

No seu aplicativo Android, depois de criar um objeto TensorImage com dados de imagem no formato correto, você pode executar o modelo com esses dados para gerar uma previsão, ou *inferência*.

Na classe `fragments/CameraFragment.kt` do aplicativo de exemplo, o objeto `imageAnalyzer` na função `bindCameraUseCases` passa automaticamente os dados ao modelo para previsões quando o aplicativo está conectado à câmera.

O aplicativo usa o método `cameraProvider.bindToLifecycle()` para lidar com o seletor de câmera, a janela de visualização e o processamento de modelo de ML. A classe `ObjectDetectorHelper.kt` passa os dados de imagem ao modelo. Para executar o modelo e gerar previsões a partir dos dados de imagem:

- Execute a previsão ao passar os dados de imagem para sua função de previsão:

    ```
    val results = objectDetector?.detect(tensorImage)
    ```

O objeto Interpreter do TensorFlow Lite recebe esses dados, executa com o modelo e produz uma lista de previsões. Para o processamento contínuo de dados pelo modelo, use o método `runForMultipleInputsOutputs()` para que os objetos Interpreter não sejam criados e depois removidos pelo sistema a cada previsão realizada.

## Processe a saída do modelo

No seu aplicativo Android, depois de executar os dados de imagem com o modelo de detecção de objetos, ele produz uma lista de previsões com que o código do aplicativo precisa lidar ao executar lógica de negócios adicional, mostrar os resultados ao usuário ou realizar outras ações.

A saída de qualquer modelo do TensorFlow Lite varia em termos de número de previsões produzidas (uma ou mais) e informações descritivas de cada previsão. No caso de um modelo de detecção de objetos, as previsões geralmente incluem dados para uma caixa delimitadora que indica onde um objeto é detectado na imagem. No código de exemplo, os resultados são passados à função `onResults` em `CameraFragment.kt`, que atua como um DetectorListener no processo de detecção de objetos.

```
interface DetectorListener {
  fun onError(error: String)
  fun onResults(
    results: MutableList<Detection>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int
  )
}
```

Para o modelo usado nesse exemplo, cada previsão inclui um local de caixa delimitadora para o objeto, um rótulo para o objeto e uma pontuação entre 0 e 1 como um Float representando a confiança da previsão, sendo 1 a pontuação mais alta. Em geral, as previsões com uma pontuação abaixo de 50% (0,5) são consideradas inconclusivas. No entanto, cabe a você decidir como lida com os resultados de previsão de valores baixos e as necessidades do seu aplicativo.

Para lidar com os resultados de previsão do modelo:

1. Use um padrão listener para passar os resultados ao código do seu aplicativo ou objetos de interface do usuário. O aplicativo de exemplo usa esse padrão para passar os resultados de detecção do objeto `ObjectDetectorHelper` para o objeto `CameraFragment`:

    ```
    objectDetectorListener.onResults(
    // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```

2. Realize ações em relação aos resultados, como exibir a previsão para o usuário. O exemplo desenha um overlay no objeto CameraPreview para mostrar o resultado:

    ```
    override fun onResults(
      results: MutableList<Detection>?,
      inferenceTime: Long,
      imageHeight: Int,
      imageWidth: Int
    ) {
        activity?.runOnUiThread {
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Pass necessary information to OverlayView for drawing on the canvas
            fragmentCameraBinding.overlay.setResults(
                results ?: LinkedList<Detection>(),
                imageHeight,
                imageWidth
            )

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }
    ```

Depois que o modelo retornar um resultado de previsão, seu aplicativo pode agir em relação a ela ao apresentar o resultado ao seu usuário ou executar lógica adicional. No caso do código de exemplo, o aplicativo desenha uma caixa delimitadora em volta do objeto identificado e exibe o nome da classe na tela.

## Próximos passos

- Descubra os usos do TensorFlow Lite nos [exemplos](../../examples).
- Saiba mais sobre como usar modelos de aprendizado de máquina com o TensorFlow Lite na seção [Modelos](../../models).
- Saiba mais sobre como implementar o aprendizado de máquina no seu aplicativo para dispositivos móveis no [Guia para desenvolvedores do TensorFlow Lite](../../guide).

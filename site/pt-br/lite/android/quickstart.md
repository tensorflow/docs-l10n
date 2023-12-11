# Guia rápido para Android

Esta página mostra como criar um app Android com o TensorFlow Lite para analisar um feed de câmera ao vivo e identificar objetos. Esse caso de uso de aprendizado de máquina é chamado de *detecção de objetos*. O app de exemplo usa a [Biblioteca Task Vision do TensorFlow Lite](../inference_with_metadata/task_library/overview#supported_tasks) pelo [Google Play Services](../inference_with_metadata/task_library/overview#supported_tasks) para permitir a execução do modelo de aprendizado de máquina de detecção de objetos, que é a abordagem recomendada para criar um aplicativo de ML com o TensorFlow Lite.

<aside class="note"> <b>Termos:</b> ao acessar ou usar o TensorFlow Lite nas APIs Google Play Services, você concorda com os <a href="./play_services#tos">Termos de Serviço</a>. Por favor, leia e entenda todos os termos e políticas aplicáveis antes de acessar as APIs</aside>

![Demonstração animada de detecção de objeto](https://storage.googleapis.com/download.tensorflow.org/tflite/examples/obj_detection_cat.gif){: .attempt-right width="250px"}

## Configure e execute o exemplo

Para a primeira parte deste exercício, baixe o [código de exemplo](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android_play_services) do GitHub e o execute usando o [Android Studio](https://developer.android.com/studio/). As seções a seguir deste documento exploram as seções relevantes do código de exemplo, para você aplicá-las aos seus próprios apps Android. Você precisa das seguintes versões destas ferramentas instaladas:

- Android Studio 4.2 ou mais recente
- Versão 21 ou mais recente do SDK do Android

Observação: este exemplo usa a câmera, então tente executá-lo em um dispositivo Android físico.

### Obtenha o código de exemplo

Crie uma cópia local do código de exemplo para compilar e executar.

Para clonar e configurar o código de exemplo:

1. Clone o repositório git
    <pre class="devsite-click-to-copy">    git clone https://github.com/tensorflow/examples.git
        </pre>
2. Configure sua instância git para usar o sparse checkout e ter somente os arquivos para o app de exemplo de detecção de objetos:
    <pre class="devsite-click-to-copy">    cd examples
        git sparse-checkout init --cone
        git sparse-checkout set lite/examples/object_detection/android_play_services
        </pre>

### Importe e execute o projeto

Use o Android Studio para criar um projeto a partir do código de exemplo baixado, compile e depois execute esse projeto.

Para importar e compilar o projeto do código de exemplo:

1. Inicie o [Android Studio](https://developer.android.com/studio).
2. Na página **Welcome** de boas-vindas ao Android, escolha **Import Project** (Importar projeto) ou selecione **File &gt; New &gt; Import Project** (Arquivo &gt; Novo &gt; Importar projeto).
3. Acesse o diretório de código de exemplo com o arquivo build.gradle (`...examples/lite/examples/object_detection/android_play_services/build.gradle`) e selecione esse diretório.

Depois de selecionar esse diretório, o Android Studio cria e compila um novo projeto. Quando o build é concluído, o Android Studio exibe uma mensagem `BUILD SUCCESSFUL` no painel de status **Build Output** (Saída do build).

Para executar o projeto:

1. No Android Studio, execute o projeto ao selecionar **Run &gt; Run…** e **MainActivity**
2. Selecione um dispositivo Android anexado com uma câmera para testar o app.

## Como funciona o app de exemplo

O app de exemplo usa o modelo de detecção de objetos pré-treinado, como [mobilenetv1.tflite](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite), no formato do TensorFlow Lite para procurar objetos em um stream de vídeo ao vivo da câmera de um dispositivo Android. O código desse recurso está principalmente nestes arquivos:

- [ObjectDetectorHelper.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper.kt): inicializa o ambiente de runtime, permite a aceleração de hardware e executa o modelo de ML para detecção de objetos.
- [CameraFragment.kt](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android_play_services/app/src/main/java/org/tensorflow/lite/examples/objectdetection/fragments/CameraFragment.kt): compila o stream de dados da imagem da câmera, prepara os dados para o modelo e exibe os resultados da detecção de objetos.

Observação: esse app de exemplo usa a [Biblioteca Task](../inference_with_metadata/task_library/overview#supported_tasks), do TensorFlow Lite, que fornece APIs fáceis de usar e específicas a tarefas para realizar operações de aprendizado de máquina comuns. Para apps com necessidades mais específicas e funções de ML personalizadas, considere usar a [API Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).

As próximas seções mostram a você os principais componentes desses arquivos de código, para você modificar um app Android a fim de adicionar essa funcionalidade.

## Crie o app {:#build_app}

As seguintes seções explicam os principais passos para criar seu próprio app Android e executar o modelo mostrado no app de exemplo. Essas instruções usam o app de exemplo anterior como ponto de referência.

Observação: para seguir essas instruções e criar seu próprio app, crie um [projeto Android básico](https://developer.android.com/studio/projects/create-project) usando o Android Studio.

### Adicione as dependências do projeto {:#add_dependencies}

No seu app Android básico, adicione as dependências do projeto para executar os modelos de aprendizado de máquina do TensorFlow Lite e acessar as funções utilitárias de dados de ML. Essas funções convertem dados como imagens em um formato de dados de tensor que pode ser processado por um modelo.

O app de exemplo usa a [Biblioteca Task Vision](../inference_with_metadata/task_library/overview#supported_tasks) do TensorFlow Lite no [Google Play Services](./play_services) para permitir a execução do modelo de aprendizado de máquina de detecção de objetos. As instruções a seguir explicam como adicionar as dependências de biblioteca necessárias para seu próprio projeto de app Android.

Para adicionar dependências de módulo:

1. No módulo do app Android que usa o TensorFlow Lite, atualize o arquivo `build.gradle` do módulo para incluir as seguintes dependências. No código de exemplo, este arquivo está localizado aqui: `...examples/lite/examples/object_detection/android_play_services/app/build.gradle`
    ```
    ...
    dependencies {
    ...
        // Tensorflow Lite dependencies
        implementation 'org.tensorflow:tensorflow-lite-task-vision-play-services:0.4.2'
        implementation 'com.google.android.gms:play-services-tflite-gpu:16.1.0'
    ...
    }
    ```
2. No Android Studio, sincronize as dependências do projeto ao selecionar: **File &gt; Sync Project with Gradle Files** (Arquivo &gt; Sincronizar projeto com arquivos gradle).

### Inicialize o Google Play Services

Ao usar o [Google Play Services](./play_services) para executar os modelos do TensorFlow Lite, você precisa inicializar o serviço antes de poder usá-lo. Se você quiser usar o suporte de aceleração de hardware com o serviço, como a aceleração de GPU, você também ativa esse suporte como parte dessa inicialização.

Para inicializar o TensorFlow Lite com o Google Play Services:

1. Crie um objeto `TfLiteInitializationOptions` e modifique-o para ativar o suporte de GPU:

    ```
    val options = TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ```

2. Use o método `TfLiteVision.initialize()` para habilitar o uso do runtime do Play Services e defina um listener para verificar se foi carregado com êxito:

    ```
    TfLiteVision.initialize(context, options).addOnSuccessListener {
        objectDetectorListener.onInitialized()
    }.addOnFailureListener {
        // Called if the GPU Delegate is not supported on the device
        TfLiteVision.initialize(context).addOnSuccessListener {
            objectDetectorListener.onInitialized()
        }.addOnFailureListener{
            objectDetectorListener.onError("TfLiteVision failed to initialize: "
                    + it.message)
        }
    }
    ```

### Inicialize o interpretador de modelo de ML

Inicialize o interpretador de modelo de aprendizado de máquina do TensorFlow Lite ao carregar o arquivo de modelo e configurar os parâmetros do modelo. Um modelo do TensorFlow Lite inclui um arquivo `.tflite` com o código do modelo. Você deve armazenar seus modelos no diretório `src/main/assets` do seu projeto de desenvolvimento, por exemplo:

```
.../src/main/assets/mobilenetv1.tflite`
```

Dica: o código do interpretador da Biblioteca Task busca modelos no diretório `src/main/assets` se você não especificar um caminho de arquivo.

Para inicializar o modelo:

1. Adicione um arquivo de modelo `.tflite` ao diretório `src/main/assets` do seu projeto de desenvolvimento, como [ssd_mobilenet_v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2).
2. Defina a variável `modelName` para especificar o nome de arquivo do modelo de ML:
    ```
    val modelName = "mobilenetv1.tflite"
    ```
3. Defina as opções para o modelo, como o limite de previsão e o tamanho do conjunto de resultados:
    ```
    val optionsBuilder =
        ObjectDetector.ObjectDetectorOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
    ```
4. Ative a aceleração de GPU com as opções e permita que o código falhe graciosamente se a aceleração não for compatível no dispositivo:
    ```
    try {
        optionsBuilder.useGpu()
    } catch(e: Exception) {
        objectDetectorListener.onError("GPU is not supported on this device")
    }

    ```
5. Use as configurações desse objeto para construir um objeto [`ObjectDetector`](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/task/vision/detector/ObjectDetector#createFromFile(Context,%20java.lang.String)) do TensorFlow Lite que contém o modelo:
    ```
    objectDetector =
        ObjectDetector.createFromFileAndOptions(
            context, modelName, optionsBuilder.build())
    ```

Para mais informações sobre como usar delegados de aceleração de hardware com o TensorFlow Lite, confira os [Delegados do TensorFlow Lite](../performance/delegates).

### Prepare os dados para o modelo

Você prepare os dados para interpretação pelo modelo ao transformar os dados existentes como imagens no formato de dados de [Tensor](../api_docs/java/org/tensorflow/lite/Tensor), para que possam ser processados pelo seu modelo. Os dados em um Tensor precisam ter dimensões específicas, ou formato, que correspondem ao formato dos usados para treinar o modelo. Dependendo do modelo usado, talvez seja necessário transformar os dados para adequar à expectativa do modelo. O app de exemplo usa um objeto [`ImageAnalysis`](https://developer.android.com/reference/androidx/camera/core/ImageAnalysis) para extrair frames de imagens do subsistema da câmera.

Para preparar os dados para processamento pelo modelo:

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
            it.setAnalyzer(cameraExecutor) { image ->
                if (!::bitmapBuffer.isInitialized) {
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width,
                        image.height,
                        Bitmap.Config.ARGB_8888
                    )
                }
                detectObjects(image)
            }
        }
    ```
3. Extraia os dados de imagens específicos de que o modelo precisa e passe as informações de rotação da imagem:
    ```
    private fun detectObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        val imageRotation = image.imageInfo.rotationDegrees
        objectDetectorHelper.detect(bitmapBuffer, imageRotation)
    }
    ```
4. Conclua quaisquer transformações de dados finais e adicione os dados de imagem a um objeto `TensorImage`, conforme mostrado no método `ObjectDetectorHelper.detect()` do app de exemplo:
    ```
    val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

    // Preprocess the image and convert it into a TensorImage for detection.
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
    ```

### Execute as previsões

Depois de criar um objeto [TensorImage](../api_docs/java/org/tensorflow/lite/support/image/TensorImage) com dados de imagem no formato correto, você pode executar o modelo com esses dados para produzir uma previsão ou *inferência*. No app de exemplo, esse código é contido no método `ObjectDetectorHelper.detect()`.

Para executar um modelo e gerar previsões a partir dos dados de imagem:

- Execute a previsão ao passar os dados de imagem para sua função de previsão:
    ```
    val results = objectDetector?.detect(tensorImage)
    ```

### Lide com a saída do modelo

Depois de executar os dados de imagem no modelo de detecção de objetos, ele produz uma lista de resultados de previsão com que o código do seu app precisa lidar ao executar lógica de negócios adicional, exibindo resultados ao usuário ou realizando outras ações. O modelo de detecção de objetos no app de exemplo produz uma lista de previsões e caixas delimitadoras para os objetos detectados. No app de exemplo, os resultados de previsão são passados para um objeto listener para processamento adicional e exibição ao usuário.

Para lidar com os resultados de previsão do modelo:

1. Use um padrão listener para passar os resultados ao código do seu app ou objetos de interface do usuário. O app de exemplo usa esse padrão para passar os resultados de detecção do objeto `ObjectDetectorHelper` para o objeto `CameraFragment`:
    ```
    objectDetectorListener.onResults( // instance of CameraFragment
        results,
        inferenceTime,
        tensorImage.height,
        tensorImage.width)
    ```
2. Realize ações em relação aos resultados, como exibir a previsão para o usuário. O app de exemplo desenha um overlay no objeto `CameraPreview` para mostrar o resultado:
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

## Próximos passos

- Saiba mais sobre as [APIs Biblioteca Task](../inference_with_metadata/task_library/overview#supported_tasks)
- Saiba mais sobre as [APIs Interpreter](https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/InterpreterApi).
- Descubra os usos do TensorFlow Lite nos [exemplos](../examples).
- Saiba mais sobre como usar e criar modelos de aprendizado de máquina com o TensorFlow Lite na seção [Modelos](../models).
- Saiba mais sobre como implementar o aprendizado de máquina no seu aplicativo para dispositivos móveis no [Guia para desenvolvedores do TensorFlow Lite](../guide).

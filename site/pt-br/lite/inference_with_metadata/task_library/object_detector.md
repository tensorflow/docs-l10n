# Integração de detectores de objetos

Os detectores de objetos conseguem identificar qual objeto dentre um conjunto conhecido de objetos pode estar presente e fornecem informações sobre suas posições na imagem ou stream de vídeo fornecido. Um detector de objetos é treinado para detectar a presença e a localização de diversas classes de objetos. Por exemplo: um modelo pode ser treinado com imagens que contenham diversos pedaços de frutas, junto com um *rótulo* que especifica a classe da fruta representada (como maçã, banana ou morango), além de dados especificando onde cada objeto aparece na imagem. Confira mais informações sobre detectores de objetos na [introdução à detecção de objetos](../../examples/object_detection/overview).

Use a API `ObjectDetector` da biblioteca Task para implantar detectores de objetos personalizados ou pré-treinados em seus aplicativos para dispositivos móveis.

## Principais recursos da API ObjectDetector

- Processamento da imagem de entrada, incluindo rotação, redimensionamento e conversão do espaço de cores.

- Idioma do mapa de rótulos.

- Limite de pontuação para filtrar resultados.

- Resultados de detecção top-k.

- Lista de permissão e proibição de rótulos.

## Modelos de detectores de objetos com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `ObjectDetector`.

- [Modelos de detecção de objetos pré-treinados no TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/object-detector/1).

- Modelos criados pela [detecção de objetos do AutoML Vision Edge](https://cloud.google.com/vision/automl/object-detection/docs).

- Modelos criados pelo [Model Maker do TensorFlow Lite para detecção de objetos](https://www.tensorflow.org/lite/guide/model_maker).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

Confira o [aplicativo de referência para detecção de objetos](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/) para ver um exemplo de como usar o `ObjectDetector` em um aplicativo para Android.

### Etapa 1 – Importe a dependência e outras configurações do Gradle

Copie o arquivo do modelo `.tflite` para o diretório de ativos do módulo para Android no qual o modelo será executado. Especifique que o arquivo não deve ser compactado e adicione a biblioteca do TensorFlow Lite ao arquivo `build.gradle` do modelo:

```java
android {
    // Other settings

    // Specify tflite file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin'
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

### Etapa 2 – Use o modelo

```java
// Initialization
ObjectDetectorOptions options =
    ObjectDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ObjectDetector objectDetector =
    ObjectDetector.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Detection> results = objectDetector.detect(image);
```

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/detector/ObjectDetector.java) para ver mais opções de configuração do `ObjectDetector`.

## Execute a inferência no iOS

### Etapa 1 – Instale as dependências

A biblioteca Task oferece suporte à instalação usando o CocoaPods, que precisa estar instalado em seu sistema. Confira as instruções no [guia de instalação do CocoaPods](https://guides.cocoapods.org/using/getting-started.html#getting-started).

Confira os detalhes de como adicionar pods a um projeto do Xcode no [guia do CocoaPods](https://guides.cocoapods.org/using/using-cocoapods.html).

Adicione o pod `TensorFlowLiteTaskVision` ao Podfile.

```
target 'MyAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskVision'
end
```

Confirme se o modelo `.tflite` que você usará para inferência está presente no pacote do aplicativo.

### Etapa 2 – Use o modelo

#### Swift

```swift
// Imports
import TensorFlowLiteTaskVision

// Initialization
guard let modelPath = Bundle.main.path(forResource: "ssd_mobilenet_v1",
                                            ofType: "tflite") else { return }

let options = ObjectDetectorOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let detector = try ObjectDetector.detector(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "cats_and_dogs.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let detectionResult = try detector.detect(mlImage: mlImage)
```

#### Objective-C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"ssd_mobilenet_v1" ofType:@"tflite"];

TFLObjectDetectorOptions *options = [[TFLObjectDetectorOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLObjectDetector *detector = [TFLObjectDetector objectDetectorWithOptions:options
                                                                     error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"dogs.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLDetectionResult *detectionResult = [detector detectWithGMLImage:gmlImage error:nil];
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLObjectDetector.h) para ver mais opções de configuração do `TFLObjectDetector`.

## Execute a inferência no Python

### Etapa 1 – Instale o pacote via pip

```
pip install tflite-support
```

### Etapa 2 – Use o modelo

```python
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(max_results=2)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/object_detector.py) para ver mais opções de configuração do `ObjectDetector`.

## Execute a inferência no C++

```c++
// Initialization
ObjectDetectorOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ObjectDetector> object_detector = ObjectDetector::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const DetectionResult result = object_detector->Detect(*frame_buffer).value();
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/object_detector.h) para ver mais opções de configuração do `ObjectDetector`.

## Exemplo de resultados

Veja abaixo um exemplo dos resultados de detecção de [ssd mobilenet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1) do TensorFlow Hub.


<img src="images/dogs.jpg" width="50%" alt="dogs">

```
Results:
 Detection #0 (red):
  Box: (x: 355, y: 133, w: 190, h: 206)
  Top-1 class:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1 (green):
  Box: (x: 103, y: 15, w: 138, h: 369)
  Top-1 class:
   index       : 17
   score       : 0.73047
   class name  : dog
```

Renderize os retângulos limítrofes na imagem de entrada:


<img src="images/detection-output.png" width="50%" alt="detection output">

Experimente a [ferramenta CLI simples de demonstração para ObjectDetector](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#object-detector) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `ObjectDetector` espera um modelo do TF Lite com os [TF Lite Model Metadata](../../models/convert/metadata) (metadados de modelo) obrigatórios. Confira exemplos de criação dos metadados para detectores de objetos na [API Metadata Writer do TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#object_detectors) (gravação de metadados).

Os modelos compatíveis de detecção de objetos devem atender aos seguintes requisitos:

- Tensor de imagem de entrada: (kTfLiteUInt8/kTfLiteFloat32).

    - Imagem de entrada de tamanho `[batch x height x width x channels]` (lote x altura x largura x canais).
    - Não há suporte à inferência de lote (`batch` precisa ser igual a 1).
    - Só há suporte a entradas RGB (`channels` precisa ser igual a 3).
    - Se o tipo for kTfLiteFloat32, as opções NormalizationOptions precisam ser adicionadas aos metadados para a normalização da entrada.

- Os tensores de saída devem ser as 4 saídas de uma operação `DetectionPostProcess`:

    - Tensor de localizações (kTfLiteFloat32)

        - Tensor de tamanho `[1 x num_results x 4]`, em que o array interno representa os retângulos limítrofes no formato [top, left, right, bottom] (superior, esquerda, direita, inferior).
        - É preciso incluir BoundingBoxProperties nos metadados e especificar `type=BOUNDARIES` e `coordinate_type=RATIO.

    - Tensor de classes (kTfLiteFloat32)

        - Tensor de tamanho `[1 x num_results]`, em que cada valor representa o índice Int de uma classe.
        - Mapa(s) de rótulos (opcionais, mas recomendados), podem ser incluídos como AssociatedFiles com tipo TENSOR_VALUE_LABELS, contendo um rótulo por linha. Confira o [arquivo de rótulos de exemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/object_detector/labelmap.txt). O primeiro AssociatedFile (se houver) é usado para preencher o campo `class_name` dos resultados. O campo `display_name` é preenchido a partir do AssociatedFile (se houver) cujo idioma coincida com o campo `display_names_locale` das opções `ObjectDetectorOptions` usadas no momento da criação ("en" por padrão, ou seja, inglês). Se nenhum estiver disponível, somente o campo `index` dos resultados será preenchido.

    - Tensor de pontuações (kTfLiteFloat32)

        - Tensor de tamanho `[1 x num_results]`, em que cada valor representa a pontuação do objeto detectado.

    - Número de tensores de detecção (kTfLiteFloat32)

        - num_results Int como um tensor de tamanho `[1]`.

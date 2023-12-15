# Integração de classificadores de imagem

A classificação de imagens é um uso comum de aprendizado de máquina para identificar o que uma imagem representa. Por exemplo: talvez a gente queira saber que tipo de animal aparece em uma determinada imagem. A tarefa de prever o que uma imagem representa é chamada de *classificação de imagem*. Um classificador de imagem é treinado para reconhecer diversas classes de imagens. Por exemplo: um modelo pode ser treinado para reconhecer fotos que representem três tipos diferentes de animais: coelhos, hamsters e cachorros. Veja mais informações sobre classificadores de imagem na [visão geral da classificação de imagens](https://www.tensorflow.org/lite/examples/image_classification/overview).

Use a API `ImageClassifier` da biblioteca Task para implantar classificadores de imagem personalizados ou pré-treinados em seus aplicativos para dispositivos móveis.

## Principais recursos da API ImageClassifier

- Processamento da imagem de entrada, incluindo rotação, redimensionamento e conversão do espaço de cores.

- Região de interesse da imagem de entrada.

- Idioma do mapa de rótulos.

- Limite de pontuação para filtrar resultados.

- Resultados de classificação top-k.

- Lista de permissão e proibição de rótulos.

## Modelos de classificador de imagem com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `ImageClassifier`.

- Modelos criados pelo [Model Maker do TensorFlow Lite para classificação de imagem](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification).

- [Modelos pré-treinados de classificação de imagens no TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1).

- Modelos criados pela [classificação de imagens do AutoML Vision Edge](https://cloud.google.com/vision/automl/docs/edge-quickstart).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

Confira o [aplicativo de referência para classificação de imagens](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) para ver um exemplo de como usar o `ImageClassifier` em um aplicativo para Android.

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

### Etapa 2 – Use o modelo

```java
// Initialization
ImageClassifierOptions options =
    ImageClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setMaxResults(1)
        .build();
ImageClassifier imageClassifier =
    ImageClassifier.createFromFileAndOptions(
        context, modelFile, options);

// Run inference
List<Classifications> results = imageClassifier.classify(image);
```

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java) para ver mais opções de configuração do `ImageClassifier`.

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
guard let modelPath = Bundle.main.path(forResource: "birds_V1",
                                            ofType: "tflite") else { return }

let options = ImageClassifierOptions(modelPath: modelPath)

// Configure any additional options:
// options.classificationOptions.maxResults = 3

let classifier = try ImageClassifier.classifier(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "sparrow.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let classificationResults = try classifier.classify(mlImage: mlImage)
```

#### Objective-C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"birds_V1" ofType:@"tflite"];

TFLImageClassifierOptions *options =
    [[TFLImageClassifierOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.classificationOptions.maxResults = 3;

TFLImageClassifier *classifier = [TFLImageClassifier imageClassifierWithOptions:options
                                                                          error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"sparrow.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLClassificationResult *classificationResult =
    [classifier classifyWithGMLImage:gmlImage error:nil];
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h) para ver mais opções de configuração do `TFLImageClassifier`.

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
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
classification_result = classifier.classify(image)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_classifier.py) para ver mais opções de configuração do `ImageClassifier`.

## Execute a inferência no C++

```c++
// Initialization
ImageClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageClassifier> image_classifier = ImageClassifier::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h

std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const ClassificationResult result = image_classifier->Classify(*frame_buffer).value();
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_classifier.h) para ver mais opções de configuração do `ImageClassifier`.

## Exemplo de resultados

Veja um exemplo dos resultados de classificação do [classificador de pássaros](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3).


<img src="images/sparrow.jpg" width="50%" alt="sparrow">

```
Results:
  Rank #0:
   index       : 671
   score       : 0.91406
   class name  : /m/01bwb9
   display name: Passer domesticus
  Rank #1:
   index       : 670
   score       : 0.00391
   class name  : /m/01bwbt
   display name: Passer montanus
  Rank #2:
   index       : 495
   score       : 0.00391
   class name  : /m/0bwm6m
   display name: Passer italiae
```

Experimente a [ferramenta CLI simples de demonstração para ImageClassifier](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-classifier) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `ImageClassifier` espera um modelo do TF Lite com os [metadados de modelo do TF Lite](https://www.tensorflow.org/lite/models/convert/metadata) obrigatórios. Confira exemplos de criação dos metadados para classificadores de imagem na [API de gravação de metadados do TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#image_classifiers).

Os modelos compatíveis de classificador de imagem devem atender aos seguintes requisitos:

- Um tensor de imagem de entrada (kTfLiteUInt8/kTfLiteFloat32).

    - Imagem de entrada de tamanho `[batch x height x width x channels]` (lote x altura x largura x canais).
    - Não há suporte à inferência de lote (`batch` precisa ser igual a 1).
    - Só há suporte a entradas RGB (`channels` precisa ser igual a 3).
    - Se o tipo for kTfLiteFloat32, as opções NormalizationOptions precisam ser adicionadas aos metadados para a normalização da entrada.

- Tensor de pontuação de saída (kTfLiteUInt8/kTfLiteFloat32).

    - Com `N` classes e 2 ou 4 dimensões, ou seja, `[1 x N]` ou `[1 x 1 x 1 x N]`
    - Mapa(s) de rótulos (opcionais, mas recomendados), como AssociatedFiles com tipo TENSOR_AXIS_LABELS, contendo um rótulo por linha. Confira o [arquivo de rótulos de exemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/labels.txt). O primeiro AssociatedFile (se houver) é usado para preencher o campo `label` (com nome igual a `class_name` no C++) dos resultados. O campo `display_name` é preenchido a partir do AssociatedFile (se houver) cujo idioma coincida com o campo `display_names_locale` das opções `ImageClassifierOptions` usadas no momento da criação ("en" por padrão, ou seja, inglês). Se nenhum estiver disponível, somente o campo `index` dos resultados será preenchido.

# Integração de segmentadores de imagem

Os segmentadores de imagens preveem se cada pixel de uma imagem está associado a uma determinada classe. Isso é diferente da <a href="../../examples/object_detection/overview">detecção de objetos</a>, que detecta objetos em regiões retangulares, e da <a href="../../examples/image_classification/overview">classificação de imagens</a>, que classifica a imagem de forma geral. Confira mais informações sobre segmentadores de imagem na [visão geral da segmentação de imagens](../../examples/segmentation/overview).

Use a API `ImageSegmenter` da biblioteca Task para implantar segmentadores de imagem personalizados ou pré-treinados em seus aplicativos para dispositivos móveis.

## Principais recursos da API ImageSegmenter

- Processamento da imagem de entrada, incluindo rotação, redimensionamento e conversão do espaço de cores.

- Idioma do mapa de rótulos.

- Dois tipos de saída, máscara de categoria e máscaras de confiança.

- Rótulo colorido para fins de exibição.

## Modelos de segmentador de imagem com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `ImageSegmenter`.

- [Modelos pré-treinados de segmentação de imagens no TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-segmenter/1).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

Confira o [aplicativo de referência para segmentação de imagens](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/) para ver um exemplo de como usar o `ImageSegmenter` em um aplicativo para Android.

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
ImageSegmenterOptions options =
    ImageSegmenterOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setOutputType(OutputType.CONFIDENCE_MASK)
        .build();
ImageSegmenter imageSegmenter =
    ImageSegmenter.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Segmentation> results = imageSegmenter.segment(image);
```

Confira o [código-fonte e o javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/segmenter/ImageSegmenter.java) para ver mais opções de configuração do `ImageSegmenter`.

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
guard let modelPath = Bundle.main.path(forResource: "deeplabv3",
                                            ofType: "tflite") else { return }

let options = ImageSegmenterOptions(modelPath: modelPath)

// Configure any additional options:
// options.outputType = OutputType.confidenceMasks

let segmenter = try ImageSegmenter.segmenter(options: options)

// Convert the input image to MLImage.
// There are other sources for MLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
guard let image = UIImage (named: "plane.jpg"), let mlImage = MLImage(image: image) else { return }

// Run inference
let segmentationResult = try segmenter.segment(mlImage: mlImage)
```

#### Objective-C

```objc
// Imports
#import <TensorFlowLiteTaskVision/TensorFlowLiteTaskVision.h>

// Initialization
NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"deeplabv3" ofType:@"tflite"];

TFLImageSegmenterOptions *options =
    [[TFLImageSegmenterOptions alloc] initWithModelPath:modelPath];

// Configure any additional options:
// options.outputType = TFLOutputTypeConfidenceMasks;

TFLImageSegmenter *segmenter = [TFLImageSegmenter imageSegmenterWithOptions:options
                                                                      error:nil];

// Convert the input image to MLImage.
UIImage *image = [UIImage imageNamed:@"plane.jpg"];

// There are other sources for GMLImage. For more details, please see:
// https://developers.google.com/ml-kit/reference/ios/mlimage/api/reference/Classes/GMLImage
GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];

// Run inference
TFLSegmentationResult *segmentationResult =
    [segmenter segmentWithGMLImage:gmlImage error:nil];
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/vision/sources/TFLImageSegmenter.h) para ver mais opções de configuração do `TFLImageSegmenter`.

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
segmentation_options = processor.SegmentationOptions(
    output_type=processor.SegmentationOptions.OutputType.CATEGORY_MASK)
options = vision.ImageSegmenterOptions(base_options=base_options, segmentation_options=segmentation_options)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Alternatively, you can create an image segmenter in the following manner:
# segmenter = vision.ImageSegmenter.create_from_file(model_path)

# Run inference
image_file = vision.TensorImage.create_from_file(image_path)
segmentation_result = segmenter.segment(image_file)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_segmenter.py) para ver mais opções de configuração do `ImageSegmenter`.

## Execute a inferência no C++

```c++
// Initialization
ImageSegmenterOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<ImageSegmenter> image_segmenter = ImageSegmenter::CreateFromOptions(options).value();

// Create input frame_buffer from your inputs, `image_data` and `image_dimension`.
// See more information here: tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h
std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      image_data, image_dimension);

// Run inference
const SegmentationResult result = image_segmenter->Segment(*frame_buffer).value();
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_segmenter.h) para ver mais opções de configuração do `ImageSegmenter`.

## Exemplo de resultados

Veja abaixo um exemplo dos resultados de segmentação de [deeplab_v3](https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/1), um modelo genérico de segmentação disponível no TensorFlow Hub.


<img src="images/plane.jpg" width="50%" alt="plane">

```
Color Legend:
 (r: 000, g: 000, b: 000):
  index       : 0
  class name  : background
 (r: 128, g: 000, b: 000):
  index       : 1
  class name  : aeroplane

# (omitting multiple lines for conciseness) ...

 (r: 128, g: 192, b: 000):
  index       : 19
  class name  : train
 (r: 000, g: 064, b: 128):
  index       : 20
  class name  : tv
Tip: use a color picker on the output PNG file to inspect the output mask with
this legend.
```

A máscara de categoria de segmentação deve ficar assim:


<img src="images/segmentation-output.png" width="30%" alt="segmentation-output">

Experimente a [ferramenta CLI simples de demonstração para ImageSegmenter](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#image-segmenter) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `ImageSegmenter` espera um modelo do TF Lite com os [TF Lite Model Metadata](../../models/convert/metadata) (metadados do modelo) obrigatórios. Confira exemplos de criação dos metadados para segmentadores de imagem na [API Metadata Writer do TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#image_segmenters) (gravação de metadados).

- Um tensor de imagem de entrada (kTfLiteUInt8/kTfLiteFloat32).

    - Imagem de entrada de tamanho `[batch x height x width x channels]` (lote x altura x largura x canais).
    - Não há suporte à inferência de lote (`batch` precisa ser igual a 1).
    - Só há suporte a entradas RGB (`channels` precisa ser igual a 3).
    - Se o tipo for kTfLiteFloat32, as opções NormalizationOptions precisam ser adicionadas aos metadados para a normalização da entrada.

- Tensor de máscara de saída (kTfLiteUInt8/kTfLiteFloat32).

    - Tensor de tamanho `[batch x mask_height x mask_width x num_classes]`, em que `batch` precisa ser igual a 1, `mask_width` e `mask_height` são as dimensões das máscaras de segmentação geradas pelo modelo, e `num_classes` é o número de classes permitidas pelo modelo.
    - Mapa(s) de rótulos (opcionais, mas recomendados), como AssociatedFiles com tipo TENSOR_AXIS_LABELS, contendo um rótulo por linha. O primeiro AssociatedFile (se houver) é usado para preencher o campo `label` (com nome igual a `class_name` no C++) dos resultados. O campo `display_name` é preenchido a partir do AssociatedFile (se houver) cujo idioma coincida com o campo `display_names_locale` das opções `ImageSegmenterOptions` usadas no momento da criação ("en" por padrão, ou seja, inglês). Se nenhum estiver disponível, somente o campo `index` dos resultados será preenchido.

# Processe dados de entrada e gere dados de saída com a TensorFlow Lite Support Library

Observação: no momento, a TensorFlow Lite Support Library só oferece suporte ao Android.

A maioria dos desenvolvedores de aplicativos para dispositivos móveis interagem com objetos tipados, como bitmaps, ou primitivos, como inteiros. Entretanto, a API do interpretador do TensorFlow Lite que executa o modelo de aprendizado de máquina no dispositivo usa tensores no formato ByteBuffer, que podem ser difíceis de depurar e manipular. A [TensorFlow Lite Android Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) (biblioteca de suporte para Android do TensorFlow Lite) foi criada para ajudar a processar as entradas e saídas de modelos do TensorFlow Lite e para facilitar o uso do interpretador do TF Lite.

## Como começar

### Importe a dependência e outras configurações do Gradle

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

    // Import tflite dependencies
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // The GPU delegate library is optional. Depend on it as needed.
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT'
    implementation 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly-SNAPSHOT'
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

Confira o [AAR da TensorFlow Lite Support Library hospedado no MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support) para ver as diferentes versões da biblioteca de suporte.

### Manipulação e conversão básicas de imagem

A TensorFlow Lite Support Library conta com um conjunto de métodos básicos de manipulação de imagem, como recorte e redimensionamento. Para usá-lo, crie um `ImagePreprocessor` e adicione as operações necessárias. Para converter a imagem para o formato de tensor exigido pelo interpretador do TensorFlow Lite, crie uma `TensorImage` a ser usada como entrada:

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .build();

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
TensorImage tensorImage = new TensorImage(DataType.UINT8);

// Analysis code for every frame
// Preprocess the image
tensorImage.load(bitmap);
tensorImage = imageProcessor.process(tensorImage);
```

O `DataType` de um tensor, bem como outras informações do modelo, pode ser lido por meio da [biblioteca de extração de metadados](../models/convert/metadata.md#read-the-metadata-from-models).

### Processamento básico de dados de áudio

A TensorFlow Lite Support Library também define uma classe `TensorAudio` que encapsula alguns métodos básicos de processamento de dados de áudio. Ela é usada principalmente junto com [AudioRecord](https://developer.android.com/reference/android/media/AudioRecord) e captura amostras de áudio em um buffer circular.

```java
import android.media.AudioRecord;
import org.tensorflow.lite.support.audio.TensorAudio;

// Create an `AudioRecord` instance.
AudioRecord record = AudioRecord(...)

// Create a `TensorAudio` object from Android AudioFormat.
TensorAudio tensorAudio = new TensorAudio(record.getFormat(), size)

// Load all audio samples available in the AudioRecord without blocking.
tensorAudio.load(record)

// Get the `TensorBuffer` for inference.
TensorBuffer buffer = tensorAudio.getTensorBuffer()
```

### Crie objetos de saída e execute o modelo

Antes de executar o modelo, precisamos criar os objetos container que armazenarão o resultado:

```java
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

Carregue o modelo e execute a inferência:

```java
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.InterpreterApi;

// Initialise the model
try{
    MappedByteBuffer tfliteModel
        = FileUtil.loadMappedFile(activity,
            "mobilenet_v1_1.0_224_quant.tflite");
    InterpreterApi tflite = new InterpreterFactory().create(
        tfliteModel, new InterpreterApi.Options());
} catch (IOException e){
    Log.e("tfliteSupport", "Error reading model", e);
}

// Running inference
if(null != tflite) {
    tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
}
```

### Como acessar o resultado

Os desenvolvedores podem acessar a saída diretamente por meio de `probabilityBuffer.getFloatArray()`. Se o modelo gerar uma saída quantizada, lembre-se de converter o resultado. Para o modelo MobileNet quantizado, o desenvolvedor precisa dividir cada valor da saída por 255 para obter a probabilidade, que vai de 0 (menos provável) a 1 (mais provável) para cada categoria.

### Opcional: mapeamento dos resultados em rótulos

Opcionalmente, os desenvolvedores podem mapear os resultados em rótulos. Primeiro, copie o arquivo de texto que contém os rótulos para o diretório de ativos do módulo. Em seguida, carregue o arquivo de rótulos por meio do seguinte código:

```java
import org.tensorflow.lite.support.common.FileUtil;

final String ASSOCIATED_AXIS_LABELS = "labels.txt";
List<String> associatedAxisLabels = null;

try {
    associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
} catch (IOException e) {
    Log.e("tfliteSupport", "Error reading label file", e);
}
```

O seguinte trecho de código demonstra como associar as probabilidades aos rótulos de categoria:

```java
import java.util.Map;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

if (null != associatedAxisLabels) {
    // Map of labels and their corresponding probability
    TensorLabel labels = new TensorLabel(associatedAxisLabels,
        probabilityProcessor.process(probabilityBuffer));

    // Create a map to access the result based on label
    Map<String, Float> floatMap = labels.getMapWithFloatValue();
}
```

## Casos de uso possíveis no momento

A versão atual da TensorFlow Lite Support Library abrange os seguintes casos de uso:

- Tipos de dado comuns (float, uint8, imagens, áudios e array desses objetos) como entradas e saídas de modelos tflite.
- Operações básicas com imagens (recortar, redimensionar e girar).
- Normalização e quantização.
- Utilitários de arquivos

Versões futuras melhorarão o suporte a aplicativos relacionados a texto.

## Arquitetura de ImageProcessor

A concepção do `ImageProcessor` possibilitou que as operações de manipulação de imagens sejam definidas de antemão e otimizadas durante o processo de criação. No momento, o `ImageProcessor` oferece suporte a três operações básicas de pré-processamento, conforme descrito nos três comentários do trecho de código abaixo:

```java
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

int width = bitmap.getWidth();
int height = bitmap.getHeight();

int size = height > width ? width : height;

ImageProcessor imageProcessor =
    new ImageProcessor.Builder()
        // Center crop the image to the largest square possible
        .add(new ResizeWithCropOrPadOp(size, size))
        // Resize using Bilinear or Nearest neighbour
        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR));
        // Rotation counter-clockwise in 90 degree increments
        .add(new Rot90Op(rotateDegrees / 90))
        .add(new NormalizeOp(127.5, 127.5))
        .add(new QuantizeOp(128.0, 1/128.0))
        .build();
```

Confira mais detalhes sobre normalização e quantização [aqui](../models/convert/metadata.md#normalization-and-quantization-parameters).

O grande objetivo da biblioteca de suporte é oferecer suporte a todas as transformações de [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image). Dessa forma, a transformação será a mesma que do TensorFlow, e a implementação será independente do sistema operacional.

Os desenvolvedores também podem criar processadores personalizados. Nesses casos, é importante se alinhar ao processo de treinamento, ou seja, o mesmo pré-processamento deve ser aplicado tanto ao treinamento quanto à inferência para aumentar a capacidade de reprodução.

## Quantização

Ao inicializar objetos de entrada ou saída, como `TensorImage` ou `TensorBuffer`, você precisa especificar seus tipos como `DataType.UINT8` ou `DataType.FLOAT32`.

```java
TensorImage tensorImage = new TensorImage(DataType.UINT8);
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
```

O `TensorProcessor` pode ser usado para quantizar tensores de entrada ou fazer a dequantização de tensores de saída. Por exemplo: ao processar uma saída `TensorBuffer` quantizada, o desenvolvedor pode usar `DequantizeOp` para fazer a dequantização do resultado para uma probabilidade de ponto flutuante entre 0 e 1:

```java
import org.tensorflow.lite.support.common.TensorProcessor;

// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    new TensorProcessor.Builder().add(new DequantizeOp(0, 1/255.0)).build();
TensorBuffer dequantizedBuffer = probabilityProcessor.process(probabilityBuffer);
```

Os parâmetros de quantização de um tensor podem ser lidos por meio da [biblioteca de extração de metadados](../models/convert/metadata.md#read-the-metadata-from-models).

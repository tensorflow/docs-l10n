# Integração do classificador de linguagem natural

A API `NLClassifier` da biblioteca Task classifica texto de entrada em diferentes categorias. É uma API versátil e configurável que pode lidar com a maioria dos modelos de classificação de texto.

## Principais recursos da API NLClassifier

- Recebe uma única string como entrada, faz a classificação com a string e gera como saída pares &lt;Label, Score&gt; (rótulo, pontuação) como resultados da classificação.

- Tokenização opcional de expressão regular disponível para texto de entrada.

- Configurável para se adaptar a diferentes modelos de classificação.

## Modelos de NLClassifier com suporte

Temos garantias de que os modelos abaixo são compatíveis com a API `NLClassifier`.

- Modelo de <a href="../../examples/text_classification/overview">classificação do sentimento de avaliações de filmes</a>.

- Modelos com a especificação `average_word_vec` criados pelo [Model Maker do TensorFlow Lite para classificação de texto](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

Confira o [aplicativo de referência para classificação de texto](https://github.com/tensorflow/examples/blob/master/lite/examples/text_classification/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/textclassification/client/TextClassificationClient.java) para ver um exemplo de como usar o `NLClassifier` em um aplicativo para Android.

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
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.4'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

### Etapa 2 – Execute a inferência usando a API

```java
// Initialization, use NLClassifierOptions to configure input and output tensors
NLClassifierOptions options =
    NLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setInputTensorName(INPUT_TENSOR_NAME)
        .setOutputScoreTensorName(OUTPUT_SCORE_TENSOR_NAME)
        .build();
NLClassifier classifier =
    NLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/NLClassifier.java) para ver mais opções de configuração do `NLClassifier`.

## Execute a inferência no Swift

### Etapa 1 – Importe o CocoaPods

Adicione o pod TensorFlowLiteTaskText ao Podfile.

```
target 'MySwiftAppWithTaskAPI' do
  use_frameworks!
  pod 'TensorFlowLiteTaskText', '~> 0.4.4'
end
```

### Etapa 2 – Execute a inferência usando a API

```swift
// Initialization
var modelOptions:TFLNLClassifierOptions = TFLNLClassifierOptions()
modelOptions.inputTensorName = inputTensorName
modelOptions.outputScoreTensorName = outputScoreTensorName
let nlClassifier = TFLNLClassifier.nlClassifier(
      modelPath: modelPath,
      options: modelOptions)

// Run inference
let categories = nlClassifier.classify(text: input)
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLNLClassifier.h).

## Execute a inferência no C++

```c++
// Initialization
NLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<NLClassifier> classifier = NLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h).

## Execute a inferência no Python

### Etapa 1 – Instale o pacote via pip

```
pip install tflite-support
```

### Etapa 2 – Use o modelo

```python
# Imports
from tflite_support.task import text

# Initialization
classifier = text.NLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/nl_classifier.py) para ver mais opções de configuração do `NLClassifier`.

## Exemplo de resultados

Veja um exemplo dos resultados da classificação do [modelo de avaliações de filmes](https://www.tensorflow.org/lite/examples/text_classification/overview).

Input: "What a waste of my time." ("Que desperdício de tempo.")

Saída:

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

Experimente a [ferramenta CLI simples de demonstração para NLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#nlclassifier) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

Dependendo do caso de uso, a API `NLClassifier` pode carregar um modelo do TF Lite com ou sem os [TF Lite Model Metadata](../../models/convert/metadata) (metadados de modelo). Confira exemplos de criação dos metadados para classificadores de linguagem natural na [API Metadata Writer do TensorFlow Lite](../../models/convert/metadata_writer_tutorial.ipynb#nl_classifiers) (gravação de metadados).

Os modelos compatíveis devem atender aos seguintes requisitos:

- Tensor de entrada: (kTfLiteString/kTfLiteInt32)

    - A entrada do modelo deve ser um tensor kTfLiteString de string de entrada não tratada ou um tensor kTfLiteInt32 para índices tokenizados de expressão regular de string de entrada não tratada.
    - Se o tipo de entrada for kTfLiteString, não são necessários [metadados](../../models/convert/metadata) para o modelo.
    - Se o tipo de entrada for kTfLiteInt32, um `RegexTokenizer` precisa ser configurado nos [metadados](https://www.tensorflow.org/lite/models/convert/metadata_writer_tutorial#natural_language_classifiers) do tensor de entrada.

- Tensor de pontuações de saída: (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)

    - Tensor de saída obrigatório para a pontuação de cada categoria classificada.

    - Se o tipo for um dos tipos Int, faça a dequantização para double/float para as plataformas correspondentes.

    - Pode ter um arquivo associado opcional nos [metadados](../../models/convert/metadata) correspondentes do tensor de saída para rótulos de categoria. O arquivo deve ser sem formação, com um rótulo por linha, e o número de rótulos deve coincidir com o número de categorias das saídas do modelo. Confira o [arquivo de rótulos de exemplo](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/python/tests/testdata/nl_classifier/labels.txt).

- Tensor de rótulos de saída: (kTfLiteString/kTfLiteInt32)

    - Tensor de saída opcional para o rótulo de cada categoria. Deve ter o mesmo tamanho que o tensor de pontuações de saída. Se este tensor não estiver presente, a API usará alguns índices como nomes de classe.

    - Será ignorado se o arquivo de rótulos associado estiver presente nos metadados do tensor de pontuações de saída.

# Integração do classificador de linguagem natural BERT

A API `BertNLClassifier` da biblioteca Task é muito similar à API `NLClassifier`, que classifica texto de entrada em diferentes categorias, exceto pelo fato de essa API ter sido personalizada para modelos relacionados a BERT que exigem as tokenizações Wordpiece e Sentencepiece fora do modelo do TF Lite.

## Principais recursos da API BertNLClassifier

- Recebe uma única string como entrada, faz a classificação com a string e gera como saída pares &lt;Label, Score&gt; (rótulo, pontuação) como resultados da classificação.

- Faz as tokenizações [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h) ou [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h) do texto de entrada fora do grafo.

## Modelos de BertNLClassifier com suporte

Os modelos abaixo são compatíveis com a API `BertNLClassifier`.

- Modelos BERT criados pelo [Model Maker do TensorFlow Lite para classificação de texto](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification).

- Modelos personalizados que atendam aos [requisitos de compatibilidade de modelos](#model-compatibility-requirements).

## Execute a inferência no Java

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

    // Import the Task Text Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-text:0.4.4'
}
```

Observação: a partir da versão 4.1 do plug-in do Gradle para Android, o arquivo .tflite será adicionado à lista noCompress (não compacte) por padrão, e a opção aaptOptions acima não é mais necessária.

### Etapa 2 – Execute a inferência usando a API

```java
// Initialization
BertNLClassifierOptions options =
    BertNLClassifierOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertNLClassifier classifier =
    BertNLClassifier.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<Category> results = classifier.classify(input);
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/nlclassifier/BertNLClassifier.java).

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
let bertNLClassifier = TFLBertNLClassifier.bertNLClassifier(
      modelPath: bertModelPath)

// Run inference
let categories = bertNLClassifier.classify(text: input)
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/nlclassifier/Sources/TFLBertNLClassifier.h).

## Execute a inferência no C++

```c++
// Initialization
BertNLClassifierOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertNLClassifier> classifier = BertNLClassifier::CreateFromOptions(options).value();

// Run inference with your input, `input_text`.
std::vector<core::Category> categories = classifier->Classify(input_text);
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_nl_classifier.h).

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
classifier = text.BertNLClassifier.create_from_file(model_path)

# Run inference
text_classification_result = classifier.classify(text)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/bert_nl_classifier.py) para ver mais opções de configuração do `BertNLClassifier`.

## Exemplo de resultados

Veja um exemplo dos resultados da classificação de avaliações de filmes usando o modelo [MobileBert](https://www.tensorflow.org/lite/models/modify/model_maker/text_classification) do Model Maker (criador de modelos).

Entrada (em inglês): "it's a charming and often affecting journey" (é uma jornada encantadora e, às vezes, comovente).

Saída:

```
category[0]: 'negative' : '0.00006'
category[1]: 'positive' : '0.99994'
```

Experimente a [ferramenta CLI simples de demonstração para BertNLClassifier](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bertnlclassifier) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `BertNLClassifier` espera um modelo do TF Lite com os [metadados de modelo do TF Lite](../../models/convert/metadata.md) obrigatórios.

Os metadados devem atender aos seguintes requisitos:

- input_process_units para o tokenizador Wordpiece/Sentencepiece.

- Três tensores de entrada com nomes "ids", "mask" e "segment_ids" para a saída do tokenizador.

- Um tensor de saída do tipo float32, com um arquivo de rótulos opcional, que deve ser um arquivo de texto sem formatação com um rótulo por linha. O número de rótulos precisa coincidir com o número de categorias da saída do modelo.

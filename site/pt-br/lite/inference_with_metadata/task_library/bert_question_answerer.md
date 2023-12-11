# Integração do respondedor de perguntas BERT

A API `BertQuestionAnswerer` da biblioteca Task carrega um modelo BERT e responde a perguntas com base no conteúdo de um determinado trecho. Confira mais informações na documentação do modelo Question-Answer <a href="../../examples/bert_qa/overview">aqui</a>.

## Principais recursos da API BertQuestionAnswerer

- Recebe duas entradas de texto como pergunta e contexto, e gera como saída uma lista de possíveis respostas.

- Faz as tokenizações Wordpiece ou Sentencepiece do texto de entrada fora do grafo.

## Modelos de BertQuestionAnswerer com suporte

Os modelos abaixo são compatíveis com a API `BertNLClassifier`.

- Modelos criados pelo [Model Maker do TensorFlow Lite para o Question-Answer BERT](https://www.tensorflow.org/lite/models/modify/model_maker/question_answer).

- [Modelos BERT pré-treinados no TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/bert-question-answerer/1).

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
BertQuestionAnswererOptions options =
    BertQuestionAnswererOptions.builder()
        .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
        .build();
BertQuestionAnswerer answerer =
    BertQuestionAnswerer.createFromFileAndOptions(
        androidContext, modelFile, options);

// Run inference
List<QaAnswer> answers = answerer.answer(contextOfTheQuestion, questionToAsk);
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/BertQuestionAnswerer.java).

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
let mobileBertAnswerer = TFLBertQuestionAnswerer.questionAnswerer(
      modelPath: mobileBertModelPath)

// Run inference
let answers = mobileBertAnswerer.answer(
      context: context, question: question)
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h).

## Execute a inferência no C++

```c++
// Initialization
BertQuestionAnswererOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_path);
std::unique_ptr<BertQuestionAnswerer> answerer = BertQuestionAnswerer::CreateFromOptions(options).value();

// Run inference with your inputs, `context_of_question` and `question_to_ask`.
std::vector<QaAnswer> positive_results = answerer->Answer(context_of_question, question_to_ask);
```

Confira mais detalhes no [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/bert_question_answerer.h).

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
answerer = text.BertQuestionAnswerer.create_from_file(model_path)

# Run inference
bert_qa_result = answerer.answer(context, question)
```

Confira o [código-fonte](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/bert_question_answerer.py) para ver mais opções de configuração do `BertQuestionAnswerer`.

## Exemplo de resultados

Veja um exemplo dos resultados de resposta do [modelo ALBERT](https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1).

Contexto: "The Amazon rainforest, alternatively, the Amazon Jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations."<br><br>(A Amazônia, também chamada de Floresta/Selva Amazônica, é uma floresta latifoliada úmida que cobre a maior parte da Bacia Amazônica da América do Sul. Esta bacia abrange 7 milhões de quilômetros quadrados, dos quais 5 milhões e meio de quilômetros quadrados são cobertos pela floresta tropical. Esta região inclui territórios pertencentes a nove nações).

Pergunta: "Where is Amazon rainforest?"<br><br>(Onde fica a Floresta Amazônica?)

Respostas:

```
answer[0]:  'South America.'
logit: 1.84847, start_index: 39, end_index: 40
answer[1]:  'most of the Amazon basin of South America.'
logit: 1.2921, start_index: 34, end_index: 40
answer[2]:  'the Amazon basin of South America.'
logit: -0.0959535, start_index: 36, end_index: 40
answer[3]:  'the Amazon biome that covers most of the Amazon basin of South America.'
logit: -0.498558, start_index: 28, end_index: 40
answer[4]:  'Amazon basin of South America.'
logit: -0.774266, start_index: 37, end_index: 40

```

Experimente a [ferramenta CLI simples de demonstração para BertQuestionAnswerer](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/text/desktop/README.md#bert-question-answerer) com seu próprio modelo e dados de teste.

## Requisitos de compatibilidade de modelos

A API `BertQuestionAnswerer` espera um modelo do TF Lite com os [metadados de modelo do TF Lite](../../models/convert/metadata) obrigatórios.

Os metadados devem atender aos seguintes requisitos:

- `input_process_units` para o tokenizador Wordpiece/Sentencepiece.

- Três tensores de entrada com nomes "ids", "mask" e "segment_ids" para a saída do tokenizador.

- Dois tensores de saída com nomes "end_logits" e "start_logits" para indicar a posição relativa da resposta no contexto.

# Inferência com o TensorFlow Lite usando metadados

Fazer a inferência em [modelos com metadados](../models/convert/metadata.md) pode ser fácil, bastando algumas linhas de código. Os metadados do TensorFlow Lite contêm uma boa descrição do que o modelo faz e de como usá-lo. Com eles, os geradores de código podem gerar automaticamente o código de inferência para você, como ao usar o [recurso Android Studio ML Binding](codegen.md#mlbinding) ou o [gerador de código do TensorFlow Lite para Android](codegen.md#codegen). Além disso, podem ser usados para configurar seu pipeline de inferência personalizado.

## Ferramentas e bibliotecas

O TensorFlow Lite conta com diversas ferramentas e bibliotecas para atender a diferentes requisitos de implantação:

### Gere a interface do modelo com os geradores de código para Android

Existem duas formas de gerar automaticamente o código encapsulador para Android necessário para um modelo do TensorFlow Lite com metadados:

1. [Android Studio ML Model Binding](codegen.md#mlbinding) é uma ferramenta disponível no Android Studio para importar um modelo do TensorFlow Lite por meio de uma interface gráfica. O Android Studio vai definir automaticamente as configurações do projeto e gerar classes encapsuladoras com base nos metadados do modelo.

2. O [gerador de código do TensorFlow Lite](codegen.md#codegen) é um arquivo executável que gera automaticamente a interface do modelo com base nos metadados. No momento, oferece suporte apenas ao Android em Java. O código encapsulador remove a necessidade de interagir diretamente com o `ByteBuffer`. Em vez disso, os desenvolvedores podem interagir com o modelo do TensorFlow Lite com objetos tipados, como `Bitmap` e `Rect`. Além disso, os usuários do Android Studio também têm acesso ao recurso codegen por meio do [Android Studio ML Binding](codegen.md#mlbinding).

### Use as APIs integradas da TensorFlow Lite Task Library

A [TensorFlow Lite Task Library](task_library/overview.md) fornece interfaces de modelo prontas para uso e otimizadas para tarefas de aprendizado de máquina populares, como classificação de imagens, pergunta e resposta, etc. As interfaces de modelo são criadas especialmente para cada tarefa alcançar o melhor desempenho e usabilidade. A biblioteca de tarefas funciona em várias plataformas e é compatível com o Java, C++ e Swift.

### Crie pipelines de inferência personalizados com a TensorFlow Lite Support Library

A [TensorFlow Lite Support Library](lite_support.md) é uma biblioteca interplataforma que ajuda a personalizar a interface do modelo e a criar pipelines de inferência. Ela contém diversos métodos utilitários e estruturas de dados para fazer pré e pós-processamento, além de conversão de dados. Além disso, tem o mesmo comportamento de módulos do TensorFlow, como TF.Image e TF.Text, garantindo a consistência do treinamento à inferência.

## Confira modelos pré-treinados com metadados

Navegue pelos [modelos hospedados do TensorFlow Lite](https://www.tensorflow.org/lite/guide/hosted_models) e pelo [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) para baixar modelos pré-treinados com metadados para tarefas de visão e texto. Além disso, veja diferentes opções de [visualização dos metadados](../models/convert/metadata.md#visualize-the-metadata).

## Repositório da TensorFlow Lite Support no GitHub

Acesse o [repositório da TensorFlow Lite Support no GitHub](https://github.com/tensorflow/tflite-support) para ver mais exemplos e código-fonte. Para fornecer feedback, crie um [novo issue no GitHub](https://github.com/tensorflow/tflite-support/issues/new).

# Ferramentas de desenvolvimento para Android

O TensorFlow Lite fornece diversas ferramentas para integrar modelos nos apps Android. Esta página descreve ferramentas de desenvolvimento para usar na criação de apps com o Kotlin, Java e C++, além do suporte para o desenvolvimento do TensorFlow Lite no Android Studio.

Ponto importante: em geral, você deve usar a [TensorFlow Lite Task Library](#task_library) para integrar o TensorFlow Lite ao seu app Android, a menos que seu caso de uso não seja compatível com essa biblioteca. Se não for compatível com a [TensorFlow Lite Task Library](#lite_lib) e a [Support Library](#support_lib).

Para começar a escrever código Android rapidamente, veja o [Guia rápido para Android](../android/quickstart)

## Ferramentas para desenvolver com Kotlin e Java

As seguintes seções descrevem as ferramentas de desenvolvimento do TensorFlow Lite que usam as linguagens Kotlin e Java.

### TensorFlow Lite Task Library {:#task_library}

A TensorFlow Lite Task Library contém um conjunto de bibliotecas específicas a tarefas poderoso e fácil de usar para os desenvolvedores de apps criarem com o TensorFlow Lite. Ela fornece interfaces de modelo prontas para uso e otimizadas para tarefas de aprendizado de máquina populares, como classificação de imagens, pergunta e resposta etc. As interfaces de modelo são criadas especialmente para cada tarefa alcançar o melhor desempenho e usabilidade. A Biblioteca Task funciona em várias plataformas e é compatível com o Java e C++.

Para usar a Biblioteca Task no seu app Android, use o AAR do MavenCentral para a [Task Vision Library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-vision) (visão), [Task Text Library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-text) (texto) e [Task Audio Library](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-task-audio) (áudio), respectivamente.

Você pode especificar isso nas suas dependências `build.gradle` da seguinte maneira:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-task-vision:+'
    implementation 'org.tensorflow:tensorflow-lite-task-text:+'
    implementation 'org.tensorflow:tensorflow-lite-task-audio:+'
}
```

Se você usa instantâneos noturnos, adicione o [repositório de instantâneo Sonatype](./lite_build#use_nightly_snapshots) ao seu projeto.

Veja a introdução na [visão geral da TensorFlow Lite Task Library](../inference_with_metadata/task_library/overview.md) para saber mais.

### TensorFlow Lite Library {:#lite_lib}

Use a biblioteca do TensorFlow Lite no seu app Android ao adicionar o [ARR hospedado no MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite) ao seu projeto de desenvolvimento.

Você pode especificar isso nas suas dependências `build.gradle` da seguinte maneira:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:+'
}
```

Se você usa instantâneos noturnos, adicione o [repositório de instantâneo Sonatype](./lite_build#use_nightly_snapshots) ao seu projeto.

Esse AAR inclui binários para todas as [ABIs Android](https://developer.android.com/ndk/guides/abis). Você pode reduzir o tamanho do binário do seu aplicativo incluindo apenas as ABIs necessárias para o suporte.

A menos que você esteja segmentando hardware específico, omita as ABIs `x86`, `x86_64` e `arm32` na maioria dos casos. Você pode definir isso com a seguinte configuração Gradle. Ela inclui especificamente apenas `armeabi-v7a` e `arm64-v8a` e deve cobrir a maioria dos dispositivos Android modernos.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

Para saber mais sobre `abiFilters`, confira as [ABIs Android](https://developer.android.com/ndk/guides/abis) na documentação de NDK do Android.

### TensorFlow Lite Support Library {:#support_lib}

A TensorFlow Lite Android Support Library facilita a integração de modelos ao seu aplicativo. Ela fornece APIs de alto nível que ajudam a transformar dados de entrada brutos no formato exigido pelo modelo e interpretar a saída do modelo, reduzindo a quantidade de código boilerplate necessária.

Ela é compatível com formatos de dados comuns para entradas e saídas, incluindo imagens e arrays. Ela também fornece unidades de pré e pós-processamento que realizam tarefas como redimensionamento e recorte de imagens.

Use a Support Library no seu app Android ao incluir o AAR da [Support Library do TensorFlow Lite hospedado no MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-support).

Você pode especificar isso nas suas dependências `build.gradle` da seguinte maneira:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:+'
}
```

Se você usa instantâneos noturnos, adicione o [repositório de instantâneo Sonatype](./lite_build#use_nightly_snapshots) ao seu projeto.

Para saber como começar, confira a [TensorFlow Lite Android Support Library](../inference_with_metadata/lite_support.md).

### Versões mínimas de SDK do Android para bibliotecas

Biblioteca | `minSdkVersion` | Requisitos do dispositivo
--- | --- | ---
tensorflow-lite | 19 | Uso da NNAPI exige
:                             :                 : API 27+                : |  |
tensorflow-lite-gpu | 19 | GLES 3.1 ou OpenCL
:                             :                 : (geralmente, só está        : |  |
:                             :                 : disponível na API 21+   : |  |
tensorflow-lite-hexagon | 19 | -
tensorflow-lite-support | 19 | -
tensorflow-lite-task-vision | 21 | android.graphics.Color
:                             :                 : as API relacionadas exigem   : |  |
:                             :                 : API 26+                : |  |
tensorflow-lite-task-text | 21 | -
tensorflow-lite-task-audio | 23 | -
tensorflow-lite-metadata | 19 | -

### Usando o Android Studio

Além das bibliotecas de desenvolvimento descritas acima, o Android Studio também fornece suporte para a integração de modelos do TensorFlow Lite, conforme descrito abaixo.

#### Vinculação de modelo de ML do Android Studio

O recurso de Vinculação de modelo de ML a partir da versão 4.1 do Android permite que você importe arquivos de modelo `.tflite` no seu app Android existente e gere classes de interface para facilitar a integração do seu código com um modelo.

Para importar um modelo do TensorFlow Lite (TFLite):

1. Clique com o botão direito no módulo onde que você quer usar o modelo TFLite ou clique em **File &gt; New &gt; Other &gt; TensorFlow Lite Model** (Arquivo &gt; Novo &gt; Outro &gt; Modelo do TensorFlow Lite).

2. Selecione o local do seu arquivo do TensorFlow Lite. Observe que a ferramenta configura a dependência do modelo com a Vinculação de modelo de ML e adiciona automaticamente todas as dependências necessárias para o arquivo `build.gradle` do seu módulo Android.

    Observação: selecione a segunda caixa de seleção para importar a GPU do TensorFlow se você quiser usar a [aceleração de GPU](../performance/gpu).

3. Clique em `Finish` para começar o processo de importação. Quando a importação for concluída, a ferramenta exibirá uma tela descrevendo o modelo, incluindo os tensores de entrada e saída.

4. Para começar a usar o modelo, selecione Kotlin ou Java, copie e cole o código na seção **Código de amostra**.

Você pode retornar à tela de informações do modelo ao clicar duas vezes no modelo do TensorFlow no diretório `ml` no Android Studio. Para mais informações sobre o uso do recurso Vinculação de modelo do Android Studio, confira as [notas da versão](https://developer.android.com/studio/releases#4.1-tensor-flow-lite-models). Para uma visão geral de como usar a vinculação de modelo no Android Studio, veja as [instruções](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md) do código de exemplo.

## Ferramentas para desenvolver com C e C++

As bibliotecas C e C++ para o TensorFlow Lite são direcionadas principalmente a desenvolvedores que usam o Kit de Desenvolvimento Nativo do Android (NDK) para criar apps. Há duas maneiras de usar o TFLite através da C++ se você quiser criar seu app com o NDK:

### API C do TFLite

Usar essa API é a abordagem *recomendada* para desenvolvedores que usam o NDK. Baixe o arquivo [ARR do TensorFlow Lite hospedado no MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow/tensorflow-lite), renomeie para `tensorflow-lite-*.zip` e descompacte. Você precisa incluir os quatro arquivos de cabeçalho nas pastas `headers/tensorflow/lite/` e `headers/tensorflow/lite/c/` e a biblioteca dinâmica `libtensorflowlite_jni.so` relevante na pasta `jni/` do seu projeto NDK.

O arquivo de cabeçalho `c_api.h` contém a documentação básica de como usar a API C do TFLite.

### API C++ do TFLite

Se você quiser usar o TFLite pela API C++, você pode criar as bibliotecas compartilhadas C++:

32bit armeabi-v7a:

```sh
bazel build -c opt --config=android_arm //tensorflow/lite:libtensorflowlite.so
```

64bit arm64-v8a:

```sh
bazel build -c opt --config=android_arm64 //tensorflow/lite:libtensorflowlite.so
```

No momento, não há uma maneira simples de extrair todos os arquivos de cabeçalho necessários, então você precisa incluir todos os arquivos em `tensorflow/lite/` a partir do repositório do TensorFlow. Além disso, você precisará dos arquivos de cabeçalho de [FlatBuffers](https://github.com/google/flatbuffers) e [Abseil](https://github.com/abseil/abseil-cpp).

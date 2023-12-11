# Operadores específicos do TensorFlow

Como a biblioteca de operadores integrados do TensorFlow Lite só tem suporte a um número limitado de operadores do TensorFlow, nem todo modelo pode ser convertido. Confira mais detalhes em [Compatibilidade de operadores](ops_compatibility.md).

Para permitir a conversão, os usuários podem ativar o uso de [operações específicas do TensorFlow](op_select_allowlist.md) em seu modelo do TensorFlow Lite. Porém, para executar modelos do TensorFlow Lite com operações do TensorFlow, é preciso buscar o runtime core do TensorFlow, o que aumenta o tamanho do binário do interpretador do TensorFlow Lite. No caso do Android, é possível evitar isso compilando seletivamente somente as operações do Tensorflow necessárias. Confira mais detalhes em [Reduza o tamanho do binário](../guide/reduce_binary_size.md).

Este documento descreve como [converter](#convert_a_model) e [executar](#run_inference) um modelo do TensorFlow Lite contendo operações do TensorFlow em uma plataforma da sua escolha. Além disso, são discutidas [métricas de tamanho e desempenho](#metrics), além das [limitações conhecidas](#known_limitations).

## Converta um modelo

O exemplo a seguir mostra como gerar um modelo do TensorFlow Lite com operações específicas do TensorFlow.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## Execute a inferência

Ao usar um modelo do TensorFlow Lite que foi convertido com suporte a operações específicas do TensorFlow, o cliente também precisa usar um runtime do TensorFlow Lite que inclua a biblioteca necessária de operações do TensorFlow.

### AAR do Android

Para reduzir o tamanho do binário, compile seus próprios arquivos AAR personalizados conforme orientado na [próxima seção](#building-the-android-aar). Se o tamanho do binário não for uma grande preocupação, recomendamos usar o [AAR pré-compilado com operações do TensorFlow hospedado em MavenCentral](https://search.maven.org/artifact/org.tensorflow/tensorflow-lite-select-tf-ops).

Você pode especificá-lo nas dependências do `build.gradle`, adicionando-o junto com o AAR padrão do TensorFlow Lite da seguinte forma:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

Para usar instantâneos noturnos, você deve adicionar o [repositório de instantâneo Sonatype](../android/lite_build.md#use_nightly_snapshots).

Após adicionar a dependência, o delegado necessário para tratar as operações de grafo do TensorFlow deve ser instalado automaticamente para os grafos que precisem dele.

*Observação*: a dependência de operações do TensorFlow é relativamente grande, então é uma boa ideia retirar as ABIs x86 desnecessárias do seu arquivo `.gradle` configurando seus `abiFilters`.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### Compilação do AAR do Android

Para reduzir o tamanho do binário ou para outros casos mais avançados, você também pode compilar a biblioteca manualmente. Pressupondo um [ambiente de compilação funcional do TensorFlow Lite](../android/quickstart.md), compile o AAR do Android com operações específicas do TensorFlow da seguinte maneira:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

Dessa forma, será gerado o arquivo AAR `bazel-bin/tmp/tensorflow-lite.aar` para as operações integradas e personalizadas do TensorFlow Lite; e será gerado o arquivo `bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` para operações do TensorFlow. Se você não tiver um ambiente de compilação funcional, pode [compilar os arquivos acima com o Docker](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker).

Em seguida, você pode importar os arquivos AAR diretamente para o seu projeto ou publicar os arquivos AAR personalizados em seu repositório Maven local:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

Por fim, no `build.gradle` do app, confira se você tem a dependência `mavenLocal()` e troque a dependência padrão do TensorFlow Lite por uma compatível com operações específicas do TensorFlow:

```build
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### Como usar o CocoaPods

O TensorFlow Lite conta com CocoaPods noturnos pré-compilados com operações específicas do TF para `arm64`, que você pode utilizar juntamente com CocoaPods `TensorFlowLiteSwift` ou `TensorFlowLiteObjC`.

*Observação*: se você precisar usar operações específicas do TF em um simulador `x86_64`, pode compilar o framework com operações específicas por conta própria. Confira mais detalhes na seção [Como usar o Bazel + Xcode](#using_bazel_xcode).

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

Após executar `pod install`, você precisa fornecer um sinalizador de vinculação adicional para forçar o carregamento do framework com operações específicas do TF em seu projeto. Em seu projeto do Xcode, acesse `Build Settings` (Configurações de compilação) -&gt; `Other Linker Flags` (Outros sinalizadores de vinculação) e acrescente:

Para versões &gt;= 2.9.0:

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.xcframework/ios-arm64/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

Para versões &lt; 2.9.0:

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

Em seguida, você poderá executar qualquer modelo convertido com `SELECT_TF_OPS` em seu aplicativo para iOS. Por exemplo: você pode modificar o [aplicativo de classificação de imagens para iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) a fim de testar o recurso de operações específicas do TF.

- Substitua o arquivo do modelo pelo convertido com `SELECT_TF_OPS` ativado.
- Adicione a dependência `TensorFlowLiteSelectTfOps` ao `Podfile` conforme orientado.
- Adicione o sinalizador de vinculação adicional conforme indicado acima.
- Execute o aplicativo de exemplo e veja se o modelo funciona corretamente.

#### Como usar o Bazel + Xcode

É possível compilar o TensorFlow Lite com operações específicas do TensorFlow para iOS usando o Bazel. Primeiro, siga as [instruções de compilação para iOS](build_ios.md) a fim de configurar o workspace do Bazel e o arquivo `.bazelrc` corretamente.

Após configurar o workspace com suporte ao iOS, você pode usar o comando abaixo para compilar o framework com operações específicas do TF, que pode ser adicionado juntamente com o `TensorFlowLiteC.framework` comum. O framework com operações específicas do TF não pode ser compilado para a arquitetura `i386`, então você precisa fornecer explicitamente a lista de arquiteturas desejadas, exceto `i386`.

```sh
bazel build -c opt --config=ios --ios_multi_cpus=arm64,x86_64 \
  //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework
```

Dessa forma, será gerado o framework no diretório `bazel-bin/tensorflow/lite/ios/`. Você pode adicionar esse novo framework ao seu projeto do Xcode seguindo etapas similares às descritas na seção [Configurações do projeto do Xcode](./build_ios.md#modify_xcode_project_settings_directly) no guia de compilação para iOS.

Após adicionar o framework ao seu projeto de aplicativo, um sinalizador de vinculação adicional deve ser especificado em seu projeto para forçar o carregamento do framework com operações específicas do TF. Em seu projeto do Xcode, acesse `Build Settings` (Configurações de compilação) -&gt; <code>Other Linker Flags</code> (Outros sinalizadores de vinculação) e acrescente:

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C/C++

Se você estiver usando o Bazel ou o [CMake](https://www.tensorflow.org/lite/guide/build_cmake) para compilar o interpretador do TensorFlow Lite, pode ativar o delegado Flex fazendo a vinculação a uma biblioteca compartilhada de delegado Flex do TensorFlow Lite. Você pode compilar usando o Bazel com o seguinte comando:

```
bazel build -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex
```

Esse comando gera a seguinte biblioteca compartilhada em `bazel-bin/tensorflow/lite/delegates/flex`.

Plataforma | Nome da biblioteca
--- | ---
Linux | `libtensorflowlite_flex.so`
macOS | `libtensorflowlite_flex.dylib`
Windows | `tensorflowlite_flex.dll`

O `TfLiteDelegate` necessário será instalado automaticamente ao criar o interpretador no runtime, desde que a biblioteca compartilhada esteja vinculada. Não é necessário instalar explicitamente a instância do delegado como costuma ser necessário para outros tipos de delegado.

**Observação:** esse recurso está disponível a partir a versão 2.7.

### Python

O TensorFlow Lite com operações específicas do TensorFlow será instalado automaticamente com o [pacote pip do TensorFlow](https://www.tensorflow.org/install/pip). Você também pode optar por instalar somente o [pacote pip do interpretador do TensorFlow Lite](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter).

Observação: o TensorFlow Lite com operações específicas do TensorFlow está disponível no pacote pip do TensorFlow desde a versão 2.3 para Linux e 2.4 para outros ambientes.

## Métricas

### Desempenho

Ao usar uma combinação de operações integradas e específicas do TensorFlow, todas as mesmas otimizações do TensorFlow Lite e operações integradas otimizadas estarão disponíveis e poderão ser usadas pelo modelo convertido.

A tabela abaixo indica o tempo médio de execução da inferência na MobileNet em um dispositivo Pixel 2. Os tempos indicados são uma média de 100 execuções. Esses alvos foram compilados para o Android usando os sinalizadores: `--config=android_arm64 -c opt`.

Build | Tempo (em ms)
--- | ---
Somente operações integradas (`TFLITE_BUILTIN`) | 260,7
Somente operações do TF (`SELECT_TF_OPS`) | 264,5

### Tamanho do binário

A tabela abaixo indica o tamanho do binário do TensorFlow Lite para cada build. Esses alvos foram compilados para o Android usando `--config=android_arm -c opt`.

Build | Tamanho do binário em C++ | Tamanho do APK para Android
--- | --- | ---
Somente operações integradas | 796 KB | 561 KB
Operações integradas + operações do TF | 23 MB | 8 MB
Operações integradas + operações do TF (1) | 4,1 MB | 1,8 MB

(1) Estas bibliotecas são compiladas seletivamente para o [modelo i3d-kinetics-400](https://tfhub.dev/deepmind/i3d-kinetics-400/1), com 8 operações integradas do TF Lite e 3 operações do TensorFlow. Confira mais detalhes na seção [Reduza o tamanho do binário do TensorFlow Lite](../guide/reduce_binary_size.md).

## Limitações conhecidas

- Tipos sem suporte: determinadas operações do TensorFlow poderão não ter suporte ao conjunto total de tipos de entrada/saída que costumam estar disponíveis no TensorFlow.

## Atualizações

- Versão 2.6
    - Suporte a operadores baseados em atributos GraphDef e melhoria das inicializações de recursos HashTable.
- Versão 2.5
    - É possível aplicar uma otimização conhecida como [quantização pós-treinamento](../performance/post_training_quantization.md).
- Versão 2.4
    - Melhoria da compatibilidade com delegados acelerados por hardware.

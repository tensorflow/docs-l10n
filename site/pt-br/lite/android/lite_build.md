# Compile o TensorFlow Lite para Android

Este documento descreve como compilar a biblioteca Android do TensorFlow Lite por conta própria. Normalmente, você não precisa compilar localmente a biblioteca Android do TensorFlow Lite. Se você só quiser usar a biblioteca nos seus projetos Android, confira o [guia rápido do Android](../android/quickstart.md) para mais detalhes.

## Use os instantâneos noturnos

Para usar os instantâneos noturnos, adicione o seguinte repositório à configuração do build Gradle raiz.

```build
allprojects {
    repositories {      // should be already there
        mavenCentral()  // should be already there
        maven {         // add this repo to use snapshots
          name 'ossrh-snapshot'
          url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
    }
}
```

## Compile o TensorFlow Lite localmente

Em alguns casos, você pode querer usar um build local do TensorFlow Lite. Por exemplo, talvez você esteja criando um binário personalizado que inclui [operações selecionadas do TensorFlow](https://www.tensorflow.org/lite/guide/ops_select) ou queira fazer alterações locais no TensorFlow Lite.

### Configure o ambiente de build usando o Docker

- Baixe o arquivo Docker. Ao baixar o arquivo Docker, você concorda que os seguintes termos de serviço regem o uso dele:

*Ao clicar em aceitar, você concorda que todo o uso do Android Studio e do Kit de Desenvolvimento Nativo do Android será regido pelo Contrato de Licença do Kit de Desenvolvimento de Software do Android disponível em https://developer.android.com/studio/terms (essa URL pode ser atualizada ou alterada pelo Google periodicamente).*

<!-- mdformat off(devsite fails if there are line-breaks in templates) -->

{% dynamic if 'tflite-android-tos' in user.acknowledged_walls and request.tld != 'cn' %} Você pode baixar o arquivo Docker <a href="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/tools/tflite-android.Dockerfile">aqui</a> {% dynamic else %} Você precisa confirmar os termos de serviço para baixar o arquivo. <button class="button-blue devsite-acknowledgement-link" data-globally-unique-wall-id="tflite-android-tos">Confirmar</button> {% dynamic endif %}

<!-- mdformat on -->

- Como opção, você pode mudar a versão do SDK ou NDK do Android. Coloque o arquivo Docker baixado em uma pasta vazia e crie sua imagem docker ao executar:

```shell
docker build . -t tflite-builder -f tflite-android.Dockerfile
```

- Inicie o container docker de maneira interativa ao montar sua pasta atual para /host_dir dentro do container (observe que /tensorflow_src é o repositório do TensorFlow dentro do container):

```shell
docker run -it -v $PWD:/host_dir tflite-builder bash
```

Se você usa o PowerShell no Windows, substitua "$PWD" por "pwd".

Se você quiser usar um repositório do TensorFlow no host, monte esse diretório host em vez disso (-v hostDir:/host_dir).

- Depois que estiver dentro do container, você pode executar o seguinte para baixar ferramentas e bibliotecas adicionais do Android (observe que você precisa aceitar a licença):

```shell
sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"
```

Agora você pode prosseguir para a seção [Configure WORKSPACE e .bazelrc](#configure_workspace_and_bazelrc) e configurar o build.

Depois de compilar as bibliotecas, você pode copiá-las para /host_dir dentro do container para acessá-las no host.

### Configure o ambiente de build sem o Docker

#### Instale o Bazel e os pré-requisitos do Android

O Bazel é o principal sistema de build para o TensorFlow. Para a compilação, ele precisa estar instalado no seu sistema com o NDK e o SDK do Android.

1. Instale a versão mais recente do [sistema de build do Bazel](https://bazel.build/versions/master/docs/install.html).
2. O NDK do Android precisa compilar o código do TensorFlow Lite (C/C++) nativo. A versão atual recomendada é 21e, que pode ser encontrada [aqui](https://developer.android.com/ndk/downloads/older_releases.html#ndk-21e-downloads).
3. O SDK do Android e as ferramentas de build podem ser obtidos [aqui](https://developer.android.com/tools/revisions/build-tools.html) ou, opcionalmente, como parte do [Android Studio](https://developer.android.com/studio/index.html). A versão recomendada das ferramentas de build para compilar o TensorFlow Lite é a API &gt;= 23.

### Configure WORKSPACE e .bazelrc

Este é um passo de configuração único necessário para compilar as bibliotecas do TFLite. Execute o script `./configure` na raiz do diretório checkout do TensorFlow e responda "Yes" quando o script pedir para configurar o `./WORKSPACE` de maneira interativa para builds do Android. O script tentará configurar usando as seguintes variáveis de ambiente:

- `ANDROID_SDK_HOME`
- `ANDROID_SDK_API_LEVEL`
- `ANDROID_NDK_HOME`
- `ANDROID_NDK_API_LEVEL`

Se essas variáveis não estiverem definidas, elas precisarão ser fornecidas interativamente no prompt do script. A configuração bem-sucedida deve gerar entradas semelhantes às seguintes no arquivo `.tf_configure.bazelrc` da pasta raiz:

```shell
build --action_env ANDROID_NDK_HOME="/usr/local/android/android-ndk-r21e"
build --action_env ANDROID_NDK_API_LEVEL="26"
build --action_env ANDROID_BUILD_TOOLS_VERSION="30.0.3"
build --action_env ANDROID_SDK_API_LEVEL="30"
build --action_env ANDROID_SDK_HOME="/usr/local/android/android-sdk-linux"
```

### Compile e instale

Depois de configurar o Bazel corretamente, você pode compilar o AAR do TensorFlow Lite a partir do diretório checkout raiz da seguinte maneira:

```sh
bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --define=android_dexmerger_tool=d8_dexmerger \
  --define=android_incremental_dexing_tool=d8_dexbuilder \
  //tensorflow/lite/java:tensorflow-lite
```

Isso gerará um arquivo AAR em `bazel-bin/tensorflow/lite/java/`. Observe que isso cria um "fat" AAR com diversas arquiteturas diferentes. Se você não precisar de todas, use o subconjunto apropriado para seu ambiente de implantação.

Você pode criar arquivos AAR menores segmentando somente um conjunto de modelos da seguinte maneira:

```sh
bash tensorflow/lite/tools/build_aar.sh \
  --input_models=model1,model2 \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

O script acima gerará o arquivo `tensorflow-lite.aar` e, opcionalmente, o arquivo `tensorflow-lite-select-tf-ops.aar`, se um dos modelos estiver usando operações do TensorFlow. Para mais detalhes, confira a seção [Reduza o tamanho binário do TensorFlow Lite](../guide/reduce_binary_size.md).

#### Adicione o AAR diretamente ao projeto

Mova o arquivo `tensorflow-lite.aar` para um diretório chamado `libs` no seu projeto. Modifique o arquivo `build.gradle` do seu app para fazer referência ao novo diretório e substituir a dependência existente do TensorFlow pela nova biblioteca local, por exemplo:

```
allprojects {
    repositories {
        mavenCentral()
        maven {  // Only for snapshot artifacts
            name 'ossrh-snapshot'
            url 'https://oss.sonatype.org/content/repositories/snapshots'
        }
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    compile(name:'tensorflow-lite', ext:'aar')
}
```

#### Instale o AAR no repositório Maven local

Execute o seguinte comando no diretório checkout raiz:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
```

No `build.gradle` do app, confira se você tem a dependência `mavenLocal()` e troque a dependência padrão do TensorFlow Lite por uma compatível com determinadas operações do TensorFlow:

```
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
}
```

Observe que a versão `0.1.100` aqui é puramente para fins de teste/desenvolvimento. Com o AAR local instalado, você pode usar as [APIs de inferência do Java do TensorFlow Lite ](../guide/inference.md) no código do seu app.

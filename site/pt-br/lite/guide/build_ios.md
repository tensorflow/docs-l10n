# Compile o TensorFlow Lite para iOS

Este documento descreve como compilar a biblioteca iOS do TensorFlow Lite por conta própria. Normalmente, você não precisa compilar localmente a biblioteca iOS do TensorFlow Lite. Se você só quiser usá-la, a forma mais fácil é utilizar as versões estáveis ou noturnas pré-compiladas do CocoaPods do TensorFlow Lite. Confira mais detalhes de como usá-las nos seus projetos para iOS no [Guia de início rápido para iOS](ios.md).

## Compilando localmente

Em alguns casos, talvez você queria usar uma build local do TensorFlow Lite, por exemplo, quando precisar fazer alterações locais no TensorFlow Lite e testá-las em seu aplicativo para iOS, ou quando preferir usar uma framework estática em vez da dinâmica que fornecemos. Para criar um framework iOS universal para o TensorFlow Lite localmente, você precisa compilá-lo usando o Bazel em um computador macOS.

### Instale o Xcode

Você precisa instalar o Xcode8 ou posteriores e as ferramentas usando `xcode-select`, caso ainda não o tenha feito:

```sh
xcode-select --install
```

Se for uma nova instalação, você precisa aceitar o contrato de licença para todos os usuários com o seguinte comando:

```sh
sudo xcodebuild -license accept
```

### Instale o Bazel

O Bazel é o principal sistema de build para o TensorFlow. Instale o Brazel conforme as [instruções no site do Bazel](https://docs.bazel.build/versions/master/install-os-x.html). Você deve escolher uma versão entre `_TF_MIN_BAZEL_VERSION` e `_TF_MAX_BAZEL_VERSION` no arquivo [`configure.py`](https://github.com/tensorflow/tensorflow/blob/master/configure.py), na raiz do repositório `tensorflow`.

### Configure WORKSPACE e .bazelrc

Execute o script `./configure` no diretório de checkout raiz do TensorFlow e responda "Yes" (Sim) quando o script perguntar se você deseja compilar o TensorFlow com suporte ao iOS.

### Compile o framework dinâmico TensorFlowLiteC (recomendado)

Observação: esta etapa não é necessária se (1) você estiver usando o Bazel para seu aplicativo ou (2) você só quiser testar as alterações locais nas APIs do Swift ou do Objective-C. Nesses casos, prossiga para a seção [Use em seu próprio aplicativo](#use_in_your_own_application) abaixo.

Quando o Bazel estiver configurado corretamente com suporte ao iOS, você pode compilar o framework `TensorFlowLiteC` com o seguinte comando:

```sh
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_framework
```

Esse comando vai gerar o arquivo `TensorFlowLiteC_framework.zip` no diretório `bazel-bin/tensorflow/lite/ios/` dentro do seu diretório raiz do TensorFlow. Por padrão, o framework gerado contém um binário "fat", contendo armv7, arm64 e x86_64 (mas sem i386). Para ver a lista completa de sinalizadores de compilação usados ao especificar `--config=ios_fat`, confira a seção de configurações do iOS no arquivo [`.bazelrc`](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc).

### Compile o framework estático TensorFlowLiteC

Por padrão, distribuímos o framework dinâmico somente via CocoaPods. Se você deseja usar o framework estático, pode compilar o framework estático `TensorFlowLiteC` com o seguinte comando:

```
bazel build --config=ios_fat -c opt --cxxopt=--std=c++17 \
  //tensorflow/lite/ios:TensorFlowLiteC_static_framework
```

O comando vai gerar um arquivo chamado `TensorFlowLiteC_static_framework.zip` no diretório `bazel-bin/tensorflow/lite/ios/` dentro do seu diretório raiz do TensorFlow. Esse framework estático pode ser usado da mesma forma que o dinâmico.

### Compile os frameworks do TF Lite seletivamente

Você pode compilar frameworks menores ao escolher como alvo somente um conjunto de modelos usando a compilação seletiva, que vai ignorar operações não usadas em seu modelo e incluir somente os kernels de operações necessários para executar o conjunto específico de modelos. Veja o comando:

```sh
bash tensorflow/lite/ios/build_frameworks.sh \
  --input_models=model1.tflite,model2.tflite \
  --target_archs=x86_64,armv7,arm64
```

O comando acima vai gerar o framework estático `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteC_framework.zip` para as operações integradas e personalizadas do TensorFlow Lite. Opcionalmente, vai gerar o framework estático `bazel-bin/tensorflow/lite/ios/tmp/TensorFlowLiteSelectTfOps_framework.zip` se os seus modelos contiverem operações específicas do TensorFlow. O sinalizador `--target_archs` pode ser usado para especificar as arquiteturas de implantação.

## Use em seu próprio aplicativo

### Desenvolvedores que usam o CocoaPods

Existem três CocoaPods para o TensorFlow Lite:

- `TensorFlowLiteSwift`: fornece APIs do Swift para o TensorFlow Lite.
- `TensorFlowLiteObjC`: fornece APIs do Objective-C para o TensorFlow Lite.
- `TensorFlowLiteC`: pod base comum, que incorpora o runtime core do TensorFlow Lite e expressa as APIs C base usadas pelos dois pods acima. Não deve ser usado diretamente pelos usuários.

Como desenvolvedor, você deve escolher o pod `TensorFlowLiteSwift` ou `TensorFlowLiteObjC` de acordo com a linguagem de programação do seu aplicativo, mas não ambos. As etapas exatas para usar builds locais do TensorFlow Lite diferem, dependendo de qual parte exata você deseja compilar.

#### Usando as APIs do Swift ou do Objective-C locais

Se você estiver usando o CocoaPods e quiser apenas testar algumas alterações locais nas [APIs do Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) ou nas [APIs do Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc) do TensorFlow Lite, siga estas etapas:

1. Faça alterações nas APIs do Swift ou do Objective-C APIs em seu checkout `tensorflow`.

2. Abra o arquivo `TensorFlowLite(Swift|ObjC).podspec` e atualize esta linha: <br> `s.dependency 'TensorFlowLiteC', "#{s.version}"` <br> para: <br> `s.dependency 'TensorFlowLiteC', "~> 0.0.1-nightly"` <br> Isso é feito para garantir que você esteja compilando suas APIs do Swift e do Objective-C com a versão noturna mais recente disponível das APIs do `TensorFlowLiteC` (compilada todas as noites entre 1 a 4 horas da manhã, Horário do Pacífico) em vez da versão estável, que pode estar desatualizada em comparação ao seu checkout `tensorflow` local. Outra opção é publicar sua própria versão do `TensorFlowLiteC` e usá-la (confira a seção [Como usar o TensorFlow lite core local](#using_local_tensorflow_lite_core) abaixo).

3. No `Podfile` do seu projeto para iOS, altere a dependência da seguinte forma para apontar para o caminho local do seu diretório raiz do `tensorflow`: <br> Para Swift: <br> `pod 'TensorFlowLiteSwift', :path => '<your_tensorflow_root_dir>'` <br> Para Objective-C: <br> `pod 'TensorFlowLiteObjC', :path => '<your_tensorflow_root_dir>'`

4. Atualize sua instalação do pod em seu diretório raiz do projeto para iOS. <br> `$ pod update`

5. Reabra o workspace gerado (`<project>.xcworkspace`) e compile novamente o aplicativo no Xcode.

#### Como usar o TensorFlow Lite core local

Você pode configurar um repositório privado de especificações do CocoaPods e publicar seu framework `TensorFlowLiteC` personalizado nesse repositório privado. Você pode copiar este [arquivo podspec](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/ios/TensorFlowLiteC.podspec) e modificar alguns valores:

```ruby
  ...
  s.version      = <your_desired_version_tag>
  ...
  # Note the `///`, two from the `file://` and one from the `/path`.
  s.source       = { :http => "file:///path/to/TensorFlowLiteC_framework.zip" }
  ...
  s.vendored_frameworks = 'TensorFlowLiteC.framework'
  ...
```

Após criar seu próprio arquivo `TensorFlowLiteC.podspec`, você pode seguir as [instruções de como usar CocoaPods privados](https://guides.cocoapods.org/making/private-cocoapods.html) para usá-lo em seu próprio projeto. Além disso, pode modificar `TensorFlowLite(Swift|ObjC).podspec` para apontar para seu pod `TensorFlowLiteC` personalizado e usar o pod do Swift ou do Objective-C no seu projeto de aplicativo.

### Desenvolvedores que usam o Bazel

Se você estiver usando o Bazel como ferramenta principal de compilação, basta adicionar a dependência `TensorFlowLite` ao seu alvo no arquivo `BUILD`.

Para Swift:

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

Para Objective-C:

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

Quando você compilar seu projeto de aplicativo, qualquer mudança feita na biblioteca do TensorFlow Lite será capturada e incorporada ao seu aplicativo.

### Modifique as configurações de projeto do Xcode diretamente

É altamente recomendável usar o CocoaPods ou o Bazel para adicionar a dependência do TensorFlow Lite ao seu projeto. Se ainda assim você desejar adicionar o framework `TensorFlowLiteC` manualmente, precisará adicionar o framework `TensorFlowLiteC` como um framework integrado ao seu projeto de aplicativo. Descompacte o arquivo `TensorFlowLiteC_framework.zip` gerado pela compilação acima para obter o diretório `TensorFlowLiteC.framework`. Esse diretório é o framework em si que o Xcode consegue entender.

Após preparar o diretório `TensorFlowLiteC.framework`, primeiro você precisa adicioná-lo como um binário integrado ao alvo do seu aplicativo. A seção específica de configurações de projeto pode diferir, dependendo da versão do Xcode.

- Xcode 11: acesse a aba "General" (Geral) do editor do projeto para o alvo do seu aplicativo e adicione `TensorFlowLiteC.framework` na seção "Frameworks, Libraries, and Embedded Content" (Frameworks, bibliotecas e conteúdo integrado).
- Xcode 10 e inferiores : acesse a aba "General" (Geral) do editor do projeto para o alvo do seu aplicativo e adicione `TensorFlowLiteC.framework` em "Embedded Binaries" (Binários integrados). O framework também deverá ser adicionado automaticamente na seção "Linked Frameworks and Libraries" (Frameworks e bibliotecas vinculados).

Quando você adiciona o framework como binário integrado, o Xcode também atualiza a entrada "Framework Search Paths" (Caminhos de pesquisa do framework) na aba "Build Settings" (Configurações da build) para incluir o diretório pai do seu framework. Caso isso não ocorra automaticamente, você precisará adicionar o diretório pai do diretório `TensorFlowLiteC.framework` manualmente.

Após fazer essas duas configurações, você deverá conseguir importar e chamar a API do C do TensorFlow Lite, definida pelos arquivos de cabeçalho no diretório `TensorFlowLiteC.framework/Headers`.

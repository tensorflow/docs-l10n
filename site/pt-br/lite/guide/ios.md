# Guia de início rápido para iOS

Para começar a usar o TensorFlow Lite no iOS, recomendamos ver o seguinte exemplo:

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">Exemplo de classificação de imagens no iOS</a>

Confira a explicação do código fonte em [Classificação de imagens com o TensorFlow Lite no iOS](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/ios/README.md).

Esse aplicativo de exemplo usa [classificação de imagens](https://www.tensorflow.org/lite/examples/image_classification/overview) para classificar continuamente o que é exibido pela câmera frontal do dispositivo, mostrando as classificações mais prováveis. Esse aplicativo permite que o usuário escolha entre um modelo de ponto flutuante ou [quantizado](https://www.tensorflow.org/lite/performance/post_training_quantization) e permite também selecionar o número de threads usados para fazer a inferência.

Observação: confira outros aplicativos para iOS que demonstram o uso do TensorFlow Lite em diversos casos de uso nos [Exemplos](https://www.tensorflow.org/lite/examples).

## Adicione o TensorFlow Lite ao seu projeto Swift ou Objective-C

O TensorFlow Lite conta com bibliotecas do iOS nativas para [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) e [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). Comece a escrever seu próprio código iOS usando o [exemplo de classificação de imagens com o Swift](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) como ponto de partida.

As seções abaixo demonstram como adicionar o Swift ou o Objective-C para TensorFlow Lite ao seu projeto:

### Desenvolvedores que usam o CocoaPods

No `Podfile`, adicione o pod do TensorFlow . Em seguida, execute `pod install`.

#### Swift

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

#### Objective-C

```ruby
pod 'TensorFlowLiteObjC'
```

#### Especificação de versões

Há versões estáveis e versões noturnas disponíveis para pods `TensorFlowLiteSwift` e `TensorFlowLiteObjC`. Se você não especificar uma restrição de versão como nos exemplos acima, o CocoaPods buscará a versão estável mais recente por padrão.

Além disso, você pode especificar uma restrição de versão. Por exemplo: se quiser usar a versão 2.10.0, pode escrever a dependência como:

```ruby
pod 'TensorFlowLiteSwift', '~> 2.10.0'
```

Dessa forma, você vai garantir que a versão 2.x.y mais recente disponível do pod `TensorFlowLiteSwift` seja usada em seu aplicativo. Já se você quiser usar as versões noturnas, pode escrever:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

A partir da versão noturna 2.4.0, por padrão, [delegados de GPU](https://www.tensorflow.org/lite/performance/gpu) e [delegados de Core ML](https://www.tensorflow.org/lite/performance/coreml_delegate) são excluídos do pod para reduzir o tamanho do binário. É possível incluí-los especificando as subspecs:

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML', 'Metal']
```

Dessa forma, você poderá usar os recursos mais recentes adicionados ao TensorFlow Lite. Quando o arquivo `Podfile.lock` for criado ao executar o comando `pod install` pela primeira vez, a versão da biblioteca noturna será mantida na versão da data atual. Se você quiser atualizar a biblioteca noturna para uma versão mais nova, precisa executar o comando `pod update`.

Confira mais informações sobre as diferentes maneiras de especificar restrições de versão em [Como especificar versões de pods](https://guides.cocoapods.org/using/the-podfile.html#specifying-pod-versions).

### Desenvolvedores que usam o Bazel

No arquivo`BUILD`, adicione a dependência `TensorFlowLite` ao seu alvo.

#### Swift

```python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

#### Objective-C

```python
objc_library(
  deps = [
      "//tensorflow/lite/objc:TensorFlowLite",
  ],
)
```

#### API do C/C++

Ou então você pode usar a [API do C](https://www.tensorflow.org/code/tensorflow/lite/c/c_api.h) ou a [API do C++](https://tensorflow.org/lite/api_docs/cc)

```python
# Using C API directly
objc_library(
  deps = [
      "//tensorflow/lite/c:c_api",
  ],
)

# Using C++ API directly
objc_library(
  deps = [
      "//tensorflow/lite:framework",
  ],
)
```

### Importe a biblioteca

Para arquivos Swift, importe o módulo TensorFlow Lite:

```swift
import TensorFlowLite
```

Para arquivos Objective-C, importe o cabeçalho guarda-chuva:

```objectivec
#import "TFLTensorFlowLite.h"
```

Ou importe o módulo se você definir `CLANG_ENABLE_MODULES = YES` em seu projeto do Xcode:

```objectivec
@import TFLTensorFlowLite;
```

Observação: para desenvolvedores que usam o CocoaPods e desejam importar o módulo TensorFlow Lite para Objective-C, é preciso incluir `use_frameworks!` no `Podfile`.

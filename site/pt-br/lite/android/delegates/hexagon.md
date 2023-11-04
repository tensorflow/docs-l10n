# Delegado Hexagon do TensorFlow Lite

Este documento explica como usar o delegado Hexagon do TensorFlow Lite no seu aplicativo usando a API Java e/ou C. O delegado usa a biblioteca Qualcomm Hexagon para executar kernels quantizados no DSP. Observe que o delegado visa *complementar* a funcionalidade NNAPI, principalmente para dispositivos em que a aceleração de DSP NNAPI não está disponível (por exemplo, em dispositivos mais antigos ou que ainda não têm um driver NNAPI de DSP).

Observação: esse delegado está em fase experimental (beta).

**Dispositivos compatíveis:**

No momento, há suporte para a seguinte arquitetura Hexagon, incluindo, sem limitação:

- Hexagon 680
    - Exemplos de SoC: Snapdragon 821, 820, 660
- Hexagon 682
    - Exemplo de SoC: Snapdragon 835
- Hexagon 685
    - Exemplos de SoC: Snapdragon 845, Snapdragon 710, QCS410, QCS610, QCS605, QCS603
- Hexagon 690
    - Exemplos de SoC: Snapdragon 855, RB5

**Modelos compatíveis:**

O delegado Hexagon é compatível com todos os modelos em conformidade com nossa [especificação de quantização simétrica de 8 bits](https://www.tensorflow.org/lite/performance/quantization_spec), incluindo aqueles gerados usando a [quantização de números inteiros pós-treinamento](https://www.tensorflow.org/lite/performance/post_training_integer_quant). Os modelos UInt8 treinados com o programa legado de [treinamento consciente de quantização](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize) também são compatíveis, por exemplo, [estas versões quantizadas](https://www.tensorflow.org/lite/guide/hosted_models#quantized_models) na nossa página Modelos hospedados.

## API Java do delegado Hexagon

```java
public class HexagonDelegate implements Delegate, Closeable {

  /*
   * Creates a new HexagonDelegate object given the current 'context'.
   * Throws UnsupportedOperationException if Hexagon DSP delegation is not
   * available on this device.
   */
  public HexagonDelegate(Context context) throws UnsupportedOperationException


  /**
   * Frees TFLite resources in C runtime.
   *
   * User is expected to call this method explicitly.
   */
  @Override
  public void close();
}
```

### Exemplo de uso

#### Etapa 1. Edite o aplicativo/build.gradle para que use o AAR do delegado Hexagon noturno

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### Etapa 2. Adicione as bibliotecas Hexagon ao seu aplicativo Android

- Baixe e execute hexagon_nn_skel.run. Ele deve fornecer 3 bibliotecas compartilhadas diferentes: "libhexagon_nn_skel.so", "libhexagon_nn_skel_v65.so" e "libhexagon_nn_skel_v66.so"
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

Observação: você precisará aceitar o contrato de licença.

Observação: desde 02/23/2021, você deve usar a v1.20.0.1.

Observação: você precisa usar as bibliotecas hexagon_nn com a versão compatível da biblioteca de interface. A biblioteca de interface faz parte do AAR e é buscada pelo bazel através da [configuração](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl). A versão na configuração do bazel é a que você deve usar.

- Inclua todas as 3 no seu aplicativo com outras bibliotecas compartilhadas. Confira [Como adicionar a biblioteca compartilhada ao seu aplicativo](#how-to-add-shared-library-to-your-app). O delegado escolherá automaticamente a que tiver o melhor desempenho dependendo do dispositivo.

Observação: se o seu aplicativo será criado tanto para dispositivos ARM de 32 bits quanto de 64, você precisará adicionar as bibliotecas compartilhadas do Hexagon a ambas as pastas de 32 e 64 bits das bibliotecas.

#### Etapa 3. Crie um delegado e inicialize um interpretador do TensorFlow Lite

```java
import org.tensorflow.lite.HexagonDelegate;

// Create the Delegate instance.
try {
  hexagonDelegate = new HexagonDelegate(activity);
  tfliteOptions.addDelegate(hexagonDelegate);
} catch (UnsupportedOperationException e) {
  // Hexagon delegate is not supported on this device.
}

tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);

// Dispose after finished with inference.
tfliteInterpreter.close();
if (hexagonDelegate != null) {
  hexagonDelegate.close();
}
```

## API C do delegado Hexagon

```c
struct TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the Hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;
  // This corresponds to powersave_level in the Hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;
  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;
  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate*
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
Void TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
Void TfLiteHexagonTearDown();
```

### Exemplo de uso

#### Etapa 1. Edite o aplicativo/build.gradle para que use o AAR do delegado Hexagon noturno

```
dependencies {
  ...
  implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
  implementation 'org.tensorflow:tensorflow-lite-hexagon:0.0.0-nightly-SNAPSHOT'
}
```

#### Etapa 2. Adicione as bibliotecas Hexagon ao seu aplicativo Android

- Baixe e execute hexagon_nn_skel.run. Ele deve fornecer 3 bibliotecas compartilhadas diferentes: "libhexagon_nn_skel.so", "libhexagon_nn_skel_v65.so" e "libhexagon_nn_skel_v66.so"
    - [v1.10.3](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_1_10_3_1.run)
    - [v1.14](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.14.run)
    - [v1.17](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.17.0.0.run)
    - [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.0.run)
    - [v1.20.0.1](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)

Observação: você precisará aceitar o contrato de licença.

Observação: desde 02/23/2021, você deve usar a v1.20.0.1.

Observação: você precisa usar as bibliotecas hexagon_nn com a versão compatível da biblioteca de interface. A biblioteca de interface faz parte do AAR e é buscada pelo bazel através da [configuração](https://github.com/tensorflow/tensorflow/blob/master/third_party/hexagon/workspace.bzl). A versão na configuração do bazel é a que você deve usar.

- Inclua todas as 3 no seu aplicativo com outras bibliotecas compartilhadas. Confira [Como adicionar a biblioteca compartilhada ao seu aplicativo](#how-to-add-shared-library-to-your-app). O delegado escolherá automaticamente a que tiver o melhor desempenho dependendo do dispositivo.

Observação: se o seu aplicativo será criado tanto para dispositivos ARM de 32 bits quanto de 64, você precisará adicionar as bibliotecas compartilhadas do Hexagon a ambas as pastas de 32 e 64 bits das bibliotecas.

#### Etapa 3. Inclua o cabeçalho C

- O arquivo de cabeçalho "hexagon_delegate.h" pode ser baixado do [GitHub](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/hexagon_delegate.h) ou extraído do AAR do delegado Hexagon.

#### Etapa 4. Crie um delegado e inicialize um interpretador do TensorFlow Lite

- No seu código, verifique se a biblioteca Hexagon nativa está carregada. Isso pode ser feito ao chamar `System.loadLibrary("tensorflowlite_hexagon_jni");`<br> na sua Atividade ou ponto de entrada Java.

- Crie um delegado, por exemplo:

```c
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Assuming shared libraries are under "/data/local/tmp/"
// If files are packaged with native lib in android App then it
// will typically be equivalent to the path provided by
// "getContext().getApplicationInfo().nativeLibraryDir"
const char[] library_directory_path = "/data/local/tmp/";
TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
::tflite::TfLiteHexagonDelegateOptions params = {0};
// 'delegate_ptr' Need to outlive the interpreter. For example,
// If use case will need to resize input or anything that can trigger
// re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
    ::tflite::TfLiteHexagonDelegateDelete(delegate);
  });
interpreter->ModifyGraphWithDelegate(delegate.get());
// After usage of delegate.
TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

## Adicione a biblioteca compartilhada ao seu aplicativo

- Crie a pasta “app/src/main/jniLibs” e um diretório para cada arquitetura alvo. Por exemplo,
    - ARM de 64 bits: `app/src/main/jniLibs/arm64-v8a`
    - ARM de 32 bits: `app/src/main/jniLibs/armeabi-v7a`
- Coloque o .so no diretório correspondente à arquitetura.

Observação: se você estiver usando o App Bundle para publicar seu aplicativo, é recomendável definir android.bundle.enableUncompressedNativeLibs=false no arquivo gradle.properties.

## Feedback

Em caso de problemas, crie um issue do [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) com todos os detalhes necessários para reprodução, incluindo o modelo do smartphone e o board usado (`adb shell getprop ro.product.device` e `adb shell getprop ro.board.platform`).

## Perguntas frequentes

- Quais operações são compatíveis com o delegado?
    - Veja a lista atual de [operações e restrições compatíveis](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/hexagon/README.md)
- Como posso saber se o modelo está usando o DSP ao ativar o delegado?
    - Duas mensagens de log serão impressas quando você ativar o delegado: uma para indicar se o delegado foi criado e outra para indicar quantos nós estão sendo executados com o delegado. <br> `Created TensorFlow Lite delegate for Hexagon.` <br> `Hexagon delegate: X nodes delegated out of Y nodes.`
- Todas as operações no modelo precisam ser compatíveis para executar o delegado?
    - Não, o modelo será particionado em subgrafos com base nas operações compatíveis. As incompatíveis serão executadas na CPU.
- Como posso criar o AAR do delegado Hexagon a partir da origem?
    - Use `bazel build -c opt --config=android_arm64 tensorflow/lite/delegates/hexagon/java:tensorflow-lite-hexagon`.
- Por que o delegado Hexagon falha ao inicializar mesmo que meu dispositivo tenha um SoC compatível?
    - Verifique se o seu dispositivo tem mesmo um SoC compatível. Execute `adb shell cat /proc/cpuinfo | grep Hardware` e veja se retorna algo como: "Hardware : Qualcomm Technologies, Inc MSMXXXX".
    - Alguns fabricantes de smartphones usam SoCs diferentes para o mesmo modelo. Por isso, o delegado Hexagon pode só funcionar em alguns dispositivos do mesmo modelo de smartphone, e não em todos.
    - Alguns fabricantes de smartphones restringem intencionalmente o uso do DSP Hexagon nos aplicativos que não são do sistema Android, tornando o delegado Hexagon incapaz de funcionar.
- Meu smartphone tem o acesso ao DSP bloqueado. Fiz root no smartphone e ainda não consigo executar o delegado. O que eu faço?
    - Desative o SELinux enforce ao executar `adb shell setenforce 0`

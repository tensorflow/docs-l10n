# Delegado de aceleração de GPU para iOS

O uso de unidades de processamento gráfico (GPUs) para executar seus modelos de aprendizado de máquina (ML) pode melhorar drasticamente o desempenho do modelo e a experiência do usuário dos seus aplicativos com tecnologia de ML. Nos dispositivos iOS, você pode ativar a execução dos seus modelos com a aceleração de GPU usando um [*delegado*](../../performance/delegates). Os delegados atuam como drivers de hardware para o TensorFlow Lite, permitindo que você execute o código do modelo em processadores com GPU.

Esta página descreve como ativar a aceleração de GPU para os modelos do TensorFlow Lite nos apps para iOS. Confira mais informações sobre como usar o delegado de GPU para o TensorFlow Lite, incluindo práticas recomendadas e técnicas avançadas, na página [delegados de GPU](../../performance/gpu).

## Use o GPU com a API Interpreter

A [API Interpreter](../../api_docs/swift/Classes/Interpreter) do TensorFlow Lite conta com um conjunto de APIs de finalidade geral para criar aplicativos de aprendizado de máquina. As instruções abaixo mostram como adicionar suporte a GPUs em um aplicativo para iOS. Este guia pressupõe que você já tenha um aplicativo para iOS que consiga executar um modelo de ML com o TensorFlow Lite.

Observação: caso você ainda não tenha um aplicativo para iOS que use o TensorFlow Lite, confira o [Guia de início rápido para iOS](https://www.tensorflow.org/lite/guide/ios) e compile o aplicativo de demonstração. Após concluir o tutorial, você pode seguir as instruções aqui para acrescentar suporte a GPUs.

### Modifique o Podfile para incluir suporte a GPUs

A partir do TensorFlow Lite versão 2.3.0, o delegado de GPU é excluído do pod para reduzir o tamanho do binário. Você pode incluí-lo especificando uma subespecificação do pod `TensorFlowLiteSwift`:

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

OU

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

Você também pode usar `TensorFlowLiteObjC` ou `TensorFlowLiteC` se quiser utilizar a API do Objective-C, que está disponível nas versões 2.4.0 e posteriores, ou a API do C.

Observação: para o TensorFlow Lite versões 2.1.0 a 2.2.0, o delegado de GPU está *incluído* no pod `TensorFlowLiteC`. Você pode escolher entre `TensorFlowLiteC` e `TensorFlowLiteSwift`, dependendo da linguagem de programação utilizada.

### Inicialize e use o delegado de GPU

Você pode usar o delegado de GPU com a [API Interpreter](../../api_docs/swift/Classes/Interpreter) do TensorFlow Lite com diversas linguagens de programação. É recomendável utilizar Swift e Objective-C, mas também é possível usar C++ e C. É obrigatório usar o C se você estiver utilizando uma versão do TensorFlow Lite abaixo da 2.4. Os exemplos de código abaixo mostram como usar o delegado com cada uma dessas linguagens.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">import TensorFlowLite

// Load model ...

// Initialize TensorFlow Lite interpreter with the GPU delegate.
let delegate = MetalDelegate()
if let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate]) {
  // Run inference ...
}
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">// Import module when using CocoaPods with module support
@import TFLTensorFlowLite;

// Or import following headers manually
#import "tensorflow/lite/objc/apis/TFLMetalDelegate.h"
#import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

// Initialize GPU delegate
TFLMetalDelegate* metalDelegate = [[TFLMetalDelegate alloc] init];

// Initialize interpreter with model path and GPU delegate
TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
NSError* error = nil;
TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                initWithModelPath:modelPath
                                          options:options
                                        delegates:@[ metalDelegate ]
                                            error:&amp;error];
if (error != nil) { /* Error handling... */ }

if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
if (error != nil) { /* Error handling... */ }

// Run inference ...
      </pre>
    </section>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-cpp">// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr&lt;Interpreter&gt; interpreter;
InterpreterBuilder(*model, op_resolver)(&amp;interpreter);

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(/*default options=*/nullptr);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter-&gt;typed_input_tensor&lt;float&gt;(0));
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter-&gt;typed_output_tensor&lt;float&gt;(0));

// Clean up.
TFLGpuDelegateDelete(delegate);
      </pre>
    </section>
    <section>
      <h3>C (antes da versão 2.4.0)</h3>
      <p></p>
<pre class="prettyprint lang-c">#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"

// Initialize model
TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

// Initialize interpreter with GPU delegate
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
TfLiteDelegate* delegate = TFLGPUDelegateCreate(nil);  // default config
TfLiteInterpreterOptionsAddDelegate(options, metal_delegate);
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
TfLiteInterpreterOptionsDelete(options);

TfLiteInterpreterAllocateTensors(interpreter);

NSMutableData *input_data = [NSMutableData dataWithLength:input_size * sizeof(float)];
NSMutableData *output_data = [NSMutableData dataWithLength:output_size * sizeof(float)];
TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

// Run inference
TfLiteTensorCopyFromBuffer(input, inputData.bytes, inputData.length);
TfLiteInterpreterInvoke(interpreter);
TfLiteTensorCopyToBuffer(output, outputData.mutableBytes, outputData.length);

// Clean up
TfLiteInterpreterDelete(interpreter);
TFLGpuDelegateDelete(metal_delegate);
TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

#### Notas de uso das linguagens de API de GPU

- Versões do TensorFlow Lite anteriores à 2.4.0 só podem usar a API do C para Objective-C.
- A API do C++ está disponível somente ao usar o Bazel ou se você compilar o TensorFlow Lite. A API do C++ não pode ser usada com CocoaPods.
- Ao usar o TensorFlow Lite com o delegado de GPU e C++, obtenha o delegado de GPU pela função `TFLGpuDelegateCreate()` e depois passe-o para `Interpreter::ModifyGraphWithDelegate()` em vez de chamar `Interpreter::AllocateTensors()`.

### Compile e teste com o modo de release

Altere para uma build de release com as configurações apropriadas do acelerador da API Metal para obter um maior desempenho e para testes finais. Esta seção explica como ativar uma build de release e definir as configurações de aceleração Metal.

Observação: para acompanhar estas instruções, é necessário ter o XCode v.10.1 ou posterior.

Para mudar para uma build de release:

1. Para editar as configurações da build, selecione **Product &gt; Scheme &gt; Edit Scheme...** (Produto &gt; Esquema &gt; Editar esquema...) e depois selecione **Run** (Executar).
2. Na guia **Info** (Informações), altere **Build Configuration** (Configuração da build) para **Release** e desmarque **Debug executable** (Depurar executável). ![setting up release](../../../images/lite/ios/iosdebug.png)
3. Clique na guia **Options** (Opções) e altere **GPU Frame Capture** (Captura de quadro de GPU) para **Disabled** (Desativada) e **Metal API Validation** (Validação da API Metal) para **Disabled** (Desativada).<br> ![setting up metal options](../../../images/lite/ios/iosmetal.png)
4. Você deve selecionar Release-only builds on 64-bit architecture (Builds somente release em arquitetura de 64 bits). Em **Project navigator &gt; tflite_camera_example &gt; PROJECT &gt; your_project_name &gt; Build Settings** (Navegador do projeto &gt; tflite_camera_example &gt; PROJETO &gt; nome_do_seu_projeto &gt; Configurações da build), defina **Build Active Architecture Only &gt; Release** (Compilar somente arquitetura ativa &gt; Release) como **Yes** (Sim). ![setting up release options](../../../images/lite/ios/iosrelease.png)

## Suporte avançado à GPU

Esta seção fala sobre usos avançados de delegado de GPU para iOS, incluindo opções de delegado, buffers de entrada e saída e uso de modelos quantizados.

### Opções de delegado para iOS

O construtor do delegado de GPU recebe uma `struct` de opções na [API da Swift](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift), na [API do Objective-C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h) e na [API do C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h). Ao passar `nullptr` (API do C) ou nada (APIs do Objective-C e da Swift) ao inicializador, as opções padrão são definidas (o que é explicado no exemplo Uso básico acima).

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">// THIS:
var options = MetalDelegate.Options()
options.isPrecisionLossAllowed = false
options.waitType = .passive
options.isQuantizationEnabled = true
let delegate = MetalDelegate(options: options)

// IS THE SAME AS THIS:
let delegate = MetalDelegate()
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">// THIS:
TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
options.precisionLossAllowed = false;
options.waitType = TFLMetalDelegateThreadWaitTypePassive;
options.quantizationEnabled = true;

TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] initWithOptions:options];

// IS THE SAME AS THIS:
TFLMetalDelegate* delegate = [[TFLMetalDelegate alloc] init];
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">// THIS:
const TFLGpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive,
  .enable_quantization = true,
};

TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);

// IS THE SAME AS THIS:
TfLiteDelegate* delegate = TFLGpuDelegateCreate(nullptr);
      </pre>
    </section>
  </devsite-selector>
</div>

Dica: embora seja conveniente usar `nullptr` ou os construtores padrão, você deve definir explicitamente as opções para evitar qualquer comportamento inesperado se os valores padrão forem alterados no futuro.

### Buffers de entrada/saída usando a API do C++

Fazer computação na GPU requer que os dados estejam disponíveis para a GPU. Em geral, este requisito exige que você faça uma cópia da memória. Se possível, você deve evitar que os dados cruzem a fronteira de memória entre CPU/GPU, pois isso pode levar um tempo considerável. Geralmente, esse cruzamento é inevitável, mas, em alguns casos, um ou o outro pode ser omitido.

Observação: a técnica abaixo está disponível somente ao usar o Bazel ou se você compilar o TensorFlow Lite. A API do C++ não pode ser usada com CocoaPods.

Se a entrada da rede for uma imagem já carregada na memória da GPU (por exemplo: uma textura de GPU contendo o feed da câmera), ela pode permanecer na memória da GPU sem nunca entrar na memória da CPU. De maneira similar, se a saída da rede estiver na forma de uma imagem renderizável, como a operação de [transferência de estilo de imagem](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), você pode exibir o resultado diretamente na tela.

Para obter o melhor desempenho, o TensorFlow Lite possibilita que os usuários leiam e escrevam diretamente no buffer de hardware do TensorFlow, evitando cópias de memória desnecessárias.

Supondo que a entrada de imagem esteja na memória da GPU, primeiro você precisa convertê-la em um objeto `MTLBuffer` para a API Metal. Você pode associar um `TfLiteTensor` a um `MTLBuffer` preparado pelo usuário por meio da função `TFLGpuDelegateBindMetalBufferToTensor()`. Atenção: essa função *precisa* ser chamada após `Interpreter::ModifyGraphWithDelegate()`. Além disso, a saída da inferência é, por padrão, copiada da memória da GPU para a memória da CPU. Para desativar esse comportamento, basta chamar `Interpreter::SetAllowBufferHandleOutput(true)` durante a inicialização.

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-swift">#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"

// ...

// Prepare GPU delegate.
auto* delegate = TFLGpuDelegateCreate(nullptr);

if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

interpreter-&gt;SetAllowBufferHandleOutput(true);  // disable default gpu-&gt;cpu copy
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter-&gt;inputs()[0], user_provided_input_buffer)) {
  return false;
}
if (!TFLGpuDelegateBindMetalBufferToTensor(
        delegate, interpreter-&gt;outputs()[0], user_provided_output_buffer)) {
  return false;
}

// Run inference.
if (interpreter-&gt;Invoke() != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

Quando o comportamento padrão é desativado, copiar a saída da inferência da memória da GPU para a memória da CPU requer uma chamada explícita a `Interpreter::EnsureTensorDataIsReadable()` para cada tensor de saída. Essa estratégia também funciona para modelos quantizados, mas você ainda precisa usar um **buffer float32 com dados float32**, pois esse buffer é vinculado ao buffer interno dequantizado.

### Modelos quantizados {:#quantized-models}

As bibliotecas de delegados de GPU do iOS *são compatíveis com os modelos quantizados por padrão*. Você não precisa fazer nenhuma alteração no código para usar modelos quantizados com o delegado de GPU. A seção a seguir explica como desativar o suporte quantizado para testes ou fins experimentais.

#### Desative o suporte a modelos quantizados

O código a seguir mostra como ***desativar*** o suporte a modelos quantizados.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">    var options = MetalDelegate.Options()
    options.isQuantizationEnabled = false
    let delegate = MetalDelegate(options: options)
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">    TFLMetalDelegateOptions* options = [[TFLMetalDelegateOptions alloc] init];
    options.quantizationEnabled = false;
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">    TFLGpuDelegateOptions options = TFLGpuDelegateOptionsDefault();
    options.enable_quantization = false;

    TfLiteDelegate* delegate = TFLGpuDelegateCreate(options);
      </pre>
    </section>
  </devsite-selector>
</div>

Para mais informações sobre como executar modelos quantizados com a aceleração de GPU, confira a visão geral do [delegado de GPU](../../performance/gpu#quantized-models).

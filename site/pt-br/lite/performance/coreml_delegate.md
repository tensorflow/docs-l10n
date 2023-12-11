# Delegado Core ML do TensorFlow Lite

O delegado Core ML do TensorFlow Lite permite executar os modelos do TensorFlow Lite no [framework Core ML](https://developer.apple.com/documentation/coreml), o que resulta em uma inferência de modelo mais rápida em dispositivos iOS.

Observação: esse delegado está em fase experimental (beta). Ele está disponível a partir do TensorFlow Lite 2.4.0 e nas versões noturnas mais recentes.

Observação: o delegado Core ML é compatível com a versão 2 ou mais recente do Core ML.

**Versões e dispositivos iOS compatíveis:**

- iOS 12 e mais recente. Nas versões mais antigas do iOS, o delegado Core ML usará automaticamente a CPU.
- Por padrão, o delegado Core ML só é ativado em dispositivos com SoC A12 e mais recente (a partir do iPhone Xs) para usar o Neural Engine para uma inferência mais rápida. Se você também quiser usar o delegado Core ML em dispositivos mais antigos, veja as [práticas recomendadas](#best-practices).

**Modelos compatíveis**

No momento, o delegado Core ML é compatível com modelos float (FP32 e FP16).

## Teste o delegado Core ML no seu próprio modelo

O delegado Core ML já está incluso na versão noturna do TensorFlow Lite CocoaPods. Para usar o delegado Core ML, altere seu pod do TensorFlow Lite para incluir a subspec `CoreML` no seu `Podfile`.

Observação: se você quiser usar a API C em vez da API Objective-C, é possível incluir o pod `TensorFlowLiteC/CoreML` para fazer isso.

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

OU

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

Observação: o delegado Core ML também pode usar a API C para código da Objective-C. Antes da versão 2.4.0 do TensorFlow Lite, essa era a única opção.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">
    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter

    // Core ML delegate will only be created for devices with Neural Engine
    if coreMLDelegate != nil {
      interpreter = try Interpreter(modelPath: modelPath,
                                    delegates: [coreMLDelegate!])
    } else {
      interpreter = try Interpreter(modelPath: modelPath)
    }
  </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">

    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    # import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"
    # import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize Core ML delegate
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc] init];

    // Initialize interpreter with model path and Core ML delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ coreMLDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
  </pre>
    </section>
    <section>
      <h3>C (até 2.3.0)</h3>
      <p></p>
<pre class="prettyprint lang-c">
    #include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

    // Initialize interpreter with model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with Core ML delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(NULL);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    // Run inference ...

    /* ... */

    // Dispose resources when it is no longer used.
    // Add following code to the section where you dispose of the delegate
    // (e.g. `dealloc` of class).

    TfLiteInterpreterDelete(interpreter);
    TfLiteCoreMlDelegateDelete(delegate);
    TfLiteModelDelete(model);
      </pre>
    </section>
  </devsite-selector>
</div>

## Práticas recomendadas

### Use o delegado Core ML em dispositivos sem Neural Engine

Por padrão, o delegado Core ML só será criado se o dispositivo tiver Neural Engine, e ele retornará `null` se o delegado não for criado. Se você quiser executar o delegado Core ML em outros ambientes (por exemplo, simulador), passe `.all` como uma opção ao criar o delegado em Swift. Em C++ (e Objective-C). Você pode passar `TfLiteCoreMlDelegateAllDevices`. O exemplo a seguir mostra como fazer isso:

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">
    var options = CoreMLDelegate.Options()
    options.enabledDevices = .all
    let coreMLDelegate = CoreMLDelegate(options: options)!
    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [coreMLDelegate])
      </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">
    TFLCoreMLDelegateOptions* coreMLOptions = [[TFLCoreMLDelegateOptions alloc] init];
    coreMLOptions.enabledDevices = TFLCoreMLDelegateEnabledDevicesAll;
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc]
                                          initWithOptions:coreMLOptions];

    // Initialize interpreter with delegate
  </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">
    TfLiteCoreMlDelegateOptions options;
    options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

### Use o delegado Metal (GPU) como fallback

Quando o delegado Core ML não for criado, como alternativa, você ainda poderá usar o [delegado Metal](https://www.tensorflow.org/lite/performance/gpu#ios) para melhorar o desempenho. O exemplo a seguir mostra como fazer isso:

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p></p>
<pre class="prettyprint lang-swift">
    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }

    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate!])
  </pre>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p></p>
<pre class="prettyprint lang-objc">
    TFLDelegate* delegate = [[TFLCoreMLDelegate alloc] init];
    if (!delegate) {
      // Add Metal delegate options if necessary
      delegate = [[TFLMetalDelegate alloc] init];
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
    <section>
      <h3>C</h3>
      <p></p>
<pre class="prettyprint lang-c">
    TfLiteCoreMlDelegateOptions options = {};
    delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    if (delegate == NULL) {
      // Add Metal delegate options if necessary
      delegate = TFLGpuDelegateCreate(NULL);
    }
    // Initialize interpreter with delegate
      </pre>
    </section>
  </devsite-selector>
</div>

A lógica de criação de delegado lê o id da máquina do dispositivo (por exemplo, iPhone11,1) para determinar a disponibilidade do Neural Engine. Veja mais detalhes no [código](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.mm). Como alternativa, você pode implementar seu próprio conjunto de dispositivos bloqueados usando outras bibliotecas como [DeviceKit](https://github.com/devicekit/DeviceKit).

### Use uma versão mais antiga do Core ML

Embora o iOS 13 seja compatível com o Core ML 3, o modelo pode funcionar melhor quando convertido com a especificação de modelo do Core ML 2. A versão de destino da conversão é definida como a mais recente por padrão, mas você pode alterar isso ao definir `coreMLVersion` (em Swift, `coreml_version` na API C) na opção de delegado como uma versão mais antiga.

## Ops compatíveis

As seguintes ops são compatíveis com o delegado Core ML.

- Add
    - Somente determinados formatos podem fazer o broadcasting. No layout de tensor do Core ML, os seguintes formatos de tensor podem fazer o broadcasting: `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- AveragePool2D
- Concat
    - A concatenação deve ser feita ao longo do eixo do canal.
- Conv2D
    - Os pesos e bias devem ser constantes.
- DepthwiseConv2D
    - Os pesos e bias devem ser constantes.
- FullyConnected (também conhecida como Dense ou InnerProduct)
    - Os pesos e bias (se presentes) devem ser constantes.
    - Só aceita casos de um único lote. As dimensões de entrada devem ser 1, exceto para a última dimensão.
- Hardswish
- Logistic (também conhecida como Sigmoid)
- MaxPool2D
- MirrorPad
    - Só aceita uma entrada 4D com o modo `REFLECT`. O preenchimento deve ser constante e é permitido somente para dimensões H e W.
- Mul
    - Somente determinados formatos podem fazer o broadcasting. No layout de tensor do Core ML, os seguintes formatos de tensor podem fazer o broadcasting: `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- Pad e PadV2
    - Só aceita uma entrada 4D. O preenchimento deve ser constante e é permitido somente para dimensões H e W.
- Relu
- ReluN1To1
- Relu6
- Reshape
    - Só é compatível se a versão Core ML de destino for 2, e não ao segmentar o Core ML 3.
- ResizeBilinear
- SoftMax
- Tanh
- TransposeConv
    - Os pesos devem ser constantes.

## Feedback

Em caso de problemas, crie um issue do [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) com todos os detalhes necessários para reprodução.

## Perguntas frequentes

- O delegado Core ML aceita fallback de CPU se um grafo contiver ops incompatíveis?
    - Sim.
- O delegado Core ML funciona no Simulador de iOS?
    - Sim. A biblioteca inclui destinos x86 e x86_64 para que possa ser executada em um simulador, mas você não verá melhoria no desempenho pela CPU.
- O TensorFlow Lite e o delegado Core ML são compatíveis com o MacOS?
    - O TensorFlow Lite só foi testado no iOS, não no MacOS.
- São compatíveis ops do TF Lite personalizadas?
    - Não, o delegado Core ML não é compatível com ops personalizadas e será usada a CPU.

## APIs

- [API Swift do delegado Core ML](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
- [API C do delegado Core ML](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    - Pode ser usada para códigos Objective-C. ~~~

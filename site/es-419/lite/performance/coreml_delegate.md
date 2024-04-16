# Delegado Core ML de Tensorflow Lite

El delegado Core ML de TensorFlow Lite permite ejecutar modelos TensorFlow Lite en [Core ML framework](https://developer.apple.com/documentation/coreml), lo que resulta en una inferencia de modelos más rápida en dispositivos iOS.

Nota: Este delegado se encuentra en fase experimental (beta). Está disponible a partir de TensorFlow Lite 2.4.0 y las últimas versiones nocturnas.

Nota: El delegado Core ML es compatible con Core ML versión 2 y posteriores.

**Versiones de iOS y dispositivos compatibles:**

- iOS 12 y posteriores. En las versiones anteriores de iOS, el delegado Core ML revertirá automáticamente a la CPU.
- De forma predeterminada, el delegado Core ML solo estará habilitado en dispositivos con SoC A12 y posteriores (iPhone Xs y posteriores) para usar Neural Engine para una inferencia más rápida. Si desea usar el delegado Core ML también en los dispositivos más antiguos, consulte las [buenas prácticas](#best-practices).

**Modelos compatibles**

El delegado Core ML admite actualmente modelos flotantes (FP32 y FP16).

## Probar el delegado Core ML en su propio modelo

El delegado Core ML ya está incluido en la versión nocturna de los CocoaPods de TensorFlow lite. Para usar el delegado Core ML, cambie su pod TensorFlow lite para incluir el subespec `CoreML` en su `Podfile`.

Nota: Si desea usar la API de C en lugar de la de Objective-C, puede incluir el pod `TensorFlowLiteC/CoreML` para hacerlo.

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

O

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

Nota: El delegado Core ML también puede usar la API de C para el código Objective-C. Antes del lanzamiento de TensorFlow Lite 2.4.0, esta era la única opción.

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
      <h3>C (hasta 2.3.0)</h3>
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

## Prácticas recomendadas

### Uso del delegado Core ML en dispositivos sin motor neuronal

De forma predeterminada, el delegado Core ML sólo se creará si el dispositivo dispone de motor neuronal, y devolverá `null` si no se crea el delegado. Si desea ejecutar el delegado Core ML en otros entornos (por ejemplo, simulador), pase `.all` como opción mientras crea el delegado en Swift. En C++ (y Objective-C), puede pasar `TfLiteCoreMlDelegateAllDevices`. El siguiente ejemplo muestra cómo hacerlo:

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

### Usar el delegado Metal(GPU) como alternativa de retroceso.

Si no se crea el delegado Core ML, puede seguir usando el delegado [Metal](https://www.tensorflow.org/lite/performance/gpu#ios) para obtener prestaciones de rendimiento. El siguiente ejemplo muestra cómo hacerlo:

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

La lógica de creación de delegados lee el id de máquina del dispositivo (por ejemplo, iPhone11,1) para determinar la disponibilidad de su motor neuronal. Consulte el [código](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.mm) para obtener más detalles. Como alternativa, puede implementar su propio conjunto de dispositivos de la denilista usando otras librerías como [DeviceKit](https://github.com/devicekit/DeviceKit).

### Usando una versión más antigua de Core ML

Aunque iOS 13 es compatible con Core ML 3, el modelo podría funcionar mejor si se convierte con la especificación de modelo Core ML 2. La versión de conversión objetivo está predeterminada a la última versión, pero puede cambiarla estableciendo `coreMLVersion` (en Swift, `coreml_version` en la API de CA) en la opción de delegado a una versión más antigua.

## Ops compatibles

Las siguientes ops son admitidas por el delegado Core ML.

- Add
    - Sólo ciertas formas son difundibles. En el diseño de tensor Core ML, las siguientes formas de tensor son difundibles. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- AveragePool2D
- Concat
    - La concatenación debe hacerse a lo largo del eje del canal.
- Conv2D
    - Las ponderaciones y los sesgos deben ser constantes.
- DepthwiseConv2D
    - Las ponderaciones y los sesgos deben ser constantes.
- FullyConnected (conocida como Dense o InnerProduct)
    - Las ponderaciones y los sesgos (si los hay) deben ser constantes.
    - Sólo admite el caso de lote único. Las dimensiones de entrada deben ser 1, excepto la última dimensión.
- Hardswish
- Logistic (conocida como Sigmoid)
- MaxPool2D
- MirrorPad
    - Sólo se admite la entrada 4D con el modo `REFLECT`. El amortiguado debe ser constante, y sólo se permite para las dimensiones H y W.
- Mul
    - Sólo ciertas formas son difundibles. En el diseño de tensor Core ML, las siguientes formas de tensor son difundibles. `[B, C, H, W]`, `[B, C, 1, 1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
- Pad y PadV2
    - Sólo se admite la entrada 4D. El amortiguado debe ser constante, y sólo se permite para las dimensiones H y W.
- Relu
- ReluN1To1
- Relu6
- Reshape
    - Sólo se admite cuando la versión Core ML objetivo es 2, no se admite cuando el objetivo es Core ML 3.
- ResizeBilinear
- SoftMax
- Tanh
- TransposeConv
    - Las ponderaciones deben ser constantes.

## Comentarios

Para las incidencias, cree una incidencia en [GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md) con todos los detalles necesarios para reproducirla.

## Preguntas frecuentes

- ¿Admite el delegado CoreML retroceder a la CPU si un grafo contiene ops no soportadas?
    - Sí
- ¿Funciona el delegado CoreML en el simulador iOS?
    - Sí. La librería incluye objetivos x86 y x86_64 para que pueda ejecutarse en un simulador, pero no verá un aumento del rendimiento respecto a la CPU.
- ¿Son compatibles TensorFlow Lite y el delegado CoreML con MacOS?
    - TensorFlow Lite sólo está probado en iOS pero no en MacOS.
- ¿Se admiten las ops personalizadas de TF Lite?
    - No, el delegado CoreML no admite ops personalizadas y éstas se revertirán a la CPU.

## APIs

- [API Swift del delegado de ML](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
- [API C del delegado Core ML](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    - Puede usarse para códigos Objective-C. ~~~

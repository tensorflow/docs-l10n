# Delegado de aceleración de GPU para iOS

Usar unidades de procesamiento gráfico (GPU) para ejecutar sus modelos de aprendizaje automático (ML) puede mejorar drásticamente el rendimiento de su modelo y la experiencia de usuario de sus aplicaciones habilitadas para ML. En los dispositivos iOS, puede usar la ejecución acelerada por GPU de sus modelos mediante un [*delegado*](../../performance/delegates). Los delegados actúan como controladores de hardware para TensorFlow Lite, permitiéndole ejecutar el código de su modelo en procesadores GPU.

Esta página describe cómo habilitar la aceleración por GPU para los modelos de TensorFlow Lite en apps de iOS. Para obtener más información sobre cómo usar el delegado de GPU para TensorFlow Lite, incluidas las mejores prácticas y técnicas avanzadas, consulte la página [Delegados de GPU](../../performance/gpu).

## Usar la GPU con la API del intérprete

La [API del Intérprete](../../api_docs/swift/Classes/Interpreter) de TensorFlow Lite ofrece un conjunto de APIs de propósito general para crear aplicaciones de aprendizaje automático. Las siguientes instrucciones le servirán de guía para añadir soporte de GPU a una app iOS. Esta guía asume que usted ya tiene una app iOS que puede ejecutar con éxito un modelo ML con TensorFlow Lite.

Nota: Si aún no tiene una app para iOS que use TensorFlow Lite, siga las instrucciones de [inicio rápido para iOS](https://www.tensorflow.org/lite/guide/ios) y cree la app demo. Tras completar el tutorial, puede seguir estas instrucciones para habilitar la compatibilidad con la GPU.

### Modifique el Podfile para incluir soporte de GPU

Desde la versión 2.3.0 de TensorFlow Lite, el delegado de GPU se excluye del pod para reducir el tamaño del binario. Puede incluirlos especificando una subespec: para el pod `TensorFlowLiteSwift`:

```ruby
pod 'TensorFlowLiteSwift/Metal', '~> 0.0.1-nightly',
```

O

```ruby
pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['Metal']
```

También puede usar `TensorFlowLiteObjC` o `TensorFlowLiteC` si desea usar Objective-C, que está disponible para las versiones 2.4.0 y superiores, o la API de C.

Nota: Para las versiones 2.1.0 a 2.2.0 de TensorFlow Lite, el delegado de GPU está *incluido* en la vaina `TensorFlowLiteC`. Puede seleccionar entre `TensorFlowLiteC` y `TensorFlowLiteSwift` dependiendo del lenguaje de programación que use.

### Inicializar y usar el delegado de la GPU

Puede usar el delegado de GPU con la [API de Intérprete](../../api_docs/swift/Classes/Interpreter) de TensorFlow Lite con varios lenguajes de programación. Los más recomendados son Swift y Objective-C, pero también puede usar C++ y C. Se requiere usar C si utiliza una versión de TensorFlow Lite anterior a la 2.4. Los siguientes ejemplos de código describen cómo usar el delegado con cada uno de estos lenguajes.

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
      <h3>C (previo a la 2.4.0)</h3>
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

#### Notas de uso del lenguaje de la API de la GPU

- Las versiones de TensorFlow Lite anteriores a la 2.4.0 sólo pueden usar la API de C para Objective-C.
- The C++ API is only available when you are using bazel or build TensorFlow Lite by yourself. C++ API can't be used with CocoaPods.La API de C++ sólo está disponible cuando se usa bazel o si usted mismo genera TensorFlow Lite. La API de C++ no se puede usar con CocoaPods.
- Cuando use TensorFlow Lite con el delegado de la GPU con C++, consiga el delegado de la GPU a través de la función `TFLGpuDelegateCreate()` y páselo después a `Interpreter::ModifyGraphWithDelegate()`, en lugar de llamar a `Interpreter::AllocateTensors()`.

### Construir y probar con el modo publicación

Cambie a una compilación a publicarse con la configuración adecuada del acelerador de la API Metal para lograr un mejor rendimiento y para las pruebas finales. Esta sección explica cómo habilitar una compilación de publicación y configurar los ajustes para la aceleración Metal.

Nota: Estas instrucciones requieren XCode v10.1 o posterior.

Para cambiar a una compilación para publicación:

1. Edite la configuración de la compilación seleccionando **Producto &gt; Esquema &gt; Editar esquema...** y, a continuación, seleccione **Ejecutar**.
2. En la pestaña **Info**, cambie **Configuración de compilación** a **Publicación** y desactive **Depurar ejecutable**. ![configurar publicación](../../../images/lite/ios/iosdebug.png)
3. Haga clic en la pestaña **Opciones** y cambie **Captura de fotogramas de la GPU** a **Desactivado** y **Validación de API Metal** a **Desactivado**.<br> ![configurar opciones de metal](../../../images/lite/ios/iosmetal.png)
4. Asegúrese de seleccionar las compilaciones de sólo publicación en la arquitectura de 64 bits. En **Navegador de proyectos &gt; tflite_camera_example &gt; PROJECT &gt; your_project_name &gt; Configuración de compilación** ponga **Compilar sólo arquitectura activa &gt; Publicar** en **Sí**. ![configurar opciones de publicación](../../../images/lite/ios/iosrelease.png)

## Compatibilidad avanzada con GPU

En esta sección se tratan los usos avanzados del delegado de la GPU para iOS, incluidas las opciones del delegado, los búferes de entrada y salida y el uso de modelos cuantizados.

### Opciones de delegado para iOS

El constructor para el delegado GPU acepta un `constructor` de opciones en la [API de Swift](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/MetalDelegate.swift), [API de Objective-C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/objc/apis/TFLMetalDelegate.h), y [API de C](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/metal_delegate.h). Pasar `nullptr` (API de C) o nada (API Objective-C y Swift) al inicializador configura las opciones predeterminadas (que se explican en el ejemplo de uso básico anterior).

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

Consejo: Aunque es conveniente usar `nullptr` o constructores predeterminados, debería configurar explícitamente las opciones para evitar cualquier comportamiento inesperado si se cambian los valores predeterminados en el futuro.

### Búferes de entrada/salida usando la API de C++

Para realizar cálculos en la GPU es necesario que los datos estén a disposición de ésta. A menudo, esto significa que debe realizar una copia de memoria. En la medida de lo posible, debe evitar que los datos crucen el límite de memoria CPU/GPU, ya que esto puede consumir una cantidad de tiempo considerable. Normalmente, este cruce es inevitable, pero en algunos casos especiales, se puede omitir uno u otro.

Nota: La siguiente técnica sólo está disponible cuando está usando Bazel o genera TensorFlow Lite usted mismo. La API de C++ no se puede usar con CocoaPods.

Si la entrada de la red es una imagen ya cargada en la memoria de la GPU (por ejemplo, una textura de la GPU que contiene la alimentación de la cámara), puede permanecer en la memoria de la GPU sin entrar nunca en la memoria de la CPU. Del mismo modo, si la salida de la red es una imagen renderizable, como una operación de [transferencia de estilo de imagen](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), puede mostrar directamente el resultado en pantalla.

Para conseguir el mejor rendimiento, TensorFlow Lite hace posible que los usuarios lean y escriban directamente en el búfer de hardware de TensorFlow y eviten las copias de memoria evitables.

Si asumimos que la imagen de entrada se encuentra en la memoria de la GPU, primero debe convertirla en un objeto `MTLBuffer` para Metal. Puede asociar un `TfLiteTensor` a un `MTLBuffer` preparado por el usuario con la función `TFLGpuDelegateBindMetalBufferToTensor()`. Observe que esta función *debe* llamarse después de `Interpreter::ModifyGraphWithDelegate()`. Además, la salida de inferencia se copia, por defecto, de la memoria de la GPU a la memoria de la CPU. Puede desactivar este comportamiento llamando a `Interpreter::SetAllowBufferHandleOutput(true)` durante la inicialización.

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

Una vez desactivado el comportamiento predeterminado, copiar la salida de inferencia de la memoria de la GPU a la memoria de la CPU requiere una llamada explícita a `Interpreter::EnsureTensorDataIsReadable()` para cada tensor de salida. Este método también funciona para modelos cuantizados, pero sigue siendo necesario usar un **búfer de tamaño float32 con datos float32**, ya que el búfer está vinculado al búfer interno descuantizado.

### Modelos cuantizados  {:#quantized-models}

Las bibliotecas del delegado de la GPU de iOS *soportan modelos cuantizados de forma predeterminada*. No es necesario cambiar el código para usar modelos cuantizados con el delegado de GPU. En la siguiente sección se explica cómo desactivar el soporte cuantizado con fines de prueba o experimentales.

#### Deshabilite el soporte de modelos cuantizados

El siguiente código muestra cómo ***deshabilitar*** el soporte para modelos cuantizados.

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

Para obtener más información sobre la ejecución de modelos cuantizados con aceleración de GPU, consulte la descripción general de [Delegado de GPU](../../performance/gpu#quantized-models).

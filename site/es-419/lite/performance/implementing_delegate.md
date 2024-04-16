# Implementar un delegado personalizado

[TDC]

## ¿Qué es un delegado TensorFlow Lite?

Un [Delegado](https://www.tensorflow.org/lite/performance/delegates) TensorFlow Lite le permite ejecutar sus modelos (en parte o en su totalidad) en otro ejecutor. Este mecanismo puede aprovechar una variedad de aceleradores en el dispositivo, como la GPU o la TPU (Unidad de Procesamiento de Tensores) Edge para la inferencia. Esto da a los desarrolladores un método flexible y desacoplado del TFLite predeterminado para acelerar la inferencia.

El siguiente diagrama resume a los delegados; más detalles en las secciones siguientes.

![Delegados TFLite](images/tflite_delegate.png "TFLite Delegates")

## ¿Cuándo debo crear un delegado personalizado?

TensorFlow Lite dispone de una amplia variedad de delegados para aceleradores objetivo como GPU, DSP, EdgeTPU y frameworks como Android NNAPI.

Crear su propio delegado es útil en los siguientes escenarios:

- Desea integrar un nuevo motor de inferencia de ML no soportado por ningún delegado existente.
- Cuenta con un acelerador de hardware personalizado que mejora el runtime para escenarios conocidos.
- Está desarrollando optimizaciones de la CPU (como la fusión de operarios) que pueden acelerar ciertos modelos.

## ¿Cómo funcionan los delegados?

Considere un grafo modelo simple como el siguiente, y un delegado "MyDelegate" que tiene una implementación más rápida para las operaciones Conv2D y Mean.

![Grafo original](../images/performance/tflite_delegate_graph_1.png "Original Graph")

Después de aplicar este "MyDelegate", el grafo original de TensorFlow Lite se actualizará como se muestra a continuación:

![Grafo con delegado](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

El grafo anterior se obtiene cuando TensorFlow Lite divide el grafo original siguiendo dos reglas:

- Las operaciones específicas que podría manejar el delegado se colocan en una partición sin dejar de satisfacer las dependencias originales del flujo de trabajo informático entre las operaciones.
- Cada partición a ser delegada sólo tiene nodos de entrada y salida que no son manejados por el delegado.

Cada partición manejada por un delegado es reemplazada por un nodo delegado (también puede llamarse kernel delegado) en el grafo original que evalúa la partición en su llamada de invocación.

Dependiendo del modelo, el grafo final puede terminar con uno o más nodos, esto último significa que algunas ops no son soportadas por el delegado. En general, no es conveniente que el delegado se encargue de varias particiones, ya que cada vez que se pasa del delegado al grafo principal, se produce una sobrecarga al pasar los resultados del subgrafo delegado al grafo principal por las copias de memoria (por ejemplo, de la GPU a la CPU). Dicha sobrecarga podría contrarrestar las ganancias de rendimiento, especialmente cuando hay una gran cantidad de copias de memoria.

## Implementar su propio delegado personalizado

El método preferido para añadir un delegado es usar la [API SimpleDelegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h).

Para crear un nuevo delegado, necesita implementar 2 interfaces y ofrecer su propia implementación para los métodos de la interfaz.

### 1 - `SimpleDelegateInterface`

Esta clase representa las capacidades del delegado, qué operaciones son compatibles, y una clase de fábrica para crear un kernel que encapsula el grafo delegado. Para más detalles, consulte la interfaz definida en este [archivo de cabecera C++](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L71). Los comentarios en el código explican cada API en detalle.

### 2 - `SimpleDelegateKernelInterface`

Esta clase encapsula la lógica para inicializar / preparar / y ejecutar la partición delegada.

Tiene: (Véase [definición](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L43))

- Init(...): que se llamará una vez para realizar cualquier inicialización puntual.
- Prepare(...): que se llama para cada instancia diferente de este nodo; esto ocurre si tiene múltiples particiones delegadas. Por lo general, usted quiere hacer asignaciones de memoria aquí, ya que esta sección se llamará cada vez que los tensores se redimensionen.
- Invoke(...): que se llamará para la inferencia.

### Ejemplo

En este ejemplo, creará un delegado muy sencillo que sólo admite 2 tipos de operaciones (ADD) y (SUB) con tensores float32 únicamente.

```
// MyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class MyDelegate : public SimpleDelegateInterface {
 public:
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports Add and Sub ops.
    if (kTfLiteBuiltinAdd != registration->builtin_code &&
        kTfLiteBuiltinSub != registration->builtin_code)
      return false;
    // This delegate only supports float32 types.
    for (int i = 0; i < node->inputs->size; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteFloat32) return false;
    }
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "MyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<MyDelegateKernel>();
  }
};
```

A continuación, cree su propio kernel de delegado heredando de la `SimpleDelegateKernelInterface`

```
// My delegate kernel.
class MyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.

    for (int i = 0; i < inputs_.size(); ++i) {
      // Get the node input tensors.
      // Add/Sub operation accepts 2 inputs.
      auto& input_tensor_1 = context->tensors[inputs_[i][0]];
      auto& input_tensor_2 = context->tensors[inputs_[i][1]];
      auto& output_tensor = context->tensors[outputs_[i][0]];
      TF_LITE_ENSURE_EQ(
          context,
          ComputeResult(context, builtin_code_[i], &input_tensor_1,
                        &input_tensor_2, &output_tensor),
          kTfLiteOk);
    }
    return kTfLiteOk;
  }

 private:
  // Computes the result of addition of 'input_tensor_1' and 'input_tensor_2'
  // and store the result in 'output_tensor'.
  TfLiteStatus ComputeResult(TfLiteContext* context, int builtin_code,
                             const TfLiteTensor* input_tensor_1,
                             const TfLiteTensor* input_tensor_2,
                             TfLiteTensor* output_tensor) {
    if (NumElements(input_tensor_1) != NumElements(input_tensor_2) ||
        NumElements(input_tensor_1) != NumElements(output_tensor)) {
      return kTfLiteDelegateError;
    }
    // This code assumes no activation, and no broadcasting needed (both inputs
    // have the same size).
    auto* input_1 = GetTensorData<float>(input_tensor_1);
    auto* input_2 = GetTensorData<float>(input_tensor_2);
    auto* output = GetTensorData<float>(output_tensor);
    for (int i = 0; i < NumElements(input_tensor_1); ++i) {
      if (builtin_code == kTfLiteBuiltinAdd)
        output[i] = input_1[i] + input_2[i];
      else
        output[i] = input_1[i] - input_2[i];
    }
    return kTfLiteOk;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;
};


```

## Comparar y evaluar al nuevo delegado

TFLite cuenta con un conjunto de herramientas que le permitirán realizar pruebas rápidas con un modelo TFLite.

- [Herramienta Model Benchmark](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark): La herramienta toma un modelo TFLite, genera entradas aleatorias y, a continuación, ejecuta repetidamente el modelo durante un número especificado de ejecuciones. Al final imprime estadísticas de latencia agregadas.
- [Herramienta Inference Diff](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff): Para un modelo dado, la herramienta genera datos gaussianos aleatorios y los pasa a través de dos intérpretes TFLite diferentes, uno ejecutando un kernel de CPU de un solo hilo y el otro usando una especificación definida por el usuario. Mide la diferencia absoluta entre los tensores de salida de cada intérprete, sobre una base por elemento. Esta herramienta también puede ser útil para depurar problemas de precisión.
- También existen herramientas de evaluación de tareas específicas, para la clasificación de imágenes y la detección de objetos. Estas herramientas pueden encontrarse [aquí](https://www.tensorflow.org/lite/performance/delegates#tools_for_evaluation).

Además, TFLite dispone de un amplio conjunto de pruebas de unidad del kernel y op que podrían reutilizarse para probar el nuevo delegado con mayor cobertura y garantizar que no se interrumpe la ruta de ejecución habitual de TFLite.

Para conseguir reutilizar las pruebas y herramientas TFLite para el nuevo delegado, puede usar cualquiera de las dos opciones siguientes:

- Usar el mecanismo [registrador de delegados](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates).
- Usar el mecanismo [delegado externo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external).

### Seleccionar el mejor enfoque

Ambos enfoques requieren algunos cambios, como se detalla a continuación. Sin embargo, el primer enfoque vincula el delegado estáticamente y requiere recompilar las herramientas de prueba, benchmarking y evaluación. En cambio, el segundo hace que el delegado sea una biblioteca compartida y requiere que exponga los métodos de creación/eliminación desde la biblioteca compartida.

Como resultado, el mecanismo de delegado externo funcionará con los [binarios precompilados de la herramienta Tensorflow Lite de TFLite](#download-links-for-nightly-pre-built-tflite-tooling-binaries). Pero es menos explícito y puede ser más complicado de configurar en pruebas de integración automatizadas. Use el enfoque del registrador de delegados para mayor claridad.

### Opción 1: Aprovechar el registrador de delegados

El [registrador de delegados](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates) mantiene una lista de proveedores de delegados, cada uno de los cuales proporciona una manera fácil de crear delegados TFLite basados en Indicadores de línea de comandos, y por lo tanto, son convenientes para las herramientas. Para conectar el nuevo delegado a todas las herramientas de Tensorflow Lite mencionadas anteriormente, primero se crea un nuevo proveedor de delegado como [este](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/hexagon_delegate_provider.cc), y luego sólo se realizan unos pocos cambios en las reglas BUILD. Un ejemplo completo de este proceso de integración se muestra a continuación (y el código se puede encontrar [aquí](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/dummy_delegate)).

Supongamos que tiene un delegado que implementa las API SimpleDelegate, y las API externas "C" de crear/eliminar este delegado "dummy" como se muestra a continuación:

```
// Returns default options for DummyDelegate.
DummyDelegateOptions TfLiteDummyDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options);

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate);
```

Para integrar el "DummyDelegate" con Benchmark Tool e Inference Tool, defina un DelegateProvider como el que se muestra a continuación:

```
class DummyDelegateProvider : public DelegateProvider {
 public:
  DummyDelegateProvider() {
    default_params_.AddParam("use_dummy_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

  std::string GetName() const final { return "DummyDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DummyDelegateProvider);

std::vector<Flag> DummyDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_dummy_delegate", params,
                                              "use the dummy delegate.")};
  return flags;
}

void DummyDelegateProvider::LogParams(const ToolParams& params) const {
  TFLITE_LOG(INFO) << "Use dummy test delegate : ["
                   << params.Get<bool>("use_dummy_delegate") << "]";
}

TfLiteDelegatePtr DummyDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_dummy_delegate")) {
    auto default_options = TfLiteDummyDelegateOptionsDefault();
    return TfLiteDummyDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

```

Las definiciones de las reglas BUILD son importantes, ya que hay que asegurarse de que la biblioteca esté siempre enlazada y no sea descartada por el optimizador.

```
#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "dummy_delegate_provider",
    srcs = ["dummy_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1, # This is required so the optimizer doesn't optimize the library away.
)
```

Añada ahora estas dos reglas contenedoras en su archivo BUILD para crear una versión de Benchmark Tool e Inference Tool, y de otras herramientas de evaluación, que pueda ejecutarse con su propio delegado.

```
cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)
```

También puede insertar este proveedor delegado en las pruebas del kernel de TFLite como se describe [aquí](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#kernel-tests).

### Opción 2: Aprovechar el delegado externo

En esta alternativa, primero se crea un adaptador de delegado externo el [external_delegate_adaptor.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc) como se muestra a continuación. Tenga en cuenta que este enfoque es ligeramente menos preferible que la opción 1, como ya ha sido [anteriormente mencionado](#comparison-between-the-two-options).

```
TfLiteDelegate* CreateDummyDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();

  // Parse key-values options to DummyDelegateOptions.
  // You can achieve this by mimicking them as command-line flags.
  std::unique_ptr<const char*> argv =
      std::unique_ptr<const char*>(new const char*[num_options + 1]);
  constexpr char kDummyDelegateParsing[] = "dummy_delegate_parsing";
  argv.get()[0] = kDummyDelegateParsing;

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.get()[i + 1] = option_args.rbegin()->c_str();
  }

  // Define command-line flags.
  // ...
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(...),
      ...,
      tflite::Flag::CreateFlag(...),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.get(), flag_list)) {
    return nullptr;
  }

  return TfLiteDummyDelegateCreate(&options);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateDummyDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteDummyDelegateDelete(delegate);
}

#ifdef __cplusplus
}
#endif  // __cplusplus
```

Ahora cree el objetivo BUILD correspondiente para compilar una biblioteca dinámica como se muestra a continuación:

```
cc_binary(
    name = "dummy_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)
```

Una vez creado este archivo .so de delegado externo, puede crear binarios o usar los ya creados para ejecutarlos con el nuevo delegado siempre que el binario esté enlazado con la biblioteca [external_delegate_provider](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159) que admite indicadores de línea de comandos como se describe [aquí](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider). Nota: este proveedor externo de delegados ya ha sido enlazado con los binarios de pruebas y herramientas existentes.

Consulte las descripciones [aquí](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#option-2-utilize-tensorflow-lite-external-delegate) para ver una ilustración de cómo evaluar el delegado ficticio mediante este enfoque de delegado externo. Puede usar comandos similares para las herramientas de prueba y evaluación mencionadas anteriormente.

Cabe destacar que el *delegado externo* es la correspondiente implementación en C++ del *delegado* en la vinculación con Python de Tensorflow Lite como se muestra [aquí](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42). Por lo tanto, la biblioteca adaptadora dinámica de delegado externo creada aquí podría usarse directamente con las APIs de Python de Tensorflow Lite.

## Recursos

### Enlaces de descarga para los binarios de herramientas TFLite precompiladas nocturnas

<table>
  <tr>
   <td>OS</td>
   <td>ARCH</td>
   <td>BINARY_NAME</td>
  </tr>
  <tr>
   <td rowspan="3">Linux</td>
   <td>x86_64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>arm</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model">benchmark_model</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td rowspan="2">Android</td>
   <td>arm</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
  <tr>
   <td>aarch64</td>
   <td>
<ul>
<li><a href="http://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model">benchmark_model</a></li>
<li>
<strong><p data-md-type="paragraph"><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk">benchmark_model.apk</a></p></strong>
</li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff">inference_diff</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification">imagenet_image_classification_eval</a></li>
<li><a href="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection">coco_object_detection_eval</a></li>
</ul>
   </td>
  </tr>
</table>

# Implementando um delegado personalizado

[TOC]

## O que é um delegado do TensorFlow Lite?

Um [delegado](https://www.tensorflow.org/lite/performance/delegates) do TensorFlow Lite permite que você execute seus modelos (inteiros ou parte deles) em outro executor. Esse mecanismo pode usar uma variedade de aceleradores no dispositivo, como GPU ou Edge TPU (unidade de processamento de tensor) para inferência. Isso fornece aos desenvolvedores um método flexível e desacoplado do padrão TFLite para acelerar a inferência.

O diagrama abaixo resume os delegados. Veja mais detalhes nas seções a seguir.

![Delegados do TFLite](images/tflite_delegate.png "TFLite Delegates")

## Quando devo criar um delegado personalizado?

O TensorFlow Lite tem uma ampla variedade de delegados para aceleradores alvo, por exemplo, GPU, DSP, EdgeTPU e frameworks como a NNAPI do Android.

É útil criar seu próprio delegado nos seguintes casos:

- Você quer integrar um novo mecanismo de inferência de ML que não é compatível com qualquer delegado existente.
- Você tem um acelerador de hardware personalizado que melhora o runtime para cenários conhecidos.
- Você está desenvolvendo otimizações de CPU (como fusão de operadores) que podem acelerar determinados modelos.

## Como funcionam os delegados?

Considere um grafo de modelo simples como o abaixo e um delegado "MyDelegate" que tem uma implementação mais rápida para operações Conv2D e Mean.

![Grafo original](../images/performance/tflite_delegate_graph_1.png "Original Graph")

Depois de aplicar o "MyDelegate", o grafo original do TensorFlow Lite será atualizado da seguinte maneira:

![Grafo com o delegado](../images/performance/tflite_delegate_graph_2.png "Graph with delegate")

O grafo acima é obtido quando o TensorFlow Lite divide o grafo original em duas regras:

- Operações específicas que podem ser processadas pelo delegado são colocadas em uma partição sem deixar de satisfazer as dependências do fluxo de trabalho de computação original entre as operações.
- Cada partição a ser delegada só tem nós de entrada e saída que não são processados pelo delegado.

Cada partição que é processada por um delegado é substituída por um nó de delegado (também pode ser chamada de kernel de delegado) no grafo original que avalia a partição na sua chamada de invocação.

Dependendo do modelo, o grafo final pode acabar com um ou mais nós, sendo que o último significa que algumas ops não são compatíveis com o delegado. Em geral, não é recomendável ter várias partições processadas pelo delegado, porque, cada vez que você alterna entre o delegado e o grafo principal, ocorre uma sobrecarga para passar os resultados do subgrafo delegado ao grafo principal que resulta das cópias na memória (por exemplo, GPU à CPU). Essa sobrecarga pode anular ganhos de desempenho, especialmente quando há um grande número de cópias na memória.

## Implementando seu próprio delegado personalizado

O método preferencial para adicionar um delegado é usando a [API SimpleDelegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h).

Para criar um novo delegado, você precisa implementar 2 interfaces e fornecer sua própria implementação para os métodos das interfaces.

### 1 - `SimpleDelegateInterface`

Essa classe representa as capacidades do delegado, quais operações são compatíveis e uma classe de fábrica para criar um kernel que encapsula o grafo delegado. Para mais detalhes, veja a interface definida neste [arquivo de cabeçalho C++](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L71). Os comentários no código explicam cada API em detalhes.

### 2 - `SimpleDelegateKernelInterface`

Essa classe encapsula a lógica para inicializar/preparar/executar a partição delegada.

Ela tem: (veja a [definição](https://github.com/tensorflow/tensorflow/blob/8a643858ce174b8bd1b4bb8fa4bfaa62f7e8c45f/tensorflow/lite/delegates/utils/simple_delegate.h#L43))

- Init(...): pode ser chamado uma vez para fazer qualquer inicialização única.
- Prepare(...): chamado para cada instância diferente desse nó — isso acontece se você tiver várias partições delegadas. Geralmente, é recomendável fazer alocações de memória aqui, já que isso será chamado sempre que os tensores forem redimensionados.
- Invoke(...): chamado para a inferência.

### Exemplo

Neste exemplo, você criará um delegado bastante simples que é compatível com apenas 2 tipos de operações (ADD) e (SUB) com somente tensores float32.

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

Em seguida, crie seu próprio kernel de delegado ao herdar de `SimpleDelegateKernelInterface`

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

## Benchmarking e avaliação do novo delegado

O TFLite tem um conjunto de ferramentas que você pode testar rapidamente em um modelo do TFLite.

- [Ferramenta de benchmarking de modelo](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark): a ferramenta aceita um modelo do TFLite, gera entradas aleatórias e executa o modelo repetidamente por um número específico de vezes. Ela imprime as estatísticas de latência agregadas no final.
- [Ferramenta de diff da inferência](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff): para um modelo específico, a ferramenta gera dados gaussianos aleatórios e os passa por dois interpretadores diferentes do TFLite, um que executa um kernel de CPU de thread único e outro que usa uma especificação definida pelo usuário. Ela mede a diferença absoluta entre os tensores de saída de cada interpretador com base em cada elemento. Essa ferramenta também pode ser útil para depurar problemas de exatidão.
- Também há ferramentas de avaliação específicas a tarefas, para classificação de imagens e detecção de objetos. Elas podem ser encontradas [aqui](https://www.tensorflow.org/lite/performance/delegates#tools_for_evaluation).

Além disso, o TFLite tem um grande conjunto de testes unitários de kernels e ops que podem ser reutilizados para testar o novo delegado com mais cobertura e garantir que o caminho de execução regular do TFLite não esteja quebrado.

Para conseguir reutilizar testes e ferramentas do TFLite para o novo delegado, você pode usar uma das seguintes opções:

- Utilizar o mecanismo [registrador de delegados](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates).
- Utilizar o mecanismo [delegado externo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external).

### Escolhendo a melhor abordagem

Ambas as abordagens exigem algumas mudanças, conforme detalhado abaixo. No entanto, a primeira abordagem vincula o delegado estaticamente e exige a reconstrução das ferramentas de teste, benchmarking e avaliação. Em contraste, a segunda transforma o delegado em uma biblioteca compartilhada e exige que você exponha os métodos de criação/exclusão da biblioteca compartilhada.

Como resultado, o mecanismo de delegado externo funcionará com os [binários das ferramentas pré-criadas do TensorFlow Lite](#download-links-for-nightly-pre-built-tflite-tooling-binaries). Porém, é menos explícito e pode ser mais complicado de configurar em testes de integração automatizados. Use a abordagem de registrador de delegados para maior clareza.

### Opção 1: use o registrador de delegados

O [registrador de delegado](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates) mantém uma lista de provedores de delegados, sendo que cada um oferece uma maneira fácil de criar delegados do TFLite com base em flags de linha de comando e, portanto, são convenientes para as ferramentas. Para conectar o novo delegado a todas as ferramentas do TensorFlow mencionadas acima, primeiro crie um novo provedor de delegado como [este](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/hexagon_delegate_provider.cc) e, depois, faça apenas algumas mudanças nas regras de BUILD. Confira abaixo um exemplo completo dessa integração (e o código pode ser encontrado [aqui](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/dummy_delegate)).

Supondo que você tenha um delegado que implementa as APIs SimpleDelegate e as APIs "C" externas de criação/exclusão desse delegado "dummy" (falso) conforme mostrado abaixo:

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

Para integrar o "DummyDelegate" à ferramenta de benchmarking e de inferência, defina um DelegateProvider da seguinte maneira:

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

As definições da regra de BUILD são importantes, porque você precisa garantir que a biblioteca esteja sempre vinculada e não seja descartada pelo otimizador.

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

Agora inclua estas duas regras de wrapper no seu arquivo BUILD para criar uma versão das ferramentas de benchmarking e inferência, além de outras ferramentas de avaliação, que podem ser executadas com seu próprio delegado.

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

Você também pode conectar esse provedor de delegado aos testes de kernels do TFLite conforme descrito [aqui](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#kernel-tests).

### Opção 2: use o delegado externo

Nesta alternativa, primeiro você cria um adaptador de delegado externo, o [external_delegate_adaptor.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc), conforme mostrado abaixo. Observe que essa abordagem é menos recomendável do que a Opção 1, como [mencionado](#comparison-between-the-two-options).

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

Agora crie o BUILD de destino para criar uma biblioteca dinâmica como mostrado a seguir:

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

Depois que esse arquivo .so do delegado externo for criado, você poderá criar binários ou usar pré-criados para a execução com o novo delegado, desde que o binário esteja vinculado a uma biblioteca [external_delegate_provider](https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159) compatível com flags de linha de comando, como descrito [aqui](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider). Observação: esse provedor de delegado externo já foi vinculado aos binários existentes de testes e ferramentas.

Consulte as descrições [aqui](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/README.md#option-2-utilize-tensorflow-lite-external-delegate) para uma ilustração de como fazer o benchmarking do delegado "dummy" por essa abordagem de delegado externo. Você pode usar comandos semelhantes para as ferramentas de teste e avaliação mencionadas anteriormente.

Vale ressaltar que o *delegado externo* é a implementação C++ correspondente do *delegado* na vinculação Python do Tensorflow, como mostrado [aqui](https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42). Portanto, a biblioteca de adaptador do delegado externo dinâmico criada aqui pode ser usada diretamente com as APIs Python do TensorFlow Lite.

## Recursos

### Links para baixar os binários das ferramentas pré-criadas noturnas do TFLite

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

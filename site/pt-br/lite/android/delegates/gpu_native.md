# Delegado de aceleração de GPU com a API C/C++

O uso de unidades de processamento gráfico (GPUs) para executar seus modelos de aprendizado de máquina (ML) pode melhorar drasticamente o desempenho e a experiência do usuário dos seus aplicativos com tecnologia de ML. Nos dispositivos Android, você pode ativar a execução dos seus modelos com a aceleração de GPU usando um [*delegado*](../../performance/delegates) e uma das seguintes APIs:

- API Interpreter - [guia](./gpu)
- API Biblioteca Task - [guia](./gpu_task)
- API nativa (C/C++) - este guia

Este guia aborda os usos avançados do delegado de GPU para a API C e C++, além do uso de modelos quantizados. Para mais informações sobre como usar o delegado de GPU para o TensorFlow Lite, incluindo práticas recomendadas e técnicas avançadas, confira a página [delegados de GPU](../../performance/gpu).

## Ative a aceleração de GPU

Use o delegado de GPU do TensorFlow Lite para Android em C ou C++ ao criar o delegado com `TfLiteGpuDelegateV2Create()` e o destruir com `TfLiteGpuDelegateV2Delete()`, conforme mostrado no código de exemplo a seguir:

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

Revise o código do objeto `TfLiteGpuDelegateOptionsV2` para criar uma instância de delegado com opções personalizadas. Você pode inicializar as opções padrão com `TfLiteGpuDelegateOptionsV2Default()` e, em seguida, modificá-las conforme necessário.

O delegado de GPU do TensorFlow Lite para Android em C ou C++ usa o sistema de build [Bazel](https://bazel.io). Você pode compilar o delegado usando o comando a seguir:

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

Ao chamar `Interpreter::ModifyGraphWithDelegate()` ou `Interpreter::Invoke()`, o autor da chamada precisa ter um `EGLContext` no thread atual, e `Interpreter::Invoke()` precisa ser chamado do mesmo `EGLContext`. Se não houver um `EGLContext`, o delegado cria um internamente, mas, depois, você precisará garantir que `Interpreter::Invoke()` seja sempre chamado do mesmo thread em que `Interpreter::ModifyGraphWithDelegate()` foi chamado.

## Modelos quantizados {:#quantized-models}

As bibliotecas de delegados GPU do Android são compatíveis com os modelos quantizados por padrão. Você não precisa fazer nenhuma alteração no código para usar modelos quantizados com o delegado de GPU. A seção a seguir explica como desativar o suporte quantizado para testes ou fins experimentais.

#### Desative o suporte ao modelo quantizado

O código a seguir mostra como ***desativar*** o suporte a modelos quantizados.

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p></p>
<pre class="prettyprint lang-c++">TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter-&gt;ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre>
    </section>
  </devsite-selector>
</div>

Para mais informações sobre como executar modelos quantizados com a aceleração de GPU, confira a visão geral do [delegado de GPU](../../performance/gpu#quantized-models).
